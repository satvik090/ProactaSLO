import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import PROMETHEUS_URL, SCRAPE_INTERVAL_SECONDS, SERVICES
from app.metric_ingestion import METRIC_NAMES
from app.prediction_engine import compute_drift, train_model
from app.slo_registry import PredictionLog

logger = logging.getLogger(__name__)

SessionFactory: async_sessionmaker | None = None
scheduler = AsyncIOScheduler(timezone=timezone.utc)


def configure_retrainer(session_factory: async_sessionmaker) -> None:
    global SessionFactory
    SessionFactory = session_factory


def start_retrainer(session_factory: async_sessionmaker | None = None) -> AsyncIOScheduler:
    if session_factory is not None:
        configure_retrainer(session_factory)
    if not scheduler.get_job("proactaslo-nightly-retraining"):
        scheduler.add_job(
            run_retraining,
            "cron",
            hour=2,
            minute=0,
            id="proactaslo-nightly-retraining",
            replace_existing=True,
        )
    if not scheduler.running:
        scheduler.start()
    return scheduler


async def run_retraining() -> None:
    if SessionFactory is None:
        logger.error("Retrainer session factory is not configured")
        return

    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)
    for service in SERVICES:
        async with SessionFactory() as session:
            result = await session.execute(
                select(PredictionLog)
                .where(PredictionLog.service_name == service)
                .where(PredictionLog.timestamp >= since)
                .where(PredictionLog.outcome.is_not(None))
                .order_by(PredictionLog.timestamp.asc())
            )
            rows = list(result.scalars().all())

        if len(rows) < 100:
            logger.warning("Skipping retraining for %s: only %s labelled rows in the last 7 days", service, len(rows))
            continue

        vectors: list[list[float]] = []
        labels: list[int] = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for row in rows:
                window_vectors = await _fetch_metric_window(client, service, row.timestamp)
                label = 1 if row.outcome == "true_positive" else 0
                vectors.extend(window_vectors)
                labels.extend([label] * len(window_vectors))

        if len(vectors) <= 30:
            logger.warning("Skipping retraining for %s: only %s metric vectors fetched", service, len(vectors))
            continue

        val_loss, epoch_count = await asyncio.to_thread(train_model, service, vectors, labels)
        drift_score = await compute_drift(service)
        logger.info(
            "Retrained %s rows=%s vectors=%s val_loss=%s epochs=%s drift=%s timestamp=%s",
            service,
            len(rows),
            len(vectors),
            val_loss,
            epoch_count,
            drift_score,
            datetime.now(timezone.utc).isoformat(),
        )


async def _fetch_metric_window(client: httpx.AsyncClient, service: str, timestamp: datetime) -> list[list[float]]:
    end = _as_utc(timestamp)
    start = end - timedelta(seconds=30 * SCRAPE_INTERVAL_SECONDS)
    histories = await asyncio.gather(
        *(_fetch_metric_history(client, service, metric_name, start, end) for metric_name in METRIC_NAMES),
        return_exceptions=True,
    )

    max_len = max((len(values) for values in histories if isinstance(values, list)), default=0)
    rows: list[list[float]] = []
    for index in range(max_len):
        vector: list[float] = []
        for values in histories:
            if isinstance(values, Exception) or index >= len(values):
                vector.append(0.0)
            else:
                vector.append(values[index])
        rows.append(vector)
    return rows


async def _fetch_metric_history(
    client: httpx.AsyncClient,
    service: str,
    metric_name: str,
    start: datetime,
    end: datetime,
) -> list[float]:
    query = f'{metric_name}{{job="{service}"}}'
    try:
        response = await client.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": SCRAPE_INTERVAL_SECONDS,
            },
        )
        response.raise_for_status()
        result = response.json().get("data", {}).get("result", [])
        if not result:
            return []
        return [float(value[1]) for value in result[0].get("values", [])]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Retrainer metric fetch failed for %s/%s: %s", service, metric_name, exc)
        return []


def _as_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


class Retrainer:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def run_once(self) -> dict[str, object]:
        await run_retraining()
        return {"status": "completed", "timestamp": datetime.now(timezone.utc).isoformat()}
