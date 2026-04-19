import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.cache import BackpressureError, MetricCache
from app.config import DEPENDENCY_MAP, PROMETHEUS_URL, RING_BUFFER_SIZE, SCRAPE_INTERVAL_SECONDS, Settings
from app.observability import ERROR_BUDGET, INGESTED_METRICS, ingestion_drop_total

logger = logging.getLogger(__name__)

METRIC_NAMES: list[str] = [
    "p50_latency",
    "p95_latency",
    "p99_latency",
    "error_rate",
    "request_rate",
    "cpu_util",
    "memory_usage",
    "queue_depth",
]

cache = MetricCache()


@dataclass
class MetricSample:
    service: str
    timestamp: float
    success_rate: float
    latency_ms: float
    request_rate: float


async def prewarm_ingestion(services: list[str]) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        await asyncio.gather(*(_bootstrap_service(client, service) for service in services), return_exceptions=True)


async def start_ingestion(services: list[str], prewarm: bool = True) -> None:
    await asyncio.gather(*(_ingest_service(service, prewarm=prewarm) for service in services))


async def _ingest_service(service: str, prewarm: bool = True) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        if prewarm:
            await _bootstrap_service(client, service)
        while True:
            await asyncio.sleep(SCRAPE_INTERVAL_SECONDS)
            await _ingest_once(client, service)


async def _bootstrap_service(client: httpx.AsyncClient, service: str) -> None:
    observed_services = _observed_services(service)
    history_by_service = await asyncio.gather(
        *(_fetch_service_history(client, observed_service) for observed_service in observed_services),
        return_exceptions=True,
    )

    rows: list[list[float]] = []
    for index in range(RING_BUFFER_SIZE):
        row: list[float] = []
        for result in history_by_service:
            if isinstance(result, Exception):
                row.extend([0.0] * len(METRIC_NAMES))
                continue
            row.extend(_values_at(result, index))
        rows.append(row)

    scaler = _fit_min_max_scaler(rows)
    scaler.update(
        {
            "metrics": METRIC_NAMES,
            "dependencies": DEPENDENCY_MAP.get(service, []),
            "feature_count": len(rows[0]) if rows else len(observed_services) * len(METRIC_NAMES),
            "updated_at": time.time(),
        }
    )
    await cache.save_scaler(service, scaler)

    for row in rows:
        try:
            await cache.write_metric(service, _normalise(row, scaler))
        except BackpressureError:
            _drop_batch(service)
            break


async def _ingest_once(client: httpx.AsyncClient, service: str) -> None:
    observed_services = _observed_services(service)
    service_vectors = await asyncio.gather(
        *(_fetch_service_instant(client, observed_service) for observed_service in observed_services),
        return_exceptions=True,
    )

    flat_vector: list[float] = []
    for observed_service, result in zip(observed_services, service_vectors):
        if isinstance(result, Exception) or result is None:
            logger.warning("Skipping ingestion batch for %s because %s metrics were unavailable", service, observed_service)
            return
        flat_vector.extend(result)

    scaler = await cache.load_scaler(service)
    if scaler is None or int(scaler.get("feature_count", 0)) != len(flat_vector):
        logger.warning("Scaler missing or stale for %s; fitting from current batch", service)
        scaler = _fit_min_max_scaler([flat_vector])
        scaler.update(
            {
                "metrics": METRIC_NAMES,
                "dependencies": DEPENDENCY_MAP.get(service, []),
                "feature_count": len(flat_vector),
                "updated_at": time.time(),
            }
        )
        await cache.save_scaler(service, scaler)

    try:
        await cache.write_metric(service, _normalise(flat_vector, scaler))
    except BackpressureError:
        _drop_batch(service)


async def _fetch_service_instant(client: httpx.AsyncClient, service: str) -> list[float] | None:
    values = await asyncio.gather(
        *(_fetch_instant_metric(client, service, metric_name) for metric_name in METRIC_NAMES),
        return_exceptions=True,
    )
    vector: list[float] = []
    for metric_name, value in zip(METRIC_NAMES, values):
        if isinstance(value, Exception) or value is None:
            logger.warning("Skipping metric %s for %s after Prometheus query failure", metric_name, service)
            return None
        vector.append(value)
    return vector


async def _fetch_instant_metric(client: httpx.AsyncClient, service: str, metric_name: str) -> float | None:
    query = f'{metric_name}{{job="{service}"}}'
    try:
        response = await client.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
        response.raise_for_status()
        result = response.json().get("data", {}).get("result", [])
        if not result:
            return None
        return float(result[0]["value"][1])
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Prometheus instant query failed for %s/%s: %s", service, metric_name, exc)
        return None


async def _fetch_service_history(client: httpx.AsyncClient, service: str) -> dict[str, list[float]]:
    metric_histories = await asyncio.gather(
        *(_fetch_range_metric(client, service, metric_name) for metric_name in METRIC_NAMES),
        return_exceptions=True,
    )
    history: dict[str, list[float]] = {}
    for metric_name, values in zip(METRIC_NAMES, metric_histories):
        if isinstance(values, Exception):
            logger.warning("Prometheus range query failed for %s/%s: %s", service, metric_name, values)
            history[metric_name] = []
        else:
            history[metric_name] = values
    return history


async def _fetch_range_metric(client: httpx.AsyncClient, service: str, metric_name: str) -> list[float]:
    end = time.time()
    start = end - (RING_BUFFER_SIZE * SCRAPE_INTERVAL_SECONDS)
    query = f'{metric_name}{{job="{service}"}}'
    try:
        response = await client.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": SCRAPE_INTERVAL_SECONDS},
        )
        response.raise_for_status()
        result = response.json().get("data", {}).get("result", [])
        if not result:
            return []
        values = [float(value[1]) for value in result[0].get("values", [])]
        return values[-RING_BUFFER_SIZE:]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Prometheus range query failed for %s/%s: %s", service, metric_name, exc)
        return []


def _observed_services(service: str) -> list[str]:
    return [service, *DEPENDENCY_MAP.get(service, [])]


def _values_at(history: dict[str, list[float]], index: int) -> list[float]:
    values: list[float] = []
    for metric_name in METRIC_NAMES:
        metric_values = history.get(metric_name, [])
        offset = len(metric_values) - RING_BUFFER_SIZE + index
        values.append(metric_values[offset] if 0 <= offset < len(metric_values) else 0.0)
    return values


def _fit_min_max_scaler(rows: list[list[float]]) -> dict[str, Any]:
    if not rows:
        return {"min": [], "max": []}
    feature_count = len(rows[0])
    mins = [min(row[index] for row in rows) for index in range(feature_count)]
    maxs = [max(row[index] for row in rows) for index in range(feature_count)]
    return {"min": mins, "max": maxs}


def _normalise(vector: list[float], scaler: dict[str, Any]) -> list[float]:
    mins = scaler.get("min", [])
    maxs = scaler.get("max", [])
    normalised: list[float] = []
    for index, value in enumerate(vector):
        minimum = float(mins[index]) if index < len(mins) else 0.0
        maximum = float(maxs[index]) if index < len(maxs) else 1.0
        denominator = maximum - minimum
        normalised.append(0.0 if denominator == 0 else max(0.0, min(1.0, (value - minimum) / denominator)))
    return normalised


def _drop_batch(service: str) -> None:
    ingestion_drop_total.labels(service=service).inc()
    logger.warning("%s dropped ingestion batch for %s due to Redis backpressure", datetime.now(timezone.utc).isoformat(), service)


class MetricIngestion:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._task: asyncio.Task[None] | None = None

    def buffers(self) -> dict[str, list[MetricSample]]:
        return {}

    def add_sample(self, sample: MetricSample) -> None:
        INGESTED_METRICS.labels(service=sample.service).inc()
        ERROR_BUDGET.labels(service=sample.service).set(max(sample.success_rate - 0.995, 0.0))

    async def scrape_forever(self) -> None:
        await start_ingestion(self._settings.service_names)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
