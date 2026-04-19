import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from confluent_kafka import Consumer, KafkaException
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import KAFKA_BROKER, PROMETHEUS_URL, SCRAPE_INTERVAL_SECONDS
from app.observability import prediction_accuracy
from app.slo_registry import PredictionLog, get_slo, log_prediction, update_outcome

logger = logging.getLogger(__name__)

consumer = Consumer(
    {
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": "proactaslo-feedback",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }
)

SessionFactory: async_sessionmaker | None = None


def configure_feedback_consumer(session_factory: async_sessionmaker) -> None:
    global SessionFactory
    SessionFactory = session_factory


async def start_feedback_consumer() -> None:
    if SessionFactory is None:
        raise RuntimeError("Feedback consumer session factory is not configured")

    consumer.subscribe(["model-feedback"])
    while True:
        message = await asyncio.to_thread(consumer.poll, 1.0)
        if message is None:
            continue
        if message.error():
            logger.error("Kafka consumer error: %s", message.error())
            continue

        try:
            payload = json.loads(message.value().decode("utf-8"))
            async with SessionFactory() as session:
                prediction_id = await log_prediction(
                    session,
                    payload["service_name"],
                    float(payload["score"]),
                    float(payload["threshold_used"]),
                    bool(payload["alert_fired"]),
                )
            original_timestamp = _parse_timestamp(payload["timestamp"])
            asyncio.create_task(check_outcome(prediction_id, payload["service_name"], original_timestamp))
            consumer.commit(message=message, asynchronous=False)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError, KafkaException) as exc:
            logger.exception("Failed to process model-feedback message: %s", exc)


async def check_outcome(prediction_id: int, service_name: str, original_timestamp: datetime) -> None:
    if SessionFactory is None:
        raise RuntimeError("Feedback consumer session factory is not configured")

    await asyncio.sleep(15 * 60)
    async with SessionFactory() as session:
        slo = await get_slo(session, service_name)
        metric = slo.metric if slo is not None else "p99_latency"
        metric_threshold = slo.threshold if slo is not None else 200.0

    violation_occurred = await _did_violate(service_name, metric, metric_threshold, original_timestamp)

    async with SessionFactory() as session:
        prediction_log = await session.get(PredictionLog, prediction_id)
        if prediction_log is None:
            logger.warning("Prediction log %s disappeared before outcome check", prediction_id)
            return

        alert_fired = bool(prediction_log.alert_fired)
        if alert_fired and violation_occurred:
            outcome = "true_positive"
        elif alert_fired and not violation_occurred:
            outcome = "false_positive"
        elif not alert_fired and not violation_occurred:
            outcome = "true_negative"
        else:
            outcome = "missed_alert"
            logger.critical("Missed SLO alert for %s at %s", service_name, original_timestamp.isoformat())

        await update_outcome(session, prediction_id, outcome)
        prediction_accuracy.labels(service=service_name, outcome=outcome).set(1.0)


async def _did_violate(service_name: str, metric: str, threshold: float, start: datetime) -> bool:
    end = start + timedelta(minutes=15)
    query = f'{metric}{{job="{service_name}"}}'
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
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
                return False
            return any(float(value[1]) > threshold for value in result[0].get("values", []))
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Prometheus outcome query failed for %s: %s", service_name, exc)
        return False


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def close_feedback_consumer() -> None:
    consumer.close()


class FeedbackConsumer:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def recent_events(self) -> list[dict[str, Any]]:
        return []

    async def consume_forever(self) -> None:
        await start_feedback_consumer()

    async def stop(self) -> None:
        close_feedback_consumer()
