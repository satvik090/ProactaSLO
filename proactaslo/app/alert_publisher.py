import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Producer

from app.config import KAFKA_BROKER
from app.observability import slo_alert_fired_total, slo_prediction_lead_time_seconds

logger = logging.getLogger(__name__)

producer = Producer({"bootstrap.servers": KAFKA_BROKER})


def _delivery_report(error: Any, message: Any) -> None:
    if error is not None:
        logger.error("Kafka delivery failed for topic %s: %s", message.topic() if message else "unknown", error)


def publish_prediction(service_name: str, score: float, threshold: float) -> bool:
    alert_fired = score > threshold
    base_message = {
        "service_name": service_name,
        "score": score,
        "threshold_used": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alert_fired": alert_fired,
    }

    _produce("model-feedback", service_name, base_message)
    _produce("audit-log", service_name, base_message)

    if alert_fired:
        violation_message = {**base_message, "predicted_violation_window_minutes": 15}
        _produce("violation-alerts", service_name, violation_message)
        slo_alert_fired_total.labels(service=service_name).inc()
        slo_prediction_lead_time_seconds.labels(service=service_name).observe(15 * 60)

    producer.flush()
    return alert_fired


def _produce(topic: str, key: str, payload: dict[str, Any]) -> None:
    producer.produce(topic, key=key, value=json.dumps(payload).encode("utf-8"), callback=_delivery_report)
    producer.poll(0)


class AlertPublisher:
    def maybe_publish(self, prediction: Any, slo: Any) -> dict[str, Any] | None:
        service_name = getattr(prediction, "service", None) or getattr(prediction, "service_name")
        score = float(getattr(prediction, "risk", getattr(prediction, "score", 0.0)))
        threshold = float(getattr(slo, "alert_threshold", getattr(slo, "threshold", 0.0)))
        alert_fired = publish_prediction(service_name, score, threshold)
        return {"service_name": service_name, "score": score, "threshold_used": threshold, "alert_fired": alert_fired} if alert_fired else None

    def flush(self) -> None:
        producer.flush()


def prediction_to_dict(prediction: Any) -> dict[str, Any]:
    if is_dataclass(prediction):
        return asdict(prediction)
    return dict(prediction)
