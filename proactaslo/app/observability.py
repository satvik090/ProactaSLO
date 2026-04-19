from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

ingestion_drop_total = Counter("ingestion_drop_total", "Metric samples dropped during ingestion.", ["service"])
slo_prediction_score = Gauge("slo_prediction_score", "Current predicted SLO breach score.", ["service"])
slo_prediction_lead_time_seconds = Histogram(
    "slo_prediction_lead_time_seconds",
    "Predicted lead time before an SLO breach.",
    ["service"],
)
slo_alert_fired_total = Counter("slo_alert_fired_total", "SLO alerts fired.", ["service"])
model_drift_signal = Gauge("model_drift_signal", "Model drift signal by service.", ["service"])
prediction_accuracy = Gauge("prediction_accuracy", "Prediction accuracy by service and outcome.", ["service", "outcome"])

REQUEST_COUNT = Counter("proactaslo_http_requests_total", "HTTP requests served by ProactaSLO.", ["path"])
INGESTED_METRICS = Counter("proactaslo_ingested_metrics_total", "Service metric samples ingested.", ["service"])
ERROR_BUDGET = Gauge("proactaslo_error_budget_remaining", "Remaining SLO error budget by service.", ["service"])
PREDICTION_LATENCY = Histogram("proactaslo_prediction_latency_seconds", "Prediction calculation latency.")

PREDICTION_RISK = slo_prediction_score


class _AlertCounterAdapter:
    def labels(self, service: str, severity: str | None = None):
        return slo_alert_fired_total.labels(service=service)


ALERTS_PUBLISHED = _AlertCounterAdapter()


def metrics_response() -> Response:
    return Response(generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")
