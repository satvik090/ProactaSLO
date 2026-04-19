from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    from os import getenv

    return getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


SERVICES: list[str] = [
    service.strip()
    for service in _env(
        "SERVICES",
        "auth,payment,order,inventory,shipping,notification,user,cart,search,recommendation,review,warehouse,gateway,scheduler,audit",
    ).split(",")
    if service.strip()
]
PROMETHEUS_URL: str = _env("PROMETHEUS_URL", "http://prometheus:9090")
KAFKA_BROKER: str = _env("KAFKA_BROKER", "kafka:9092")
REDIS_URL: str = _env("REDIS_URL", "redis://redis:6379")
POSTGRES_DSN: str = _env("POSTGRES_DSN", "postgresql://postgres:postgres@postgres:5432/proactaslo")
SCRAPE_INTERVAL_SECONDS: int = _env_int("SCRAPE_INTERVAL_SECONDS", 15)
RING_BUFFER_SIZE: int = _env_int("RING_BUFFER_SIZE", 120)
PREDICTION_TTL_SECONDS: int = _env_int("PREDICTION_TTL_SECONDS", 30)
ALERT_THRESHOLD_DEFAULT: float = _env_float("ALERT_THRESHOLD_DEFAULT", 0.75)
MODEL_DIR: Path = Path(_env("MODEL_DIR", "./models"))

DEPENDENCY_MAP: dict[str, list[str]] = {
    "auth": [],
    "payment": ["auth", "order"],
    "order": ["inventory", "cart"],
    "inventory": ["warehouse"],
    "shipping": ["order", "warehouse"],
    "notification": ["auth"],
    "user": ["auth"],
    "cart": ["user", "inventory"],
    "search": ["inventory", "recommendation"],
    "recommendation": ["user", "review"],
    "review": ["user", "order"],
    "warehouse": [],
    "gateway": ["auth", "payment", "order", "search"],
    "scheduler": ["notification", "audit"],
    "audit": [],
}


@dataclass(frozen=True)
class Settings:
    services: list[str] = field(default_factory=lambda: SERVICES)
    prometheus_url: str = PROMETHEUS_URL
    kafka_broker: str = KAFKA_BROKER
    redis_url: str = REDIS_URL
    postgres_dsn: str = POSTGRES_DSN
    scrape_interval_seconds: int = SCRAPE_INTERVAL_SECONDS
    ring_buffer_size: int = RING_BUFFER_SIZE
    prediction_ttl_seconds: int = PREDICTION_TTL_SECONDS
    alert_threshold_default: float = ALERT_THRESHOLD_DEFAULT
    model_dir: Path = MODEL_DIR
    dependency_map: dict[str, list[str]] = field(default_factory=lambda: DEPENDENCY_MAP)

    @property
    def service_names(self) -> list[str]:
        return self.services


def get_settings() -> Settings:
    return Settings()
