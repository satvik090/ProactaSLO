import asyncio
import json
from typing import Any

from redis.asyncio import Redis

from app.config import PREDICTION_TTL_SECONDS, REDIS_URL, RING_BUFFER_SIZE


class BackpressureError(RuntimeError):
    pass


class MetricCache:
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        ring_buffer_size: int = RING_BUFFER_SIZE,
        prediction_ttl_seconds: int = PREDICTION_TTL_SECONDS,
        backpressure_timeout_seconds: float = 0.5,
    ) -> None:
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self._ring_buffer_size = ring_buffer_size
        self._prediction_ttl_seconds = prediction_ttl_seconds
        self._backpressure_timeout_seconds = backpressure_timeout_seconds

    async def write_metric(self, service: str, vector: list[float]) -> None:
        key = f"metrics:{service}"

        async def write() -> None:
            pipe = self._redis.pipeline()
            pipe.rpush(key, json.dumps(vector))
            pipe.ltrim(key, -self._ring_buffer_size, -1)
            await pipe.execute()

        try:
            await asyncio.wait_for(write(), timeout=self._backpressure_timeout_seconds)
        except TimeoutError as exc:
            raise BackpressureError(f"Redis write for {service} exceeded 500ms") from exc

    async def read_metrics(self, service: str, n: int) -> list[list[float]]:
        values = await self._redis.lrange(f"metrics:{service}", -n, -1)
        return [json.loads(value) for value in values]

    async def get_prediction(self, service: str) -> float | None:
        value = await self._redis.get(f"pred:{service}")
        return None if value is None else float(value)

    async def set_prediction(self, service: str, score: float) -> None:
        await self._redis.set(f"pred:{service}", score, ex=self._prediction_ttl_seconds)

    async def save_scaler(self, service: str, scaler: dict[str, Any]) -> None:
        await self._redis.set(f"scaler:{service}", json.dumps(scaler))

    async def load_scaler(self, service: str) -> dict[str, Any] | None:
        value = await self._redis.get(f"scaler:{service}")
        return None if value is None else json.loads(value)

    async def close(self) -> None:
        await self._redis.aclose()


class PredictionCache(MetricCache):
    async def get(self, key: str) -> dict[str, Any] | None:
        value = await self._redis.get(f"response:{key}")
        return None if value is None else json.loads(value)

    async def set(self, key: str, value: dict[str, Any]) -> None:
        await self._redis.set(f"response:{key}", json.dumps(value), ex=self._prediction_ttl_seconds)
