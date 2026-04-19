import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import generate_latest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.responses import Response

from app.alert_publisher import publish_prediction
from app.config import POSTGRES_DSN, SERVICES
from app.feedback_consumer import close_feedback_consumer, configure_feedback_consumer, start_feedback_consumer
from app.metric_ingestion import prewarm_ingestion, start_ingestion
from app.prediction_engine import load_all_models, predict as predict_score
from app.retrainer import scheduler as retrainer_scheduler
from app.retrainer import start_retrainer
from app.slo_registry import SLO, create_slo, create_tables, get_effective_threshold, get_slo, log_prediction, update_budget


class SLOCreate(BaseModel):
    service_name: str
    metric: str
    threshold: float
    window_minutes: int
    budget_total: float
    budget_consumed: float = 0.0
    alert_threshold_override: float | None = None


class BudgetUpdate(BaseModel):
    consumed: float


def _asyncpg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    return dsn


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_async_engine(_asyncpg_dsn(POSTGRES_DSN), pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    app.state.engine = engine
    app.state.session_factory = session_factory

    await create_tables(engine)
    await prewarm_ingestion(SERVICES)

    ingestion_task = asyncio.create_task(start_ingestion(SERVICES, prewarm=False))
    configure_feedback_consumer(session_factory)
    feedback_task = asyncio.create_task(start_feedback_consumer())
    start_retrainer(session_factory)
    await load_all_models(SERVICES)

    try:
        yield
    finally:
        ingestion_task.cancel()
        feedback_task.cancel()
        await asyncio.gather(ingestion_task, feedback_task, return_exceptions=True)
        close_feedback_consumer()
        if retrainer_scheduler.running:
            retrainer_scheduler.shutdown(wait=False)
        await engine.dispose()


app = FastAPI(title="ProactaSLO", version="1.0.0", lifespan=lifespan)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with app.state.session_factory() as session:
        yield session


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/slos")
async def list_slos(session: AsyncSession = Depends(get_session)) -> list[dict[str, object]]:
    result = await session.execute(select(SLO).order_by(SLO.service_name))
    return [_serialize_slo(slo) for slo in result.scalars().all()]


@app.post("/slos")
async def create_slo_endpoint(payload: SLOCreate, session: AsyncSession = Depends(get_session)) -> dict[str, object]:
    existing = await get_slo(session, payload.service_name)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"SLO already exists for {payload.service_name}")
    slo = await create_slo(session, payload.dict())
    return _serialize_slo(slo)


@app.get("/slos/{service_name}")
async def get_slo_endpoint(service_name: str, session: AsyncSession = Depends(get_session)) -> dict[str, object]:
    slo = await get_slo(session, service_name)
    if slo is None:
        raise HTTPException(status_code=404, detail=f"SLO not found for {service_name}")
    return _serialize_slo(slo)


@app.put("/slos/{service_name}/budget")
async def update_budget_endpoint(
    service_name: str,
    payload: BudgetUpdate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    slo = await update_budget(session, service_name, payload.consumed)
    if slo is None:
        raise HTTPException(status_code=404, detail=f"SLO not found for {service_name}")
    return _serialize_slo(slo)


@app.get("/predict/{service_name}")
async def predict_endpoint(service_name: str, session: AsyncSession = Depends(get_session)) -> dict[str, object]:
    score = await predict_score(service_name)
    threshold = await get_effective_threshold(session, service_name)
    alert_fired = publish_prediction(service_name, score, threshold)
    await log_prediction(session, service_name, score, threshold, alert_fired)
    return {
        "service_name": service_name,
        "score": score,
        "threshold": threshold,
        "alert_fired": alert_fired,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain")


def _serialize_slo(slo: SLO) -> dict[str, object]:
    return {
        "service_name": slo.service_name,
        "metric": slo.metric,
        "threshold": slo.threshold,
        "window_minutes": slo.window_minutes,
        "budget_total": slo.budget_total,
        "budget_consumed": slo.budget_consumed,
        "alert_threshold_override": slo.alert_threshold_override,
    }
