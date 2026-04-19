from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import declarative_base

from app.config import ALERT_THRESHOLD_DEFAULT

Base = declarative_base()


class SLO(Base):
    __tablename__ = "slos"

    service_name = Column(String, primary_key=True)
    metric = Column(String, nullable=False)
    threshold = Column(Float, nullable=False)
    window_minutes = Column(Integer, nullable=False)
    budget_total = Column(Float, nullable=False)
    budget_consumed = Column(Float, nullable=False, default=0.0)
    alert_threshold_override = Column(Float, nullable=True)


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    service_name = Column(String, nullable=False, index=True)
    score = Column(Float, nullable=False)
    threshold_used = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    alert_fired = Column(Boolean, nullable=False)
    outcome = Column(String, nullable=True)


async def create_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_slo(session: AsyncSession, service_name: str) -> SLO | None:
    return await session.get(SLO, service_name)


async def create_slo(session: AsyncSession, data: dict) -> SLO:
    slo = SLO(
        service_name=data["service_name"],
        metric=data["metric"],
        threshold=float(data["threshold"]),
        window_minutes=int(data["window_minutes"]),
        budget_total=float(data["budget_total"]),
        budget_consumed=float(data.get("budget_consumed", 0.0)),
        alert_threshold_override=data.get("alert_threshold_override"),
    )
    session.add(slo)
    await session.commit()
    await session.refresh(slo)
    return slo


async def update_budget(session: AsyncSession, service_name: str, consumed: float) -> SLO | None:
    slo = await get_slo(session, service_name)
    if slo is None:
        return None
    slo.budget_consumed = consumed
    await session.commit()
    await session.refresh(slo)
    return slo


async def log_prediction(
    session: AsyncSession,
    service_name: str,
    score: float,
    threshold_used: float,
    alert_fired: bool,
) -> int:
    row = PredictionLog(
        service_name=service_name,
        score=score,
        threshold_used=threshold_used,
        alert_fired=alert_fired,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row.id


async def update_outcome(session: AsyncSession, prediction_id: int, outcome: str) -> None:
    row = await session.get(PredictionLog, prediction_id)
    if row is None:
        return
    row.outcome = outcome
    await session.commit()


async def get_recent_predictions(session: AsyncSession, service_name: str, since: datetime) -> list[PredictionLog]:
    result = await session.execute(
        select(PredictionLog)
        .where(PredictionLog.service_name == service_name)
        .where(PredictionLog.timestamp >= since)
        .order_by(PredictionLog.timestamp.desc())
    )
    return list(result.scalars().all())


async def get_effective_threshold(session: AsyncSession, service_name: str) -> float:
    slo = await get_slo(session, service_name)
    if slo is None:
        return ALERT_THRESHOLD_DEFAULT
    if slo.budget_total > 0 and slo.budget_consumed / slo.budget_total > 0.8:
        return slo.alert_threshold_override if slo.alert_threshold_override is not None else 0.6
    return ALERT_THRESHOLD_DEFAULT
