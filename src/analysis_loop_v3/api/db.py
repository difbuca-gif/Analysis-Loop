"""PostgreSQL 런 레지스트리 — 목록·필터링 전용 색인(진리원 아님).

status/stop_reason은 체크포인터 조회로 매번 다시 채우는 파생값이다. 이 DB가
사라져도 분석 상태는 checkpoints.sqlite에 남는다.
homelab 데이터 웨어하우스와는 별도 DB — 운영 메타데이터를 분석 데이터의
마이그레이션 주기에 묶지 않는다.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, String, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://analysis:analysis@localhost:5432/analysis_loop_v3",
)

class Base(DeclarativeBase):
    pass

def _utcnow() -> datetime:
    return datetime.now(UTC)

class RunRecord(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    objective: Mapped[str] = mapped_column(String)
    dataset_path: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="RUNNING")
    stop_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
    )

_engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
_SessionLocal = async_sessionmaker(_engine, expire_on_commit=False)

async def init_db() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def dispose_engine() -> None:
    await _engine.dispose()

async def ping() -> None:
    async with _engine.connect() as conn:
        await conn.execute(select(1))

def _to_dict(record: RunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "objective": record.objective,
        "dataset_path": record.dataset_path,
        "status": record.status,
        "stop_reason": record.stop_reason,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }

async def upsert_run(
    *, run_id: str, objective: str, dataset_path: str, status: str,
) -> dict[str, Any]:
    async with _SessionLocal() as session:  # type: AsyncSession
        record = await session.get(RunRecord, run_id)
        if record is None:
            record = RunRecord(
                run_id=run_id, objective=objective,
                dataset_path=dataset_path, status=status,
            )
            session.add(record)
        else:
            record.status = status
        await session.commit()
        await session.refresh(record)
        return _to_dict(record)

async def update_status(run_id: str, *, status: str, stop_reason: str | None = None) -> None:
    async with _SessionLocal() as session:
        record = await session.get(RunRecord, run_id)
        if record is None:
            # 레지스트리에 없는 run은 체크포인터 상태만 유지한다.
            return
        record.status = status
        record.stop_reason = stop_reason
        await session.commit()

async def list_runs(*, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    async with _SessionLocal() as session:
        stmt = select(RunRecord).order_by(RunRecord.created_at.desc()).limit(limit)
        if status:
            stmt = stmt.where(RunRecord.status == status)
        result = await session.execute(stmt)
        return [_to_dict(r) for r in result.scalars().all()]
