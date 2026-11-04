"""PostgreSQL run registry: 목록/필터링용 보조 인덱스.

status/stop_reason은 체크포인터 조회로 다시 채우는 파생값이다. 이 DB가 사라져도
분석 상태는 checkpoints.sqlite에 남는다.

DB engine은 module import 시점이 아니라 init_db()에서 만든다. Application이 .env를
로드한 뒤 DATABASE_URL을 읽도록 해 import 순서와 설정 로딩을 분리한다.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, String, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_DATABASE_URL = (
    "postgresql+asyncpg://analysis:analysis@localhost:5432/analysis_loop_v3"
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def _require_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("database is not initialized; call init_db() first")
    return _engine


def _require_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("database is not initialized; call init_db() first")
    return _session_factory


async def init_db() -> None:
    """현재 환경 설정으로 engine/session factory를 만들고 schema를 준비한다."""
    global _engine, _session_factory

    if _engine is None:
        _engine = create_async_engine(
            _database_url(),
            pool_pre_ping=True,
        )
        _session_factory = async_sessionmaker(
            _engine,
            expire_on_commit=False,
        )

    async with _engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Application shutdown에서 DB 자원을 해제하고 재초기화 가능한 상태로 만든다."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def ping() -> None:
    engine = _require_engine()
    async with engine.connect() as connection:
        await connection.execute(select(1))


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
    *,
    run_id: str,
    objective: str,
    dataset_path: str,
    status: str,
) -> dict[str, Any]:
    session_factory = _require_session_factory()
    async with session_factory() as session:
        record = await session.get(RunRecord, run_id)
        if record is None:
            record = RunRecord(
                run_id=run_id,
                objective=objective,
                dataset_path=dataset_path,
                status=status,
            )
            session.add(record)
        else:
            record.status = status

        await session.commit()
        await session.refresh(record)
        return _to_dict(record)


async def update_status(
    run_id: str,
    *,
    status: str,
    stop_reason: str | None = None,
) -> None:
    session_factory = _require_session_factory()
    async with session_factory() as session:
        record = await session.get(RunRecord, run_id)
        if record is None:
            return
        record.status = status
        record.stop_reason = stop_reason
        await session.commit()


async def list_runs(
    *,
    status: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    session_factory = _require_session_factory()
    async with session_factory() as session:
        statement = (
            select(RunRecord)
            .order_by(RunRecord.created_at.desc())
            .limit(limit)
        )
        if status:
            statement = statement.where(RunRecord.status == status)
        result = await session.execute(statement)
        return [_to_dict(record) for record in result.scalars().all()]
