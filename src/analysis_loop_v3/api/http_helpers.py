"""FastAPI 요청 경계에서 사용하는 검증/조회/응답 변환 helper."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from ..contracts import InvalidRunId, validate_run_id
from ..data.postgres import PostgresSource, validate_read_query
from ..service import RunHandle
from .runs import ServerContext


def get_server_context(request: Request) -> ServerContext:
    """Application lifespan에서 준비한 ServerContext를 반환한다."""
    return request.app.state.ctx


def checked_dataset_path(raw: str) -> str:
    """API 입력 dataset_path를 설정된 허용 루트 아래로 제한한다."""
    root = os.environ.get("ANALYSIS_DATA_ROOT")
    if not root:
        return raw

    base = Path(root).expanduser().resolve()
    resolved = Path(raw).expanduser().resolve()
    if not resolved.is_relative_to(base):
        raise HTTPException(
            status_code=400,
            detail=f"dataset_path는 {base} 아래여야 한다",
        )
    return str(resolved)


def postgres_source(query: str) -> PostgresSource:
    """SQL 읽기 전용 규칙과 DB 설정을 검증해 PostgresSource를 만든다."""
    try:
        validate_read_query(query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    database_url = os.environ.get("AIA_POSTGRES_URL")
    if not database_url:
        raise HTTPException(
            status_code=503,
            detail="서버 환경 변수 AIA_POSTGRES_URL이 설정되지 않았다",
        )
    try:
        return PostgresSource(database_url=database_url, query=query)
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL 연결 설정이 올바르지 않다",
        ) from exc


def checked_run_id(run_id: str) -> str:
    """HTTP path parameter의 run_id를 검증한다."""
    try:
        return validate_run_id(run_id)
    except InvalidRunId as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def snapshot_or_404(ctx: ServerContext, run_id: str) -> RunHandle:
    """Checkpoint snapshot이 없으면 HTTP 404로 변환한다."""
    validate_run_id(run_id)
    try:
        return await ctx.reader.snapshot(thread_id=run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def tail_events(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    """JSONL event log의 마지막 limit개 유효 레코드만 읽는다."""
    if not path.is_file():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:]


def shape_snapshot(
    handle: RunHandle,
    events: list[dict[str, Any]],
    *,
    active: bool,
) -> dict[str, Any]:
    """Checkpoint State를 HTTP 응답 형태로 변환한다."""
    state = handle.state
    return {
        "run_id": handle.run_id,
        "status": handle.status,
        "active": active,
        "stop_code": state.get("stop_code"),
        "stop_reason": state.get("stop_reason"),
        "objective": state.get("objective"),
        "iteration": state.get("iteration"),
        "max_iterations": state.get("max_iterations"),
        "goal_expansions": state.get("goal_expansions"),
        "sub_goals": state.get("sub_goals") or [],
        "intent": state.get("intent"),
        "manifest": state.get("manifest"),
        "review": state.get("review"),
        "evidence": state.get("evidence") or [],
        "rejected": state.get("rejected") or [],
        "issues": state.get("issues") or [],
        "validation_warnings": state.get("validation_warnings") or [],
        "report_available": bool(state.get("report_ref")),
        "events": events,
    }


# 기존 app.py helper 이름과의 호환용 alias.
_ctx = get_server_context
_checked_dataset_path = checked_dataset_path
_postgres_source = postgres_source
_checked_run_id = checked_run_id
_snapshot_or_404 = snapshot_or_404
_tail_events = tail_events
_shape_snapshot = shape_snapshot
