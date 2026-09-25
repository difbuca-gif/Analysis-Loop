"""FastAPI 진입점. AnalysisService를 HTTP로 노출한다.

상태를 소유하지 않는다. 조회는 체크포인터에서 다시 읽고, 변경은 background task로
그래프를 잇는다. Postgres는 목록 색인일 뿐 진리원이 아니다.
CLI와 workspace를 공유하므로 CLI로 시작한 run을 API로 재개할 수 있다(반대도).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..contracts import ArtifactRef, InvalidRunId, validate_run_id
from ..data.postgres import PostgresSource, validate_read_query
from ..env import load_project_env
from ..service import AnalysisService, RunHandle, open_checkpointer
from . import db
from .auth import require_api_token
from .runs import RunConflictError, RunRegistry, ServerContext, build_reader_service, drive
from .schemas import ResumeRequest, RunSummary, StartRunRequest

logger = logging.getLogger("analysis_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_project_env()
    await db.init_db()
    workspace = Path(os.environ.get("ANALYSIS_WORKSPACE", "runs"))
    async with open_checkpointer(workspace / "checkpoints.sqlite") as checkpointer:
        from ..runtime import ArtifactStore

        app.state.ctx = ServerContext(
            workspace=workspace,
            checkpointer=checkpointer,
            artifacts=ArtifactStore(workspace / "artifacts"),
            registry=RunRegistry(),
            reader=await build_reader_service(workspace, checkpointer),
        )
        yield
    await db.dispose_engine()

app = FastAPI(title="Analysis Loop v3 API", version="1.0.0", lifespan=lifespan)

def _ctx(request: Request) -> ServerContext:
    return request.app.state.ctx

def _checked_dataset_path(raw: str) -> str:
    """API로 들어온 데이터셋 경로를 허용 루트 아래로 제한한다.

    `ANALYSIS_DATA_ROOT`가 없으면 제한하지 않는다(단일 머신 개발). 서버에 올릴
    때는 반드시 채운다. 인증이 있어도 토큰 하나로 서버의 아무 파일이나 분석
    대상으로 삼을 수 있는 상태를 남기지 않기 위한 두 번째 방어선이다.
    CLI는 서버 관리자가 직접 쓰는 것이라 이 제한을 받지 않는다.
    """
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

def _postgres_source(query: str) -> PostgresSource:
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
        raise HTTPException(status_code=503, detail="PostgreSQL 연결 설정이 올바르지 않다") from exc

def _checked_run_id(run_id: str) -> str:
    """경로 파라미터도 거른다. GET /runs/{run_id}가 events/{run_id}.jsonl을 읽으므로
    검증 없이 두면 파일 읽기 프리미티브가 된다."""
    try:
        return validate_run_id(run_id)
    except InvalidRunId as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

async def _snapshot_or_404(ctx: ServerContext, run_id: str) -> RunHandle:
    validate_run_id(run_id)
    try:
        return await ctx.reader.snapshot(thread_id=run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

def _tail_events(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:]

def _shape_snapshot(
    handle: RunHandle, events: list[dict[str, Any]], *, active: bool,
) -> dict[str, Any]:
    """State 필드를 그대로 노출한다(이름을 다시 짓지 않는다). 화면용 표현은 프런트가.

    active는 status와 별개다. 취소되면 체크포인트 status는 RUNNING인 채 남을 수
    있고(CancelledError는 노드 완료를 안 기다린다), 실제 구동 여부는 레지스트리만 안다.
    """
    state = handle.state
    return {
        "run_id": handle.run_id,
        "status": handle.status,
        "active": active,
        # stop_code는 프런트가 분기해도 되는 계약, stop_reason은 표시 전용 문자열이다.
        # 프런트가 stop_reason 문자열을 파싱해 분기하기 시작하면 안 된다.
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
        # rejected와 나눠 노출한다. 대시보드가 눈에 띄게 표시해야 하는 쪽이다.
        "issues": state.get("issues") or [],
        "validation_warnings": state.get("validation_warnings") or [],
        "report_available": bool(state.get("report_ref")),
        "events": events,
    }

@app.get("/health")
async def health(request: Request) -> JSONResponse:
    checks = {"api": "ok"}
    healthy = True
    try:
        await db.ping()
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        healthy = False
        checks["postgres"] = "error"
        logger.warning("PostgreSQL healthcheck failed: %s", exc)
    return JSONResponse(status_code=200 if healthy else 503, content=checks)

@app.post("/runs", response_model=RunSummary, status_code=201, dependencies=[Depends(require_api_token)])
async def create_run(payload: StartRunRequest, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    if payload.dataset_path is not None:
        dataset_path = _checked_dataset_path(payload.dataset_path)
        postgres_source = None
        source_label = dataset_path
    else:
        dataset_path = None
        postgres_source = _postgres_source(payload.sql_query or "")
        source_label = "postgresql"

    run_id = payload.run_id or uuid4().hex[:12]

    async def action(service: AnalysisService) -> RunHandle:
        return await service.start(
            dataset_path=dataset_path,
            postgres_source=postgres_source,
            objective=payload.objective,
            max_iterations=payload.max_iterations,
            time_budget_seconds=payload.time_budget_seconds,
            run_id=run_id,
        )

    try:
        ctx.registry.start(run_id, lambda: drive(ctx, run_id, action))
    except RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # service.start()의 "이미 존재하는 run_id" 같은 검증은 background task 안에서
    # 일어난다. 여기서 미리 걸러내지 않는다. 실패하면 status=FAILED로 db에
    # 반영되므로 클라이언트는 뒤이은 GET /runs/{run_id}로 확인한다.
    return await db.upsert_run(
        run_id=run_id, objective=payload.objective,
        dataset_path=source_label, status="RUNNING",
    )

@app.get("/runs", dependencies=[Depends(require_api_token)])
async def list_runs(
    request: Request,
    status: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Postgres 목록을 훑고 각 항목을 체크포인터로 다시 확인해 최신 status로 갱신한다.

    iteration/evidence_count/active는 이미 읽은 스냅샷에서 붙인다.
    """
    ctx = _ctx(request)
    records = await db.list_runs(status=None, limit=limit)
    enriched: list[dict[str, Any]] = []
    for record in records:
        try:
            snap = await ctx.reader.snapshot(thread_id=record["run_id"])
        except ValueError:
            # 체크포인트를 못 읽는 항목도 같은 필드 모양으로 돌려준다. 소비자가
            # 키 존재 여부로 분기하지 않아도 되게.
            enriched.append({
                **record, "iteration": None, "evidence_count": None,
                "issue_count": None, "active": False, "stop_code": None,
            })
            continue
        actual_reason = snap.state.get("stop_reason")
        if snap.status != record["status"] or actual_reason != record.get("stop_reason"):
            await db.update_status(record["run_id"], status=snap.status, stop_reason=actual_reason)
            record = {**record, "status": snap.status, "stop_reason": actual_reason}
        enriched.append({
            **record,
            "iteration": snap.state.get("iteration"),
            "evidence_count": len(snap.state.get("evidence") or []),
            "issue_count": len(snap.state.get("issues") or []),
            "active": ctx.registry.is_active(record["run_id"]),
            # stop_code는 체크포인트의 최신 상태에서 읽는다.
            "stop_code": snap.state.get("stop_code"),
        })
    if status:
        enriched = [r for r in enriched if r["status"] == status]
    return enriched

@app.get("/runs/{run_id}", dependencies=[Depends(require_api_token)])
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    run_id = _checked_run_id(run_id)
    ctx = _ctx(request)
    handle = await _snapshot_or_404(ctx, run_id)
    events = _tail_events(ctx.workspace / "events" / f"{run_id}.jsonl")
    return _shape_snapshot(handle, events, active=ctx.registry.is_active(run_id))

@app.post("/runs/{run_id}/resume", dependencies=[Depends(require_api_token)])
async def resume_run(
    run_id: str, request: Request, payload: ResumeRequest | None = None,
) -> dict[str, Any]:
    run_id = _checked_run_id(run_id)
    ctx = _ctx(request)
    await _snapshot_or_404(ctx, run_id)
    budget = payload.time_budget_seconds if payload else None

    async def action(service: AnalysisService) -> RunHandle:
        return await service.resume(thread_id=run_id, time_budget_seconds=budget)

    try:
        ctx.registry.start(run_id, lambda: drive(ctx, run_id, action))
    except RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await db.update_status(run_id, status="RUNNING")
    return {"run_id": run_id, "status": "RUNNING"}

@app.post("/runs/{run_id}/stop", dependencies=[Depends(require_api_token)])
async def stop_run(run_id: str, request: Request) -> dict[str, Any]:
    run_id = _checked_run_id(run_id)
    ctx = _ctx(request)
    await _snapshot_or_404(ctx, run_id)
    cancelled = ctx.registry.cancel(run_id)
    if not cancelled:
        raise HTTPException(
            status_code=409, detail=f"run {run_id}를 구동 중인 task가 없다(이미 멈춰 있음)",
        )
    await db.update_status(run_id, status="STOPPING", stop_reason="사용자 중지 요청")
    return {"run_id": run_id, "status": "STOPPING"}

@app.get("/runs/{run_id}/report", response_class=HTMLResponse, dependencies=[Depends(require_api_token)])
async def get_report(run_id: str, request: Request) -> str:
    run_id = _checked_run_id(run_id)
    ctx = _ctx(request)
    handle = await _snapshot_or_404(ctx, run_id)
    report_ref = handle.state.get("report_ref")
    if not report_ref:
        raise HTTPException(status_code=404, detail=f"run {run_id}에 아직 보고서가 없다")
    ref = ArtifactRef.model_validate(report_ref)
    try:
        return ctx.artifacts.read_text(ref)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc

@app.get("/runs/{run_id}/code", dependencies=[Depends(require_api_token)])
async def get_code(
    run_id: str, request: Request, evidence_id: str | None = None,
) -> dict[str, Any]:
    """코드 지연 조회 — state엔 ArtifactRef만 있어 펼쳐볼 때만 파일에서 읽는다.

    evidence_id를 주면 그 근거의 program_ref를, 없으면 현재 회차의 것을 읽는다.
    """
    run_id = _checked_run_id(run_id)
    ctx = _ctx(request)
    handle = await _snapshot_or_404(ctx, run_id)
    ref_raw: dict[str, Any] | None
    if evidence_id:
        ref_raw = None
        for item in handle.state.get("evidence") or []:
            if item.get("evidence_id") == evidence_id:
                ref_raw = item.get("program_ref")
                break
        else:
            raise HTTPException(
                status_code=404, detail=f"evidence {evidence_id}를 run {run_id}에서 찾을 수 없다",
            )
    else:
        ref_raw = handle.state.get("program_ref")
    if not ref_raw:
        return {"code": None}
    ref = ArtifactRef.model_validate(ref_raw)
    try:
        return {"code": ctx.artifacts.read_text(ref)}
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
