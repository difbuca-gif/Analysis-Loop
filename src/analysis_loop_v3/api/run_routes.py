"""Run lifecycle과 Artifact 조회 HTTP endpoints."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from ..contracts import ArtifactRef
from ..service import AnalysisService, RunHandle
from . import db
from .auth import require_api_token
from .http_helpers import (
    checked_dataset_path,
    checked_run_id,
    get_server_context,
    postgres_source,
    shape_snapshot,
    snapshot_or_404,
    tail_events,
)
from .runs import RunConflictError, drive
from .schemas import ResumeRequest, RunSummary, StartRunRequest

router = APIRouter(dependencies=[Depends(require_api_token)])


@router.post("/runs", response_model=RunSummary, status_code=201)
async def create_run(
    payload: StartRunRequest,
    request: Request,
) -> dict[str, Any]:
    ctx = get_server_context(request)
    if payload.dataset_path is not None:
        dataset_path = checked_dataset_path(payload.dataset_path)
        source = None
        source_label = dataset_path
    else:
        dataset_path = None
        source = postgres_source(payload.sql_query or "")
        source_label = "postgresql"

    run_id = payload.run_id or uuid4().hex[:12]

    async def action(service: AnalysisService) -> RunHandle:
        return await service.start(
            dataset_path=dataset_path,
            postgres_source=source,
            objective=payload.objective,
            max_iterations=payload.max_iterations,
            time_budget_seconds=payload.time_budget_seconds,
            run_id=run_id,
        )

    try:
        ctx.registry.start(run_id, lambda: drive(ctx, run_id, action))
    except RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return await db.upsert_run(
        run_id=run_id,
        objective=payload.objective,
        dataset_path=source_label,
        status="RUNNING",
    )


@router.get("/runs")
async def list_runs(
    request: Request,
    status: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Postgres 목록을 최신 checkpoint 상태와 함께 반환한다."""
    ctx = get_server_context(request)
    records = await db.list_runs(status=None, limit=limit)
    enriched: list[dict[str, Any]] = []

    for record in records:
        try:
            snapshot = await ctx.reader.snapshot(thread_id=record["run_id"])
        except ValueError:
            enriched.append({
                **record,
                "iteration": None,
                "evidence_count": None,
                "issue_count": None,
                "active": False,
                "stop_code": None,
            })
            continue

        actual_reason = snapshot.state.get("stop_reason")
        if (
            snapshot.status != record["status"]
            or actual_reason != record.get("stop_reason")
        ):
            await db.update_status(
                record["run_id"],
                status=snapshot.status,
                stop_reason=actual_reason,
            )
            record = {
                **record,
                "status": snapshot.status,
                "stop_reason": actual_reason,
            }

        enriched.append({
            **record,
            "iteration": snapshot.state.get("iteration"),
            "evidence_count": len(snapshot.state.get("evidence") or []),
            "issue_count": len(snapshot.state.get("issues") or []),
            "active": ctx.registry.is_active(record["run_id"]),
            "stop_code": snapshot.state.get("stop_code"),
        })

    if status:
        enriched = [record for record in enriched if record["status"] == status]
    return enriched


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    run_id = checked_run_id(run_id)
    ctx = get_server_context(request)
    handle = await snapshot_or_404(ctx, run_id)
    events = tail_events(ctx.workspace / "events" / f"{run_id}.jsonl")
    return shape_snapshot(
        handle,
        events,
        active=ctx.registry.is_active(run_id),
    )


@router.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: str,
    request: Request,
    payload: ResumeRequest | None = None,
) -> dict[str, Any]:
    run_id = checked_run_id(run_id)
    ctx = get_server_context(request)
    await snapshot_or_404(ctx, run_id)
    budget = payload.time_budget_seconds if payload else None

    async def action(service: AnalysisService) -> RunHandle:
        return await service.resume(
            thread_id=run_id,
            time_budget_seconds=budget,
        )

    try:
        ctx.registry.start(run_id, lambda: drive(ctx, run_id, action))
    except RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await db.update_status(run_id, status="RUNNING")
    return {"run_id": run_id, "status": "RUNNING"}


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str, request: Request) -> dict[str, Any]:
    run_id = checked_run_id(run_id)
    ctx = get_server_context(request)
    await snapshot_or_404(ctx, run_id)

    if not ctx.registry.cancel(run_id):
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id}를 구동 중인 task가 없다(이미 멈춰 있음)",
        )

    await db.update_status(
        run_id,
        status="STOPPING",
        stop_reason="사용자 중지 요청",
    )
    return {"run_id": run_id, "status": "STOPPING"}


@router.get("/runs/{run_id}/report", response_class=HTMLResponse)
async def get_report(run_id: str, request: Request) -> str:
    run_id = checked_run_id(run_id)
    ctx = get_server_context(request)
    handle = await snapshot_or_404(ctx, run_id)
    report_ref = handle.state.get("report_ref")
    if not report_ref:
        raise HTTPException(
            status_code=404,
            detail=f"run {run_id}에 아직 보고서가 없다",
        )

    ref = ArtifactRef.model_validate(report_ref)
    try:
        return ctx.artifacts.read_text(ref)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc


@router.get("/runs/{run_id}/code")
async def get_code(
    run_id: str,
    request: Request,
    evidence_id: str | None = None,
) -> dict[str, Any]:
    """현재 회차 또는 특정 Evidence의 프로그램 코드를 지연 조회한다."""
    run_id = checked_run_id(run_id)
    ctx = get_server_context(request)
    handle = await snapshot_or_404(ctx, run_id)

    ref_raw: dict[str, Any] | None
    if evidence_id:
        ref_raw = None
        for item in handle.state.get("evidence") or []:
            if item.get("evidence_id") == evidence_id:
                ref_raw = item.get("program_ref")
                break
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"evidence {evidence_id}를 run {run_id}에서 찾을 수 없다"
                ),
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
