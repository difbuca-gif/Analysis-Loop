"""최종 보고서 생성, 검증, semantic review orchestration."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from ...contracts import ArtifactKind, EvidenceRecord, RunStatus, StopCode
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...report_validation import validate_report, validate_semantic_review
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    current_evidence,
    get_evidence,
    get_evidence_relations,
    get_profile,
    get_sub_goals,
    pending_agenda_tasks,
    unresolved_contradictions,
)
from ._shared import _event


def reportable_evidence(state: AnalysisGraphState) -> list[EvidenceRecord]:
    """현재 데이터에서 사용 가능하고 superseded되지 않은 Evidence."""
    return current_evidence(state)


def run_summary(state: AnalysisGraphState) -> dict[str, Any]:
    """보고서 생성 입력과 JSON fallback에 공통으로 사용할 실행 요약."""
    evidence = state.get("evidence") or []
    reportable = reportable_evidence(state)
    return {
        "run_id": state.get("run_id"),
        "objective": state.get("objective"),
        "status": state.get("status"),
        "stop_code": state.get("stop_code"),
        "iterations": state.get("iteration", 0),
        "evidence_count": len(evidence),
        "rejected_count": len(state.get("rejected") or []),
        "agenda": state.get("agenda") or [],
        "pending_agenda": [
            item.model_dump(mode="json")
            for item in pending_agenda_tasks(state)
        ],
        "evidence_relations": [
            item.model_dump(mode="json")
            for item in get_evidence_relations(state)
        ],
        "active_evidence_ids": [item.evidence_id for item in reportable],
        "incomplete_goal_ids": [
            goal.goal_id
            for goal in get_sub_goals(state)
            if goal.status != "converged"
        ],
        "pending_task_ids": [
            task.task_id for task in pending_agenda_tasks(state)
        ],
        "unresolved_contradictions": [
            list(pair) for pair in unresolved_contradictions(state)
        ],
        "stop_reason": state.get("stop_reason"),
        "goals": [
            {
                "goal_id": goal.goal_id,
                "question": goal.question,
                "status": goal.status,
                "evidence_count": len(goal.evidence_ids),
                "closing_note": goal.closing_note,
            }
            for goal in get_sub_goals(state)
        ],
        "failure_issues": state.get("validation_errors") or [],
        "claims": [
            {
                "evidence_id": item["evidence_id"],
                "question": item["question"],
                "claim": item["claim"],
            }
            for item in evidence
        ],
    }


async def review_report(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    markdown: str,
    summary: dict[str, Any],
    evidence: list[EvidenceRecord],
    cited_ids: set[str],
    results: dict[str, Any],
    timeout_seconds: float = DEFAULT_READ_TIMEOUT,
) -> dict[str, Any]:
    """구조·수치 검사를 통과한 초안만 의미 검토한다."""
    outcome: dict[str, Any] = {
        "passed": False,
        "status": "unreviewed",
        "report_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
    }
    missing = cited_ids - results.keys()
    reviewer = getattr(deps.critic, "review_report", None)
    if missing:
        outcome["error"] = f"인용 근거의 원본 결과를 읽을 수 없다: {sorted(missing)}"
    elif not callable(reviewer):
        outcome["error"] = "보고서 의미 검토 서비스가 없다"
    else:
        try:
            profile = get_profile(state)
            raw = await asyncio.wait_for(
                reviewer(
                    markdown=markdown,
                    run_summary=summary,
                    evidence=[
                        item.model_dump(mode="json")
                        for item in evidence
                        if item.evidence_id in cited_ids
                    ],
                    results={key: results[key] for key in sorted(cited_ids)},
                    dataset_profile=(
                        profile.model_dump(mode="json") if profile else None
                    ),
                    timeout_seconds=timeout_seconds,
                ),
                timeout=timeout_seconds,
            )
            if raw is None:
                reason = getattr(deps.critic, "last_error", None) or "응답이 없다"
                raise ValueError(reason)
            review = validate_semantic_review(
                raw,
                markdown=markdown,
                evidence_ids=cited_ids,
            )
            outcome.update(
                passed=review.verdict == "accept",
                status="accepted" if review.verdict == "accept" else "rejected",
                review=review.model_dump(mode="json"),
            )
        except Exception as exc:  # noqa: BLE001 - 의미 검토 실패는 승인하지 않는다.
            outcome["error"] = (
                f"보고서 의미 검토 실패: {type(exc).__name__}: {exc}"[:500]
            )

    _event(state, deps, "REPORT_REVIEWED", outcome)
    return outcome


def _collect_numeric_values(value: Any) -> list[float]:
    """원본 Result의 numeric leaf를 재귀적으로 모은다."""
    numbers: list[float] = []
    if isinstance(value, bool):
        return numbers
    if isinstance(value, (int, float)):
        numbers.append(float(value))
    elif isinstance(value, dict):
        for sub in value.values():
            numbers.extend(_collect_numeric_values(sub))
    elif isinstance(value, (list, tuple)):
        for sub in value:
            numbers.extend(_collect_numeric_values(sub))
    return numbers


async def write_report(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    summary: dict[str, Any],
    *,
    timeout_seconds: float = DEFAULT_READ_TIMEOUT,
) -> tuple[Any, bool]:
    """검증된 Markdown을 채택하고 실패하면 구조화 JSON으로 fallback한다."""
    all_evidence = get_evidence(state)
    reportable = reportable_evidence(state)
    numeric_support: dict[str, list[float]] = {}
    original_results: dict[str, Any] = {}

    for item in reportable:
        try:
            payload = deps.artifacts.read_json(item.result_ref)
        except (OSError, ValueError, TypeError):
            numeric_support[item.evidence_id] = []
            continue
        original_results[item.evidence_id] = payload
        numeric_support[item.evidence_id] = _collect_numeric_values(payload)

    report_md = None
    if (
        deps.critic is not None
        and state.get("stop_code") != StopCode.BUDGET_EXHAUSTED.value
    ):
        report_md = await deps.critic.generate_report(
            run_summary=summary,
            evidence=[item.model_dump(mode="json") for item in reportable],
            rejected=state.get("rejected") or [],
            timeout_seconds=timeout_seconds,
        )

    if report_md:
        validation = validate_report(
            report_md,
            evidence=all_evidence,
            relations=get_evidence_relations(state),
            allowed_evidence_ids={item.evidence_id for item in reportable},
            numeric_support=numeric_support,
            stop_code=state.get("stop_code"),
            incomplete_goal_ids=summary["incomplete_goal_ids"],
            pending_task_ids=summary["pending_task_ids"],
            unresolved_contradictions=[
                tuple(pair) for pair in summary["unresolved_contradictions"]
            ],
        )
        _event(state, deps, "REPORT_VALIDATED", {
            "passed": validation["passed"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "citation_count": validation["citation_count"],
        })

        if validation["passed"]:
            semantic = await review_report(
                state,
                deps,
                markdown=report_md,
                summary=summary,
                evidence=reportable,
                cited_ids=set(validation["cited_evidence_ids"]),
                results=original_results,
                timeout_seconds=timeout_seconds,
            )
            if semantic["passed"]:
                ref = deps.artifacts.put_text(
                    report_md,
                    kind=ArtifactKind.REPORT,
                    suffix=".md",
                    summary={
                        "evidence_count": len(reportable),
                        "citation_count": validation["citation_count"],
                        "semantic_review_passed": True,
                        "report_sha256": semantic["report_sha256"],
                    },
                )
                return ref, True
            summary = {**summary, "report_review": semantic}

        summary = {**summary, "report_validation": validation}

    ref = deps.artifacts.put_json(
        summary,
        kind=ArtifactKind.REPORT,
        summary={"evidence_count": len(reportable)},
    )
    return ref, False


async def finalize(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    timeout_seconds: float = DEFAULT_READ_TIMEOUT,
) -> dict[str, Any]:
    """최종 보고서 Artifact와 종료 메타데이터를 만든다."""
    summary = run_summary(state)
    ref, is_markdown = await write_report(
        state,
        deps,
        summary,
        timeout_seconds=timeout_seconds,
    )

    _event(state, deps, "FINALIZED", {
        "evidence_count": summary["evidence_count"],
        "has_md_report": is_markdown,
    })
    return {
        "report_ref": ref.model_dump(mode="json"),
        "status": state.get("status") or RunStatus.COMPLETED.value,
        "stop_code": state.get("stop_code") or StopCode.UNKNOWN.value,
        "stop_reason": state.get("stop_reason") or "완료",
    }


_reportable_evidence = reportable_evidence
_run_summary = run_summary
_review_report = review_report
_write_report = write_report
