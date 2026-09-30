"""진행 여부 판정과 최종 보고서 생성."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from ...contracts import (
    AgendaTask,
    ArtifactKind,
    EvidenceRecord,
    FindingDigest,
    RunStatus,
    StopCode,
)
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...report_validation import validate_report, validate_semantic_review
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    all_goals_closed,
    current_evidence,
    get_agenda,
    get_evidence,
    get_evidence_relations,
    get_profile,
    get_sub_goals,
    pending_agenda_tasks,
    remaining_seconds,
    terminal_update,
    unresolved_contradictions,
)
from ._shared import _event

# 확장 반복 상한. 분석 예산 소진 목적이 아닌 무한 반복 방지용
MAX_GOAL_EXPANSIONS = 5
# 남은 시간이 임계값 미만이면 확장 판단 없이 종료
# LLM 호출 자체의 예산 초과 방지
MIN_SECONDS_FOR_EXPANSION = 90.0

def _repair_contradiction_agenda(
    state: AnalysisGraphState,
) -> tuple[list[dict[str, Any]] | None, list[tuple[str, str]]]:
    """관계 정보는 있으나 Agenda가 없는 상태 복구."""
    unresolved = unresolved_contradictions(state)
    if not unresolved:
        return None, []

    agenda = get_agenda(state)
    covered = {
        frozenset(task.source_evidence_ids)
        for task in agenda
        if task.kind == "resolve_contradiction"
        and task.status in {"pending", "in_progress"}
    }
    missing = [pair for pair in unresolved if frozenset(pair) not in covered]
    if not missing:
        return None, []

    evidence_by_id = {item.evidence_id: item for item in get_evidence(state)}
    updated = list(agenda)
    for left, right in missing:
        left_record = evidence_by_id.get(left)
        right_record = evidence_by_id.get(right)
        left_goal = left_record.goal_id if left_record else None
        right_goal = right_record.goal_id if right_record else None
        goal_id = left_goal if left_goal == right_goal else (left_goal or right_goal)
        updated.append(AgendaTask(
            kind="resolve_contradiction",
            goal_id=goal_id,
            question=f"Evidence {left}와 {right}의 모순을 해소한다",
            rationale="Evidence graph에 미해결 contradiction 관계가 남아 있다",
            priority=100,
            source_evidence_ids=[left, right],
        ))
    return [item.model_dump(mode="json") for item in updated], missing

def _terminal_after_expansion(
    state: AnalysisGraphState,
    *,
    reason_suffix: str,
) -> dict[str, Any]:
    """추가 분석 축 부재 확인 후 종료 코드 결정."""
    goals = get_sub_goals(state)
    abandoned = sum(1 for goal in goals if goal.status == "abandoned")
    limited_tasks = sum(1 for task in get_agenda(state) if task.status == "data_limited")
    converged = sum(1 for goal in goals if goal.status == "converged")
    summary = (
        f"목표 축 {len(goals)}개 처리"
        f"(수렴 {converged} / 포기 {abandoned})"
    )
    if abandoned or limited_tasks:
        return terminal_update(
            status=RunStatus.COMPLETED,
            stop_code=StopCode.DATA_LIMITED,
            reason=f"{summary} — 데이터 한계 작업 {limited_tasks}개{reason_suffix}",
        )
    return terminal_update(
        status=RunStatus.COMPLETED,
        stop_code=StopCode.GOAL_SATISFIED,
        reason=f"{summary}{reason_suffix}",
    )


async def assess_progress(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """시간·회차·Agenda·SubGoal 순으로 진행/종료 상태 판정.

    시간/회차 상한은 미해결 Agenda보다 우선하는 hard limit이다.
    """
    if remaining_seconds(state) <= 0:
        return terminal_update(
            status=RunStatus.COMPLETED,
            stop_code=StopCode.BUDGET_EXHAUSTED,
            reason="시간 예산 소진",
        )

    max_iterations = state.get("max_iterations", 25)
    if state.get("iteration", 0) >= max_iterations:
        return terminal_update(
            status=RunStatus.COMPLETED,
            stop_code=StopCode.MAX_ITERATIONS_REACHED,
            reason=f"최대 {max_iterations}회 분석 완료",
        )

    if pending_agenda_tasks(state):
        return {}

    repaired_agenda, missing_conflicts = _repair_contradiction_agenda(state)
    if repaired_agenda is not None:
        _event(state, deps, "AGENDA_REPAIRED_FROM_RELATION", {
            "contradictions": [list(pair) for pair in missing_conflicts],
        })
        return {"agenda": repaired_agenda}

    if all_goals_closed(state):
        expansions = state.get("goal_expansions", 0)
        seconds = remaining_seconds(state)
        # 추가 가치 판정 실패 시 GOAL_SATISFIED 미설정
        if seconds < MIN_SECONDS_FOR_EXPANSION:
            return terminal_update(
                status=RunStatus.COMPLETED,
                stop_code=StopCode.BUDGET_EXHAUSTED,
                reason=(
                    "현재 목표 축은 닫혔지만 추가 분석 가치 판단에 필요한 "
                    f"시간 예산이 부족하다(남은 {max(0.0, seconds):.1f}초)"
                ),
            )
        if expansions >= MAX_GOAL_EXPANSIONS:
            return terminal_update(
                status=RunStatus.COMPLETED,
                stop_code=StopCode.EXPANSION_LIMIT_REACHED,
                reason=(
                    f"목표 축은 닫혔지만 추가 축 확장 안전 상한 "
                    f"{MAX_GOAL_EXPANSIONS}회에 도달했다"
                ),
            )
        # 추가 가치 판정은 expand_goals에서 수행
        return {}

    # 컬럼 사용 여부만으로 종료하지 않음
    # 강건성·반증·대체 방법은 별도 판정
    return {}

async def expand_goals(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """모든 기존 SubGoal 종료 후 추가 분석 축 평가."""
    goals = get_sub_goals(state)
    expansions = state.get("goal_expansions", 0)

    def _finish(reason: str) -> dict[str, Any]:
        # 추가 SubGoal 부재 확인 후 종료
        return _terminal_after_expansion(
            state, reason_suffix=f" — {reason}",
        )

    def _unverified(reason: str) -> dict[str, Any]:
        """모든 SubGoal 종료 후 추가 가치 판정 대기 상태.

        판정 미완료 상태는 GOAL_SATISFIED와 구분해 기록한다.
        """
        _event(state, deps, "EXPANSION_UNVERIFIED", {"reason": reason})
        return terminal_update(
            status=RunStatus.COMPLETED,
            stop_code=StopCode.SERVICE_UNAVAILABLE,
            reason=reason,
        )

    def _summary() -> str:
        converged = sum(1 for g in goals if g.status == "converged")
        return (
            f"목표 축 {len(goals)}개를 모두 처리했다"
            f"(수렴 {converged} / 포기 {len(goals) - converged})"
        )

    if deps.planner is None:
        return _unverified(f"{_summary()} — planner가 없어 확장 여부를 확인하지 못함")

    profile = get_profile(state)
    assert profile is not None
    findings = [
        FindingDigest.from_evidence(e, data_applicability="exact_snapshot")
        for e in current_evidence(state)
    ]

    remaining_iterations = max(
        0, state.get("max_iterations", 25) - state.get("iteration", 0)
    )
    seconds = remaining_seconds(state)
    new_goals = await deps.planner.expand_goals(
        objective=state.get("objective", ""),
        profile=profile,
        existing_goals=goals,
        findings=findings,
        timeout_seconds=min(DEFAULT_READ_TIMEOUT, max(1.0, seconds)),
        remaining_seconds=seconds,
        remaining_iterations=remaining_iterations,
        expansion_round=expansions + 1,
        max_expansions=MAX_GOAL_EXPANSIONS,
    )
    if not new_goals:
        # 빈 응답은 명시적 stop일 때만 정상 종료
        # contract/provider 실패와 구분
        if getattr(deps.planner, "last_failure_kind", None):
            reason = getattr(deps.planner, "last_error", None) or "확장 판단 실패"
            return _unverified(f"{_summary()} — 확장 판단 실패: {reason}")
        reason = getattr(deps.planner, "last_expansion_reason", None) or (
            "추가 분석이 현재 결론을 실질적으로 바꿀 정보 가치를 찾지 못함"
        )
        _event(state, deps, "EXPANSION_DECIDED", {
            "decision": "stop",
            "reason": reason,
            "expansion_round": expansions + 1,
        })
        return _finish(reason)

    # 신규 SubGoal 수는 잔여 회차 수로 제한
    if remaining_iterations <= 0:
        return terminal_update(
            status=RunStatus.COMPLETED,
            stop_code=StopCode.MAX_ITERATIONS_REACHED,
            reason="새 목표 축 후보가 있지만 남은 분석 회차가 없다",
        )
    if len(new_goals) > remaining_iterations:
        skipped = [g.question for g in new_goals[remaining_iterations:]]
        new_goals = new_goals[:remaining_iterations]
        _event(state, deps, "GOAL_EXPANSION_TRIMMED", {
            "remaining_iterations": remaining_iterations,
            "skipped_questions": skipped,
        })

    _event(state, deps, "EXPANSION_DECIDED", {
        "decision": "expand",
        "reason": getattr(deps.planner, "last_expansion_reason", None) or "",
        "new_goal_ids": [g.goal_id for g in new_goals],
        "expansion_round": expansions + 1,
    })
    _event(state, deps, "GOALS_EXPANDED", {
        "new_goal_ids": [g.goal_id for g in new_goals],
        "questions": [g.question for g in new_goals],
        "expansion_round": expansions + 1,
    })
    merged = [*goals, *new_goals]
    return {
        "sub_goals": [g.model_dump(mode="json") for g in merged],
        "goal_expansions": expansions + 1,
    }

def _reportable_evidence(state: AnalysisGraphState) -> list[EvidenceRecord]:
    """현재 데이터에서 사용 가능하고 superseded되지 않은 Evidence."""
    return current_evidence(state)

def _run_summary(state: AnalysisGraphState) -> dict[str, Any]:
    """보고서 생성 입력. Critic 실패 시 구조화 JSON 폴백에 사용."""
    evidence = state.get("evidence") or []
    reportable = _reportable_evidence(state)
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
            item.model_dump(mode="json") for item in pending_agenda_tasks(state)
        ],
        "evidence_relations": [
            item.model_dump(mode="json") for item in get_evidence_relations(state)
        ],
        "active_evidence_ids": [item.evidence_id for item in reportable],
        "incomplete_goal_ids": [
            goal.goal_id for goal in get_sub_goals(state)
            if goal.status != "converged"
        ],
        "pending_task_ids": [
            task.task_id for task in pending_agenda_tasks(state)
        ],
        "unresolved_contradictions": [
            list(pair) for pair in unresolved_contradictions(state)
        ],
        "stop_reason": state.get("stop_reason"),
        # 진행률은 Evidence 수가 아닌 SubGoal 상태 기준
        "goals": [
            {
                "goal_id": g.goal_id,
                "question": g.question,
                "status": g.status,
                "evidence_count": len(g.evidence_ids),
                "closing_note": g.closing_note,
            }
            for g in get_sub_goals(state)
        ],
        "failure_issues": state.get("validation_errors") or [],
        # 보고서 주장별 evidence_id 참조 필수
        "claims": [
            {"evidence_id": e["evidence_id"], "question": e["question"], "claim": e["claim"]}
            for e in evidence
        ],
    }

async def _review_report(
    state: AnalysisGraphState, deps: RuntimeDeps, *, markdown: str,
    summary: dict[str, Any], evidence: list[EvidenceRecord],
    cited_ids: set[str], results: dict[str, Any],
) -> dict[str, Any]:
    """구조·수치 검사를 통과한 초안에만 의미 검토를 요청한다. 실패하면 미검토다."""
    outcome: dict[str, Any] = {
        "passed": False, "status": "unreviewed",
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
            raw = await asyncio.wait_for(reviewer(
                markdown=markdown, run_summary=summary,
                evidence=[item.model_dump(mode="json") for item in evidence
                          if item.evidence_id in cited_ids],
                results={key: results[key] for key in sorted(cited_ids)},
                dataset_profile=profile.model_dump(mode="json") if profile else None,
                timeout_seconds=DEFAULT_READ_TIMEOUT,
            ), timeout=DEFAULT_READ_TIMEOUT)
            if raw is None:
                reason = getattr(deps.critic, "last_error", None) or "응답이 없다"
                raise ValueError(reason)
            review = validate_semantic_review(raw, markdown=markdown, evidence_ids=cited_ids)
            outcome.update(
                passed=review.verdict == "accept",
                status="accepted" if review.verdict == "accept" else "rejected",
                review=review.model_dump(mode="json"),
            )
        except Exception as exc:  # noqa: BLE001 - 검토 실패로 보고서를 승인하지 않는다.
            outcome["error"] = f"보고서 의미 검토 실패: {type(exc).__name__}: {exc}"[:500]
    _event(state, deps, "REPORT_REVIEWED", outcome)
    return outcome


async def _write_report(
    state: AnalysisGraphState, deps: RuntimeDeps, summary: dict[str, Any],
) -> tuple[Any, bool]:
    """검증된 Markdown 채택. 실패 시 구조화 JSON 요약으로 폴백."""
    all_evidence = get_evidence(state)
    reportable = _reportable_evidence(state)
    numeric_support: dict[str, list[float]] = {}
    original_results: dict[str, Any] = {}
    for item in reportable:
        try:
            payload = deps.artifacts.read_json(item.result_ref)
        except (OSError, ValueError, TypeError):
            numeric_support[item.evidence_id] = []
            continue
        original_results[item.evidence_id] = payload

        def collect(value: Any) -> list[float]:
            numbers: list[float] = []
            if isinstance(value, bool):
                return numbers
            if isinstance(value, (int, float)):
                numbers.append(float(value))
            elif isinstance(value, dict):
                for sub in value.values():
                    numbers.extend(collect(sub))
            elif isinstance(value, (list, tuple)):
                for sub in value:
                    numbers.extend(collect(sub))
            return numbers

        numeric_support[item.evidence_id] = collect(payload)

    report_md = None
    if deps.critic is not None and state.get("stop_code") != StopCode.BUDGET_EXHAUSTED.value:
        report_md = await deps.critic.generate_report(
            run_summary=summary,
            evidence=[item.model_dump(mode="json") for item in reportable],
            rejected=state.get("rejected") or [],
            timeout_seconds=DEFAULT_READ_TIMEOUT,
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
            semantic = await _review_report(
                state, deps, markdown=report_md, summary=summary, evidence=reportable,
                cited_ids=set(validation["cited_evidence_ids"]), results=original_results,
            )
            if semantic["passed"]:
                ref = deps.artifacts.put_text(
                    report_md, kind=ArtifactKind.REPORT, suffix=".md",
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
        summary, kind=ArtifactKind.REPORT,
        summary={"evidence_count": len(reportable)},
    )
    return ref, False

async def finalize(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """최종 보고서 Artifact와 종료 메타데이터 생성.

    정상 완료뿐 아니라 예산 소진·실패 경로에서도 동일한 결과 조회 인터페이스를 유지한다.
    """
    summary = _run_summary(state)
    ref, is_markdown = await _write_report(state, deps, summary)

    _event(state, deps, "FINALIZED", {
        "evidence_count": summary["evidence_count"], "has_md_report": is_markdown,
    })
    return {
        "report_ref": ref.model_dump(mode="json"),
        "status": state.get("status") or RunStatus.COMPLETED.value,
        # stop_code 누락 경로는 UNKNOWN으로 노출
        "stop_code": state.get("stop_code") or StopCode.UNKNOWN.value,
        "stop_reason": state.get("stop_reason") or "완료",
    }
