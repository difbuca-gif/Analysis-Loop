"""계속할지 끝낼지 정하고, 끝난다면 보고서를 남기는 단계."""

from __future__ import annotations

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
from ...report_validation import validate_report
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

# 확장이 무한 반복되는 것만 막는 안전망("예산을 채운다"가 아니라 "고도화하다 끝나면 멈춘다").
MAX_GOAL_EXPANSIONS = 5
# 시간이 이보다 적게 남았으면 확장을 시도하지 않고 바로 종료한다. 확장
# 판단(LLM 호출) 자체가 남은 예산을 넘길 수 있어서다.
MIN_SECONDS_FOR_EXPANSION = 90.0

def _repair_contradiction_agenda(
    state: AnalysisGraphState,
) -> tuple[list[dict[str, Any]] | None, list[tuple[str, str]]]:
    """관계는 있는데 Agenda가 빠진 비정상 상태를 종료 전에 복구한다."""
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
    """'더 볼 축 없음'이 확인된 뒤 결과 수준에 맞는 종료 코드를 고른다."""
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

# 진척 평가 / 종료

async def assess_progress(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """시간·회차·목표 상태로 종료 여부를 정한다."""
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
        # 추가 가치 판단을 하지 못했으면 목표 달성이라고 과장하지 않는다.
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
        # 실제 '더 할 가치가 없는가' 판단은 expand_goals가 수행한다.
        return {}

    # 컬럼을 한 번씩 사용했다는 이유만으로 끝내지 않는다. 한 프로그램이 모든 컬럼을
    # 선택해도 강건성·반증·대체 방법 검토가 남을 수 있다.
    return {}

async def expand_goals(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """기존 목표가 모두 닫힌 뒤 추가 분석 축을 찾는다."""
    goals = get_sub_goals(state)
    expansions = state.get("goal_expansions", 0)

    def _finish(reason: str) -> dict[str, Any]:
        # 추가 목표가 없음을 확인한 뒤 종료한다.
        return _terminal_after_expansion(
            state, reason_suffix=f" — {reason}",
        )

    def _unverified(reason: str) -> dict[str, Any]:
        """축은 닫혔지만 "더 볼 게 없는지"를 확인하지 못했다.

        GOAL_SATISFIED로 보고하면 갖지 않은 지식을 주장하는 것이다. stop_code를
        도입한 이유가 정확히 이 구분이다. 일은 됐으므로 status는 COMPLETED로 둔다.
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
        # 빈 응답은 명시적 stop 판단일 때만 정상 종료다. 계약/공급자 실패를
        # "더 볼 게 없다"로 오인하면 GOAL_SATISFIED가 거짓이 된다.
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

    # 남은 회차보다 많은 축을 열면 새 축을 만들자마자 MAX_ITERATIONS로 끝난다.
    # 모델 판단을 존중하되 실행 가능한 수만 남기는 것은 런타임의 책임이다.
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
    """현재 데이터에서 직접 쓸 수 있고 superseded되지 않은 Evidence만."""
    return current_evidence(state)

def _run_summary(state: AnalysisGraphState) -> dict[str, Any]:
    """보고서 생성기에 넘길 요약. Critic이 실패하면 이것이 그대로 보고서가 된다."""
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
        # 목표 대비 진척. 근거 개수가 아니라 축별 상태로 보고한다.
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
        # 보고서의 모든 주장은 evidence_id를 참조한다.
        "claims": [
            {"evidence_id": e["evidence_id"], "question": e["question"], "claim": e["claim"]}
            for e in evidence
        ],
    }

async def _write_report(
    state: AnalysisGraphState, deps: RuntimeDeps, summary: dict[str, Any],
) -> tuple[Any, bool]:
    """검증된 Markdown만 채택하고, 실패하면 구조화 JSON 요약으로 폴백한다."""
    all_evidence = get_evidence(state)
    reportable = _reportable_evidence(state)
    numeric_support: dict[str, list[float]] = {}
    for item in reportable:
        try:
            payload = deps.artifacts.read_json(item.result_ref)
        except (OSError, ValueError, TypeError):
            numeric_support[item.evidence_id] = []
            continue

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
            ref = deps.artifacts.put_text(
                report_md, kind=ArtifactKind.REPORT, suffix=".md",
                summary={
                    "evidence_count": len(reportable),
                    "citation_count": validation["citation_count"],
                },
            )
            return ref, True
        summary = {**summary, "report_validation": validation}
    ref = deps.artifacts.put_json(
        summary, kind=ArtifactKind.REPORT,
        summary={"evidence_count": len(reportable)},
    )
    return ref, False

async def finalize(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    summary = _run_summary(state)
    ref, is_markdown = await _write_report(state, deps, summary)

    _event(state, deps, "FINALIZED", {
        "evidence_count": summary["evidence_count"], "has_md_report": is_markdown,
    })
    return {
        "report_ref": ref.model_dump(mode="json"),
        "status": state.get("status") or RunStatus.COMPLETED.value,
        # 종료 코드가 없으면 UNKNOWN으로 남겨 누락 경로를 드러낸다.
        "stop_code": state.get("stop_code") or StopCode.UNKNOWN.value,
        "stop_reason": state.get("stop_reason") or "완료",
    }
