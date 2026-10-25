"""진행/종료 판정에서 사용하는 순수 정책 helper."""

from __future__ import annotations

from typing import Any

from ...contracts import AgendaTask, RunStatus, StopCode
from ...state import (
    AnalysisGraphState,
    get_agenda,
    get_evidence,
    get_sub_goals,
    terminal_update,
    unresolved_contradictions,
)

MAX_GOAL_EXPANSIONS = 5
MIN_SECONDS_FOR_EXPANSION = 90.0


def repair_contradiction_agenda(
    state: AnalysisGraphState,
) -> tuple[list[dict[str, Any]] | None, list[tuple[str, str]]]:
    """Evidence relation은 있으나 Agenda가 없는 contradiction을 복구한다."""
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


def terminal_after_expansion(
    state: AnalysisGraphState,
    *,
    reason_suffix: str,
) -> dict[str, Any]:
    """추가 분석 축이 없을 때 적절한 종료 코드를 만든다."""
    goals = get_sub_goals(state)
    abandoned = sum(1 for goal in goals if goal.status == "abandoned")
    limited_tasks = sum(
        1 for task in get_agenda(state) if task.status == "data_limited"
    )
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


_repair_contradiction_agenda = repair_contradiction_agenda
_terminal_after_expansion = terminal_after_expansion
