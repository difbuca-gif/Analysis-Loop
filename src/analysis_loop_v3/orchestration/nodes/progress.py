"""분석 진행 여부와 Goal 확장 판단.

종료 정책은 이 파일의 순수 helper로, 보고서 생성은 reporting.py로 구분한다.
기존 private helper import 경로는 compatibility wrapper로 유지한다.
"""

from __future__ import annotations

from typing import Any

from ...contracts import AgendaTask, ArtifactRef, FindingDigest, RunStatus, StopCode
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    all_goals_closed,
    current_evidence,
    get_agenda,
    get_evidence,
    get_profile,
    get_sub_goals,
    pending_agenda_tasks,
    remaining_seconds,
    terminal_update,
    unresolved_contradictions,
)
from ..node_updates import FinalizeUpdate, ProgressUpdate
from . import reporting as _reporting
from ._shared import _event

MAX_GOAL_EXPANSIONS = 5
MIN_SECONDS_FOR_EXPANSION = 90.0


async def assess_progress(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> ProgressUpdate:
    """시간·회차·Agenda·SubGoal 순서로 진행/종료 상태를 판정한다."""
    if state.get("status") in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
        return {}

    if remaining_seconds(state) <= 0:
        return {
            **terminal_update(
                status=RunStatus.COMPLETED,
                stop_code=StopCode.BUDGET_EXHAUSTED,
                reason="시간 예산 소진",
            )
        }

    max_iterations = state.get("max_iterations", 25)
    if state.get("iteration", 0) >= max_iterations:
        return {
            **terminal_update(
                status=RunStatus.COMPLETED,
                stop_code=StopCode.MAX_ITERATIONS_REACHED,
                reason=f"최대 {max_iterations}회 분석 완료",
            )
        }

    if pending_agenda_tasks(state):
        return {}

    repaired_agenda, missing_conflicts = repair_contradiction_agenda(state)
    if repaired_agenda is not None:
        _event(
            state,
            deps,
            "AGENDA_REPAIRED_FROM_RELATION",
            {
                "contradictions": [list(pair) for pair in missing_conflicts],
            },
        )
        return {"agenda": repaired_agenda}

    if all_goals_closed(state):
        expansions = state.get("goal_expansions", 0)
        seconds = remaining_seconds(state)
        if seconds < MIN_SECONDS_FOR_EXPANSION:
            return {
                **terminal_update(
                    status=RunStatus.COMPLETED,
                    stop_code=StopCode.BUDGET_EXHAUSTED,
                    reason=(
                        "현재 목표 축은 닫혔지만 추가 분석 가치 판단에 필요한 "
                        f"시간 예산이 부족하다(남은 {max(0.0, seconds):.1f}초)"
                    ),
                )
            }
        if expansions >= MAX_GOAL_EXPANSIONS:
            return {
                **terminal_update(
                    status=RunStatus.COMPLETED,
                    stop_code=StopCode.EXPANSION_LIMIT_REACHED,
                    reason=(
                        f"목표 축은 닫혔지만 추가 축 확장 안전 상한 {MAX_GOAL_EXPANSIONS}회에 도달했다"
                    ),
                )
            }
        return {}

    return {}


async def expand_goals(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> ProgressUpdate:
    """모든 기존 SubGoal 종료 후 추가 분석 축의 정보 가치를 평가한다."""
    goals = get_sub_goals(state)
    expansions = state.get("goal_expansions", 0)

    def finish(reason: str) -> ProgressUpdate:
        return terminal_after_expansion(
            state,
            reason_suffix=f" — {reason}",
        )

    def unverified(reason: str) -> ProgressUpdate:
        _event(state, deps, "EXPANSION_UNVERIFIED", {"reason": reason})
        return {
            **terminal_update(
                status=RunStatus.COMPLETED,
                stop_code=StopCode.SERVICE_UNAVAILABLE,
                reason=reason,
            )
        }

    def summary() -> str:
        converged = sum(1 for goal in goals if goal.status == "converged")
        return (
            f"목표 축 {len(goals)}개를 모두 처리했다"
            f"(수렴 {converged} / 포기 {len(goals) - converged})"
        )

    if deps.planner is None:
        return unverified(f"{summary()} — planner가 없어 확장 여부를 확인하지 못함")

    profile = get_profile(state)
    assert profile is not None
    findings = [
        FindingDigest.from_evidence(
            evidence,
            data_applicability="exact_snapshot",
        )
        for evidence in current_evidence(state)
    ]

    remaining_iterations = max(
        0,
        state.get("max_iterations", 25) - state.get("iteration", 0),
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
        if getattr(deps.planner, "last_failure_kind", None):
            reason = getattr(deps.planner, "last_error", None) or "확장 판단 실패"
            return unverified(f"{summary()} — 확장 판단 실패: {reason}")

        reason = (
            getattr(deps.planner, "last_expansion_reason", None)
            or "추가 분석이 현재 결론을 실질적으로 바꿀 정보 가치를 찾지 못함"
        )
        _event(
            state,
            deps,
            "EXPANSION_DECIDED",
            {
                "decision": "stop",
                "reason": reason,
                "expansion_round": expansions + 1,
            },
        )
        return finish(reason)

    if remaining_iterations <= 0:
        return {
            **terminal_update(
                status=RunStatus.COMPLETED,
                stop_code=StopCode.MAX_ITERATIONS_REACHED,
                reason="새 목표 축 후보가 있지만 남은 분석 회차가 없다",
            )
        }

    if len(new_goals) > remaining_iterations:
        skipped = [goal.question for goal in new_goals[remaining_iterations:]]
        new_goals = new_goals[:remaining_iterations]
        _event(
            state,
            deps,
            "GOAL_EXPANSION_TRIMMED",
            {
                "remaining_iterations": remaining_iterations,
                "skipped_questions": skipped,
            },
        )

    _event(
        state,
        deps,
        "EXPANSION_DECIDED",
        {
            "decision": "expand",
            "reason": getattr(deps.planner, "last_expansion_reason", None) or "",
            "new_goal_ids": [goal.goal_id for goal in new_goals],
            "expansion_round": expansions + 1,
        },
    )
    _event(
        state,
        deps,
        "GOALS_EXPANDED",
        {
            "new_goal_ids": [goal.goal_id for goal in new_goals],
            "questions": [goal.question for goal in new_goals],
            "expansion_round": expansions + 1,
        },
    )

    merged = [*goals, *new_goals]
    return {
        "sub_goals": [goal.model_dump(mode="json") for goal in merged],
        "goal_expansions": expansions + 1,
    }


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
        if task.kind == "resolve_contradiction" and task.status in {"pending", "in_progress"}
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
        updated.append(
            AgendaTask(
                kind="resolve_contradiction",
                goal_id=goal_id,
                question=f"Evidence {left}와 {right}의 모순을 해소한다",
                rationale="Evidence graph에 미해결 contradiction 관계가 남아 있다",
                priority=100,
                source_evidence_ids=[left, right],
            )
        )

    return [item.model_dump(mode="json") for item in updated], missing


def terminal_after_expansion(
    state: AnalysisGraphState,
    *,
    reason_suffix: str,
) -> ProgressUpdate:
    """추가 분석 축이 없을 때 적절한 종료 코드를 만든다."""
    goals = get_sub_goals(state)
    abandoned = sum(1 for goal in goals if goal.status == "abandoned")
    limited_tasks = sum(1 for task in get_agenda(state) if task.status == "data_limited")
    converged = sum(1 for goal in goals if goal.status == "converged")
    summary = f"목표 축 {len(goals)}개 처리(수렴 {converged} / 포기 {abandoned})"

    if abandoned or limited_tasks:
        return {
            **terminal_update(
                status=RunStatus.COMPLETED,
                stop_code=StopCode.DATA_LIMITED,
                reason=f"{summary} — 데이터 한계 작업 {limited_tasks}개{reason_suffix}",
            )
        }
    return {
        **terminal_update(
            status=RunStatus.COMPLETED,
            stop_code=StopCode.GOAL_SATISFIED,
            reason=f"{summary}{reason_suffix}",
        )
    }


_repair_contradiction_agenda = repair_contradiction_agenda
_terminal_after_expansion = terminal_after_expansion


# -- compatibility wrappers -------------------------------------------------
# 기존 테스트/호출부가 progress 모듈의 helper를 직접 참조해도 동작하게 유지한다.
_reportable_evidence = _reporting.reportable_evidence
_run_summary = _reporting.run_summary


async def finalize(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> FinalizeUpdate:
    return await _reporting.finalize(
        state,
        deps,
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )


async def _review_report(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    **kwargs: Any,
) -> dict[str, Any]:
    return await _reporting.review_report(
        state,
        deps,
        timeout_seconds=DEFAULT_READ_TIMEOUT,
        **kwargs,
    )


async def _write_report(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    summary: dict[str, Any],
) -> tuple[ArtifactRef, bool]:
    return await _reporting.write_report(
        state,
        deps,
        summary,
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )


__all__ = [
    "MAX_GOAL_EXPANSIONS",
    "MIN_SECONDS_FOR_EXPANSION",
    "assess_progress",
    "expand_goals",
    "finalize",
    "_repair_contradiction_agenda",
    "_terminal_after_expansion",
    "_reportable_evidence",
    "_run_summary",
    "_review_report",
    "_write_report",
]
