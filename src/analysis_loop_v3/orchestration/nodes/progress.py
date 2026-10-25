"""분석 진행 여부와 Goal 확장 판단.

종료 정책은 progress_policy.py, 보고서 생성은 reporting.py에 분리한다.
기존 private helper import 경로는 compatibility wrapper로 유지한다.
"""

from __future__ import annotations

from typing import Any

from ...contracts import FindingDigest, RunStatus, StopCode
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    all_goals_closed,
    current_evidence,
    get_profile,
    get_sub_goals,
    pending_agenda_tasks,
    remaining_seconds,
    terminal_update,
)
from ..node_updates import FinalizeUpdate, ProgressUpdate
from . import reporting as _reporting
from ._shared import _event
from .progress_policy import (
    MAX_GOAL_EXPANSIONS,
    MIN_SECONDS_FOR_EXPANSION,
    _repair_contradiction_agenda,
    _terminal_after_expansion,
)


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

    repaired_agenda, missing_conflicts = _repair_contradiction_agenda(state)
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
        return _terminal_after_expansion(
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


# -- compatibility wrappers -------------------------------------------------
# 기존 테스트/호출부가 progress 모듈의 helper를 직접 참조해도 동작하게 유지한다.
_reportable_evidence = _reporting.reportable_evidence
_run_summary = _reporting.run_summary


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
) -> tuple[Any, bool]:
    return await _reporting.write_report(
        state,
        deps,
        summary,
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )


async def finalize(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> FinalizeUpdate:
    return await _reporting.finalize(
        state,
        deps,
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
