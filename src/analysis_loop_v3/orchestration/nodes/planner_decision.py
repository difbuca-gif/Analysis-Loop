"""Planner 응답 이후 Goal 진행, 실패, Intent 거부를 결정하는 정책."""

from __future__ import annotations

from typing import Any

from ...contracts import (
    AnalysisCoverage,
    DatasetProfile,
    LoopIssue,
    ResearchIntent,
    RunStatus,
    StopCode,
    SubGoal,
)
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    current_evidence,
    get_evidence,
    pending_agenda_tasks,
    terminal_update,
)
from ...validation.checks import validate_intent_references
from ..failure_policy import is_terminal_service_failure
from ._shared import STUCK_THRESHOLD, _event, bump_error_streak
from .goal_policy import apply_goal_verdict
from .intent_policy import (
    agenda_intent_error,
    repeated_action_error,
    reuse_evidence_error,
)

NO_INTENT_CODE = "planner_no_intent"


def stalled(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    goal_update: dict[str, Any],
    reason: str | None = None,
    planner_error: str | None = None,
) -> dict[str, Any]:
    """Intent 미생성 회차를 기록하고 연속 실패 시 종료한다."""
    streak = bump_error_streak(state, [{"code": NO_INTENT_CODE}])
    update: dict[str, Any] = {
        **goal_update,
        "intent": None,
        "error_streak": streak,
        "last_planner_error": planner_error,
    }
    if streak.get("count", 0) < STUCK_THRESHOLD:
        return update

    issue = LoopIssue(
        label=NO_INTENT_CODE,
        count=int(streak["count"]),
        iteration=int(state.get("iteration", 0)),
        messages=[m for m in (reason, planner_error) if m][:2],
    )
    _event(state, deps, "PLANNER_STUCK", issue.model_dump(mode="json"))
    return {
        **update,
        "issues": [issue.model_dump(mode="json")],
        **terminal_update(
            status=RunStatus.FAILED,
            stop_code=StopCode.PLANNER_STUCK,
            reason=(
                f"플래너가 {streak['count']}회 연속 분석 의도를 만들지 못했다"
            ),
        ),
    }


def goal_progress(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    goals: list[SubGoal],
    verdict: dict[str, Any],
) -> tuple[list[SubGoal], dict[str, Any]]:
    """Intent 생성 여부와 독립적으로 SubGoal 판정을 반영한다."""
    evidence_by_goal: dict[str, list[str]] = {}
    for record in current_evidence(state):
        if record.goal_id:
            evidence_by_goal.setdefault(record.goal_id, []).append(record.evidence_id)

    attempts_by_goal = {
        goal: len(ids)
        for goal, ids in evidence_by_goal.items()
    }
    for record in state.get("rejected") or []:
        if goal := record.get("goal_id"):
            attempts_by_goal[goal] = attempts_by_goal.get(goal, 0) + 1

    blocked_goal_ids = {
        item.goal_id
        for item in pending_agenda_tasks(state)
        if item.goal_id
    }
    updated, changes = apply_goal_verdict(
        goals,
        verdict,
        evidence_by_goal,
        attempts_by_goal,
        blocked_goal_ids=blocked_goal_ids,
    )

    update: dict[str, Any] = {}
    if unreviewed := verdict.get("unreviewed"):
        _event(state, deps, "GOALS_UNREVIEWED", {"goal_ids": unreviewed})
    if changes:
        _event(state, deps, "GOAL_STATUS_CHANGED", {"changes": changes})
        update["sub_goals"] = [
            goal.model_dump(mode="json")
            for goal in updated
        ]
    return updated, update


def no_intent(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    verdict: dict[str, Any],
    updated_goals: list[SubGoal],
    goal_update: dict[str, Any],
) -> dict[str, Any]:
    """Planner가 Intent를 내지 않았을 때 완료/실패/재시도를 구분한다."""
    if verdict.get("finished") and all(
        not goal.is_open for goal in updated_goals
    ):
        pending = pending_agenda_tasks(state)
        if pending:
            return stalled(
                state,
                deps,
                goal_update=goal_update,
                reason="미해결 Agenda 작업이 남아 있는데 종료를 선언했다",
                planner_error=(
                    "pending task를 먼저 처리하라: "
                    + ", ".join(item.task_id for item in pending[:3])
                ),
            )
        return {
            **goal_update,
            "intent": None,
            "last_planner_error": None,
        }

    reason = getattr(deps.planner, "last_error", None)
    if not reason:
        return stalled(
            state,
            deps,
            goal_update=goal_update,
            reason="플래너가 의도를 내지 않았다",
        )

    _event(state, deps, "PLANNER_FAILED", {"reason": reason})
    kind = getattr(deps.planner, "last_failure_kind", None)
    if is_terminal_service_failure(kind):
        return {
            **goal_update,
            "intent": None,
            **terminal_update(
                status=RunStatus.FAILED,
                stop_code=StopCode.SERVICE_UNAVAILABLE,
                reason=f"planner 실패: {reason}",
            ),
        }

    return stalled(
        state,
        deps,
        goal_update=goal_update,
        reason="플래너 응답이 계약을 어겼다",
        planner_error=reason,
    )


def rejected_intent(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    intent: ResearchIntent,
    *,
    profile: DatasetProfile,
    goals: list[SubGoal],
    evidence_ids: set[str],
    coverage: AnalysisCoverage,
    updated_goals: list[SubGoal],
    goal_update: dict[str, Any],
) -> dict[str, Any] | None:
    """Intent의 참조·Agenda·중복·재사용 정책을 검사한다."""
    try:
        validate_intent_references(
            intent,
            profile,
            goals=goals,
            evidence_ids=evidence_ids,
        )
    except ValueError as exc:
        _event(
            state,
            deps,
            "INTENT_REFERENCE_INVALID",
            {"reason": str(exc)},
        )
        return stalled(
            state,
            deps,
            goal_update=goal_update,
            reason="의도가 없는 것을 참조했다",
            planner_error=f"참조 오류로 거부됨: {exc}",
        )

    if intent.task_id:
        pending = {
            item.task_id: item
            for item in pending_agenda_tasks(state)
        }
        task = pending.get(intent.task_id)
        if task is None:
            return stalled(
                state,
                deps,
                goal_update=goal_update,
                reason="존재하지 않거나 이미 완료된 Agenda 작업을 선택했다",
                planner_error=f"task_id={intent.task_id!r}를 다시 선택하라",
            )
        if error := agenda_intent_error(task, intent):
            reason, planner_error = error
            return stalled(
                state,
                deps,
                goal_update=goal_update,
                reason=reason,
                planner_error=planner_error,
            )

    if error := repeated_action_error(state, intent):
        reason, planner_error = error
        return stalled(
            state,
            deps,
            goal_update=goal_update,
            reason=reason,
            planner_error=planner_error,
        )

    closed = {
        goal.goal_id
        for goal in updated_goals
        if not goal.is_open
    }
    if intent.goal_id and goal_update and intent.goal_id in closed:
        _event(
            state,
            deps,
            "INTENT_TARGETS_CLOSED_GOAL",
            {"goal_id": intent.goal_id},
        )
        return stalled(
            state,
            deps,
            goal_update=goal_update,
            reason="닫힌 축을 다시 겨냥했다",
            planner_error=(
                f"방금 응답이 이미 판정으로 닫힌 축 {intent.goal_id}를 다시 겨냥했다. "
                "아직 열려 있는 다른 축을 고르거나 goal_id를 비워라"
            ),
        )

    if (
        intent.execution_mode == "reuse_evidence"
        and (error := reuse_evidence_error(state, intent))
    ):
        reason, planner_error = error
        return stalled(
            state,
            deps,
            goal_update=goal_update,
            reason=reason,
            planner_error=planner_error,
        )

    if intent.dedupe_key() in coverage.tested_dedupe_keys:
        _event(
            state,
            deps,
            "DUPLICATE_INTENT_DROPPED",
            {"question": intent.question},
        )
        return stalled(
            state,
            deps,
            goal_update=goal_update,
            reason="같은 분석을 다시 제안했다",
            planner_error=(
                f"직전과 같은 분석을 다시 제안했다(중복): {intent.question!r}. "
                "이미 다룬 목적·축·컬럼 조합이다. 다른 각도를 골라라"
            ),
        )
    return None


_stalled = stalled
_goal_progress = goal_progress
_no_intent = no_intent
_rejected_intent = rejected_intent
