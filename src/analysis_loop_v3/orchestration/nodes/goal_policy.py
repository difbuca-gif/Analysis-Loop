"""Planner의 SubGoal 판정 적용 규칙."""

from __future__ import annotations

from typing import Any

from ...contracts import SubGoal


def apply_goal_verdict(
    goals: list[SubGoal],
    verdict: dict[str, Any],
    evidence_by_goal: dict[str, list[str]],
    attempts_by_goal: dict[str, int] | None = None,
    blocked_goal_ids: set[str] | None = None,
) -> tuple[list[SubGoal], list[str]]:
    """Planner verdict를 Goal 상태에 적용하되 조기 종료를 막는다.

    converged에는 Evidence가 필요하고, abandoned에는 최소 한 번의 시도가 필요하다.
    pending Agenda가 남은 Goal은 converged로 닫지 않는다.
    """
    attempts_by_goal = attempts_by_goal or {}
    blocked_goal_ids = blocked_goal_ids or set()
    note = str(verdict.get("closing_note") or "")
    converged = set(verdict.get("converged") or [])
    abandoned = set(verdict.get("abandoned") or [])
    changed: list[str] = []

    updated: list[SubGoal] = []
    for goal in goals:
        if not goal.is_open:
            updated.append(goal)
            continue

        if goal.goal_id in converged:
            if goal.goal_id in blocked_goal_ids:
                changed.append(f"{goal.goal_id}:converged_rejected(required_followup)")
                updated.append(goal)
                continue
            if evidence_by_goal.get(goal.goal_id):
                updated.append(goal.model_copy(update={
                    "status": "converged",
                    "closing_note": note,
                }))
                changed.append(f"{goal.goal_id}:converged")
                continue
            changed.append(f"{goal.goal_id}:converged_rejected")

        elif goal.goal_id in abandoned:
            if attempts_by_goal.get(goal.goal_id, 0) > 0:
                updated.append(goal.model_copy(update={
                    "status": "abandoned",
                    "closing_note": note,
                }))
                changed.append(f"{goal.goal_id}:abandoned")
                continue
            changed.append(f"{goal.goal_id}:abandon_rejected(시도 부족)")

        updated.append(goal)

    return updated, changed


# 기존 public helper 이름 호환.
_apply_goal_verdict = apply_goal_verdict
