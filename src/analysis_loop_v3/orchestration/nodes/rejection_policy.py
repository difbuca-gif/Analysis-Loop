"""거부된 분석 시도의 사유와 반복 실패 분류."""

from __future__ import annotations

from typing import Any

from ...contracts import LoopIssue, ResearchIntent
from ...state import AnalysisGraphState
from ._shared import is_stuck


def rejection_reasons(
    state: AnalysisGraphState,
    review: dict[str, Any],
) -> list[str]:
    """검증 오류를 우선하고, 없으면 Critic 사유를 반환한다."""
    reasons = [
        str(error.get("message", error))
        for error in state.get("validation_errors") or []
    ]
    if reasons:
        return reasons
    return [str(reason) for reason in review.get("reasons") or []]


def stuck_issue(
    state: AnalysisGraphState,
    intent: ResearchIntent | None,
    reasons: list[str],
) -> LoopIssue | None:
    """동일 오류 반복으로 중단된 회차를 운영 신호로 변환한다."""
    if not is_stuck(state):
        return None
    streak = state.get("error_streak") or {}
    return LoopIssue(
        label=str(streak.get("key") or "unknown"),
        count=int(streak.get("count", 0)),
        iteration=int(state.get("iteration", 0)),
        intent_id=intent.intent_id if intent else None,
        question=intent.question if intent else "",
        messages=reasons[:3],
    )


_rejection_reasons = rejection_reasons
_stuck_issue = stuck_issue
