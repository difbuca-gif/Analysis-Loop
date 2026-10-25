"""LangGraph 조건부 라우팅 정책.

Node 구현과 그래프 조립에서 분리해, State만 보고 다음 Node를 결정하는
순수 라우팅 함수만 둔다.
"""

from __future__ import annotations

from typing import Literal

from ..contracts import RunStatus
from ..state import AnalysisGraphState, all_goals_closed, pending_agenda_tasks
from .nodes._shared import MAX_RUNTIME_REPAIRS, MAX_STATIC_REPAIRS, is_stuck


def _has_blocking_errors(state: AnalysisGraphState) -> bool:
    return bool(state.get("validation_errors"))


def _error_codes(state: AnalysisGraphState) -> set[str]:
    return {str(e.get("code")) for e in state.get("validation_errors") or []}


def route_after_plan(
    state: AnalysisGraphState,
) -> Literal[
    "lookup_program",
    "reuse_evidence",
    "inspect_data",
    "lookup_evidence",
    "mark_data_limited",
    "assess_progress",
]:
    if state.get("status") in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
        return "assess_progress"
    intent = state.get("intent") or {}
    if not intent:
        return "assess_progress"
    if intent.get("execution_mode") == "reuse_evidence":
        return "reuse_evidence"
    if intent.get("execution_mode") == "inspect_data":
        return "inspect_data"
    if intent.get("execution_mode") == "lookup_evidence":
        return "lookup_evidence"
    if intent.get("execution_mode") == "mark_data_limited":
        return "mark_data_limited"
    return "lookup_program"


def route_after_lookup(
    state: AnalysisGraphState,
) -> Literal["validate_code", "generate_code"]:
    return "validate_code" if state.get("program_ref") else "generate_code"


def route_after_generate(
    state: AnalysisGraphState,
) -> Literal["validate_code", "generate_code", "reject_attempt", "finalize"]:
    if state.get("status") in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
        return "finalize"
    if state.get("program_ref") and not _has_blocking_errors(state):
        return "validate_code"
    if is_stuck(state) or state.get("static_repairs", 0) >= MAX_STATIC_REPAIRS:
        return "reject_attempt"
    return "generate_code"


def route_after_validate_code(
    state: AnalysisGraphState,
) -> Literal["execute", "generate_code", "reject_attempt"]:
    if not _has_blocking_errors(state):
        return "execute"
    if is_stuck(state) or state.get("static_repairs", 0) >= MAX_STATIC_REPAIRS:
        return "reject_attempt"
    return "generate_code"


def route_after_execute(
    state: AnalysisGraphState,
) -> Literal["validate_result", "generate_code", "reject_attempt"]:
    if state.get("candidate_ref"):
        return "validate_result"
    if "execution_refused" in _error_codes(state):
        return "reject_attempt"
    if state.get("runtime_repairs", 0) <= MAX_RUNTIME_REPAIRS:
        return "generate_code"
    return "reject_attempt"


def route_after_validate_result(
    state: AnalysisGraphState,
) -> Literal["critique", "generate_code", "reject_attempt"]:
    if not _has_blocking_errors(state):
        return "critique"
    if state.get("runtime_repairs", 0) <= MAX_RUNTIME_REPAIRS:
        return "generate_code"
    return "reject_attempt"


def route_after_critique(
    state: AnalysisGraphState,
) -> Literal["commit_evidence", "reject_attempt", "finalize"]:
    if state.get("status") in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
        return "finalize"
    review = state.get("review") or {}
    verdict = review.get("verdict")
    claim = str(review.get("claim") or "").strip()
    if verdict != "accept" or not claim:
        return "reject_attempt"
    return "commit_evidence"


def route_after_assess(
    state: AnalysisGraphState,
) -> Literal["plan_intent", "expand_goals", "finalize"]:
    if state.get("status") in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
        return "finalize"
    if pending_agenda_tasks(state):
        return "plan_intent"
    if all_goals_closed(state):
        return "expand_goals"
    return "plan_intent"


def route_after_expand_goals(
    state: AnalysisGraphState,
) -> Literal["plan_intent", "finalize"]:
    if state.get("status") in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
        return "finalize"
    return "plan_intent"


__all__ = [
    "route_after_plan",
    "route_after_lookup",
    "route_after_generate",
    "route_after_validate_code",
    "route_after_execute",
    "route_after_validate_result",
    "route_after_critique",
    "route_after_assess",
    "route_after_expand_goals",
]
