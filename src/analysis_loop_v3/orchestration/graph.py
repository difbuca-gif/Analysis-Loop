"""LangGraph 노드와 조건부 라우팅을 조립한다.

종료 판단은 assess_progress가 담당하고 RECURSION_LIMIT은 비정상 루프의 안전망이다.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from ..contracts import RunStatus
from ..runtime import RuntimeDeps
from ..state import AnalysisGraphState, all_goals_closed, pending_agenda_tasks
from . import nodes

RECURSION_LIMIT = 500

# 라우팅은 State만 입력으로 받는 순수 함수다.

def _has_blocking_errors(state: AnalysisGraphState) -> bool:
    return bool(state.get("validation_errors"))

def _error_codes(state: AnalysisGraphState) -> set[str]:
    return {str(e.get("code")) for e in state.get("validation_errors") or []}

def route_after_plan(
    state: AnalysisGraphState,
) -> Literal[
    "lookup_program", "reuse_evidence", "inspect_data", "lookup_evidence",
    "mark_data_limited", "assess_progress"
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

def route_after_lookup(state: AnalysisGraphState) -> Literal["validate_code", "generate_code"]:
    return "validate_code" if state.get("program_ref") else "generate_code"

def route_after_generate(
    state: AnalysisGraphState,
) -> Literal["validate_code", "generate_code", "reject_attempt", "finalize"]:
    if state.get("status") in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
        return "finalize"
    if state.get("program_ref") and not _has_blocking_errors(state):
        return "validate_code"
    if nodes.is_stuck(state) or state.get("static_repairs", 0) >= nodes.MAX_STATIC_REPAIRS:
        return "reject_attempt"
    return "generate_code"

def route_after_validate_code(
    state: AnalysisGraphState,
) -> Literal["execute", "generate_code", "reject_attempt"]:
    if not _has_blocking_errors(state):
        return "execute"
    if nodes.is_stuck(state) or state.get("static_repairs", 0) >= nodes.MAX_STATIC_REPAIRS:
        return "reject_attempt"
    return "generate_code"

def route_after_execute(
    state: AnalysisGraphState,
) -> Literal["validate_result", "generate_code", "reject_attempt"]:
    if state.get("candidate_ref"):
        return "validate_result"
    # 실행 거부는 코드 재생성으로 해결되지 않는다.
    if "execution_refused" in _error_codes(state):
        return "reject_attempt"
    # 현재 실패까지 포함해 허용 횟수 안이면 한 번 더 재생성한다.
    if state.get("runtime_repairs", 0) <= nodes.MAX_RUNTIME_REPAIRS:
        return "generate_code"
    return "reject_attempt"

def route_after_validate_result(
    state: AnalysisGraphState,
) -> Literal["critique", "generate_code", "reject_attempt"]:
    if not _has_blocking_errors(state):
        return "critique"
    if state.get("runtime_repairs", 0) <= nodes.MAX_RUNTIME_REPAIRS:
        return "generate_code"
    return "reject_attempt"

def route_after_critique(
    state: AnalysisGraphState,
) -> Literal["commit_evidence", "reject_attempt"]:
    review = state.get("review") or {}
    verdict = review.get("verdict")
    claim = str(review.get("claim") or "").strip()
    # accept와 비어 있지 않은 claim이 모두 필요하다.
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
    # 기존 목표가 끝나도 추가 분석 가치가 있는지 한 번 확인한다.
    if all_goals_closed(state):
        return "expand_goals"
    return "plan_intent"

def route_after_expand_goals(
    state: AnalysisGraphState,
) -> Literal["plan_intent", "finalize"]:
    if state.get("status") in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
        return "finalize"
    return "plan_intent"

# 조립

def build_graph(deps: RuntimeDeps, *, checkpointer: Any | None = None):
    """RuntimeDeps를 클로저로 묶은 그래프. deps는 직렬화 불가라 State에 넣지 않는다."""
    builder = StateGraph(AnalysisGraphState)

    def bind(fn):
        """deps를 클로저로 감춘다. config는 langgraph가 알아서 전달한다."""
        async def node(state: AnalysisGraphState) -> dict[str, Any]:
            return await fn(state, deps)
        node.__name__ = fn.__name__
        return node

    for fn in (
        nodes.profile_data, nodes.decompose_objective, nodes.plan_intent,
        nodes.reuse_evidence, nodes.inspect_data, nodes.lookup_evidence,
        nodes.mark_data_limited,
        nodes.lookup_program, nodes.generate_code, nodes.validate_code,
        nodes.execute, nodes.validate_result, nodes.critique, nodes.commit_evidence,
        nodes.reject_attempt,
        nodes.assess_progress, nodes.expand_goals, nodes.finalize,
    ):
        builder.add_node(fn.__name__, bind(fn))

    builder.add_edge(START, "profile_data")
    # 목표 분해는 별도 노드로 두어 재개 시 중복 실행을 피한다.
    builder.add_edge("profile_data", "decompose_objective")
    builder.add_edge("decompose_objective", "plan_intent")

    builder.add_conditional_edges("plan_intent", route_after_plan)
    builder.add_edge("reuse_evidence", "assess_progress")
    builder.add_edge("inspect_data", "assess_progress")
    builder.add_edge("lookup_evidence", "assess_progress")
    builder.add_edge("mark_data_limited", "assess_progress")
    builder.add_conditional_edges("lookup_program", route_after_lookup)
    builder.add_conditional_edges("generate_code", route_after_generate)
    builder.add_conditional_edges("validate_code", route_after_validate_code)
    builder.add_conditional_edges("execute", route_after_execute)
    builder.add_conditional_edges("validate_result", route_after_validate_result)
    builder.add_conditional_edges("critique", route_after_critique)

    builder.add_edge("commit_evidence", "assess_progress")
    builder.add_edge("reject_attempt", "assess_progress")

    builder.add_conditional_edges("assess_progress", route_after_assess)
    builder.add_conditional_edges("expand_goals", route_after_expand_goals)
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
