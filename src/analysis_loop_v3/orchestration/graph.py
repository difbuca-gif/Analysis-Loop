"""Main Graph: 분석 계획, 비코드 작업, 진행·종료와 실행 서브그래프 연결."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..runtime import RuntimeDeps
from ..state import AnalysisGraphState
from . import nodes
from .execution_graph import build_execution_subgraph
from .routes import (
    route_after_assess,
    route_after_critique as route_after_critique,
    route_after_execute as route_after_execute,
    route_after_expand_goals,
    route_after_generate as route_after_generate,
    route_after_lookup as route_after_lookup,
    route_after_plan,
    route_after_validate_code as route_after_validate_code,
    route_after_validate_result as route_after_validate_result,
)

RECURSION_LIMIT = 500


def route_after_main_plan(state: AnalysisGraphState) -> str:
    next_node = route_after_plan(state)
    return "execution" if next_node == "lookup_program" else next_node


def apply_execution_delta(state: AnalysisGraphState) -> dict[str, Any]:
    """자식의 변경분을 한 번만 부모 State에 반영한다."""
    return {**(state.get("execution_delta") or {}), "execution_delta": None}


def build_graph(deps: RuntimeDeps, *, checkpointer: Any | None = None):
    """부모 그래프를 구성하고 자식에게 checkpoint 저장소를 상속한다."""
    builder = StateGraph(AnalysisGraphState)

    def bind(fn):
        async def node(state: AnalysisGraphState) -> dict[str, Any]:
            return await fn(state, deps)

        node.__name__ = fn.__name__
        return node

    for fn in (
        nodes.profile_data,
        nodes.decompose_objective,
        nodes.plan_intent,
        nodes.reuse_evidence,
        nodes.inspect_data,
        nodes.lookup_evidence,
        nodes.mark_data_limited,
        nodes.assess_progress,
        nodes.expand_goals,
        nodes.finalize,
    ):
        builder.add_node(fn.__name__, bind(fn))

    builder.add_node("execution", build_execution_subgraph(deps))
    builder.add_node("apply_execution_delta", apply_execution_delta)
    builder.add_edge(START, "profile_data")
    builder.add_edge("profile_data", "decompose_objective")
    builder.add_edge("decompose_objective", "plan_intent")
    builder.add_conditional_edges("plan_intent", route_after_main_plan)
    builder.add_edge("reuse_evidence", "assess_progress")
    builder.add_edge("inspect_data", "assess_progress")
    builder.add_edge("lookup_evidence", "assess_progress")
    builder.add_edge("mark_data_limited", "assess_progress")
    builder.add_edge("execution", "apply_execution_delta")
    builder.add_edge("apply_execution_delta", "assess_progress")
    builder.add_conditional_edges("assess_progress", route_after_assess)
    builder.add_conditional_edges("expand_goals", route_after_expand_goals)
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
