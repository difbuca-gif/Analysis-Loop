"""LangGraph 노드 조립과 실행 흐름 연결.

조건부 라우팅 정책은 routes.py에 분리한다.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..runtime import RuntimeDeps
from ..state import AnalysisGraphState
from . import nodes
from .routes import (
    route_after_assess,
    route_after_critique,
    route_after_execute,
    route_after_expand_goals,
    route_after_generate,
    route_after_lookup,
    route_after_plan,
    route_after_validate_code,
    route_after_validate_result,
)

RECURSION_LIMIT = 500


def build_graph(deps: RuntimeDeps, *, checkpointer: Any | None = None):
    """RuntimeDeps를 클로저로 캡처한 그래프 생성."""
    builder = StateGraph(AnalysisGraphState)

    def bind(fn):
        """LangGraph config 전달과 분리해 deps를 클로저에 캡처."""
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
        nodes.lookup_program,
        nodes.generate_code,
        nodes.validate_code,
        nodes.execute,
        nodes.validate_result,
        nodes.critique,
        nodes.commit_evidence,
        nodes.reject_attempt,
        nodes.assess_progress,
        nodes.expand_goals,
        nodes.finalize,
    ):
        builder.add_node(fn.__name__, bind(fn))

    builder.add_edge(START, "profile_data")
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
