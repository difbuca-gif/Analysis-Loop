"""Code execution subgraph: generation, validation, review, evidence commit/reject.

Parent and child both use append-only Evidence reducers. Export only the list
suffix added during this execution, not the complete child checkpoint state.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from ..runtime import RuntimeDeps
from ..state import AnalysisGraphState
from . import nodes
from .routes import (
    route_after_critique,
    route_after_execute,
    route_after_generate,
    route_after_lookup,
    route_after_validate_code,
    route_after_validate_result,
)

_APPEND_FIELDS = ("evidence", "evidence_relations", "rejected", "issues")
_RESULT_FIELDS = (
    "program_ref", "manifest", "candidate_ref", "figure_refs",
    "validation_errors", "validation_warnings", "static_repairs",
    "runtime_repairs", "error_streak", "review", "status", "stop_code",
    "stop_reason", "service_failure", "intent", "agenda", "sub_goals",
    "coverage",
)


class ExecutionGraphState(AnalysisGraphState, total=False):
    _entry_lengths: dict[str, int]


class ExecutionGraphOutput(TypedDict):
    execution_delta: dict[str, Any]


def capture_execution_entry(state: ExecutionGraphState) -> dict[str, Any]:
    values = cast(dict[str, Any], state)
    return {"_entry_lengths": {key: len(values.get(key) or []) for key in _APPEND_FIELDS}}


def export_execution_delta(state: ExecutionGraphState) -> ExecutionGraphOutput:
    values = cast(dict[str, Any], state)
    update: dict[str, Any] = {key: values[key] for key in _RESULT_FIELDS if key in values}
    for key in _APPEND_FIELDS:
        appended_list = cast(list[dict[str, Any]], values.get(key) or [])
        entry_length = state["_entry_lengths"][key]
        if len(appended_list) < entry_length:
            raise ValueError(f"append-only state decreased during execution: {key}")
        appended = appended_list[entry_length:]
        if appended:
            update[key] = appended
    return {"execution_delta": update}


def _after_generate(state: ExecutionGraphState) -> str:
    next_node = route_after_generate(state)
    return "export_execution_delta" if next_node == "finalize" else next_node


def _after_critique(state: ExecutionGraphState) -> str:
    next_node = route_after_critique(state)
    return "export_execution_delta" if next_node == "finalize" else next_node


def build_execution_subgraph(deps: RuntimeDeps) -> Any:
    """Reuse the existing static/runtime repair limits and Critic routing."""
    builder = StateGraph(
        ExecutionGraphState,
        input_schema=AnalysisGraphState,
        output_schema=ExecutionGraphOutput,
    )

    def bind(
        fn: Callable[..., Awaitable[Any]],
    ) -> Callable[[ExecutionGraphState], Awaitable[dict[str, Any]]]:
        async def node(state: ExecutionGraphState) -> dict[str, Any]:
            return cast(dict[str, Any], await fn(state, deps))

        node.__name__ = fn.__name__
        return node

    builder.add_node("capture_execution_entry", capture_execution_entry)
    for fn in (
        nodes.lookup_program,
        nodes.generate_code,
        nodes.validate_code,
        nodes.execute,
        nodes.validate_result,
        nodes.critique,
        nodes.commit_evidence,
        nodes.reject_attempt,
    ):
        # LangGraph's overloaded node protocol rejects this valid async wrapper in mypy.
        builder.add_node(fn.__name__, cast(Any, bind(fn)))
    builder.add_node("export_execution_delta", export_execution_delta)
    builder.add_edge(START, "capture_execution_entry")
    builder.add_edge("capture_execution_entry", "lookup_program")
    builder.add_conditional_edges("lookup_program", route_after_lookup)
    builder.add_conditional_edges("generate_code", _after_generate)
    builder.add_conditional_edges("validate_code", route_after_validate_code)
    builder.add_conditional_edges("execute", route_after_execute)
    builder.add_conditional_edges("validate_result", route_after_validate_result)
    builder.add_conditional_edges("critique", _after_critique)
    builder.add_edge("commit_evidence", "export_execution_delta")
    builder.add_edge("reject_attempt", "export_execution_delta")
    builder.add_edge("export_execution_delta", END)
    return builder.compile()
