"""Regression tests for Main Graph and Execution Subgraph boundaries."""

import asyncio

from analysis_loop_v3.orchestration import nodes
from analysis_loop_v3.orchestration.execution_graph import build_execution_subgraph
from analysis_loop_v3.orchestration.graph import apply_execution_delta, route_after_main_plan


def _stub(name, update, calls):
    async def node(state, deps):
        calls.append(name)
        return dict(update)

    node.__name__ = name
    return node


def test_plan_sends_compute_to_execution_only():
    assert route_after_main_plan({"intent": {"execution_mode": "compute"}}) == "execution"
    assert route_after_main_plan({"intent": {"execution_mode": "inspect_data"}}) == "inspect_data"


def test_subgraph_exports_only_new_evidence(harness, monkeypatch):
    state, deps = harness
    state["evidence"] = [{"evidence_id": "old"}]
    calls = []
    for name, value in (
        ("lookup_program", {}),
        ("validate_code", {}),
        ("execute", {}),
        ("validate_result", {}),
        ("critique", {"review": {"verdict": "accept", "claim": "confirmed"}}),
        ("commit_evidence", {"evidence": [{"evidence_id": "new"}], "intent": None}),
    ):
        monkeypatch.setattr(nodes, name, _stub(name, value, calls))
    out = asyncio.run(build_execution_subgraph(deps).ainvoke(state))["execution_delta"]
    assert out["evidence"] == [{"evidence_id": "new"}]
    assert out["intent"] is None
    assert calls == [
        "lookup_program", "validate_code", "execute", "validate_result",
        "critique", "commit_evidence",
    ]
    assert apply_execution_delta({"execution_delta": out})["evidence"] == [
        {"evidence_id": "new"}
    ]


def test_subgraph_static_failure_rejects_at_limit(harness, monkeypatch):
    from analysis_loop_v3.orchestration.nodes._shared import MAX_STATIC_REPAIRS

    state, deps = harness
    calls = []
    for name, value in (
        ("lookup_program", {}),
        ("validate_code", {
            "validation_errors": [{"code": "bad"}],
            "static_repairs": MAX_STATIC_REPAIRS,
        }),
        ("reject_attempt", {"rejected": [{"intent_id": "bad"}], "intent": None}),
    ):
        monkeypatch.setattr(nodes, name, _stub(name, value, calls))
    out = asyncio.run(build_execution_subgraph(deps).ainvoke(state))["execution_delta"]
    assert calls == ["lookup_program", "validate_code", "reject_attempt"]
    assert out["rejected"] == [{"intent_id": "bad"}]
