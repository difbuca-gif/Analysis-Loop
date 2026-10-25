from analysis_loop_v3.runtime_protocols import ServiceResult

"""예산 갱신은 중단된 그래프의 실행 위치와 완료 상태를 보존해야 한다."""

import asyncio
import time
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from analysis_loop_v3.contracts import SCHEMA_VERSION
from analysis_loop_v3.orchestration import nodes
from analysis_loop_v3.orchestration.graph import build_graph
from analysis_loop_v3.service import AnalysisService, open_checkpointer


def _service_with_completed_prefix(state, deps, monkeypatch):
    # Use real graph execution to create a checkpoint with recorded channel reads.
    # A synthetic aupdate_state on an empty graph has no executed-node history.
    completed = []

    def stub(name):
        async def node(state, deps):
            completed.append(name)
            return {}

        node.__name__ = name
        return node

    for name in (
        "profile_data",
        "decompose_objective",
        "plan_intent",
        "lookup_program",
        "validate_code",
        "execute",
        "validate_result",
    ):
        monkeypatch.setattr(nodes, name, stub(name))
    deps.critic.review = AsyncMock(
        return_value=ServiceResult(error="offline", failure_kind="provider_unavailable")
    )
    return AnalysisService(deps, InMemorySaver()), completed


@pytest.mark.parametrize(
    "last_node,expected_next",
    [
        ("validate_code", "execute"),
        ("execute", "validate_result"),
        ("validate_result", "critique"),
        ("finalize", None),
    ],
)
def test_budget_extension_preserves_checkpoint_position(
    harness, monkeypatch, last_node, expected_next
):
    state, deps = harness
    state.update(
        schema_version=SCHEMA_VERSION,
        status="RUNNING",
        iteration=3,
        deadline_at=time.time() + 60,
    )
    service, _completed = _service_with_completed_prefix(state, deps, monkeypatch)
    config = {"configurable": {"thread_id": state["run_id"]}}
    # Isolate scheduling; the integration test below also executes the resumed graph.
    service._drive = AsyncMock(return_value=None)

    async def check():
        await service.graph.ainvoke(
            state, config, interrupt_before=[expected_next] if expected_next else []
        )
        before = await service.graph.aget_state(config)
        original = deepcopy(before.values)
        await service.resume(thread_id=state["run_id"], time_budget_seconds=300)
        after = await service.graph.aget_state(config)
        assert before.next == after.next == ((expected_next,) if expected_next else ())
        assert after.values["deadline_at"] > original.pop("deadline_at")
        assert {k: v for k, v in after.values.items() if k != "deadline_at"} == original
        service._drive.assert_awaited_once_with(None, thread_id=state["run_id"])

    asyncio.run(check())


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_does_not_change_checkpoint(harness, budget):
    state, deps = harness
    state["schema_version"] = SCHEMA_VERSION
    service = AnalysisService(deps, InMemorySaver())
    config = {"configurable": {"thread_id": state["run_id"]}}
    service._drive = AsyncMock()

    async def check():
        await service.graph.aupdate_state(config, state, as_node="validate_result")
        before = await service.graph.aget_state(config)
        with pytest.raises(ValueError, match="0보다 커야"):
            await service.resume(thread_id=state["run_id"], time_budget_seconds=budget)
        after = await service.graph.aget_state(config)
        assert before.config == after.config
        assert before.next == after.next
        assert before.values == after.values
        service._drive.assert_not_awaited()

    asyncio.run(check())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_resume_with_budget_runs_only_remaining_nodes(harness, monkeypatch, backend):
    state, deps = harness
    state.update(
        schema_version=SCHEMA_VERSION, status="RUNNING", iteration=3, deadline_at=time.time() + 60
    )
    # The first remaining step is critique; its outage should end the graph.
    deps.critic.review = AsyncMock(
        return_value=ServiceResult(error="offline", failure_kind="provider_unavailable")
    )
    service, completed = _service_with_completed_prefix(state, deps, monkeypatch)
    config = {"configurable": {"thread_id": state["run_id"]}}

    async def check():
        await service.graph.ainvoke(state, config, interrupt_before=["critique"])
        before = list(completed)
        deps.events.clear()
        handle = await service.resume(thread_id=state["run_id"], time_budget_seconds=300)
        assert completed == before
        assert handle.status == "FAILED"
        assert handle.state["stop_reason"] == "critic 실패: offline"
        assert handle.state["iteration"] == 3
        assert (await service.graph.aget_state(config)).next == ()

    async def run():
        if backend == "sqlite":
            async with open_checkpointer(deps.workdir_root / "resume.sqlite") as saver:
                service.graph = build_graph(deps, checkpointer=saver)
                await check()
        else:
            await check()

    asyncio.run(run())
    assert [event["event"] for event in deps.events] == [
        "CRITIC_FAILED",
        "REPORT_FALLBACK",
        "FINALIZED",
    ]
    deps.critic.review.assert_awaited_once()
