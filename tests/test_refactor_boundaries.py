"""실패 경계와 판단/관측 분리의 회귀 검증."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from analysis_loop_v3.contracts import ArtifactRef, RunStatus, StopCode, SubGoal
from analysis_loop_v3.orchestration.failure_policy import service_failure_action
from analysis_loop_v3.orchestration.graph import build_graph
from analysis_loop_v3.orchestration.nodes.planner_decision import (
    goal_progress,
    no_intent,
    stalled,
)
from analysis_loop_v3.orchestration.nodes.planning import _no_intent
from analysis_loop_v3.orchestration.nodes.progress import assess_progress
from analysis_loop_v3.orchestration.nodes.reporting import finalize
from analysis_loop_v3.orchestration.nodes.review import critique
from analysis_loop_v3.orchestration.routes import route_after_critique


@pytest.mark.parametrize("kind", ["configuration", "provider_unavailable"])
def test_critic_service_outage_stops_without_replanning(harness, kind):
    state, deps = harness
    deps.critic.review = AsyncMock(return_value=None)
    deps.critic.last_error = "service unavailable"
    deps.critic.last_failure_kind = kind
    state.update(asyncio.run(critique(state, deps)))

    assert state["status"] == RunStatus.FAILED.value
    assert state["stop_code"] == StopCode.SERVICE_UNAVAILABLE.value
    assert route_after_critique(state) == "finalize"
    # Terminal reason survives budget/iteration checks.
    state["deadline_at"] = 0
    assert asyncio.run(assess_progress(state, deps)) == {}
    update = asyncio.run(finalize(state, deps))
    assert update["status"] == RunStatus.FAILED.value
    deps.critic.generate_report.assert_not_awaited()
    deps.critic.review_report.assert_not_awaited()
    summary = deps.artifacts.read_json(ArtifactRef.model_validate(update["report_ref"]))
    assert summary["stop_reason"] == "critic 실패: service unavailable"


def test_missing_critic_is_terminal(harness):
    state, deps = harness
    deps.critic = None
    update = asyncio.run(critique(state, deps))
    assert update["stop_code"] == StopCode.SERVICE_UNAVAILABLE.value
    assert route_after_critique({**state, **update}) == "finalize"


@pytest.mark.parametrize("kind", ["contract", "response_invalid", None])
def test_critic_bad_response_rejects_only_this_attempt(harness, kind):
    state, deps = harness
    deps.critic.review = AsyncMock(return_value=None)
    deps.critic.last_failure_kind = kind
    update = asyncio.run(critique(state, deps))
    assert "status" not in update
    assert route_after_critique({**state, **update}) == "reject_attempt"


def test_reporting_failure_keeps_json_fallback(harness):
    state, deps = harness

    # Simulate a failure arising in report generation, after analysis is complete.
    async def fail_report(**kwargs):
        deps.critic.last_failure_kind = "provider_unavailable"

    deps.critic.generate_report = AsyncMock(side_effect=fail_report)
    state.update(status="COMPLETED", stop_code="goal_satisfied")
    update = asyncio.run(finalize(state, deps))
    assert update["status"] == "COMPLETED"
    assert (
        deps.artifacts.read_json(ArtifactRef.model_validate(update["report_ref"]))["status"]
        == "COMPLETED"
    )
    assert any(event["event"] == "REPORT_FALLBACK" for event in deps.events)


def test_decision_returns_events_without_mutating_state():
    state = {"error_streak": {"key": "planner_no_intent", "count": 4}, "iteration": 2}
    original = deepcopy(state)
    decision = stalled(state, goal_update={}, reason="no intent")
    assert state == original
    assert decision.update["stop_code"] == StopCode.PLANNER_STUCK.value
    assert [event.name for event in decision.events] == ["PLANNER_STUCK"]


def test_nested_decision_preserves_event_order_and_node_records_it(harness):
    state, deps = harness
    state["error_streak"] = {"key": "planner_no_intent", "count": 4}
    goal = SubGoal(goal_id="g1", question="why?")
    kwargs = {"verdict": {}, "updated_goals": [goal], "goal_update": {}}
    deps.planner = SimpleNamespace(last_error="bad response", last_failure_kind="contract")
    decision = no_intent(state, failure_reason="bad response", failure_kind="contract", **kwargs)
    assert not deps.events
    assert [event.name for event in decision.events] == ["PLANNER_FAILED", "PLANNER_STUCK"]
    update = _no_intent(state, deps, **kwargs)
    assert update == decision.update
    assert [event["event"] for event in deps.events] == ["PLANNER_FAILED", "PLANNER_STUCK"]


def test_goal_policy_returns_unreviewed_event_as_data():
    goal = SubGoal(goal_id="g1", question="why?")
    original = goal.model_dump()
    updated, decision = goal_progress({}, [goal], {"unreviewed": ["g1"]})
    assert goal.model_dump() == original
    assert updated == [goal]
    assert decision.update == {}
    assert decision.events[0].name == "GOALS_UNREVIEWED"


def test_report_failure_policy_does_not_stop_completed_analysis():
    assert service_failure_action("configuration") == "stop_run"
    assert service_failure_action("contract") == "reject_attempt"
    assert service_failure_action("configuration", stage="report") == "fallback_report"


def test_checkpoint_resume_critic_outage_reaches_graph_end(harness):
    state, deps = harness
    deps.critic.review = AsyncMock(return_value=None)
    deps.critic.last_error = "offline"
    deps.critic.last_failure_kind = "provider_unavailable"
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "critic-outage"}}

    async def resume():
        await graph.aupdate_state(config, state, as_node="validate_result")
        result = await graph.ainvoke(None, config)
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ()
        return result

    result = asyncio.run(resume())
    assert result["status"] == "FAILED"
    assert result["stop_code"] == StopCode.SERVICE_UNAVAILABLE.value
    assert result["evidence"] == []
    assert result["report_ref"]
    assert [event["event"] for event in deps.events] == [
        "CRITIC_FAILED",
        "REPORT_FALLBACK",
        "FINALIZED",
    ]
    deps.critic.review.assert_awaited_once()
    deps.critic.generate_report.assert_not_awaited()
