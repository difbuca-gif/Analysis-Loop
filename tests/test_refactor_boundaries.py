"""실패 경계와 판단/관측 분리의 회귀 검증."""

import asyncio
from copy import deepcopy
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
from analysis_loop_v3.runtime_protocols import ServiceResult


@pytest.mark.parametrize("kind", ["configuration", "provider_unavailable"])
def test_critic_service_outage_stops_without_replanning(harness, kind):
    state, deps = harness
    deps.critic.review = AsyncMock(
        return_value=ServiceResult(error="service unavailable", failure_kind=kind)
    )
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
    deps.critic.review = AsyncMock(return_value=ServiceResult(failure_kind=kind))
    update = asyncio.run(critique(state, deps))
    assert "status" not in update
    assert route_after_critique({**state, **update}) == "reject_attempt"


def test_reporting_failure_keeps_json_fallback(harness):
    state, deps = harness

    # Simulate a failure arising in report generation, after analysis is complete.
    async def fail_report(**kwargs):
        return ServiceResult(error="offline", failure_kind="provider_unavailable")

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
    decision = no_intent(state, failure_reason="bad response", failure_kind="contract", **kwargs)
    assert not deps.events
    assert [event.name for event in decision.events] == ["PLANNER_FAILED", "PLANNER_STUCK"]
    update = _no_intent(
        state, deps, failure_reason="bad response", failure_kind="contract", **kwargs
    )
    assert update == decision.update
    assert [event["event"] for event in deps.events] == ["PLANNER_FAILED", "PLANNER_STUCK"]


def test_goal_policy_returns_unreviewed_event_as_data():
    goal = SubGoal(goal_id="g1", question="why?")
    original = goal.model_dump()
    updated, decision = goal_progress(
        goals=[goal],
        verdict={"unreviewed": ["g1"]},
        evidence=[],
        rejected_attempts=[],
        pending_tasks=[],
    )
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
    deps.critic.review = AsyncMock(
        return_value=ServiceResult(error="offline", failure_kind="provider_unavailable")
    )
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


def test_plan_intent_exposes_cycle_updates_and_preserves_accumulated_state(harness):
    from types import SimpleNamespace

    from analysis_loop_v3.contracts import ResearchIntent
    from analysis_loop_v3.orchestration.nodes.planning import plan_intent
    from analysis_loop_v3.runtime_protocols import PlannerProposal
    from analysis_loop_v3.state import cycle_reset_update

    state, deps = harness
    state.update(
        iteration=3,
        static_repairs=4,
        runtime_repairs=1,
        review={"verdict": "reject"},
        last_planner_error="old error",
    )
    original = deepcopy(state)
    intent = ResearchIntent.model_validate(state["intent"])
    deps.planner = SimpleNamespace(
        propose_intent=AsyncMock(return_value=ServiceResult(value=PlannerProposal(intent=intent)))
    )
    update = asyncio.run(plan_intent(state, deps))
    assert state == original
    assert update["iteration"] == 4
    assert update["intent"] == original["intent"]
    for key, value in cycle_reset_update().items():
        assert update[key] == value
    assert not {"evidence", "rejected", "coverage"}.intersection(update)
    assert [event["event"] for event in deps.events] == [
        "PLANNER_CONTEXT_SELECTED",
        "INTENT_PROPOSED",
    ]


@pytest.mark.parametrize(
    "kind,expected_status",
    [
        ("provider_unavailable", "FAILED"),
        ("configuration", "FAILED"),
        ("contract", None),
    ],
)
def test_planner_failure_is_read_from_call_result(harness, kind, expected_status):
    from types import SimpleNamespace

    from analysis_loop_v3.orchestration.nodes.planning import plan_intent

    state, deps = harness
    deps.planner = SimpleNamespace(
        propose_intent=AsyncMock(
            return_value=ServiceResult(error="planner problem", failure_kind=kind)
        )
    )
    update = asyncio.run(plan_intent(state, deps))
    assert update.get("status") == expected_status
    if expected_status:
        assert update["service_failure"] == {
            "role": "planner",
            "kind": kind,
            "message": "planner problem",
        }
    else:
        assert update["last_planner_error"] == "planner problem"


def test_planner_contract_failure_preserves_goal_verdict(harness):
    from types import SimpleNamespace

    from analysis_loop_v3.contracts import DatasetProfile
    from analysis_loop_v3.llm.planner_service import LLMPlanner

    state, _deps = harness
    client = SimpleNamespace(
        generate_json=AsyncMock(
            return_value={
                "goal_review": [{"goal_id": "g1", "verdict": "converged"}],
                "question": "next question",
                "candidate_columns": 42,
            }
        )
    )
    planner = LLMPlanner(client)
    result = asyncio.run(
        planner.propose_intent(
            objective="test",
            profile=DatasetProfile.model_validate(state["profile"]),
            unexplored=[],
            column_usage={},
            prior_findings=[],
            unresolved_findings=[],
            sub_goals=[SubGoal(goal_id="g1", question="why?")],
            timeout_seconds=1,
        )
    )
    assert result.failure_kind == "contract"
    assert result.value.intent is None
    assert result.value.verdict["converged"] == ["g1"]


def test_concurrent_service_calls_keep_their_own_errors():
    import json
    from types import SimpleNamespace

    from analysis_loop_v3.llm.critic_service import LLMCritic

    async def check():
        failing_call_started = asyncio.Event()
        successful_call_finished = asyncio.Event()

        async def generate_text(**kwargs):
            objective = json.loads(kwargs["prompt"])["실행 요약"]["objective"]
            if objective == "fail":
                failing_call_started.set()
                await successful_call_finished.wait()
                raise RuntimeError("this call failed")
            await failing_call_started.wait()
            successful_call_finished.set()
            return "valid report"

        critic = LLMCritic(SimpleNamespace(generate_text=generate_text))
        failed, succeeded = await asyncio.gather(
            *[
                critic.generate_report(
                    run_summary={"objective": objective},
                    evidence=[],
                    rejected=[],
                    timeout_seconds=1,
                )
                for objective in ["fail", "success"]
            ]
        )
        assert failed.value is None
        assert "this call failed" in failed.error
        assert succeeded.value == "valid report"
        assert succeeded.error is succeeded.failure_kind is None
        assert not hasattr(critic, "last_error")

    asyncio.run(check())


def test_codegen_returns_validated_value_without_shared_failure_state(harness):
    from types import SimpleNamespace

    from analysis_loop_v3.contracts import DatasetProfile, ResearchIntent
    from analysis_loop_v3.llm.codegen_service import LLMCodegen

    state, deps = harness
    code = deps.artifacts.read_text(ArtifactRef.model_validate(state["program_ref"]))
    client = SimpleNamespace(
        generate_json=AsyncMock(return_value={**state["manifest"], "code": code})
    )
    codegen = LLMCodegen(client)
    result = asyncio.run(
        codegen.generate(
            intent=ResearchIntent.model_validate(state["intent"]),
            profile=DatasetProfile.model_validate(state["profile"]),
            previous_errors=[],
            timeout_seconds=1,
        )
    )
    assert result.value.code == code
    assert result.error is result.failure_kind is None
    assert not hasattr(codegen, "last_error")


def test_missing_cli_executable_returns_configuration_failure_and_cleans_temp_file(
    harness, tmp_path, monkeypatch
):
    import tempfile

    from analysis_loop_v3.contracts import DatasetProfile, ResearchIntent
    from analysis_loop_v3.llm.agy_codegen import AgyCodegen

    state, _deps = harness
    create_temp = tempfile.NamedTemporaryFile
    monkeypatch.setattr(
        tempfile, "NamedTemporaryFile", lambda **kwargs: create_temp(dir=tmp_path, **kwargs)
    )
    codegen = AgyCodegen(executable=str(tmp_path / "missing-agy"))
    result = asyncio.run(
        codegen.generate(
            intent=ResearchIntent.model_validate(state["intent"]),
            profile=DatasetProfile.model_validate(state["profile"]),
            previous_errors=[],
            timeout_seconds=1,
        )
    )
    assert result.value is None
    assert result.failure_kind == "configuration"
    assert "agy CLI 실행 실패" in result.error
    assert not list(tmp_path.glob("*.json"))
