import pytest

from analysis_loop_v3.contracts import RunStatus
from analysis_loop_v3.orchestration.nodes._shared import (
    MAX_RUNTIME_REPAIRS,
    MAX_STATIC_REPAIRS,
)
from analysis_loop_v3.orchestration.routes import (
    route_after_assess,
    route_after_critique,
    route_after_execute,
    route_after_expand_goals,
    route_after_generate,
    route_after_plan,
    route_after_validate_code,
    route_after_validate_result,
)


def test_route_after_plan_uses_execution_mode():
    assert route_after_plan({"intent": {"execution_mode": "compute"}}) == "lookup_program"
    assert route_after_plan({"intent": {"execution_mode": "reuse_evidence"}}) == "reuse_evidence"
    assert route_after_plan({"intent": {}}) == "assess_progress"


def test_route_after_generate_uses_program_and_repairs():
    assert (
        route_after_generate({"program_ref": {"artifact_id": "p"}, "validation_errors": []})
        == "validate_code"
    )
    assert (
        route_after_generate(
            {
                "program_ref": None,
                "validation_errors": [{"code": "bad"}],
                "static_repairs": MAX_STATIC_REPAIRS,
            }
        )
        == "reject_attempt"
    )


def test_route_after_validate_code_retries_or_executes():
    assert route_after_validate_code({"validation_errors": []}) == "execute"
    assert (
        route_after_validate_code(
            {
                "validation_errors": [{"code": "bad"}],
                "static_repairs": 0,
            }
        )
        == "generate_code"
    )


def test_route_after_execute_distinguishes_refusal_and_runtime_failure():
    assert route_after_execute({"candidate_ref": {"artifact_id": "result"}}) == "validate_result"
    assert (
        route_after_execute(
            {
                "candidate_ref": None,
                "validation_errors": [{"code": "execution_refused"}],
            }
        )
        == "reject_attempt"
    )
    assert (
        route_after_execute(
            {
                "candidate_ref": None,
                "validation_errors": [{"code": "runtime_error"}],
                "runtime_repairs": 0,
            }
        )
        == "generate_code"
    )


def test_route_after_critique_requires_accept_and_claim():
    assert (
        route_after_critique({"review": {"verdict": "accept", "claim": "supported"}})
        == "commit_evidence"
    )
    assert route_after_critique({"review": {"verdict": "accept", "claim": ""}}) == "reject_attempt"
    assert route_after_critique({"review": {"verdict": "reject", "claim": "x"}}) == "reject_attempt"


def test_terminal_state_routes_to_finalize():
    terminal = {"status": RunStatus.COMPLETED.value}
    assert route_after_assess(terminal) == "finalize"
    assert route_after_expand_goals(terminal) == "finalize"


def test_running_expand_goals_routes_back_to_planner():
    assert route_after_expand_goals({"status": RunStatus.RUNNING.value}) == "plan_intent"


@pytest.mark.parametrize(
    "count,expected",
    [
        (MAX_STATIC_REPAIRS - 1, "generate_code"),
        (MAX_STATIC_REPAIRS, "reject_attempt"),
        (MAX_STATIC_REPAIRS + 1, "reject_attempt"),
    ],
)
@pytest.mark.parametrize("route", [route_after_generate, route_after_validate_code])
def test_static_failure_limit_includes_the_current_failure(route, count, expected):
    state = {"validation_errors": [{"code": "bad_code"}], "static_repairs": count}
    assert route(state) == expected


@pytest.mark.parametrize(
    "count,expected",
    [
        (MAX_RUNTIME_REPAIRS - 1, "generate_code"),
        (MAX_RUNTIME_REPAIRS, "generate_code"),
        (MAX_RUNTIME_REPAIRS + 1, "reject_attempt"),
    ],
)
@pytest.mark.parametrize("route", [route_after_execute, route_after_validate_result])
def test_runtime_failure_limit_allows_the_first_repair(route, count, expected):
    state = {"validation_errors": [{"code": "bad_result"}], "runtime_repairs": count}
    assert route(state) == expected
