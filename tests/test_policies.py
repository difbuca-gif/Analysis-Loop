from analysis_loop_v3.contracts import AgendaTask, ResearchIntent, SubGoal
from analysis_loop_v3.orchestration.nodes.intent_policy import (
    agenda_intent_error,
    repeated_action_error,
)
from analysis_loop_v3.orchestration.nodes.planner_decision import apply_goal_verdict
from analysis_loop_v3.orchestration.nodes.review import rejection_reasons


def test_goal_cannot_converge_without_evidence():
    goal = SubGoal(goal_id="g1", question="원인은 무엇인가?")

    updated, changes = apply_goal_verdict(
        [goal],
        {"converged": ["g1"], "closing_note": "done"},
        evidence_by_goal={},
    )

    assert updated[0].status == "open"
    assert "g1:converged_rejected" in changes


def test_goal_converges_with_evidence():
    goal = SubGoal(goal_id="g1", question="원인은 무엇인가?")

    updated, changes = apply_goal_verdict(
        [goal],
        {"converged": ["g1"], "closing_note": "done"},
        evidence_by_goal={"g1": ["e1"]},
    )

    assert updated[0].status == "converged"
    assert updated[0].closing_note == "done"
    assert "g1:converged" in changes


def test_goal_cannot_close_while_followup_is_pending():
    goal = SubGoal(goal_id="g1", question="원인은 무엇인가?")

    updated, changes = apply_goal_verdict(
        [goal],
        {"converged": ["g1"]},
        evidence_by_goal={"g1": ["e1"]},
        blocked_goal_ids={"g1"},
    )

    assert updated[0].status == "open"
    assert "g1:converged_rejected(required_followup)" in changes


def test_followup_task_cannot_be_treated_as_new_exploration():
    task = AgendaTask(
        kind="follow_up",
        goal_id="g1",
        question="조건별로 다시 확인",
        source_evidence_ids=["e1"],
    )
    intent = ResearchIntent(
        question="다른 분석",
        hypothesis="차이가 있을 수 있다",
        goal_id="g1",
        task_id=task.task_id,
        purpose="explore",
    )

    error = agenda_intent_error(task, intent)

    assert error is not None
    assert "후속 작업" in error[0]


def test_mark_data_limited_requires_prior_attempt():
    task = AgendaTask(
        kind="follow_up",
        goal_id="g1",
        question="추가 데이터가 필요한가?",
        source_evidence_ids=["e1"],
    )
    intent = ResearchIntent(
        question="현재 데이터 한계 확인",
        hypothesis="현재 데이터로는 확인할 수 없다",
        goal_id="g1",
        task_id=task.task_id,
        purpose="deepen",
        parent_evidence_ids=["e1"],
        execution_mode="mark_data_limited",
    )

    error = agenda_intent_error(task, intent)

    assert error is not None
    assert "시도하지 않은" in error[0]


def test_repeated_inspection_is_rejected():
    state = {
        "inspection_history": [
            {"columns": ["a", "b"]},
        ]
    }
    intent = ResearchIntent(
        question="컬럼 상태 확인",
        hypothesis="분포를 확인할 필요가 있다",
        execution_mode="inspect_data",
        candidate_columns=["b", "a"],
    )

    error = repeated_action_error(state, intent)

    assert error is not None
    assert "이미 점검" in error[0]


def test_validation_errors_take_priority_over_critic_reasons():
    state = {
        "validation_errors": [
            {"code": "bad_result", "message": "result contract failed"},
        ]
    }
    review = {"reasons": ["critic rejected"]}

    assert rejection_reasons(state, review) == ["result contract failed"]
