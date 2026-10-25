"""프로파일링, 목표 분해, 분석 의도 결정.

세부 정책은 planner_context.py / goal_policy.py / intent_policy.py /
planner_decision.py에 분리한다. 이 파일은 LangGraph Node orchestration만 담당한다.
"""

from __future__ import annotations

from typing import Any

from ...contracts import ArtifactKind, RunStatus, StopCode
from ...data.profiling import load_dataframe, profile_dataframe
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    cycle_reset_update,
    evidence_data_applicability,
    get_agenda,
    get_coverage,
    get_evidence,
    get_evidence_relations,
    get_intent,
    get_profile,
    get_ref,
    get_sub_goals,
    pending_agenda_tasks,
    remaining_seconds,
    terminal_update,
)
from ._shared import _event
from .goal_policy import _apply_goal_verdict
from .intent_policy import (
    _agenda_intent_error,
    _repeated_action_error,
    _reuse_evidence_error,
)
from .planner_context import _select_planner_findings
from .planner_decision import (
    NO_INTENT_CODE,
    _goal_progress,
    _no_intent,
    _rejected_intent,
    _stalled,
)


async def profile_data(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """Dataset을 profile하고 원본 profile은 Artifact, compact profile은 State에 둔다."""
    dataset = get_ref(state, "dataset_ref")
    assert dataset is not None, "dataset_ref 없이 그래프를 시작할 수 없다"

    frame = load_dataframe(dataset.path)
    profile = profile_dataframe(frame, dataset_id=dataset.artifact_id)
    ref = deps.artifacts.put_json(
        profile.model_dump(mode="json"),
        kind=ArtifactKind.PROFILE,
        summary={
            "rows": profile.row_count,
            "columns": len(profile.columns),
        },
    )
    _event(
        state,
        deps,
        "PROFILED",
        {
            "rows": profile.row_count,
            "columns": len(profile.columns),
        },
    )

    compact = profile.model_dump(mode="json")
    for column in compact["columns"]:
        # 원시 sample은 Artifact에만 두고 반복 Planner context에서는 제외한다.
        column["sample_values"] = []
    return {
        "profile": compact,
        "profile_ref": ref.model_dump(mode="json"),
    }


async def decompose_objective(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> dict[str, Any]:
    """분석 목표를 SubGoal로 분해한다. 실행당 한 번만 수행한다."""
    if state.get("sub_goals"):
        return {}

    profile = get_profile(state)
    assert profile is not None

    if deps.planner is None:
        return {}

    goals = await deps.planner.decompose_objective(
        objective=state.get("objective", ""),
        profile=profile,
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )
    if not goals:
        reason = getattr(deps.planner, "last_error", None) or "축을 만들지 못했다"
        _event(state, deps, "DECOMPOSE_FAILED", {"reason": reason})
        return {}

    _event(
        state,
        deps,
        "OBJECTIVE_DECOMPOSED",
        {
            "goals": [
                {"id": goal.goal_id, "question": goal.question}
                for goal in goals
            ]
        },
    )
    return {
        "sub_goals": [goal.model_dump(mode="json") for goal in goals]
    }


async def plan_intent(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> dict[str, Any]:
    """Planner 응답을 검증하고 다음 실행 Intent와 SubGoal 상태를 확정한다."""
    profile = get_profile(state)
    assert profile is not None
    coverage = get_coverage(state)

    if deps.planner is None:
        return terminal_update(
            status=RunStatus.FAILED,
            stop_code=StopCode.SERVICE_UNAVAILABLE,
            reason="planner LLM이 없다",
        )

    all_evidence = get_evidence(state)
    goals = get_sub_goals(state)
    findings, context_reasons = _select_planner_findings(
        state,
        goals=goals,
    )
    _event(
        state,
        deps,
        "PLANNER_CONTEXT_SELECTED",
        {
            "selected_evidence_ids": [
                finding.evidence_id for finding in findings
            ],
            "reasons": context_reasons,
            "total_evidence": len(all_evidence),
        },
    )

    required_followups = pending_agenda_tasks(state)
    intent, verdict = await deps.planner.propose_intent(
        objective=state.get("objective", ""),
        profile=profile,
        sub_goals=goals,
        unexplored=coverage.unexplored(profile),
        column_usage=coverage.column_usage,
        prior_findings=findings,
        unresolved_findings=[
            finding for finding in findings if finding.needs_follow_up()
        ],
        rejected_attempts=[
            {
                "goal_id": rejected.get("goal_id"),
                "question": rejected.get("question", ""),
                "stage": rejected.get("stage"),
                "reasons": rejected.get("reasons") or [],
            }
            for rejected in state.get("rejected") or []
        ],
        previous_generation_error=state.get("last_planner_error"),
        required_followups=required_followups,
        recent_inspection=state.get("inspection_summary"),
        recent_evidence_lookup=state.get("evidence_lookup_summary") or [],
        evidence_relations=[
            relation.model_dump(mode="json")
            for relation in get_evidence_relations(state)
        ],
        remaining_seconds=max(0.0, remaining_seconds(state)),
        remaining_iterations=max(
            0,
            int(state.get("max_iterations", 25))
            - int(state.get("iteration", 0)),
        ),
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )

    updated_goals, goal_update = _goal_progress(
        state,
        deps,
        goals,
        verdict,
    )

    if intent is None:
        return _no_intent(
            state,
            deps,
            verdict=verdict,
            updated_goals=updated_goals,
            goal_update=goal_update,
        )

    rejection = _rejected_intent(
        state,
        deps,
        intent,
        profile=profile,
        goals=goals,
        evidence_ids={evidence.evidence_id for evidence in all_evidence},
        coverage=coverage,
        updated_goals=updated_goals,
        goal_update=goal_update,
    )
    if rejection is not None:
        return rejection

    _event(
        state,
        deps,
        "INTENT_PROPOSED",
        {
            "intent_id": intent.intent_id,
            "question": intent.question,
            "goal_id": intent.goal_id,
            "purpose": intent.purpose,
            "parent_evidence_ids": intent.parent_evidence_ids,
            "task_id": intent.task_id,
            "execution_mode": intent.execution_mode,
            "success_criterion": intent.success_criterion,
        },
        iteration=state.get("iteration", 0) + 1,
    )

    agenda_update: list[dict[str, Any]] | None = None
    if intent.task_id and intent.execution_mode == "compute":
        agenda_update = []
        for item in get_agenda(state):
            if item.task_id == intent.task_id:
                item = item.model_copy(
                    update={
                        "status": "in_progress",
                        "selected_intent_id": intent.intent_id,
                        "attempt_count": item.attempt_count + 1,
                    }
                )
            agenda_update.append(item.model_dump(mode="json"))

        _event(
            state,
            deps,
            "AGENDA_TASK_SELECTED",
            {
                "task_id": intent.task_id,
                "intent_id": intent.intent_id,
                "goal_id": intent.goal_id,
            },
        )

    return {
        **goal_update,
        **({"agenda": agenda_update} if agenda_update is not None else {}),
        **(
            {"evidence_lookup_summary": []}
            if intent.execution_mode != "lookup_evidence"
            else {}
        ),
        "intent": intent.model_dump(mode="json"),
        "iteration": state.get("iteration", 0) + 1,
        **cycle_reset_update(),
    }


async def reuse_evidence(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> dict[str, Any]:
    """검토 완료 Evidence를 추가 계산 없이 현재 SubGoal에 연결한다."""
    intent = get_intent(state)
    assert intent is not None and intent.execution_mode == "reuse_evidence"
    assert intent.goal_id is not None

    evidence_by_id = {
        item.evidence_id: item for item in get_evidence(state)
    }
    selected = [
        evidence_by_id[evidence_id]
        for evidence_id in intent.reuse_evidence_ids
    ]
    invalid = {
        item.evidence_id: evidence_data_applicability(state, item)
        for item in selected
        if evidence_data_applicability(state, item) != "exact_snapshot"
    }
    if invalid:
        raise ValueError(
            f"현재 데이터 스냅샷에 직접 재사용할 수 없는 Evidence: {invalid}"
        )

    updated = []
    for goal in get_sub_goals(state):
        if goal.goal_id == intent.goal_id:
            linked = list(
                dict.fromkeys(
                    [*goal.evidence_ids, *intent.reuse_evidence_ids]
                )
            )
            goal = goal.model_copy(update={"evidence_ids": linked})
        updated.append(goal)

    _event(
        state,
        deps,
        "EVIDENCE_REUSED",
        {
            "intent_id": intent.intent_id,
            "goal_id": intent.goal_id,
            "evidence_ids": [
                item.evidence_id for item in selected
            ],
        },
    )
    return {
        "sub_goals": [
            goal.model_dump(mode="json") for goal in updated
        ],
        "intent": None,
    }


async def mark_data_limited(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> dict[str, Any]:
    """실행 시도 후 현재 데이터로 해결 불가능한 Agenda 작업을 닫는다."""
    intent = get_intent(state)
    assert intent is not None and intent.execution_mode == "mark_data_limited"
    assert intent.task_id is not None

    updated: list[dict[str, Any]] = []
    resolved = False
    for item in get_agenda(state):
        if item.task_id == intent.task_id:
            if item.attempt_count < 1:
                raise ValueError(
                    "시도하지 않은 Agenda 작업은 data_limited로 닫을 수 없다"
                )
            item = item.model_copy(
                update={
                    "status": "data_limited",
                    "selected_intent_id": None,
                    "resolution_reason": (
                        intent.rationale.strip() or intent.question
                    ),
                }
            )
            resolved = True
        updated.append(item.model_dump(mode="json"))

    if not resolved:
        raise ValueError(
            f"Agenda task를 찾을 수 없다: {intent.task_id}"
        )

    _event(
        state,
        deps,
        "AGENDA_TASK_DATA_LIMITED",
        {
            "task_id": intent.task_id,
            "reason": intent.rationale.strip() or intent.question,
        },
    )
    return {"agenda": updated, "intent": None}


__all__ = [
    "profile_data",
    "decompose_objective",
    "plan_intent",
    "reuse_evidence",
    "mark_data_limited",
    "NO_INTENT_CODE",
    "_apply_goal_verdict",
    "_select_planner_findings",
    "_agenda_intent_error",
    "_repeated_action_error",
    "_reuse_evidence_error",
    "_stalled",
    "_goal_progress",
    "_no_intent",
    "_rejected_intent",
]
