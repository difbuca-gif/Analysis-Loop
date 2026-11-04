"""프로파일링, 목표 분해, 분석 의도 결정.

세부 정책은 planner_context.py / intent_policy.py /
planner_decision.py에 분리한다. 이 파일은 LangGraph Node orchestration만 담당한다.
"""

from __future__ import annotations

from typing import Any

from ...contracts import (
    AnalysisCoverage,
    ArtifactKind,
    DatasetProfile,
    ResearchIntent,
    RunStatus,
    StopCode,
    SubGoal,
)
from ...data.profiling import load_dataframe, profile_dataframe
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    current_evidence,
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
from ..node_updates import PlanningUpdate
from . import planner_decision as decisions
from ._shared import _event
from .intent_policy import (
    _agenda_intent_error,
    _repeated_action_error,
    _reuse_evidence_error,
)
from .planner_context import select_planner_findings

_select_planner_findings = select_planner_findings
from .planner_decision import NO_INTENT_CODE
from .planner_decision import apply_goal_verdict as _apply_goal_verdict


async def profile_data(state: AnalysisGraphState, deps: RuntimeDeps) -> PlanningUpdate:
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
) -> PlanningUpdate:
    """분석 목표를 SubGoal로 분해한다. 실행당 한 번만 수행한다."""
    if state.get("sub_goals"):
        return {}

    profile = get_profile(state)
    assert profile is not None

    if deps.planner is None:
        return {}

    result = await deps.planner.decompose_objective(
        objective=state.get("objective", ""),
        profile=profile,
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )
    goals = result.value
    if not goals:
        reason = result.error or "축을 만들지 못했다"
        _event(state, deps, "DECOMPOSE_FAILED", {"reason": reason})
        return {}

    _event(
        state,
        deps,
        "OBJECTIVE_DECOMPOSED",
        {"goals": [{"id": goal.goal_id, "question": goal.question} for goal in goals]},
    )
    return {"sub_goals": [goal.model_dump(mode="json") for goal in goals]}


async def plan_intent(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> PlanningUpdate:
    """Planner 응답을 검증하고 다음 실행 Intent와 SubGoal 상태를 확정한다."""
    profile = get_profile(state)
    assert profile is not None
    coverage = get_coverage(state)

    if deps.planner is None:
        return {
            **terminal_update(
                status=RunStatus.FAILED,
                stop_code=StopCode.SERVICE_UNAVAILABLE,
                reason="planner LLM이 없다",
            )
        }

    all_evidence = get_evidence(state)
    goals = get_sub_goals(state)
    findings, context_reasons = select_planner_findings(
        state,
        goals=goals,
    )
    _event(
        state,
        deps,
        "PLANNER_CONTEXT_SELECTED",
        {
            "selected_evidence_ids": [finding.evidence_id for finding in findings],
            "reasons": context_reasons,
            "total_evidence": len(all_evidence),
        },
    )

    required_followups = pending_agenda_tasks(state)
    next_iteration = state.get("iteration", 0) + 1
    remaining_iterations = max(0, state.get("max_iterations", 25) - state.get("iteration", 0))
    unresolved_findings = [finding for finding in findings if finding.needs_follow_up()]
    evidence_relations = [
        relation.model_dump(mode="json") for relation in get_evidence_relations(state)
    ]
    rejected_attempts = [
        {
            "goal_id": rejected.get("goal_id"),
            "question": rejected.get("question", ""),
            "stage": rejected.get("stage"),
            "reasons": rejected.get("reasons") or [],
        }
        for rejected in state.get("rejected") or []
    ]
    result = await deps.planner.propose_intent(
        objective=state.get("objective", ""),
        profile=profile,
        sub_goals=goals,
        unexplored=coverage.unexplored(profile),
        column_usage=coverage.column_usage,
        prior_findings=findings,
        unresolved_findings=unresolved_findings,
        rejected_attempts=rejected_attempts,
        previous_generation_error=state.get("last_planner_error"),
        required_followups=required_followups,
        recent_inspection=state.get("inspection_summary"),
        recent_evidence_lookup=state.get("evidence_lookup_summary") or [],
        evidence_relations=evidence_relations,
        remaining_seconds=max(0.0, remaining_seconds(state)),
        remaining_iterations=remaining_iterations,
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )

    proposal = result.value
    intent = proposal.intent if proposal is not None else None
    verdict = proposal.verdict if proposal is not None else {}
    updated_goals, goal_decision = decisions.goal_progress(
        goals=goals,
        verdict=verdict,
        evidence=current_evidence(state),
        rejected_attempts=state.get("rejected") or [],
        pending_tasks=required_followups,
    )
    for event in goal_decision.events:
        _event(state, deps, event.name, event.payload)
    goal_update = goal_decision.update
    assert goal_update is not None

    if intent is None:
        decision = decisions.no_intent(
            state,
            verdict=verdict,
            updated_goals=updated_goals,
            goal_update=goal_update,
            failure_reason=result.error,
            failure_kind=result.failure_kind,
        )
        for event in decision.events:
            _event(state, deps, event.name, event.payload)
        assert decision.update is not None
        if decision.update.get("stop_code") == StopCode.SERVICE_UNAVAILABLE.value:
            decision.update["service_failure"] = {
                "role": "planner",
                "kind": result.failure_kind,
                "message": result.error or "planner 실패",
            }
        return decision.update

    decision = decisions.rejected_intent(
        state,
        intent,
        profile=profile,
        goals=goals,
        evidence_ids={evidence.evidence_id for evidence in all_evidence},
        coverage=coverage,
        updated_goals=updated_goals,
        goal_update=goal_update,
    )
    for event in decision.events:
        _event(state, deps, event.name, event.payload)
    if decision.update is not None:
        return decision.update

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
        iteration=next_iteration,
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

    update: PlanningUpdate = {**cycle_reset_update()}
    update.update(goal_update)
    update["intent"] = intent.model_dump(mode="json")
    update["iteration"] = next_iteration
    if agenda_update is not None:
        update["agenda"] = agenda_update
    if intent.execution_mode != "lookup_evidence":
        update["evidence_lookup_summary"] = []
    return update


async def reuse_evidence(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> PlanningUpdate:
    """검토 완료 Evidence를 추가 계산 없이 현재 SubGoal에 연결한다."""
    intent = get_intent(state)
    assert intent is not None and intent.execution_mode == "reuse_evidence"
    assert intent.goal_id is not None

    evidence_by_id = {item.evidence_id: item for item in get_evidence(state)}
    selected = [evidence_by_id[evidence_id] for evidence_id in intent.reuse_evidence_ids]
    invalid = {
        item.evidence_id: evidence_data_applicability(state, item)
        for item in selected
        if evidence_data_applicability(state, item) != "exact_snapshot"
    }
    if invalid:
        raise ValueError(f"현재 데이터 스냅샷에 직접 재사용할 수 없는 Evidence: {invalid}")

    updated = []
    for goal in get_sub_goals(state):
        if goal.goal_id == intent.goal_id:
            linked = list(dict.fromkeys([*goal.evidence_ids, *intent.reuse_evidence_ids]))
            goal = goal.model_copy(update={"evidence_ids": linked})
        updated.append(goal)

    _event(
        state,
        deps,
        "EVIDENCE_REUSED",
        {
            "intent_id": intent.intent_id,
            "goal_id": intent.goal_id,
            "evidence_ids": [item.evidence_id for item in selected],
        },
    )
    return {
        "sub_goals": [goal.model_dump(mode="json") for goal in updated],
        "intent": None,
    }


async def mark_data_limited(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> PlanningUpdate:
    """실행 시도 후 현재 데이터로 해결 불가능한 Agenda 작업을 닫는다."""
    intent = get_intent(state)
    assert intent is not None and intent.execution_mode == "mark_data_limited"
    assert intent.task_id is not None

    updated: list[dict[str, Any]] = []
    resolved = False
    for item in get_agenda(state):
        if item.task_id == intent.task_id:
            if item.attempt_count < 1:
                raise ValueError("시도하지 않은 Agenda 작업은 data_limited로 닫을 수 없다")
            item = item.model_copy(
                update={
                    "status": "data_limited",
                    "selected_intent_id": None,
                    "resolution_reason": (intent.rationale.strip() or intent.question),
                }
            )
            resolved = True
        updated.append(item.model_dump(mode="json"))

    if not resolved:
        raise ValueError(f"Agenda task를 찾을 수 없다: {intent.task_id}")

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


def _record_decision(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    decision: decisions.PlannerDecision,
) -> PlanningUpdate | None:
    for event in decision.events:
        _event(state, deps, event.name, event.payload)
    return decision.update


def _stalled(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    goal_update: PlanningUpdate,
    reason: str | None = None,
    planner_error: str | None = None,
) -> PlanningUpdate:
    update = _record_decision(
        state,
        deps,
        decisions.stalled(
            state,
            goal_update=goal_update,
            reason=reason,
            planner_error=planner_error,
        ),
    )
    assert update is not None
    return update


def _goal_progress(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    goals: list[SubGoal],
    verdict: dict[str, Any],
) -> tuple[list[SubGoal], PlanningUpdate]:
    updated, decision = decisions.goal_progress(
        goals=goals,
        verdict=verdict,
        evidence=current_evidence(state),
        rejected_attempts=state.get("rejected") or [],
        pending_tasks=pending_agenda_tasks(state),
    )
    update = _record_decision(state, deps, decision)
    assert update is not None
    return updated, update


def _no_intent(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    verdict: dict[str, Any],
    updated_goals: list[SubGoal],
    goal_update: PlanningUpdate,
    failure_reason: str | None = None,
    failure_kind: str | None = None,
) -> PlanningUpdate:
    decision = decisions.no_intent(
        state,
        failure_reason=failure_reason,
        failure_kind=failure_kind,
        verdict=verdict,
        updated_goals=updated_goals,
        goal_update=goal_update,
    )
    update = _record_decision(state, deps, decision)
    assert update is not None
    return update


def _rejected_intent(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    intent: ResearchIntent,
    *,
    profile: DatasetProfile,
    goals: list[SubGoal],
    evidence_ids: set[str],
    coverage: AnalysisCoverage,
    updated_goals: list[SubGoal],
    goal_update: PlanningUpdate,
) -> PlanningUpdate | None:
    return _record_decision(
        state,
        deps,
        decisions.rejected_intent(
            state,
            intent,
            profile=profile,
            goals=goals,
            evidence_ids=evidence_ids,
            coverage=coverage,
            updated_goals=updated_goals,
            goal_update=goal_update,
        ),
    )


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
