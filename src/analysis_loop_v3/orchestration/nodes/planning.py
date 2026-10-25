"""프로파일링, 목표 분해, 분석 의도 결정.

세부 정책은 planner_context.py / goal_policy.py / intent_policy.py에 분리한다.
이 파일은 LangGraph Node의 orchestration만 담당한다.
"""

from __future__ import annotations

from typing import Any

from ...contracts import (
    AnalysisCoverage,
    ArtifactKind,
    DatasetProfile,
    LoopIssue,
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
from ...validation.checks import validate_intent_references
from ._shared import STUCK_THRESHOLD, _event, bump_error_streak
from .goal_policy import _apply_goal_verdict
from .intent_policy import (
    _agenda_intent_error,
    _repeated_action_error,
    _reuse_evidence_error,
)
from .planner_context import _select_planner_findings


async def profile_data(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    dataset = get_ref(state, "dataset_ref")
    assert dataset is not None, "dataset_ref 없이 그래프를 시작할 수 없다"

    frame = load_dataframe(dataset.path)
    profile = profile_dataframe(frame, dataset_id=dataset.artifact_id)
    ref = deps.artifacts.put_json(
        profile.model_dump(mode="json"),
        kind=ArtifactKind.PROFILE,
        summary={"rows": profile.row_count, "columns": len(profile.columns)},
    )
    _event(state, deps, "PROFILED", {
        "rows": profile.row_count, "columns": len(profile.columns),
    })
    compact = profile.model_dump(mode="json")
    for column in compact["columns"]:
        column["sample_values"] = []
    return {"profile": compact, "profile_ref": ref.model_dump(mode="json")}


async def decompose_objective(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
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

    _event(state, deps, "OBJECTIVE_DECOMPOSED", {
        "goals": [{"id": g.goal_id, "question": g.question} for g in goals],
    })
    return {"sub_goals": [g.model_dump(mode="json") for g in goals]}


NO_INTENT_CODE = "planner_no_intent"


def _stalled(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    goal_update: dict[str, Any],
    reason: str | None = None,
    planner_error: str | None = None,
) -> dict[str, Any]:
    """Intent 미생성 회차를 기록하고 연속 실패 시 종료한다."""
    streak = bump_error_streak(state, [{"code": NO_INTENT_CODE}])
    update: dict[str, Any] = {
        **goal_update,
        "intent": None,
        "error_streak": streak,
        "last_planner_error": planner_error,
    }
    if streak.get("count", 0) < STUCK_THRESHOLD:
        return update

    issue = LoopIssue(
        label=NO_INTENT_CODE,
        count=int(streak["count"]),
        iteration=int(state.get("iteration", 0)),
        messages=[m for m in (reason, planner_error) if m][:2],
    )
    _event(state, deps, "PLANNER_STUCK", issue.model_dump(mode="json"))
    return {
        **update,
        "issues": [issue.model_dump(mode="json")],
        **terminal_update(
            status=RunStatus.FAILED,
            stop_code=StopCode.PLANNER_STUCK,
            reason=f"플래너가 {streak['count']}회 연속 분석 의도를 만들지 못했다",
        ),
    }


def _goal_progress(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    goals: list[SubGoal],
    verdict: dict[str, Any],
) -> tuple[list[SubGoal], dict[str, Any]]:
    """Intent 생성 여부와 독립적으로 SubGoal 판정을 반영한다."""
    evidence_by_goal: dict[str, list[str]] = {}
    for record in current_evidence(state):
        if record.goal_id:
            evidence_by_goal.setdefault(record.goal_id, []).append(record.evidence_id)

    attempts_by_goal = {goal: len(ids) for goal, ids in evidence_by_goal.items()}
    for record in state.get("rejected") or []:
        if goal := record.get("goal_id"):
            attempts_by_goal[goal] = attempts_by_goal.get(goal, 0) + 1

    blocked_goal_ids = {
        item.goal_id for item in pending_agenda_tasks(state) if item.goal_id
    }
    updated, changes = _apply_goal_verdict(
        goals,
        verdict,
        evidence_by_goal,
        attempts_by_goal,
        blocked_goal_ids=blocked_goal_ids,
    )
    update: dict[str, Any] = {}
    if unreviewed := verdict.get("unreviewed"):
        _event(state, deps, "GOALS_UNREVIEWED", {"goal_ids": unreviewed})
    if changes:
        _event(state, deps, "GOAL_STATUS_CHANGED", {"changes": changes})
        update["sub_goals"] = [g.model_dump(mode="json") for g in updated]
    return updated, update


def _no_intent(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    verdict: dict[str, Any],
    updated_goals: list[SubGoal],
    goal_update: dict[str, Any],
) -> dict[str, Any]:
    """Planner가 Intent를 내지 않았을 때 완료/실패/재시도를 구분한다."""
    if verdict.get("finished") and all(not goal.is_open for goal in updated_goals):
        pending = pending_agenda_tasks(state)
        if pending:
            return _stalled(
                state,
                deps,
                goal_update=goal_update,
                reason="미해결 Agenda 작업이 남아 있는데 종료를 선언했다",
                planner_error=(
                    "pending task를 먼저 처리하라: "
                    + ", ".join(item.task_id for item in pending[:3])
                ),
            )
        return {**goal_update, "intent": None, "last_planner_error": None}

    reason = getattr(deps.planner, "last_error", None)
    if not reason:
        return _stalled(
            state,
            deps,
            goal_update=goal_update,
            reason="플래너가 의도를 내지 않았다",
        )

    _event(state, deps, "PLANNER_FAILED", {"reason": reason})
    kind = getattr(deps.planner, "last_failure_kind", None)
    if kind in {"provider_unavailable", "configuration"}:
        return {
            **goal_update,
            "intent": None,
            **terminal_update(
                status=RunStatus.FAILED,
                stop_code=StopCode.SERVICE_UNAVAILABLE,
                reason=f"planner 실패: {reason}",
            ),
        }
    return _stalled(
        state,
        deps,
        goal_update=goal_update,
        reason="플래너 응답이 계약을 어겼다",
        planner_error=reason,
    )


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
    goal_update: dict[str, Any],
) -> dict[str, Any] | None:
    """Intent의 참조·Agenda·중복·재사용 정책을 검사한다."""
    try:
        validate_intent_references(
            intent,
            profile,
            goals=goals,
            evidence_ids=evidence_ids,
        )
    except ValueError as exc:
        _event(state, deps, "INTENT_REFERENCE_INVALID", {"reason": str(exc)})
        return _stalled(
            state,
            deps,
            goal_update=goal_update,
            reason="의도가 없는 것을 참조했다",
            planner_error=f"참조 오류로 거부됨: {exc}",
        )

    if intent.task_id:
        pending = {item.task_id: item for item in pending_agenda_tasks(state)}
        task = pending.get(intent.task_id)
        if task is None:
            return _stalled(
                state,
                deps,
                goal_update=goal_update,
                reason="존재하지 않거나 이미 완료된 Agenda 작업을 선택했다",
                planner_error=f"task_id={intent.task_id!r}를 다시 선택하라",
            )
        if error := _agenda_intent_error(task, intent):
            reason, planner_error = error
            return _stalled(
                state,
                deps,
                goal_update=goal_update,
                reason=reason,
                planner_error=planner_error,
            )

    if error := _repeated_action_error(state, intent):
        reason, planner_error = error
        return _stalled(
            state,
            deps,
            goal_update=goal_update,
            reason=reason,
            planner_error=planner_error,
        )

    closed = {g.goal_id for g in updated_goals if not g.is_open}
    if intent.goal_id and goal_update and intent.goal_id in closed:
        _event(state, deps, "INTENT_TARGETS_CLOSED_GOAL", {"goal_id": intent.goal_id})
        return _stalled(
            state,
            deps,
            goal_update=goal_update,
            reason="닫힌 축을 다시 겨냥했다",
            planner_error=(
                f"방금 응답이 이미 판정으로 닫힌 축 {intent.goal_id}를 다시 겨냥했다. "
                "아직 열려 있는 다른 축을 고르거나 goal_id를 비워라"
            ),
        )

    if (
        intent.execution_mode == "reuse_evidence"
        and (error := _reuse_evidence_error(state, intent))
    ):
        reason, planner_error = error
        return _stalled(
            state,
            deps,
            goal_update=goal_update,
            reason=reason,
            planner_error=planner_error,
        )

    if intent.dedupe_key() in coverage.tested_dedupe_keys:
        _event(state, deps, "DUPLICATE_INTENT_DROPPED", {"question": intent.question})
        return _stalled(
            state,
            deps,
            goal_update=goal_update,
            reason="같은 분석을 다시 제안했다",
            planner_error=(
                f"직전과 같은 분석을 다시 제안했다(중복): {intent.question!r}. "
                "이미 다룬 목적·축·컬럼 조합이다. 다른 각도를 골라라"
            ),
        )
    return None


async def plan_intent(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
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
    findings, context_reasons = _select_planner_findings(state, goals=goals)
    _event(state, deps, "PLANNER_CONTEXT_SELECTED", {
        "selected_evidence_ids": [f.evidence_id for f in findings],
        "reasons": context_reasons,
        "total_evidence": len(all_evidence),
    })
    required_followups = pending_agenda_tasks(state)
    intent, verdict = await deps.planner.propose_intent(
        objective=state.get("objective", ""),
        profile=profile,
        sub_goals=goals,
        unexplored=coverage.unexplored(profile),
        column_usage=coverage.column_usage,
        prior_findings=findings,
        unresolved_findings=[f for f in findings if f.needs_follow_up()],
        rejected_attempts=[
            {
                "goal_id": r.get("goal_id"),
                "question": r.get("question", ""),
                "stage": r.get("stage"),
                "reasons": r.get("reasons") or [],
            }
            for r in state.get("rejected") or []
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
            int(state.get("max_iterations", 25)) - int(state.get("iteration", 0)),
        ),
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )

    updated_goals, goal_update = _goal_progress(state, deps, goals, verdict)

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
        evidence_ids={e.evidence_id for e in all_evidence},
        coverage=coverage,
        updated_goals=updated_goals,
        goal_update=goal_update,
    )
    if rejection is not None:
        return rejection

    _event(state, deps, "INTENT_PROPOSED", {
        "intent_id": intent.intent_id,
        "question": intent.question,
        "goal_id": intent.goal_id,
        "purpose": intent.purpose,
        "parent_evidence_ids": intent.parent_evidence_ids,
        "task_id": intent.task_id,
        "execution_mode": intent.execution_mode,
        "success_criterion": intent.success_criterion,
    }, iteration=state.get("iteration", 0) + 1)

    agenda_update: list[dict[str, Any]] | None = None
    if intent.task_id and intent.execution_mode == "compute":
        agenda_update = []
        for item in get_agenda(state):
            if item.task_id == intent.task_id:
                item = item.model_copy(update={
                    "status": "in_progress",
                    "selected_intent_id": intent.intent_id,
                    "attempt_count": item.attempt_count + 1,
                })
            agenda_update.append(item.model_dump(mode="json"))
        _event(state, deps, "AGENDA_TASK_SELECTED", {
            "task_id": intent.task_id,
            "intent_id": intent.intent_id,
            "goal_id": intent.goal_id,
        })

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


async def reuse_evidence(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """검토 완료 Evidence를 추가 계산 없이 현재 SubGoal에 연결한다."""
    intent = get_intent(state)
    assert intent is not None and intent.execution_mode == "reuse_evidence"
    assert intent.goal_id is not None

    evidence_by_id = {item.evidence_id: item for item in get_evidence(state)}
    selected = [evidence_by_id[eid] for eid in intent.reuse_evidence_ids]
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

    _event(state, deps, "EVIDENCE_REUSED", {
        "intent_id": intent.intent_id,
        "goal_id": intent.goal_id,
        "evidence_ids": [item.evidence_id for item in selected],
    })
    return {
        "sub_goals": [goal.model_dump(mode="json") for goal in updated],
        "intent": None,
    }


async def mark_data_limited(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
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
            item = item.model_copy(update={
                "status": "data_limited",
                "selected_intent_id": None,
                "resolution_reason": intent.rationale.strip() or intent.question,
            })
            resolved = True
        updated.append(item.model_dump(mode="json"))
    if not resolved:
        raise ValueError(f"Agenda task를 찾을 수 없다: {intent.task_id}")

    _event(state, deps, "AGENDA_TASK_DATA_LIMITED", {
        "task_id": intent.task_id,
        "reason": intent.rationale.strip() or intent.question,
    })
    return {"agenda": updated, "intent": None}
