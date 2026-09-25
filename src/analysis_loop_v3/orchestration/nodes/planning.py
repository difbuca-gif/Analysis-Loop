"""무엇을 물을지 정하는 단계 — 프로파일, 목표 분해, 연구 의도 제안."""

from __future__ import annotations

from typing import Any

from ...contracts import (
    AgendaTask,
    AnalysisCoverage,
    ArtifactKind,
    DatasetProfile,
    FindingDigest,
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
    superseded_evidence_ids,
    terminal_update,
)
from ...validation.checks import validate_intent_references
from ._shared import STUCK_THRESHOLD, _event, bump_error_streak

# 프로파일

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
        # 원시 샘플은 artifact에만 보관한다.
        column["sample_values"] = []
    return {"profile": compact, "profile_ref": ref.model_dump(mode="json")}

# 목표 분해 — 목표와 개별 질문 사이의 층

async def decompose_objective(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """목표를 답할 수 있는 축으로 쪼갠다(실행당 1회).

    축이 있어야 "목표를 다 다뤘는가"를 물을 수 있다(없으면 종료가 전부 대리 지표).
    분해 실패는 치명적이지 않다. 대리 지표로 돌아갈 뿐이라 실행을 막지 않는다.
    """
    if state.get("sub_goals"):
        # 재개 시 기존 목표 축을 유지한다.
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

def _apply_goal_verdict(
    goals: list[SubGoal],
    verdict: dict[str, Any],
    evidence_by_goal: dict[str, list[str]],
    attempts_by_goal: dict[str, int] | None = None,
    blocked_goal_ids: set[str] | None = None,
) -> tuple[list[SubGoal], list[str]]:
    """planner의 축 판정을 결정론적 조건과 함께 반영한다(조기 종료 방지).

    converged엔 근거를 요구한다. abandoned엔 근거 대신 시도를 요구한다. 실측:
    첫 회차에 시도 없이 축을 포기해 근거 0개로 끝난 적이 있다.
    """
    attempts_by_goal = attempts_by_goal or {}
    blocked_goal_ids = blocked_goal_ids or set()
    note = str(verdict.get("closing_note") or "")
    converged = set(verdict.get("converged") or [])
    abandoned = set(verdict.get("abandoned") or [])
    changed: list[str] = []

    updated: list[SubGoal] = []
    for goal in goals:
        if not goal.is_open:
            updated.append(goal)
            continue
        if goal.goal_id in converged:
            if goal.goal_id in blocked_goal_ids:
                changed.append(f"{goal.goal_id}:converged_rejected(required_followup)")
                updated.append(goal)
                continue
            if evidence_by_goal.get(goal.goal_id):
                updated.append(goal.model_copy(update={
                    "status": "converged", "closing_note": note,
                }))
                changed.append(f"{goal.goal_id}:converged")
                continue
            # 근거 없이 닫겠다는 선언은 무시한다.
            changed.append(f"{goal.goal_id}:converged_rejected")
        elif goal.goal_id in abandoned:
            if attempts_by_goal.get(goal.goal_id, 0) > 0:
                updated.append(goal.model_copy(update={
                    "status": "abandoned", "closing_note": note,
                }))
                changed.append(f"{goal.goal_id}:abandoned")
                continue
            changed.append(f"{goal.goal_id}:abandon_rejected(시도 부족)")
        updated.append(goal)
    return updated, changed

# 의도 생성 실패는 하나의 연속 실패 코드로 집계한다.
NO_INTENT_CODE = "planner_no_intent"

def _stalled(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    goal_update: dict[str, Any],
    reason: str | None = None,
    planner_error: str | None = None,
) -> dict[str, Any]:
    """의도 없이 회차를 끝낸다. 같은 일이 STUCK_THRESHOLD회 반복되면 런을 멈춘다.

    iteration은 의도가 나올 때만 오른다. 여기서 멈추지 않으면 max_iterations가
    영영 걸리지 않고 재귀 한도까지 플래너를 계속 부른다(실측: 상한 3회짜리 실행이
    249회 호출).
    """
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

# Planner context — 모든 Evidence를 매번 넣지 않고 지금 필요한 기억을 고른다.

PLANNER_CONTEXT_TARGET = 16

def _select_planner_findings(
    state: AnalysisGraphState,
    *,
    goals: list[SubGoal],
) -> tuple[list[FindingDigest], dict[str, list[str]]]:
    evidence = get_evidence(state)
    tasks = pending_agenda_tasks(state)
    superseded_ids = superseded_evidence_ids(state)
    by_id = {item.evidence_id: item for item in evidence}
    applicability = {
        item.evidence_id: evidence_data_applicability(state, item)
        for item in evidence
    }
    chosen: dict[str, Any] = {}
    reasons: dict[str, list[str]] = {}

    def add(evidence_id: str, reason: str, *, mandatory: bool = False) -> None:
        item = by_id.get(evidence_id)
        if item is None:
            return
        if applicability[evidence_id] == "incompatible_schema" and not mandatory:
            return
        if evidence_id not in chosen and len(chosen) >= PLANNER_CONTEXT_TARGET and not mandatory:
            return
        chosen.setdefault(evidence_id, item)
        reasons.setdefault(evidence_id, [])
        if reason not in reasons[evidence_id]:
            reasons[evidence_id].append(reason)

    # Agenda 출처와 명시적 조회 결과는 항상 보존한다.
    for task in tasks:
        for evidence_id in task.source_evidence_ids:
            add(evidence_id, f"agenda:{task.kind}", mandatory=True)
    for digest in state.get("evidence_lookup_summary") or []:
        if evidence_id := digest.get("evidence_id"):
            add(str(evidence_id), "explicit_lookup", mandatory=True)

    open_goal_ids = {goal.goal_id for goal in goals if goal.is_open}
    seen_goals: set[str] = set()
    for item in reversed(evidence):
        if item.evidence_id in superseded_ids:
            continue
        if applicability[item.evidence_id] != "exact_snapshot":
            continue
        if item.goal_id in open_goal_ids and item.goal_id not in seen_goals:
            add(item.evidence_id, "latest_open_goal", mandatory=True)
            seen_goals.add(item.goal_id)

    # 다른 데이터 스냅샷의 근거는 재검증 후보로만 사용한다.
    for item in reversed(evidence):
        if (
            item.evidence_id not in superseded_ids
            and applicability[item.evidence_id] == "revalidation_required"
        ):
            add(item.evidence_id, "dataset_changed")

    # 미해결 근거를 우선 포함한다.
    for item in reversed(evidence):
        if item.evidence_id in superseded_ids:
            continue
        if item.confidence == "low" or item.open_questions or item.contradicts:
            add(item.evidence_id, "unresolved")
            for peer in item.contradicts:
                add(peer, "contradiction_peer")

    # 남은 문맥은 현재 스냅샷의 최근 근거로 채운다.
    for item in reversed(evidence):
        if (
            item.evidence_id not in superseded_ids
            and applicability[item.evidence_id] == "exact_snapshot"
        ):
            add(item.evidence_id, "recent")

    return [
        FindingDigest.from_evidence(
            item, data_applicability=applicability[item.evidence_id],
        )
        for item in chosen.values()
    ], reasons

# 연구 의도

def _goal_progress(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    goals: list[SubGoal],
    verdict: dict[str, Any],
) -> tuple[list[SubGoal], dict[str, Any]]:
    """축 판정을 반영한다. 의도 생성이 실패해도 이 판단은 살아남아야 한다.
    그러지 않으면 이미 닫힌 축을 계속 다시 고른다."""
    evidence_by_goal: dict[str, list[str]] = {}
    for record in current_evidence(state):
        if record.goal_id:
            evidence_by_goal.setdefault(record.goal_id, []).append(record.evidence_id)
    # 채택 근거와 거부 시도를 모두 시도 횟수로 본다.
    attempts_by_goal = {goal: len(ids) for goal, ids in evidence_by_goal.items()}
    for record in state.get("rejected") or []:
        if goal := record.get("goal_id"):
            attempts_by_goal[goal] = attempts_by_goal.get(goal, 0) + 1

    blocked_goal_ids = {
        item.goal_id for item in pending_agenda_tasks(state) if item.goal_id
    }
    updated, changes = _apply_goal_verdict(
        goals, verdict, evidence_by_goal, attempts_by_goal,
        blocked_goal_ids=blocked_goal_ids,
    )
    update: dict[str, Any] = {}
    if unreviewed := verdict.get("unreviewed"):
        # 누락된 목표 축은 이벤트로 남긴다.
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
    """플래너가 의도를 못 냈을 때. 정상 완료인지, 고칠 수 없는 실패인지, 되먹여
    다시 시도할 것인지 가른다."""
    if verdict.get("finished") and all(not goal.is_open for goal in updated_goals):
        pending = pending_agenda_tasks(state)
        if pending:
            return _stalled(
                state, deps, goal_update=goal_update,
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
            state, deps, goal_update=goal_update,
            reason="플래너가 의도를 내지 않았다",
        )

    _event(state, deps, "PLANNER_FAILED", {"reason": reason})
    kind = getattr(deps.planner, "last_failure_kind", None)
    # 공급자/설정 오류는 즉시 종료하고 계약 오류만 재시도한다.
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
        state, deps, goal_update=goal_update,
        reason="플래너 응답이 계약을 어겼다", planner_error=reason,
    )

def _agenda_intent_error(
    task: AgendaTask,
    intent: ResearchIntent,
) -> tuple[str, str] | None:
    """Agenda 작업과 Planner Intent의 계약 위반을 사람이 읽는 이유와 함께 반환한다."""
    if task.kind == "resolve_contradiction" and intent.purpose != "resolve_contradiction":
        return (
            "모순 해소 작업의 purpose가 맞지 않는다",
            "resolve_contradiction task는 purpose=resolve_contradiction이어야 한다",
        )
    if task.kind == "follow_up" and intent.purpose == "explore":
        return (
            "후속 작업을 새 탐색으로 처리했다",
            "follow_up task는 deepen 또는 challenge 목적을 사용하라",
        )
    if task.kind == "revalidate" and intent.purpose != "challenge":
        return (
            "재검증 작업을 강건성 확인이 아닌 다른 목적으로 처리했다",
            "revalidate task는 purpose=challenge이어야 한다",
        )
    if intent.execution_mode not in {"compute", "mark_data_limited"}:
        return (
            "Agenda 작업은 새 Evidence 또는 명시적 data_limited 판정으로 닫아야 한다",
            "task_id가 있으면 execution_mode=compute 또는 mark_data_limited를 사용하라",
        )
    if intent.execution_mode == "mark_data_limited" and task.attempt_count < 1:
        return (
            "시도하지 않은 Agenda 작업을 data_limited로 닫으려 했다",
            "먼저 task를 compute로 실제 시도한 뒤 data_limited를 판단하라",
        )
    if task.goal_id and intent.goal_id != task.goal_id:
        return (
            "Agenda 작업과 다른 목표 축을 선택했다",
            f"task_id={task.task_id}는 goal_id={task.goal_id}에 속한다",
        )
    missing_sources = set(task.source_evidence_ids) - set(intent.parent_evidence_ids)
    if missing_sources:
        return (
            "Agenda 출처 근거를 부모 근거로 연결하지 않았다",
            f"parent_evidence_ids에 {sorted(missing_sources)}를 포함하라",
        )
    return None

def _repeated_action_error(
    state: AnalysisGraphState,
    intent: ResearchIntent,
) -> tuple[str, str] | None:
    """이미 같은 inspect/lookup을 수행했는지 검사한다."""
    if intent.execution_mode == "inspect_data":
        inspected = {
            tuple(sorted(item.get("columns") or []))
            for item in state.get("inspection_history") or []
        }
        key = tuple(sorted(set(intent.candidate_columns)))
        if key in inspected:
            return (
                "같은 컬럼 조합을 이미 점검했다",
                f"inspect_data 반복 금지: {list(key)}",
            )

    if intent.execution_mode == "lookup_evidence":
        lookup_key = {
            "goal_id": intent.goal_id,
            "columns": sorted(set(intent.candidate_columns)),
            "parents": sorted(set(intent.parent_evidence_ids)),
        }
        previous = {
            (
                item.get("goal_id"),
                tuple(item.get("columns") or []),
                tuple(item.get("parents") or []),
            )
            for item in state.get("evidence_lookup_history") or []
        }
        key = (
            lookup_key["goal_id"],
            tuple(lookup_key["columns"]),
            tuple(lookup_key["parents"]),
        )
        if key in previous:
            return (
                "같은 조건으로 Evidence를 이미 조회했다",
                f"lookup_evidence 반복 금지: {lookup_key}",
            )
    return None

def _reuse_evidence_error(
    state: AnalysisGraphState,
    intent: ResearchIntent,
) -> tuple[str, str] | None:
    """현재 실행에서 직접 재사용할 수 없는 Evidence를 설명한다."""
    obsolete = sorted(
        set(intent.reuse_evidence_ids) & superseded_evidence_ids(state)
    )
    if obsolete:
        return (
            "superseded Evidence를 재사용하려 했다",
            f"현재 결론에서 대체된 Evidence는 재사용할 수 없다: {obsolete}",
        )

    by_id = {item.evidence_id: item for item in get_evidence(state)}
    stale: dict[str, str] = {}
    for evidence_id in intent.reuse_evidence_ids:
        item = by_id.get(evidence_id)
        if item is None:
            continue
        applicability = evidence_data_applicability(state, item)
        if applicability != "exact_snapshot":
            stale[evidence_id] = applicability
    if stale:
        return (
            "현재 데이터 스냅샷과 다른 Evidence를 직접 재사용하려 했다",
            (
                "다른 데이터 버전의 Evidence는 reuse_evidence가 아니라 "
                f"재검증 대상으로 사용하라: {stale}"
            ),
        )
    return None

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
    """의도를 버려야 할 이유를 찾는다. 없으면 None. 어느 경우든 축 진척은 보존한다."""
    try:
        validate_intent_references(
            intent, profile, goals=goals,
            evidence_ids=evidence_ids,
        )
    except ValueError as exc:
        _event(state, deps, "INTENT_REFERENCE_INVALID", {"reason": str(exc)})
        return _stalled(
            state, deps, goal_update=goal_update, reason="의도가 없는 것을 참조했다",
            planner_error=f"참조 오류로 거부됨: {exc}",
        )

    if intent.task_id:
        pending = {item.task_id: item for item in pending_agenda_tasks(state)}
        task = pending.get(intent.task_id)
        if task is None:
            return _stalled(
                state, deps, goal_update=goal_update,
                reason="존재하지 않거나 이미 완료된 Agenda 작업을 선택했다",
                planner_error=f"task_id={intent.task_id!r}를 다시 선택하라",
            )
        if error := _agenda_intent_error(task, intent):
            reason, planner_error = error
            return _stalled(
                state, deps, goal_update=goal_update,
                reason=reason, planner_error=planner_error,
            )

    if error := _repeated_action_error(state, intent):
        reason, planner_error = error
        return _stalled(
            state, deps, goal_update=goal_update,
            reason=reason, planner_error=planner_error,
        )

    # 같은 응답에서 방금 닫은 축을 다시 고른 경우.
    closed = {g.goal_id for g in updated_goals if not g.is_open}
    if intent.goal_id and goal_update and intent.goal_id in closed:
        _event(state, deps, "INTENT_TARGETS_CLOSED_GOAL", {"goal_id": intent.goal_id})
        return _stalled(
            state, deps, goal_update=goal_update, reason="닫힌 축을 다시 겨냥했다",
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
            state, deps, goal_update=goal_update,
            reason=reason, planner_error=planner_error,
        )

    if intent.dedupe_key() in coverage.tested_dedupe_keys:
        _event(state, deps, "DUPLICATE_INTENT_DROPPED", {"question": intent.question})
        return _stalled(
            state, deps, goal_update=goal_update, reason="같은 분석을 다시 제안했다",
            planner_error=(
                f"직전과 같은 분석을 다시 제안했다(중복): {intent.question!r}. "
                "이미 다룬 목적·축·컬럼 조합이다. 다른 각도를 골라라"
            ),
        )
    return None

async def plan_intent(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
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
        # 규칙 검사는 분석 주제가 아니라 계약 위반만 제한한다.
        unexplored=coverage.unexplored(profile),
        column_usage=coverage.column_usage,
        prior_findings=findings,
        unresolved_findings=[f for f in findings if f.needs_follow_up()],
        # 거부 사유를 다음 계획에 전달한다.
        rejected_attempts=[
            {
                "goal_id": r.get("goal_id"),
                "question": r.get("question", ""),
                "stage": r.get("stage"),
                "reasons": r.get("reasons") or [],
            }
            for r in state.get("rejected") or []
        ],
        # 직전 거부 사유를 전달해 동일 오류 반복을 줄인다.
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
            0, int(state.get("max_iterations", 25)) - int(state.get("iteration", 0))
        ),
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )

    updated_goals, goal_update = _goal_progress(state, deps, goals, verdict)

    if intent is None:
        return _no_intent(
            state, deps, verdict=verdict,
            updated_goals=updated_goals, goal_update=goal_update,
        )

    rejection = _rejected_intent(
        state, deps, intent, profile=profile, goals=goals,
        evidence_ids={e.evidence_id for e in all_evidence},
        coverage=coverage, updated_goals=updated_goals, goal_update=goal_update,
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
        # 사이클 상태 초기화는 state.py에서 관리한다.
        **cycle_reset_update(),
    }

async def reuse_evidence(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """검토가 끝난 기존 Evidence를 새 계산 없이 현재 목표 축에 연결한다.

    새 claim을 만드는 기능이 아니다. 이미 검토된 Evidence가 현재 목표를 직접
    충족한다고 Planner가 판단했을 때 계보만 연결한다.
    """
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
    """실제 시도 뒤 현재 데이터로 작업을 해결할 수 없음을 명시적으로 기록한다."""
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
