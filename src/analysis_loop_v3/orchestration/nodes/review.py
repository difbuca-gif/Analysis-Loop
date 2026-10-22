"""Critic 검토와 Evidence 채택/거부 경계."""

from __future__ import annotations

from typing import Any

from ...contracts import (
    AgendaTask,
    ArtifactRef,
    CriticReview,
    DataLineage,
    EvidenceRecord,
    EvidenceRelation,
    FindingDigest,
    GeneratedAnalysis,
    LoopIssue,
    ResearchIntent,
)
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    active_evidence,
    get_agenda,
    get_coverage,
    get_evidence,
    get_evidence_relations,
    get_intent,
    get_profile,
    get_ref,
    get_sub_goals,
)
from ...validation.checks import validate_critic_references
from ...validation.numeric import bind_review_to_result
from ._shared import _event, _load_generated, is_stuck

# 결정론적 검증 통과 결과만 Critic 검토

async def critique(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    if deps.critic is None:
        # Critic 부재 시 미검토 상태로 기록
        _event(state, deps, "CRITIC_UNAVAILABLE")
        return {"review": {"verdict": "unreviewed", "reasons": ["critic LLM 없음"]}}

    intent = get_intent(state)
    ref = get_ref(state, "candidate_ref")
    assert intent is not None and ref is not None
    generated = _load_generated(state, deps)
    payload = deps.artifacts.read_json(ref)

    review = await deps.critic.review(
        intent=intent,
        manifest=generated,
        result_summary=payload if isinstance(payload, dict) else {"value": payload},
        timeout_seconds=DEFAULT_READ_TIMEOUT,
        # superseded Evidence 제외
        prior_findings=[
            FindingDigest.from_evidence(e)
            for e in active_evidence(state)
        ],
        validation_warnings=state.get("validation_warnings", []),
    )
    if review is None:
        # Critic 호출 실패 사유 기록
        reason = getattr(deps.critic, "last_error", None) or "이유 불명"
        _event(state, deps, "CRITIC_FAILED", {"reason": reason})
        return {"review": {"verdict": "unreviewed", "reasons": ["critic 호출 실패"]}}
    try:
        parsed = CriticReview.model_validate(review)
        validate_critic_references(
            parsed,
            generated,
            evidence_ids={e.evidence_id for e in get_evidence(state)},
        )
        parsed = bind_review_to_result(parsed, payload)
    except ValueError as exc:
        _event(
            state, deps, "CRITIC_RESPONSE_INVALID",
            {"reason": str(exc)[:300]},
        )
        return {
            "review": {
                "verdict": "unreviewed",
                "reasons": [f"critic 응답 계약 위반: {str(exc)[:200]}"],
            }
        }
    _event(state, deps, "CRITIQUED", {
        "intent_id": intent.intent_id,
        "verdict": parsed.verdict,
        "confidence": parsed.confidence,
        "open_questions": parsed.open_questions,
        "contradicts": parsed.contradicts,
        "revalidates": parsed.revalidates,
        "revalidation_required": parsed.revalidation_required,
        "supersedes": parsed.supersedes,
        "reconciles": parsed.reconciles,
        "caveat_count": len(parsed.caveats),
        # 게이팅 전 효과 추정 개수 기록
        "effect_count": len(parsed.effect_summary),
    })
    return {"review": parsed.model_dump(mode="json")}


def _build_evidence(
    state: AnalysisGraphState,
    *,
    intent: ResearchIntent,
    review: CriticReview,
    generated: GeneratedAnalysis,
    result_ref: ArtifactRef,
    program_ref: ArtifactRef | None,
    dataset_ref: ArtifactRef,
    profile: Any,
) -> EvidenceRecord:
    """원본 EvidenceRecord 생성. 프롬프트용 축약은 FindingDigest에서 처리."""
    warning_caveats = [
        f"[검증 경고:{item.get('code', 'unknown')!s}] {item.get('message', '')!s}"
        for item in state.get("validation_warnings", [])
    ]
    return EvidenceRecord(
        intent_id=intent.intent_id,
        question=intent.question,
        claim=review.claim.strip(),
        result_ref=result_ref,
        program_ref=program_ref,
        figure_refs=state.get("figure_refs") or [],
        method=generated.method,
        protocol=generated.protocol,
        data_lineage=DataLineage.from_analysis(
            dataset_ref=dataset_ref, profile=profile, manifest=generated,
        ),
        columns_used=generated.selected_columns,
        caveats=list(dict.fromkeys([*review.caveats, *warning_caveats])),
        # 다음 회차용 요약
        effect_summary=review.effect_summary,
        uncertainty_summary=review.uncertainty_summary,
        uncertainty_source_paths=review.uncertainty_source_paths,
        confidence=review.confidence,
        # Critic 미해결 질문 보존
        open_questions=review.open_questions,
        contradicts=review.contradicts,
        purpose=intent.purpose,
        parent_evidence_ids=intent.parent_evidence_ids,
        goal_id=intent.goal_id,
    )

def _cache_program(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    intent: ResearchIntent,
    generated: GeneratedAnalysis,
    program_ref: ArtifactRef,
) -> None:
    """동일 schema/Intent 재실행 시 코드 생성 캐시 사용."""
    profile = get_profile(state)
    assert profile is not None
    cached = deps.artifacts.save_validated_program(
        schema_fingerprint=profile.schema_fingerprint,
        intent_reuse_signature=intent.reuse_signature(),
        code_ref=program_ref,
        manifest=generated.manifest_without_code(),
    )
    _event(state, deps, "PROGRAM_CACHED", {"cache_key": cached.cache_key})

def _goals_with_evidence(
    state: AnalysisGraphState, goal_id: str, evidence_id: str,
) -> list[dict[str, Any]] | None:
    """Evidence를 SubGoal에 연결해 convergence 판정 근거를 유지."""
    goals = [
        g.model_copy(update={"evidence_ids": [*g.evidence_ids, evidence_id]})
        if g.goal_id == goal_id else g
        for g in get_sub_goals(state)
    ]
    return [g.model_dump(mode="json") for g in goals] if goals else None

def _revalidation_changed_conditions(
    state: AnalysisGraphState,
    *,
    source_ids: list[str],
    evidence: EvidenceRecord,
) -> list[str]:
    """독립 재검증으로 인정할 실행 조건 변경 여부 확인."""
    by_id = {item.evidence_id: item for item in get_evidence(state)}
    changed: set[str] = set()
    for source_id in source_ids:
        source = by_id.get(source_id)
        if source is None:
            continue
        if source.method != evidence.method:
            changed.add("method")
        if source.protocol != evidence.protocol:
            changed.add("protocol")
        if source.data_lineage.dataset_sha256 != evidence.data_lineage.dataset_sha256:
            changed.add("dataset")
        if (
            source.data_lineage.transform_signature
            != evidence.data_lineage.transform_signature
        ):
            changed.add("transform")
    return sorted(changed)

def _agenda_after_commit(
    state: AnalysisGraphState,
    *,
    intent: ResearchIntent,
    review: CriticReview,
    evidence: EvidenceRecord,
) -> tuple[list[AgendaTask], list[AgendaTask], str | None]:
    """선택 작업 종료 후 Critic의 미해결 질문·모순을 Agenda로 등록."""
    updated: list[AgendaTask] = []
    completed_id: str | None = None

    # contradiction 종료에는 supersede/reconcile 관계 필요
    superseded = set(review.supersedes)
    reconciled = set(review.reconciles)
    for item in get_agenda(state):
        selected = item.task_id == intent.task_id
        contradiction_resolved = (
            item.kind == "resolve_contradiction"
            and (
                bool(superseded.intersection(item.source_evidence_ids))
                or set(item.source_evidence_ids).issubset(reconciled)
            )
        )
        revalidation_changes = (
            _revalidation_changed_conditions(
                state, source_ids=item.source_evidence_ids, evidence=evidence,
            )
            if selected and item.kind == "revalidate"
            else []
        )
        revalidation_resolved = (
            selected
            and item.kind == "revalidate"
            and set(item.source_evidence_ids).issubset(set(review.revalidates))
            and bool(revalidation_changes)
        )
        should_complete = (
            selected and item.kind == "follow_up"
        ) or contradiction_resolved or revalidation_resolved

        if should_complete and item.status != "completed":
            item = item.model_copy(update={
                "status": "completed",
                "selected_intent_id": intent.intent_id,
                "completion_evidence_id": evidence.evidence_id,
                "resolution_reason": review.claim.strip(),
            })
            if selected:
                completed_id = item.task_id
        elif selected and item.kind in {"resolve_contradiction", "revalidate"}:
            # 해결 관계가 없으면 다음 회차에서 재처리
            if item.kind == "resolve_contradiction":
                reason = "accepted evidence did not resolve contradiction"
            elif set(item.source_evidence_ids).issubset(set(review.revalidates)):
                reason = (
                    "critic marked revalidation, but method/protocol/data/transform "
                    "conditions did not change"
                )
            else:
                reason = "accepted evidence did not explicitly revalidate source evidence"
            item = item.model_copy(update={
                "status": "pending",
                "selected_intent_id": None,
                "resolution_reason": reason,
            })
        updated.append(item)

    def normalized(text: str) -> str:
        return " ".join(text.split()).casefold()

    unresolved_followups = {
        (item.goal_id, normalized(item.question))
        for item in updated
        if item.status != "completed" and item.kind == "follow_up"
    }
    unresolved_conflicts = {
        frozenset(item.source_evidence_ids)
        for item in updated
        if item.status != "completed" and item.kind == "resolve_contradiction"
    }
    unresolved_revalidations = {
        (item.goal_id, normalized(item.question))
        for item in updated
        if item.status != "completed" and item.kind == "revalidate"
    }
    requested_revalidations = {
        normalized(str(question))
        for question in review.revalidation_required
        if str(question).strip()
    }

    created: list[AgendaTask] = []
    for raw_question in review.open_questions:
        question = str(raw_question).strip()
        if not question:
            continue
        key = (intent.goal_id, normalized(question))
        if key in unresolved_followups or normalized(question) in requested_revalidations:
            continue
        task = AgendaTask(
            kind="follow_up",
            goal_id=intent.goal_id,
            question=question,
            rationale="Critic이 현재 claim의 신뢰도·범위를 바꿀 수 있는 후속 질문으로 지정",
            priority=70,
            source_evidence_ids=[evidence.evidence_id],
        )
        updated.append(task)
        created.append(task)
        unresolved_followups.add(key)

    for raw_question in review.revalidation_required:
        question = str(raw_question).strip()
        if not question:
            continue
        key = (intent.goal_id, normalized(question))
        if key in unresolved_revalidations:
            continue
        task = AgendaTask(
            kind="revalidate",
            goal_id=intent.goal_id,
            question=question,
            rationale=(
                "Critic이 현재 Evidence의 신뢰도·범위를 독립 조건에서 다시 확인하도록 지정"
            ),
            priority=80,
            source_evidence_ids=[evidence.evidence_id],
        )
        updated.append(task)
        created.append(task)
        unresolved_revalidations.add(key)

    for target_id in review.contradicts:
        pair = frozenset({evidence.evidence_id, target_id})
        if len(pair) < 2 or pair in unresolved_conflicts:
            continue
        task = AgendaTask(
            kind="resolve_contradiction",
            goal_id=intent.goal_id,
            question=f"Evidence {evidence.evidence_id}와 {target_id}의 모순을 해소한다",
            rationale="Critic이 같은 실행의 기존 Evidence와 실제 모순을 선언",
            priority=90,
            source_evidence_ids=[evidence.evidence_id, target_id],
        )
        updated.append(task)
        created.append(task)
        unresolved_conflicts.add(pair)

    return updated, created, completed_id

def _relations_after_commit(
    state: AnalysisGraphState,
    *,
    intent: ResearchIntent,
    review: CriticReview,
    evidence: EvidenceRecord,
) -> list[EvidenceRelation]:
    """Critic/Intent가 명시한 Evidence 관계만 추가. 기존 Evidence는 불변."""
    existing = {
        (item.source_evidence_id, item.target_evidence_id, item.kind)
        for item in get_evidence_relations(state)
    }
    candidates: list[tuple[str, str]] = []
    candidates.extend(("derived_from", target) for target in intent.parent_evidence_ids)
    candidates.extend(("contradicts", target) for target in review.contradicts)
    allowed_revalidates = set(review.revalidates)
    # 실행 조건이 동일한 반복은 revalidation으로 인정하지 않음
    if intent.task_id:
        task = next(
            (item for item in get_agenda(state) if item.task_id == intent.task_id),
            None,
        )
        if (
            task is not None
            and task.kind == "revalidate"
            and not _revalidation_changed_conditions(
                state, source_ids=task.source_evidence_ids, evidence=evidence,
            )
        ):
            allowed_revalidates.difference_update(task.source_evidence_ids)
    candidates.extend(("revalidates", target) for target in allowed_revalidates)
    candidates.extend(("supersedes", target) for target in review.supersedes)
    candidates.extend(("reconciles", target) for target in review.reconciles)

    created: list[EvidenceRelation] = []
    for kind, target in candidates:
        key = (evidence.evidence_id, target, kind)
        if target == evidence.evidence_id or key in existing:
            continue
        relation = EvidenceRelation(
            source_evidence_id=evidence.evidence_id,
            target_evidence_id=target,
            kind=kind,
        )
        created.append(relation)
        existing.add(key)
    return created

async def commit_evidence(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """Critic 승인 결과를 Evidence와 관련 상태에 일관되게 반영.

    Evidence 추가와 함께 program cache, coverage, Agenda, Evidence relation,
    SubGoal 연결을 같은 graph update에서 갱신한다.
    """
    intent = get_intent(state)
    result_ref = get_ref(state, "candidate_ref")
    program_ref = get_ref(state, "program_ref")
    dataset = get_ref(state, "dataset_ref")
    profile = get_profile(state)
    assert (
        intent is not None and result_ref is not None
        and dataset is not None and profile is not None
    )
    generated = _load_generated(state, deps)
    review = CriticReview.model_validate(state.get("review") or {})
    # 체크포인트 재개나 직접 노드 호출도 같은 채택 경계를 통과해야 한다.
    if review.verdict != "accept":
        raise ValueError("accept 판정만 Evidence로 저장할 수 있다")
    validate_critic_references(
        review, generated, evidence_ids={e.evidence_id for e in get_evidence(state)},
    )
    review = bind_review_to_result(review, deps.artifacts.read_json(result_ref))

    evidence = _build_evidence(
        state, intent=intent, review=review, generated=generated,
        result_ref=result_ref, program_ref=program_ref,
        dataset_ref=dataset, profile=profile,
    )
    if program_ref is not None:
        # 채택된 Evidence의 프로그램만 재사용 캐시에 저장
        _cache_program(
            state, deps, intent=intent, generated=generated, program_ref=program_ref,
        )

    coverage = get_coverage(state).with_record(generated.selected_columns)
    coverage = coverage.model_copy(update={
        "tested_dedupe_keys": [*coverage.tested_dedupe_keys, intent.dedupe_key()],
    })
    agenda, created_tasks, completed_task_id = _agenda_after_commit(
        state, intent=intent, review=review, evidence=evidence,
    )
    relations = _relations_after_commit(
        state, intent=intent, review=review, evidence=evidence,
    )
    update: dict[str, Any] = {
        "evidence": [evidence.model_dump(mode="json")],
        "evidence_relations": [item.model_dump(mode="json") for item in relations],
        "coverage": coverage.model_dump(mode="json"),
        "agenda": [item.model_dump(mode="json") for item in agenda],
        "intent": None,
    }
    if intent.goal_id:
        goals = _goals_with_evidence(state, intent.goal_id, evidence.evidence_id)
        if goals:
            update["sub_goals"] = goals

    if completed_task_id:
        _event(state, deps, "AGENDA_TASK_COMPLETED", {
            "task_id": completed_task_id,
            "completion_evidence_id": evidence.evidence_id,
        })
    for task in created_tasks:
        _event(state, deps, "AGENDA_TASK_CREATED", {
            "task_id": task.task_id,
            "kind": task.kind,
            "source_evidence_ids": task.source_evidence_ids,
            "goal_id": task.goal_id,
            "question": task.question,
        })
    for relation in relations:
        _event(state, deps, "EVIDENCE_RELATION_ADDED", relation.model_dump(mode="json"))

    _event(state, deps, "EVIDENCE_COMMITTED", {
        "evidence_id": evidence.evidence_id,
        "intent_id": intent.intent_id,
        "goal_id": intent.goal_id,
        "purpose": intent.purpose,
        "parent_evidence_ids": intent.parent_evidence_ids,
        "contradicts": evidence.contradicts,
    })
    return update

def _rejection_reasons(state: AnalysisGraphState, review: dict[str, Any]) -> list[str]:
    """거부 사유 반환. 검증 오류 우선, 없으면 Critic 사유 사용."""
    reasons = [str(e.get("message", e)) for e in state.get("validation_errors") or []]
    if reasons:
        return reasons
    return [str(reason) for reason in review.get("reasons") or []]

def _stuck_issue(
    state: AnalysisGraphState, intent: ResearchIntent | None, reasons: list[str],
) -> LoopIssue | None:
    """동일 오류 반복으로 중단된 회차를 운영 신호로 분류."""
    if not is_stuck(state):
        return None
    streak = state.get("error_streak") or {}
    return LoopIssue(
        label=str(streak.get("key") or "unknown"),
        count=int(streak.get("count", 0)),
        iteration=int(state.get("iteration", 0)),
        intent_id=intent.intent_id if intent else None,
        question=intent.question if intent else "",
        messages=reasons[:3],
    )

async def reject_attempt(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """실패/거부된 회차를 기록하고 재계획 가능한 상태로 복원.

    진행 중 Agenda 작업은 pending으로 반환하고, 거부 시도는 dedupe 완료 상태에
    포함하지 않아 수정된 Intent로 재시도할 수 있게 유지한다.
    """
    intent = get_intent(state)
    review = state.get("review") or {}
    reasons = _rejection_reasons(state, review)

    payload: dict[str, Any] = {"intent": None}
    if intent is not None and intent.task_id:
        agenda: list[dict[str, Any]] = []
        released = False
        for item in get_agenda(state):
            if item.task_id == intent.task_id and item.status == "in_progress":
                item = item.model_copy(update={
                    "status": "pending",
                    "selected_intent_id": None,
                })
                released = True
            agenda.append(item.model_dump(mode="json"))
        if released:
            payload["agenda"] = agenda
            _event(state, deps, "AGENDA_TASK_RELEASED", {
                "task_id": intent.task_id,
                "intent_id": intent.intent_id,
            })

    if intent is not None:
        # 거부 시도는 재시도 가능하도록 완료 키에서 제외
        payload["rejected"] = [{
            "intent_id": intent.intent_id,
            "dedupe_key": intent.dedupe_key(),
            "stage": (
                "critic" if review else "result" if state.get("candidate_ref") else "code"
            ),
            "reasons": reasons,
            "goal_id": intent.goal_id,
            "question": intent.question,
        }]

    if issue := _stuck_issue(state, intent, reasons):
        payload["issues"] = [issue.model_dump(mode="json")]
        payload["error_streak"] = {}
        _event(state, deps, "STUCK_ON_ERROR", issue.model_dump(mode="json"))

    _event(state, deps, "ATTEMPT_REJECTED", {"reasons": reasons[:3]})
    return payload
