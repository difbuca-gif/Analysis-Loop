"""Critic 검토와 Evidence 채택/거부 경계.

Evidence/Agenda/Reject 세부 정책은 별도 helper 모듈로 분리한다.
이 파일은 LangGraph Node orchestration만 담당한다.
"""

from __future__ import annotations

from typing import Any

from ...contracts import CriticReview, FindingDigest, RunStatus, StopCode
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    active_evidence,
    get_agenda,
    get_coverage,
    get_evidence,
    get_intent,
    get_profile,
    get_ref,
    terminal_update,
)
from ...validation.checks import validate_critic_references
from ...validation.numeric import bind_review_to_result
from ..failure_policy import service_failure_action
from ..node_updates import CritiqueUpdate, EvidenceCommitUpdate, RejectionUpdate
from ._shared import _event, _load_generated
from .rejection_policy import _rejection_reasons, _stuck_issue
from .review_agenda import _agenda_after_commit
from .review_evidence import (
    _build_evidence,
    _cache_program,
    _goals_with_evidence,
    _relations_after_commit,
)


async def critique(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> CritiqueUpdate:
    """결정론적 검증을 통과한 결과만 Critic에게 의미 검토를 요청한다."""
    if deps.critic is None:
        _event(state, deps, "CRITIC_UNAVAILABLE")
        return {
            "review": {"verdict": "unreviewed", "reasons": ["critic LLM 없음"]},
            **terminal_update(
                status=RunStatus.FAILED,
                stop_code=StopCode.SERVICE_UNAVAILABLE,
                reason="critic LLM이 없다",
            ),
        }

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
        prior_findings=[FindingDigest.from_evidence(e) for e in active_evidence(state)],
        validation_warnings=state.get("validation_warnings", []),
    )
    if review is None:
        reason = getattr(deps.critic, "last_error", None) or "이유 불명"
        _event(state, deps, "CRITIC_FAILED", {"reason": reason})
        update: CritiqueUpdate = {
            "review": {"verdict": "unreviewed", "reasons": ["critic 호출 실패"]},
        }
        kind = getattr(deps.critic, "last_failure_kind", None)
        if service_failure_action(kind) == "stop_run":
            update.update(
                terminal_update(
                    status=RunStatus.FAILED,
                    stop_code=StopCode.SERVICE_UNAVAILABLE,
                    reason=f"critic 실패: {reason}",
                )
            )
        return update

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
            state,
            deps,
            "CRITIC_RESPONSE_INVALID",
            {"reason": str(exc)[:300]},
        )
        return {
            "review": {
                "verdict": "unreviewed",
                "reasons": [f"critic 응답 계약 위반: {str(exc)[:200]}"],
            }
        }

    _event(
        state,
        deps,
        "CRITIQUED",
        {
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
            "effect_count": len(parsed.effect_summary),
        },
    )
    return {"review": parsed.model_dump(mode="json")}


async def commit_evidence(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> EvidenceCommitUpdate:
    """Critic 승인 결과를 Evidence와 관련 상태에 일관되게 반영한다."""
    intent = get_intent(state)
    result_ref = get_ref(state, "candidate_ref")
    program_ref = get_ref(state, "program_ref")
    dataset = get_ref(state, "dataset_ref")
    profile = get_profile(state)
    assert (
        intent is not None
        and result_ref is not None
        and dataset is not None
        and profile is not None
    )

    generated = _load_generated(state, deps)
    review = CriticReview.model_validate(state.get("review") or {})
    if review.verdict != "accept":
        raise ValueError("accept 판정만 Evidence로 저장할 수 있다")

    validate_critic_references(
        review,
        generated,
        evidence_ids={e.evidence_id for e in get_evidence(state)},
    )
    review = bind_review_to_result(
        review,
        deps.artifacts.read_json(result_ref),
    )

    evidence = _build_evidence(
        state,
        intent=intent,
        review=review,
        generated=generated,
        result_ref=result_ref,
        program_ref=program_ref,
        dataset_ref=dataset,
        profile=profile,
    )

    if program_ref is not None:
        _cache_program(
            state,
            deps,
            intent=intent,
            generated=generated,
            program_ref=program_ref,
        )

    coverage = get_coverage(state).with_record(generated.selected_columns)
    coverage = coverage.model_copy(
        update={
            "tested_dedupe_keys": [
                *coverage.tested_dedupe_keys,
                intent.dedupe_key(),
            ],
        }
    )

    agenda, created_tasks, completed_task_id = _agenda_after_commit(
        state,
        intent=intent,
        review=review,
        evidence=evidence,
    )
    relations = _relations_after_commit(
        state,
        intent=intent,
        review=review,
        evidence=evidence,
    )

    update: EvidenceCommitUpdate = {
        "evidence": [evidence.model_dump(mode="json")],
        "evidence_relations": [item.model_dump(mode="json") for item in relations],
        "coverage": coverage.model_dump(mode="json"),
        "agenda": [item.model_dump(mode="json") for item in agenda],
        "intent": None,
    }
    if intent.goal_id:
        goals = _goals_with_evidence(
            state,
            intent.goal_id,
            evidence.evidence_id,
        )
        if goals:
            update["sub_goals"] = goals

    if completed_task_id:
        _event(
            state,
            deps,
            "AGENDA_TASK_COMPLETED",
            {
                "task_id": completed_task_id,
                "completion_evidence_id": evidence.evidence_id,
            },
        )
    for task in created_tasks:
        _event(
            state,
            deps,
            "AGENDA_TASK_CREATED",
            {
                "task_id": task.task_id,
                "kind": task.kind,
                "source_evidence_ids": task.source_evidence_ids,
                "goal_id": task.goal_id,
                "question": task.question,
            },
        )
    for relation in relations:
        _event(
            state,
            deps,
            "EVIDENCE_RELATION_ADDED",
            relation.model_dump(mode="json"),
        )

    _event(
        state,
        deps,
        "EVIDENCE_COMMITTED",
        {
            "evidence_id": evidence.evidence_id,
            "intent_id": intent.intent_id,
            "goal_id": intent.goal_id,
            "purpose": intent.purpose,
            "parent_evidence_ids": intent.parent_evidence_ids,
            "contradicts": evidence.contradicts,
        },
    )
    return update


async def reject_attempt(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> RejectionUpdate:
    """실패/거부된 회차를 기록하고 재계획 가능한 상태로 복원한다."""
    intent = get_intent(state)
    review = state.get("review") or {}
    reasons = _rejection_reasons(state, review)

    payload: RejectionUpdate = {"intent": None}
    if intent is not None and intent.task_id:
        agenda: list[dict[str, Any]] = []
        released = False
        for item in get_agenda(state):
            if item.task_id == intent.task_id and item.status == "in_progress":
                item = item.model_copy(
                    update={
                        "status": "pending",
                        "selected_intent_id": None,
                    }
                )
                released = True
            agenda.append(item.model_dump(mode="json"))
        if released:
            payload["agenda"] = agenda
            _event(
                state,
                deps,
                "AGENDA_TASK_RELEASED",
                {
                    "task_id": intent.task_id,
                    "intent_id": intent.intent_id,
                },
            )

    if intent is not None:
        payload["rejected"] = [
            {
                "intent_id": intent.intent_id,
                "dedupe_key": intent.dedupe_key(),
                "stage": (
                    "critic" if review else "result" if state.get("candidate_ref") else "code"
                ),
                "reasons": reasons,
                "goal_id": intent.goal_id,
                "question": intent.question,
            }
        ]

    if issue := _stuck_issue(state, intent, reasons):
        payload["issues"] = [issue.model_dump(mode="json")]
        payload["error_streak"] = {}
        _event(state, deps, "STUCK_ON_ERROR", issue.model_dump(mode="json"))

    _event(state, deps, "ATTEMPT_REJECTED", {"reasons": reasons[:3]})
    return payload
