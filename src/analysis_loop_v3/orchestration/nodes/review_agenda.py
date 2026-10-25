"""Evidence 채택 후 Agenda 상태 전이 규칙."""

from __future__ import annotations

from ...contracts import AgendaTask, CriticReview, EvidenceRecord, ResearchIntent
from ...state import AnalysisGraphState, get_agenda
from .review_evidence import revalidation_changed_conditions


def agenda_after_commit(
    state: AnalysisGraphState,
    *,
    intent: ResearchIntent,
    review: CriticReview,
    evidence: EvidenceRecord,
) -> tuple[list[AgendaTask], list[AgendaTask], str | None]:
    """채택된 Evidence로 기존 작업을 판정한 뒤 새 작업을 추가한다.

    - follow_up: 선택된 작업이면 완료한다.
    - resolve_contradiction: 출처가 supersedes/reconciles로 해소되면 완료한다.
      이 관계는 선택하지 않은 모순 작업도 해결할 수 있다.
    - revalidate: 선택된 작업의 모든 출처를 명시적으로 재검증하고,
      방법·프로토콜·데이터·변환 중 조건 차이가 있어야 완료한다.
    - 선택한 모순/재검증 작업이 해소되지 않았으면 pending으로 돌린다.
    기존 작업 판정 후 미해결 질문·재검증 요청·모순에 해당하는 작업을 중복 없이 추가한다.
    """
    updated: list[AgendaTask] = []
    completed_id: str | None = None

    superseded = set(review.supersedes)
    reconciled = set(review.reconciles)
    for item in get_agenda(state):
        selected = item.task_id == intent.task_id
        contradiction_resolved = item.kind == "resolve_contradiction" and (
            bool(superseded.intersection(item.source_evidence_ids))
            or set(item.source_evidence_ids).issubset(reconciled)
        )
        revalidation_changes = (
            revalidation_changed_conditions(
                state,
                source_ids=item.source_evidence_ids,
                evidence=evidence,
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
            (selected and item.kind == "follow_up")
            or contradiction_resolved
            or revalidation_resolved
        )

        if should_complete and item.status != "completed":
            item = item.model_copy(
                update={
                    "status": "completed",
                    "selected_intent_id": intent.intent_id,
                    "completion_evidence_id": evidence.evidence_id,
                    "resolution_reason": review.claim.strip(),
                }
            )
            if selected:
                completed_id = item.task_id
        elif selected and item.kind in {"resolve_contradiction", "revalidate"}:
            if item.kind == "resolve_contradiction":
                reason = "accepted evidence did not resolve contradiction"
            elif set(item.source_evidence_ids).issubset(set(review.revalidates)):
                reason = (
                    "critic marked revalidation, but method/protocol/data/transform "
                    "conditions did not change"
                )
            else:
                reason = "accepted evidence did not explicitly revalidate source evidence"
            item = item.model_copy(
                update={
                    "status": "pending",
                    "selected_intent_id": None,
                    "resolution_reason": reason,
                }
            )
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
            rationale=("Critic이 현재 Evidence의 신뢰도·범위를 독립 조건에서 다시 확인하도록 지정"),
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


_agenda_after_commit = agenda_after_commit
