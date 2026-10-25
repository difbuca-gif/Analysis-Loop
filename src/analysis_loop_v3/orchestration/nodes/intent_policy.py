"""Planner가 제안한 ResearchIntent의 정책 검증 helper."""

from __future__ import annotations

from ...contracts import AgendaTask, ResearchIntent
from ...state import (
    AnalysisGraphState,
    evidence_data_applicability,
    get_evidence,
    superseded_evidence_ids,
)


def agenda_intent_error(
    task: AgendaTask,
    intent: ResearchIntent,
) -> tuple[str, str] | None:
    """Agenda와 Planner Intent 간 계약 위반 사유를 반환한다."""
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


def repeated_action_error(
    state: AnalysisGraphState,
    intent: ResearchIntent,
) -> tuple[str, str] | None:
    """동일 inspect/lookup 실행을 반복하는지 확인한다."""
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


def reuse_evidence_error(
    state: AnalysisGraphState,
    intent: ResearchIntent,
) -> tuple[str, str] | None:
    """현재 실행에서 Evidence를 직접 재사용할 수 없는 사유를 반환한다."""
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


# 기존 planning.py 내부 helper 이름 호환.
_agenda_intent_error = agenda_intent_error
_repeated_action_error = repeated_action_error
_reuse_evidence_error = reuse_evidence_error
