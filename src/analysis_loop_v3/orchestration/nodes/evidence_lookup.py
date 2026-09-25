"""전체 Evidence를 프롬프트에 싣지 않고 구조적 키로 과거 근거를 조회한다."""

from __future__ import annotations

from typing import Any

from ...contracts import FindingDigest
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    active_evidence,
    evidence_data_applicability,
    get_evidence_relations,
    get_intent,
)
from ._shared import _event

LOOKUP_LIMIT = 8



async def lookup_evidence(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """goal/컬럼/계보를 기준으로 오래된 active Evidence를 찾는다.

    Claim 텍스트 유사도로 분석 정답을 판단하지 않는다. durable 구조 필드만 사용한다.
    """
    intent = get_intent(state)
    assert intent is not None and intent.execution_mode == "lookup_evidence"

    evidence = active_evidence(state)
    applicability = {
        item.evidence_id: evidence_data_applicability(state, item)
        for item in evidence
    }
    # 스키마가 달라진 근거는 현재 분석 후보로 직접 비교하지 않는다.
    evidence = [
        item for item in evidence
        if applicability[item.evidence_id] != "incompatible_schema"
    ]
    requested_columns = set(intent.candidate_columns)
    parent_ids = set(intent.parent_evidence_ids)

    relation_neighbors: set[str] = set()
    if parent_ids:
        for relation in get_evidence_relations(state):
            if relation.source_evidence_id in parent_ids:
                relation_neighbors.add(relation.target_evidence_id)
            if relation.target_evidence_id in parent_ids:
                relation_neighbors.add(relation.source_evidence_id)

    ranked: list[tuple[int, int, Any]] = []
    for index, item in enumerate(evidence):
        score = 30 if applicability[item.evidence_id] == "exact_snapshot" else 10
        if item.evidence_id in parent_ids:
            score += 100
        if item.evidence_id in relation_neighbors:
            score += 80
        if intent.goal_id and item.goal_id == intent.goal_id:
            score += 40
        score += 10 * len(requested_columns.intersection(item.columns_used))
        if score > 0:
            ranked.append((score, index, item))

    # 구조적 조건으로 아무것도 못 찾았으면 최신 active Evidence만 짧게 보여 준다.
    if not ranked:
        selected = evidence[-LOOKUP_LIMIT:]
    else:
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected = [item for _, _, item in ranked[:LOOKUP_LIMIT]]

    digests = [
        FindingDigest.from_evidence(
            item, data_applicability=applicability[item.evidence_id],
        ).model_dump(mode="json")
        for item in selected
    ]
    history = {
        "intent_id": intent.intent_id,
        "goal_id": intent.goal_id,
        "columns": sorted(requested_columns),
        "parents": sorted(parent_ids),
        "selected_evidence_ids": [item.evidence_id for item in selected],
    }
    _event(state, deps, "EVIDENCE_LOOKED_UP", {
        "intent_id": intent.intent_id,
        "selected_evidence_ids": history["selected_evidence_ids"],
        "goal_id": intent.goal_id,
        "columns": history["columns"],
    })
    return {
        "evidence_lookup_summary": digests,
        "evidence_lookup_history": [history],
        "intent": None,
    }
