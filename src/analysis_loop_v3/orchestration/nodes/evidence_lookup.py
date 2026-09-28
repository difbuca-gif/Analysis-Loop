"""구조적 key 기반 과거 Evidence 조회."""

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
    """goal/column/lineage 기준 active Evidence 조회.

    Claim 텍스트 유사도로 분석 정답을 판단하지 않는다. durable 구조 필드만 사용한다.
    """
    intent = get_intent(state)
    assert intent is not None and intent.execution_mode == "lookup_evidence"

    evidence = active_evidence(state)
    applicability = {
        item.evidence_id: evidence_data_applicability(state, item)
        for item in evidence
    }
    # incompatible schema Evidence는 직접 비교 대상에서 제외
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

    # 구조 조건 결과가 없으면 최신 active Evidence만 제한적으로 반환
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
