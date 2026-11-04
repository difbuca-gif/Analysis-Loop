"""Planner에 전달할 Evidence 컨텍스트 선택 정책."""

from __future__ import annotations

from typing import Any

from ...contracts import FindingDigest, SubGoal
from ...state import (
    AnalysisGraphState,
    evidence_data_applicability,
    get_evidence,
    pending_agenda_tasks,
    superseded_evidence_ids,
)

PLANNER_CONTEXT_TARGET = 16


def select_planner_findings(
    state: AnalysisGraphState,
    *,
    goals: list[SubGoal],
) -> tuple[list[FindingDigest], dict[str, list[str]]]:
    """현재 판단에 필요한 Evidence만 Planner 컨텍스트로 선택한다.

    Agenda/명시 조회/열린 SubGoal의 최신 Evidence는 우선 보존하고,
    이후 재검증·미해결·최신 Evidence 순으로 목표 개수까지 채운다.
    """
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
        if (
            evidence_id not in chosen
            and len(chosen) >= PLANNER_CONTEXT_TARGET
            and not mandatory
        ):
            return
        chosen.setdefault(evidence_id, item)
        reasons.setdefault(evidence_id, [])
        if reason not in reasons[evidence_id]:
            reasons[evidence_id].append(reason)

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

    for item in reversed(evidence):
        if (
            item.evidence_id not in superseded_ids
            and applicability[item.evidence_id] == "revalidation_required"
        ):
            add(item.evidence_id, "dataset_changed")

    for item in reversed(evidence):
        if item.evidence_id in superseded_ids:
            continue
        if item.confidence == "low" or item.open_questions or item.contradicts:
            add(item.evidence_id, "unresolved")
            for peer in item.contradicts:
                add(peer, "contradiction_peer")

    for item in reversed(evidence):
        if (
            item.evidence_id not in superseded_ids
            and applicability[item.evidence_id] == "exact_snapshot"
        ):
            add(item.evidence_id, "recent")

    return [
        FindingDigest.from_evidence(
            item,
            data_applicability=applicability[item.evidence_id],
        )
        for item in chosen.values()
    ], reasons


# 기존 planning.py 내부 이름과 호환.
_select_planner_findings = select_planner_findings
