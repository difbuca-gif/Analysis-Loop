"""체크포인트와 이벤트 로그를 조립해 한 실행의 감사 요약을 만든다."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import AgendaTask, ArtifactRef, EvidenceRecord, EvidenceRelation, SubGoal


def read_event_log(path: Path | str) -> tuple[list[dict[str, Any]], list[str]]:
    """JSONL 이벤트를 읽는다. 손상된 행은 전체 감사를 막지 않고 issue로 남긴다."""
    source = Path(path)
    if not source.is_file():
        return [], [f"event log not found: {source}"]

    events: list[dict[str, Any]] = []
    issues: list[str] = []
    for lineno, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append(f"event log line {lineno}: invalid json ({exc.msg})")
            continue
        if not isinstance(item, dict):
            issues.append(f"event log line {lineno}: object가 아니다")
            continue
        events.append(item)
    return events, issues


def _artifact_issue(ref: ArtifactRef) -> str | None:
    path = Path(ref.path)
    if not path.is_file():
        return f"artifact missing: {ref.artifact_id} ({ref.kind.value})"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"artifact unreadable: {ref.artifact_id} ({exc})"
    if digest != ref.sha256:
        return (
            f"artifact hash mismatch: {ref.artifact_id} "
            f"(expected={ref.sha256[:12]}, actual={digest[:12]})"
        )
    return None


def _all_artifact_refs(
    state: dict[str, Any], evidence: list[EvidenceRecord],
) -> list[ArtifactRef]:
    refs: dict[str, ArtifactRef] = {}

    def add(raw: Any) -> None:
        if not raw:
            return
        try:
            ref = raw if isinstance(raw, ArtifactRef) else ArtifactRef.model_validate(raw)
        except (TypeError, ValidationError):
            return
        refs.setdefault(ref.artifact_id, ref)

    for key in (
        "dataset_ref", "profile_ref", "program_ref", "candidate_ref",
        "report_ref", "inspection_ref",
    ):
        add(state.get(key))
    for item in evidence:
        add(item.result_ref)
        add(item.program_ref)
        for figure in item.figure_refs:
            add(figure)
    return list(refs.values())


def _latency_by_role(events: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    rows: dict[str, list[float]] = defaultdict(list)
    for event in events:
        role = event.get("role")
        latency = event.get("latency_seconds")
        if not isinstance(role, str) or not isinstance(latency, (int, float)):
            continue
        rows[role].append(float(latency))
    return {
        role: {
            "calls": len(values),
            "total_seconds": round(sum(values), 3),
            "max_seconds": round(max(values), 3),
        }
        for role, values in sorted(rows.items())
    }


def _active_evidence_ids(
    evidence: list[EvidenceRecord], relations: list[EvidenceRelation],
) -> set[str]:
    superseded = {
        relation.target_evidence_id
        for relation in relations
        if relation.kind == "supersedes"
    }
    return {item.evidence_id for item in evidence if item.evidence_id not in superseded}


def build_run_audit(
    state: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    event_parse_issues: list[str] | None = None,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """State/Evidence/Artifact/Event 계보와 실제 행동 경로를 검사한다."""
    evidence = [EvidenceRecord.model_validate(item) for item in state.get("evidence") or []]
    relations = [
        EvidenceRelation.model_validate(item)
        for item in state.get("evidence_relations") or []
    ]
    agenda = [AgendaTask.model_validate(item) for item in state.get("agenda") or []]
    goals = [SubGoal.model_validate(item) for item in state.get("sub_goals") or []]

    evidence_ids = {item.evidence_id for item in evidence}
    goal_ids = {goal.goal_id for goal in goals}
    issues = list(event_parse_issues or [])
    warnings: list[str] = []

    for item in evidence:
        for parent_id in item.parent_evidence_ids:
            if parent_id not in evidence_ids:
                issues.append(
                    f"evidence {item.evidence_id}: missing parent_evidence_id {parent_id}"
                )
        if item.goal_id and item.goal_id not in goal_ids:
            issues.append(f"evidence {item.evidence_id}: unknown goal_id {item.goal_id}")

    for relation in relations:
        if relation.source_evidence_id not in evidence_ids:
            issues.append(
                f"relation {relation.relation_id}: missing source "
                f"{relation.source_evidence_id}"
            )
        if relation.target_evidence_id not in evidence_ids:
            issues.append(
                f"relation {relation.relation_id}: missing target "
                f"{relation.target_evidence_id}"
            )

    for task in agenda:
        for source_id in task.source_evidence_ids:
            if source_id not in evidence_ids:
                issues.append(f"agenda {task.task_id}: missing source Evidence {source_id}")
        if task.goal_id and task.goal_id not in goal_ids:
            issues.append(f"agenda {task.task_id}: unknown goal_id {task.goal_id}")
        if task.completion_evidence_id and task.completion_evidence_id not in evidence_ids:
            issues.append(
                f"agenda {task.task_id}: missing completion Evidence "
                f"{task.completion_evidence_id}"
            )

    proposed_intents = {
        str(event.get("intent_id"))
        for event in events
        if event.get("event") == "INTENT_PROPOSED" and event.get("intent_id")
    }
    if events:
        for item in evidence:
            if item.intent_id not in proposed_intents:
                warnings.append(
                    f"evidence {item.evidence_id}: INTENT_PROPOSED event가 없다 "
                    f"(intent={item.intent_id})"
                )

    if verify_artifacts:
        for ref in _all_artifact_refs(state, evidence):
            if issue := _artifact_issue(ref):
                issues.append(issue)

    event_counts = Counter(str(event.get("event") or "<missing>") for event in events)
    action_events = {
        "INTENT_PROPOSED", "EVIDENCE_REUSED", "DATA_INSPECTED",
        "EVIDENCE_LOOKED_UP", "EVIDENCE_COMMITTED", "ATTEMPT_REJECTED",
        "FOLLOW_UP_CREATED", "FOLLOW_UP_SELECTED", "FOLLOW_UP_COMPLETED",
        "TASK_CREATED", "TASK_SELECTED", "TASK_COMPLETED", "TASK_DATA_LIMITED",
        "GOALS_EXPANDED", "GOAL_STATUS_CHANGED",
    }
    path = [
        {
            "event": event.get("event"),
            "iteration": event.get("iteration"),
            "intent_id": event.get("intent_id"),
            "goal_id": event.get("goal_id"),
            "task_id": event.get("task_id"),
            "evidence_id": event.get("evidence_id"),
        }
        for event in events
        if event.get("event") in action_events
    ]

    active_ids = _active_evidence_ids(evidence, relations)
    agenda_status = Counter(task.status for task in agenda)
    agenda_kind = Counter(task.kind for task in agenda)
    relation_kind = Counter(relation.kind for relation in relations)

    return {
        "ok": not issues,
        "run": {
            "run_id": state.get("run_id"),
            "status": state.get("status"),
            "stop_code": state.get("stop_code"),
            "stop_reason": state.get("stop_reason"),
            "iterations": state.get("iteration", 0),
        },
        "goals": [
            {
                "goal_id": goal.goal_id,
                "status": goal.status,
                "evidence_ids": goal.evidence_ids,
                "closing_note": goal.closing_note,
            }
            for goal in goals
        ],
        "evidence": {
            "total": len(evidence),
            "active": len(active_ids),
            "superseded": len(evidence) - len(active_ids),
        },
        "agenda": {
            "total": len(agenda),
            "by_status": dict(sorted(agenda_status.items())),
            "by_kind": dict(sorted(agenda_kind.items())),
            "pending_task_ids": [
                task.task_id for task in agenda
                if task.status in {"pending", "in_progress"}
            ],
        },
        "relations": {
            "total": len(relations),
            "by_kind": dict(sorted(relation_kind.items())),
        },
        "events": {
            "total": len(events),
            "by_name": dict(sorted(event_counts.items())),
            "latency_by_role": _latency_by_role(events),
            "action_path": path,
        },
        "issues": issues,
        "warnings": warnings,
    }
