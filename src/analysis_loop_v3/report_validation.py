"""최종 Markdown이 현재 run의 Evidence와 종료 상태를 실제로 반영하는지 검사한다."""

from __future__ import annotations

import math
import re
from typing import Any

from .contracts import EvidenceRecord, EvidenceRelation

_CITATION = re.compile(r"\[증거:\s*([A-Za-z0-9_-]+)\s*\]")
_GOAL_MARKER = re.compile(r"\[목표:\s*([A-Za-z0-9_-]+)\s*\]")
_TASK_MARKER = re.compile(r"\[작업:\s*([A-Za-z0-9_-]+)\s*\]")
_STOP_MARKER = re.compile(r"\[종료:\s*([A-Z0-9_]+)\s*\]")
_NUMBER = re.compile(r"(?<![A-Za-z0-9])(-?\d+\.\d+%?|-?\d+%)(?![A-Za-z0-9])")
_REQUIRED_SECTIONS = (
    "원래 목표와 실행 상태",
    "핵심 결론",
    "목표 축별 근거",
    "한계와 미완료 사항",
)
_CITED_SECTIONS = {"핵심 결론", "목표 축별 근거"}


def _numeric_values(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            out.append(number)
        return out
    if isinstance(value, dict):
        for sub in value.values():
            out.extend(_numeric_values(sub))
    elif isinstance(value, (list, tuple)):
        for sub in value:
            out.extend(_numeric_values(sub))
    return out


def _evidence_numbers(record: EvidenceRecord) -> list[float]:
    values = [float(effect.value) for effect in record.effect_summary]
    values.extend(_numeric_values(record.uncertainty_summary))
    return values


def _number_supported(token: str, allowed: list[float]) -> bool:
    percent = token.endswith("%")
    value = float(token.rstrip("%"))
    candidates = [value]
    if percent:
        candidates.append(value / 100.0)
    return any(
        math.isclose(candidate, allowed_value, rel_tol=1e-6, abs_tol=1e-9)
        for candidate in candidates
        for allowed_value in allowed
    )


def _section_lines(markdown: str) -> dict[str, list[tuple[int, str]]]:
    sections: dict[str, list[tuple[int, str]]] = {
        section: [] for section in _REQUIRED_SECTIONS
    }
    current: str | None = None
    in_fence = False
    for lineno, raw in enumerate(markdown.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith(chr(96) * 3):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("#"):
            current = next(
                (section for section in _REQUIRED_SECTIONS if section in stripped),
                None,
            )
            continue
        if current is not None:
            sections[current].append((lineno, raw))
    return sections


def validate_report(
    markdown: str,
    *,
    evidence: list[EvidenceRecord],
    relations: list[EvidenceRelation],
    allowed_evidence_ids: set[str] | None = None,
    numeric_support: dict[str, list[float]] | None = None,
    stop_code: str | None = None,
    incomplete_goal_ids: list[str] | None = None,
    pending_task_ids: list[str] | None = None,
    unresolved_contradictions: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """구조·근거·수치·미완료 상태가 최종 보고서와 맞는지 검사한다.

    numeric_support가 주어지면 실제 Result Artifact까지 읽은 strict mode다. 이 경우
    인용 Evidence 어디에도 없는 수치는 warning이 아니라 보고서 채택 실패다.
    """
    known = {item.evidence_id: item for item in evidence}
    superseded = {
        relation.target_evidence_id
        for relation in relations
        if relation.kind == "supersedes"
    }
    allowed = (
        set(allowed_evidence_ids)
        if allowed_evidence_ids is not None
        else set(known) - superseded
    )
    errors: list[str] = []
    warnings: list[str] = []
    sections = _section_lines(markdown)

    for section in _REQUIRED_SECTIONS:
        if not any(line.strip() for _, line in sections[section]):
            errors.append(f"필수 보고서 섹션이 없거나 비어 있다: {section}")

    all_citations = _CITATION.findall(markdown)
    if allowed and not all_citations:
        errors.append("사용 가능한 Evidence가 있는데 보고서에 [증거: ...] 인용이 하나도 없다")

    for evidence_id in sorted(set(all_citations)):
        if evidence_id not in known:
            errors.append(f"존재하지 않는 Evidence 인용: {evidence_id}")
        elif evidence_id in superseded:
            errors.append(f"superseded Evidence를 현재 결론에 인용했다: {evidence_id}")
        elif evidence_id not in allowed:
            errors.append(
                f"현재 데이터 스냅샷의 직접 근거로 사용할 수 없는 Evidence 인용: {evidence_id}"
            )

    for section in _CITED_SECTIONS:
        for lineno, raw in sections[section]:
            stripped = raw.strip()
            if not stripped or set(stripped) <= {"|", "-", ":", " "}:
                continue
            citations = _CITATION.findall(stripped)
            if not citations:
                errors.append(f"{section} 섹션 {lineno}행에 Evidence 인용이 없다")
                continue

            allowed_numbers: list[float] = []
            for evidence_id in citations:
                record = known.get(evidence_id)
                if record is None or evidence_id not in allowed:
                    continue
                allowed_numbers.extend(_evidence_numbers(record))
                if numeric_support is not None:
                    allowed_numbers.extend(numeric_support.get(evidence_id, []))
            for token in _NUMBER.findall(stripped):
                if allowed_numbers and _number_supported(token, allowed_numbers):
                    continue
                message = (
                    f"{lineno}행 수치 {token}를 인용 Evidence의 "
                    "effect/uncertainty/result 값에서 확인하지 못했다"
                )
                if numeric_support is None:
                    warnings.append(message)
                else:
                    errors.append(message)

    if stop_code:
        markers = _STOP_MARKER.findall(markdown)
        if markers != [stop_code]:
            errors.append(
                f"실행 상태 섹션에 정확한 종료 표식 [종료: {stop_code}]가 필요하다"
            )

    limits_text = "\n".join(line for _, line in sections["한계와 미완료 사항"])
    limit_goals = set(_GOAL_MARKER.findall(limits_text))
    limit_tasks = set(_TASK_MARKER.findall(limits_text))
    limit_citations = set(_CITATION.findall(limits_text))

    for goal_id in incomplete_goal_ids or []:
        if goal_id not in limit_goals:
            errors.append(f"미완료/포기 목표가 한계 섹션에 표시되지 않았다: {goal_id}")
    for task_id in pending_task_ids or []:
        if task_id not in limit_tasks:
            errors.append(f"미완료 Agenda 작업이 한계 섹션에 표시되지 않았다: {task_id}")
    for left, right in unresolved_contradictions or []:
        missing = sorted({left, right} - limit_citations)
        if missing:
            errors.append(
                "미해결 contradiction이 한계 섹션에 양쪽 Evidence 인용으로 "
                f"드러나지 않았다: {left} vs {right} (누락 {missing})"
            )

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "citation_count": len(all_citations),
        "cited_evidence_ids": sorted(set(all_citations)),
    }
