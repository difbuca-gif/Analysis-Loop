"""LangGraph Node가 반환하는 State partial update 타입.

AnalysisGraphState 전체를 각 Node의 반환 타입으로 쓰지 않는다.
함수 시그니처만 보고도 어떤 필드를 바꿀 수 있는지 알 수 있게 한다.
런타임 동작에는 영향을 주지 않는 타입 계약이다.
"""

from __future__ import annotations

from typing import Any, TypedDict


class TerminalUpdate(TypedDict, total=False):
    status: str
    stop_code: str | None
    stop_reason: str | None


class CycleResetUpdate(TypedDict, total=False):
    last_planner_error: str | None
    program_ref: dict[str, Any] | None
    manifest: dict[str, Any] | None
    candidate_ref: dict[str, Any] | None
    figure_refs: list[dict[str, Any]] | None
    review: dict[str, Any] | None
    validation_errors: list[dict[str, Any]]
    error_streak: dict[str, Any]
    validation_warnings: list[dict[str, Any]]
    static_repairs: int
    runtime_repairs: int


class PlanningUpdate(CycleResetUpdate, TerminalUpdate, total=False):
    profile_ref: dict[str, Any]
    profile: dict[str, Any]
    sub_goals: list[dict[str, Any]]
    intent: dict[str, Any] | None
    agenda: list[dict[str, Any]]
    evidence_lookup_summary: list[dict[str, Any]]
    iteration: int
    issues: list[dict[str, Any]]


class ProgramLookupUpdate(TypedDict, total=False):
    program_ref: dict[str, Any] | None
    manifest: dict[str, Any] | None
    validation_errors: list[dict[str, Any]]


class CodegenUpdate(ProgramLookupUpdate, total=False):
    candidate_ref: dict[str, Any] | None
    validation_warnings: list[dict[str, Any]]
    error_streak: dict[str, Any]
    static_repairs: int
    status: str
    stop_code: str | None
    stop_reason: str | None


class ValidationUpdate(TypedDict, total=False):
    validation_errors: list[dict[str, Any]]
    validation_warnings: list[dict[str, Any]]
    error_streak: dict[str, Any]
    static_repairs: int
    runtime_repairs: int


class ExecutionUpdate(ValidationUpdate, total=False):
    candidate_ref: dict[str, Any] | None
    figure_refs: list[dict[str, Any]]


class CritiqueUpdate(TerminalUpdate, total=False):
    review: dict[str, Any]


class EvidenceCommitUpdate(TypedDict, total=False):
    evidence: list[dict[str, Any]]
    evidence_relations: list[dict[str, Any]]
    coverage: dict[str, Any]
    agenda: list[dict[str, Any]]
    sub_goals: list[dict[str, Any]]
    intent: None


class RejectionUpdate(TypedDict, total=False):
    intent: None
    agenda: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    error_streak: dict[str, Any]


class ProgressUpdate(TypedDict, total=False):
    agenda: list[dict[str, Any]]
    sub_goals: list[dict[str, Any]]
    goal_expansions: int
    status: str
    stop_code: str | None
    stop_reason: str | None


class FinalizeUpdate(TypedDict, total=False):
    report_ref: dict[str, Any]
    status: str
    stop_code: str
    stop_reason: str
