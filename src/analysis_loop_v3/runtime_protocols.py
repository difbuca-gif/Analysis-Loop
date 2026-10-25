"""Planner / Codegen / Critic / Sandbox 런타임 인터페이스."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypedDict, TypeVar

from .contracts import (
    AgendaTask,
    DatasetProfile,
    FindingDigest,
    GeneratedAnalysis,
    ResearchIntent,
    SubGoal,
)

_ValueT = TypeVar("_ValueT")
FailureKind = Literal["provider_unavailable", "configuration", "contract", "response_invalid"]


@dataclass(frozen=True)
class ServiceResult(Generic[_ValueT]):
    """한 호출의 값과 실패 정보를 함께 반환한다. 서비스 인스턴스에 실패 상태를 남기지 않는다."""

    value: _ValueT | None = None
    error: str | None = None
    failure_kind: FailureKind | None = None
    reason: str | None = None  # 정상적인 확장 중단/확장 선택 이유


@dataclass(frozen=True)
class PlannerProposal:
    intent: ResearchIntent | None = None
    verdict: dict[str, Any] = field(default_factory=dict)


class ServiceFailure(TypedDict):
    role: Literal["planner", "codegen", "critic"]
    kind: FailureKind | None
    message: str


class PlannerService(Protocol):
    """연구 질문과 다음 행동을 결정하는 Planner 인터페이스."""

    async def propose_intent(
        self,
        *,
        objective: str,
        profile: DatasetProfile,
        unexplored: list[str],
        column_usage: dict[str, int],
        prior_findings: list[FindingDigest],
        unresolved_findings: list[FindingDigest],
        sub_goals: list[SubGoal],
        timeout_seconds: float,
        required_followups: list[AgendaTask] | None = None,
        recent_inspection: dict[str, Any] | None = None,
        recent_evidence_lookup: list[dict[str, Any]] | None = None,
        evidence_relations: list[dict[str, Any]] | None = None,
        remaining_seconds: float | None = None,
        remaining_iterations: int | None = None,
        rejected_attempts: list[dict[str, Any]] | None = None,
        previous_generation_error: str | None = None,
    ) -> ServiceResult[PlannerProposal]: ...

    async def decompose_objective(
        self,
        *,
        objective: str,
        profile: DatasetProfile,
        timeout_seconds: float,
    ) -> ServiceResult[list[SubGoal]]: ...

    async def expand_goals(
        self,
        *,
        objective: str,
        profile: DatasetProfile,
        existing_goals: list[SubGoal],
        findings: list[FindingDigest],
        timeout_seconds: float,
        remaining_seconds: float,
        remaining_iterations: int,
        expansion_round: int,
        max_expansions: int,
    ) -> ServiceResult[list[SubGoal]]: ...


class CodegenService(Protocol):
    """분석 코드와 실행 Manifest를 생성하는 인터페이스."""

    async def generate(
        self,
        *,
        intent: ResearchIntent,
        profile: DatasetProfile,
        previous_errors: list[str],
        timeout_seconds: float,
    ) -> ServiceResult[GeneratedAnalysis]: ...


class CriticService(Protocol):
    """결과 의미 검토와 보고서 생성/검토 인터페이스."""

    async def review(
        self,
        *,
        intent: ResearchIntent,
        manifest: GeneratedAnalysis,
        result_summary: dict[str, Any],
        timeout_seconds: float,
        prior_findings: list[FindingDigest] | None = None,
        validation_warnings: list[dict[str, Any]] | None = None,
    ) -> ServiceResult[dict[str, Any]]: ...

    async def generate_report(
        self,
        *,
        run_summary: dict[str, Any],
        evidence: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        timeout_seconds: float,
    ) -> ServiceResult[str]: ...

    async def review_report(
        self,
        *,
        markdown: str,
        run_summary: dict[str, Any],
        evidence: list[dict[str, Any]],
        results: dict[str, Any],
        dataset_profile: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> ServiceResult[dict[str, Any]]: ...


@dataclass(slots=True)
class SandboxResult:
    ok: bool
    result_path: Path | None
    stdout: str
    stderr: str
    duration_seconds: float
    refused_reason: str | None = None


class SandboxService(Protocol):
    """생성 코드 실행기 인터페이스."""

    async def run(
        self,
        *,
        code: str,
        dataset_path: str,
        workdir: Path,
        timeout_seconds: float,
        execution_id: str | None = None,
    ) -> SandboxResult: ...
