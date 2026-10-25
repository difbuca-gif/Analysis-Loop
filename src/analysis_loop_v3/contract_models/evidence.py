"""Evidence, Agenda, Coverage, Critic 계약."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .analysis import AnalysisProtocol, DataLineage
from .base import DataApplicability, ResultPath
from .data import ArtifactRef, DatasetProfile


class EffectEstimate(BaseModel):
    """원본 결과 항목에 연결된 단일 효과 추정치."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    metric: str
    value: float = Field(strict=True, allow_inf_nan=False)
    directional: bool = Field(default=False, strict=True)
    unit: str | None = None
    source_path: ResultPath | None = Field(default=None, min_length=1)


class EvidenceRecord(BaseModel):
    """채택 Evidence의 불변 레코드."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(default_factory=lambda: uuid4().hex)
    intent_id: str
    question: str
    claim: str

    result_ref: ArtifactRef
    program_ref: ArtifactRef | None = None
    figure_refs: list[ArtifactRef] = Field(default_factory=list)
    method: str
    protocol: AnalysisProtocol = Field(default_factory=AnalysisProtocol)
    data_lineage: DataLineage

    columns_used: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    effect_summary: list[EffectEstimate] = Field(default_factory=list)
    uncertainty_summary: dict[str, Any] = Field(default_factory=dict)
    uncertainty_source_paths: dict[str, ResultPath] = Field(default_factory=dict)
    confidence: Literal[
        "high", "medium", "low", "unreviewed"
    ] = "unreviewed"
    open_questions: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    purpose: str = "explore"
    goal_id: str | None = None
    parent_evidence_ids: list[str] = Field(default_factory=list)


class FindingDigest(BaseModel):
    """Planner 입력용 Evidence 요약."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    question: str
    claim: str
    purpose: str = "explore"
    protocol: AnalysisProtocol = Field(default_factory=AnalysisProtocol)
    dataset_sha256: str | None = None
    transform_signature: str | None = None
    data_applicability: DataApplicability = "unknown"
    effect_summary: list[EffectEstimate] = Field(default_factory=list)
    uncertainty_summary: dict[str, Any] = Field(default_factory=dict)
    confidence: str = "unreviewed"
    caveats: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)

    @classmethod
    def from_evidence(
        cls,
        evidence: EvidenceRecord,
        *,
        data_applicability: DataApplicability = "unknown",
    ) -> FindingDigest:
        return cls(
            evidence_id=evidence.evidence_id,
            question=evidence.question,
            claim=evidence.claim,
            purpose=evidence.purpose,
            protocol=evidence.protocol,
            dataset_sha256=evidence.data_lineage.dataset_sha256,
            transform_signature=evidence.data_lineage.transform_signature,
            data_applicability=data_applicability,
            effect_summary=evidence.effect_summary,
            uncertainty_summary=evidence.uncertainty_summary,
            confidence=evidence.confidence,
            caveats=evidence.caveats,
            open_questions=evidence.open_questions,
            contradicts=evidence.contradicts,
        )

    def needs_follow_up(self) -> bool:
        return (
            self.confidence in {"low", "unreviewed"}
            or bool(self.open_questions)
            or bool(self.contradicts)
        )


TaskKind = Literal[
    "follow_up",
    "resolve_contradiction",
    "revalidate",
]
TaskStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "data_limited",
]


class AgendaTask(BaseModel):
    """미해결 분석 작업."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: TaskKind
    goal_id: str | None = None
    question: str = Field(min_length=1)
    rationale: str = ""
    priority: int = Field(default=50, ge=0, le=100)
    source_evidence_ids: list[str] = Field(default_factory=list)
    status: TaskStatus = "pending"
    selected_intent_id: str | None = None
    attempt_count: int = Field(default=0, ge=0)
    completion_evidence_id: str | None = None
    resolution_reason: str = ""

    @model_validator(mode="after")
    def _task_is_consistent(self) -> AgendaTask:
        if (
            self.kind in {"follow_up", "revalidate"}
            and not self.source_evidence_ids
        ):
            raise ValueError(
                f"{self.kind} 작업에는 source_evidence_ids가 필요하다"
            )
        if (
            self.kind == "resolve_contradiction"
            and len(set(self.source_evidence_ids)) < 2
        ):
            raise ValueError(
                "resolve_contradiction에는 서로 다른 Evidence 2개 이상이 필요하다"
            )
        if self.status == "in_progress" and not self.selected_intent_id:
            raise ValueError(
                "in_progress task에는 selected_intent_id가 필요하다"
            )
        if self.status == "completed" and not self.completion_evidence_id:
            raise ValueError(
                "completed task에는 completion_evidence_id가 필요하다"
            )
        if self.status == "data_limited":
            if self.attempt_count < 1:
                raise ValueError(
                    "data_limited task는 최소 1회 실제 시도 이후에만 닫을 수 있다"
                )
            if not self.resolution_reason.strip():
                raise ValueError(
                    "data_limited task에는 resolution_reason이 필요하다"
                )
        return self


RelationKind = Literal[
    "derived_from",
    "contradicts",
    "revalidates",
    "supersedes",
    "reconciles",
]


class EvidenceRelation(BaseModel):
    """불변 Evidence 간 관계 기록."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: str = Field(default_factory=lambda: uuid4().hex)
    source_evidence_id: str = Field(min_length=1)
    target_evidence_id: str = Field(min_length=1)
    kind: RelationKind
    note: str = ""

    @model_validator(mode="after")
    def _not_self_relation(self) -> EvidenceRelation:
        if self.source_evidence_id == self.target_evidence_id:
            raise ValueError(
                "Evidence는 자기 자신과 관계를 만들 수 없다"
            )
        return self


class RejectedAttempt(BaseModel):
    """거부된 시도 기록. 중복 접근 방지용."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_id: str
    dedupe_key: str
    stage: Literal["code", "result", "manifest", "critic"]
    reasons: list[str] = Field(default_factory=list)
    goal_id: str | None = None
    question: str = ""


class AnalysisCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    column_usage: dict[str, int] = Field(default_factory=dict)
    tested_dedupe_keys: list[str] = Field(default_factory=list)

    def with_record(self, columns: list[str]) -> AnalysisCoverage:
        column_usage = dict(self.column_usage)
        for name in columns:
            column_usage[name] = column_usage.get(name, 0) + 1
        return self.model_copy(
            update={"column_usage": column_usage}
        )

    def unexplored(self, profile: DatasetProfile) -> list[str]:
        return [
            name
            for name in profile.analyzable_columns()
            if self.column_usage.get(name, 0) == 0
        ]


class ValidatedProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_key: str
    intent_reuse_signature: str
    code_ref: ArtifactRef
    schema_fingerprint: str
    manifest: dict[str, Any]


class CriticReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    claim: str = ""
    reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    open_questions: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    revalidates: list[str] = Field(default_factory=list)
    revalidation_required: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    reconciles: list[str] = Field(default_factory=list)
    effect_summary: list[EffectEstimate] = Field(default_factory=list)
    uncertainty_summary: dict[str, Any] = Field(default_factory=dict)
    uncertainty_source_paths: dict[str, ResultPath] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _accepted_review_requires_claim(self) -> CriticReview:
        if self.verdict == "accept" and not self.claim.strip():
            raise ValueError(
                "accept 판정에는 실제 결과가 뒷받침하는 claim이 필요하다"
            )
        return self
