"""분석 루프의 영속 계약과 경계 모델."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

# 수치 출처를 검사하지 않은 이전 체크포인트는 새 실행에서 재사용하지 않는다.
SCHEMA_VERSION = 9

ResultPath = list[StrictStr | StrictInt]

# 결과 크기 상한

MAX_RESULT_BYTES = 64 * 1024

# run_id 경로 구분자 금지
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

class InvalidRunId(ValueError):
    """디렉터리 이름으로 쓸 수 없는 run_id."""

def validate_run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value or ""):
        raise InvalidRunId(
            f"run_id는 영문·숫자로 시작하고 영문·숫자·밑줄·하이픈만 쓸 수 있다"
            f"(최대 64자): {value!r}"
        )
    return value

# Artifact 참조 타입

class ArtifactKind(str, Enum):
    DATASET = "dataset"
    PROFILE = "profile"
    PROGRAM = "program"
    RESULT = "result"
    REPORT = "report"
    FIGURE = "figure"
    INSPECTION = "inspection"

class ArtifactRef(BaseModel):
    """실제 바이트는 ArtifactStore에, State에는 이 참조만.

    sha256은 재개 시 "그 파일이 맞는지" 확인용 — 경로만으로는 알 수 없다.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    kind: ArtifactKind
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    # summary에는 소형 메타데이터만 저장
    summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def _summary_must_stay_small(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 20:
            raise ValueError(
                f"ArtifactRef.summary는 최대 20개 키다(실제 {len(value)}개). "
                "요약이 아니라 데이터를 넣고 있는지 확인하라."
            )
        return value

# 데이터 프로파일

class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    non_null: int = Field(ge=0)
    null_count: int = Field(ge=0)
    unique_count: int = Field(ge=0)
    sample_values: list[str] = Field(default_factory=list, max_length=5)

class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    row_count: int = Field(ge=0)
    columns: list[ColumnProfile]
    # 프로그램 재사용용 schema fingerprint
    schema_fingerprint: str

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def analyzable_columns(self) -> list[str]:
        """실제 존재하는 모든 컬럼. 의미상 제외 여부는 Planner가 판단한다."""
        return self.column_names()

# ResearchIntent: Planner가 정의하는 연구 의도

# SubGoal: 목표 진행 상태 단위

GoalStatus = Literal[
    "open",       # 아직 답하지 못했다
    "converged",  # 충분한 근거가 쌓여 닫혔다
    "abandoned",  # 데이터로는 답할 수 없다고 판단했다
]

class SubGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 짧고 안정적인 목표 ID.
    goal_id: str = Field(pattern=r"^g[0-9]+$")
    question: str = Field(min_length=1)
    rationale: str = ""
    # 목표 종료 기준.
    success_criterion: str = ""
    status: GoalStatus = "open"
    # 목표에 연결된 근거.
    evidence_ids: list[str] = Field(default_factory=list)
    # 종료 사유.
    closing_note: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == "open"

# 분석 회차 목적.
IntentPurpose = Literal[
    "explore",                # 아직 안 본 영역을 넓힌다
    "deepen",                 # 기존 근거를 더 파고든다 (세분화·상호작용·조건부)
    "challenge",              # 기존 근거를 반증하거나 강건성을 검사한다
    "resolve_contradiction",  # 서로 어긋나는 근거를 조정한다
]
DescriptionItem = str | dict[str, Any]
DataApplicability = Literal[
    "exact_snapshot",
    "revalidation_required",
    "incompatible_schema",
    "unknown",
]

class ResearchIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str = Field(default_factory=lambda: uuid4().hex)
    question: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    # 현재 질문을 선택한 이유.
    rationale: str = ""

    # 비어 있으면 전체 목표를 대상으로 본다.
    goal_id: str | None = None
    purpose: IntentPurpose = "explore"
    # 기존 근거를 다루는 회차의 부모 근거.
    parent_evidence_ids: list[str] = Field(default_factory=list)
    # 연결된 Agenda 작업.
    task_id: str | None = None
    # 회차 실행 방식.
    execution_mode: Literal[
        "compute", "reuse_evidence", "inspect_data", "lookup_evidence",
        "mark_data_limited"
    ] = "compute"
    reuse_evidence_ids: list[str] = Field(default_factory=list)
    # 회차 성공 기준.
    success_criterion: str = ""
    candidate_columns: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    excluded_columns: list[str] = Field(default_factory=list)

    # 기대하는 근거 유형.
    desired_evidence: list[DescriptionItem] = Field(default_factory=list)
    # 분석 제약.
    constraints: list[DescriptionItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _purpose_needs_parent(self) -> ResearchIntent:
        """explore가 아니면 상대할 근거를 지목해야 한다.

        안 막으면 purpose만 'deepen'이고 내용은 무관한 새 질문이 된다(라벨만 고도화).
        """
        if self.purpose != "explore" and not self.parent_evidence_ids:
            raise ValueError(
                f"purpose={self.purpose}이면 parent_evidence_ids가 있어야 한다"
            )
        return self

    @model_validator(mode="after")
    def _execution_mode_is_consistent(self) -> ResearchIntent:
        if self.execution_mode == "reuse_evidence":
            if not self.reuse_evidence_ids:
                raise ValueError("reuse_evidence에는 reuse_evidence_ids가 필요하다")
            if self.task_id:
                raise ValueError("Agenda 작업은 새 Evidence로 닫아야 하므로 재사용할 수 없다")
            missing = set(self.reuse_evidence_ids) - set(self.parent_evidence_ids)
            if missing:
                raise ValueError(
                    "reuse_evidence_ids는 parent_evidence_ids에도 있어야 한다: "
                    f"{sorted(missing)}"
                )
        elif self.reuse_evidence_ids:
            raise ValueError("reuse_evidence가 아닌 행동에는 reuse_evidence_ids를 넣지 않는다")

        if self.execution_mode == "inspect_data":
            if self.task_id:
                raise ValueError("Agenda 작업 수행 중 inspect_data로 우회할 수 없다")
            if not self.candidate_columns:
                raise ValueError("inspect_data에는 확인할 candidate_columns가 필요하다")

        if self.execution_mode == "lookup_evidence":
            if self.task_id:
                raise ValueError("Agenda 작업은 source Evidence가 이미 있어 lookup_evidence가 필요 없다")
            if not (self.goal_id or self.candidate_columns or self.parent_evidence_ids):
                raise ValueError(
                    "lookup_evidence에는 goal_id, candidate_columns, "
                    "parent_evidence_ids 중 하나가 필요하다"
                )

        if self.execution_mode == "mark_data_limited" and not self.task_id:
            raise ValueError("mark_data_limited에는 task_id가 필요하다")
        return self

    @model_validator(mode="after")
    def _required_within_candidates(self) -> ResearchIntent:
        # required 컬럼은 candidate에 자동 포함
        missing = set(self.required_columns) - set(self.candidate_columns)
        if missing:
            self.candidate_columns = [
                *self.candidate_columns,
                *sorted(missing),
            ]
        # required/excluded 충돌 거부
        overlap = set(self.required_columns) & set(self.excluded_columns)
        if overlap:
            raise ValueError(f"required와 excluded가 겹친다: {sorted(overlap)}")
        return self

    def dedupe_key(self) -> str:
        """동일 목표 내 분석 요청 중복 판정."""
        return self._analysis_signature(include_goal=True)

    def reuse_signature(self) -> str:
        """goal_id와 무관한 코드 재사용용 분석 요청 서명."""
        return self._analysis_signature(include_goal=False)

    def _analysis_signature(self, *, include_goal: bool) -> str:
        # 질문과 제약까지 포함해 분석 요청을 식별한다.
        excluded = {"intent_id"} if include_goal else {"intent_id", "goal_id"}
        payload = self.model_dump(mode="json", exclude=excluded)
        for field in (
            "candidate_columns", "required_columns", "excluded_columns",
            "parent_evidence_ids", "reuse_evidence_ids",
        ):
            payload[field] = sorted(set(payload[field]))
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        # 구버전 캐시 재사용 금지
        return "intent-numeric-v1:" + hashlib.sha256(encoded).hexdigest()

# GeneratedAnalysis: Codegen 실행 Manifest

SchemaType = Literal["mapping", "list", "number", "string", "boolean"]

class AnalysisProtocol(BaseModel):
    """계산 조건 재현성 계약.

    특정 통계기법을 강제하지 않고, Codegen이 실제로 선택한 표본·평가·반복 조건을
    Evidence와 함께 보존한다. 이후 revalidation이 "무엇을 바꿨는가"를 비교하는 기준이다.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sampling_strategy: str | None = None
    exclusion_rules: list[DescriptionItem] = Field(default_factory=list)
    comparison_groups: list[DescriptionItem] = Field(default_factory=list)
    random_seed: int | None = None
    repetitions: int | None = Field(default=None, ge=1)
    evaluation_strategy: str | None = None
    # 일반화 성능 평가 구조.
    evaluation_design: Literal[
        "holdout", "cross_validation", "out_of_fold", "time_split",
        "group_split", "nested_cv", "external_validation", "custom"
    ] | None = None
    # 전처리 학습 범위.
    preprocessing_fit_scope: Literal[
        "train_only", "full_data_safe", "not_applicable", "custom"
    ] | None = None
    temporal_order_preserved: bool | None = None
    group_isolation_preserved: bool | None = None
    uncertainty_method: str | None = None
    hypothesis_testing: bool = False
    planned_comparisons: int | None = Field(default=None, ge=1)
    multiple_testing: str | None = None
    class_balance_strategy: str | None = None
    interpretation_scope: Literal[
        "descriptive", "associational", "predictive", "causal"
    ] = "descriptive"

class GeneratedAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str

    selected_columns: list[str] = Field(min_length=1)
    method: str = Field(min_length=1)
    method_reason: str = ""

    assumptions: list[str] = Field(default_factory=list)

    # 누수 검사는 target/predictor 역할 기준
    target: str | None = None
    # 관심 예측 변수.
    feature_columns: list[str] = Field(default_factory=list)
    # 통제·보정 변수.
    adjustment_columns: list[str] = Field(default_factory=list)
    # 시간축 컬럼.
    time_column: str | None = None
    # 분할 간 격리가 필요한 단위.
    split_unit_columns: list[str] = Field(default_factory=list)
    prediction_task: Literal[
        "classification", "regression", "forecasting", "ranking", "other"
    ] | None = None
    group_by: list[str] = Field(default_factory=list)
    # 코드 생성 파생 컬럼
    derived_columns: list[str] = Field(default_factory=list)
    transformations: list[DescriptionItem] = Field(default_factory=list)
    split_strategy: str | None = None
    protocol: AnalysisProtocol = Field(default_factory=AnalysisProtocol)
    result_granularity: Literal["aggregate", "model_metric"] = "aggregate"
    expected_outputs: list[str] = Field(default_factory=list)
    expected_figures: list[str] = Field(default_factory=list)
    # 결과 최상위 key/type
    expected_output_schema: dict[str, SchemaType] = Field(default_factory=dict)

    code: str = Field(min_length=1)

    @model_validator(mode="after")
    def _target_is_not_a_feature(self) -> GeneratedAnalysis:
        """target의 predictor 사용은 누수로 판정.

        selected_columns에 target이 있는 건 정상이다(접근해야 학습된다). 막을 것은 역할.
        """
        if not self.target:
            return self
        leaked = [
            role for role, columns in (
                ("feature_columns", self.feature_columns),
                ("adjustment_columns", self.adjustment_columns),
            ) if self.target in columns
        ]
        if leaked:
            raise ValueError(
                f"target '{self.target}'이 {', '.join(leaked)}에 들어 있다 — 데이터 누수다"
            )
        return self

    def available_columns(self, source_columns: set[str]) -> set[str]:
        """사용 가능 컬럼: 원본 + 파생."""
        return source_columns | set(self.derived_columns)

    def predictor_columns(self) -> list[str]:
        """모형 입력 변수: 관심 + 통제."""
        seen: dict[str, None] = {}
        for name in [*self.feature_columns, *self.adjustment_columns]:
            seen.setdefault(name, None)
        return list(seen)

    def is_predictive(self) -> bool:
        """모형 적합 여부. 분할·누수 검사 적용 조건.

        target 유무로 판단하면 안 된다(카이제곱·그룹 비교도 target을 선언한다).
        기준은 result_granularity — model_metric일 때만 분할과 누수를 따진다.
        """
        return self.result_granularity == "model_metric" and bool(self.target)

    def manifest_without_code(self) -> dict[str, Any]:
        """State 저장용 Manifest. 코드 본문 제외."""
        return self.model_dump(exclude={"code"})

class DataLineage(BaseModel):
    """Evidence의 데이터 스냅샷·변환 조건 식별.

    전체 전처리 설명을 State에 복사하지 않고 canonical hash만 남긴다. 실제 계산은
    program_ref/code와 protocol이 보존하므로 이 값은 버전 비교와 재검증 판단용이다.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_artifact_id: str = Field(min_length=1)
    dataset_sha256: str = Field(min_length=64, max_length=64)
    schema_fingerprint: str = Field(min_length=1)
    source_rows: int = Field(ge=0)
    transform_signature: str = Field(min_length=64, max_length=64)

    @classmethod
    def from_analysis(
        cls,
        *,
        dataset_ref: ArtifactRef,
        profile: DatasetProfile,
        manifest: GeneratedAnalysis,
    ) -> DataLineage:
        payload = {
            "selected_columns": sorted(set(manifest.selected_columns)),
            "derived_columns": sorted(set(manifest.derived_columns)),
            "transformations": manifest.transformations,
            "sampling_strategy": manifest.protocol.sampling_strategy,
            "exclusion_rules": manifest.protocol.exclusion_rules,
        }
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            dataset_artifact_id=dataset_ref.artifact_id,
            dataset_sha256=dataset_ref.sha256,
            schema_fingerprint=profile.schema_fingerprint,
            source_rows=profile.row_count,
            transform_signature=hashlib.sha256(encoded).hexdigest(),
        )

# Validation

class ValidationSeverity(str, Enum):
    # 재시도 없이 폐기
    FATAL = "fatal"
    # 재생성 가능한 오류
    REPAIRABLE = "repairable"
    # warning 기록 후 진행
    WARNING = "warning"

class ValidationCategory(str, Enum):
    """검증 이슈 분류. severity는 차단 여부, category는 검증 범위.

    분리해야 품질 휴리스틱이 안전 규칙처럼 과잉 집행되지 않는다.
    """

    SAFETY = "safety"
    INTEGRITY = "integrity"
    QUALITY = "quality"
    RESOURCE = "resource"

class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.REPAIRABLE
    category: ValidationCategory = ValidationCategory.INTEGRITY
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def repairable(self) -> bool:
        return self.severity is ValidationSeverity.REPAIRABLE

class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["code", "result", "manifest"]
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def fatal(self) -> bool:
        return any(i.severity is ValidationSeverity.FATAL for i in self.issues)

    @property
    def repairable(self) -> bool:
        """blocking issue 없이 repairable issue가 있을 때 재생성."""
        return not self.fatal and any(i.repairable for i in self.issues)

    def messages(self) -> list[str]:
        return [f"[{i.code}] {i.message}" for i in self.issues]

# Evidence: 채택 후 불변

class EffectEstimate(BaseModel):
    """원본 결과 항목에 연결된 단일 효과 추정치."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    metric: str
    value: float = Field(strict=True, allow_inf_nan=False)
    directional: bool = Field(default=False, strict=True)
    unit: str | None = None
    # 과거 결과 열람과 single-model baseline은 허용하되 새 Critic 채택 시 필수 검사한다.
    source_path: ResultPath | None = Field(default=None, min_length=1)

class EvidenceRecord(BaseModel):
    """채택 Evidence의 불변 레코드.

    실제 수치는 result_ref에 있고, 요약은 원본 경로·메타데이터·값을 대조한 뒤 저장한다.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(default_factory=lambda: uuid4().hex)
    intent_id: str
    question: str
    # Evidence가 뒷받침하는 claim
    claim: str

    result_ref: ArtifactRef
    program_ref: ArtifactRef | None = None
    figure_refs: list[ArtifactRef] = Field(default_factory=list)
    method: str
    protocol: AnalysisProtocol = Field(default_factory=AnalysisProtocol)
    # Evidence data lineage
    data_lineage: DataLineage

    columns_used: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    # 다음 회차용 요약
    effect_summary: list[EffectEstimate] = Field(default_factory=list)
    uncertainty_summary: dict[str, Any] = Field(default_factory=dict)
    uncertainty_source_paths: dict[str, ResultPath] = Field(default_factory=dict)
    confidence: Literal["high", "medium", "low", "unreviewed"] = "unreviewed"
    open_questions: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    # Evidence 생성 회차 목적
    purpose: str = "explore"
    # 연결 SubGoal
    goal_id: str | None = None
    parent_evidence_ids: list[str] = Field(default_factory=list)

class FindingDigest(BaseModel):
    """Planner 입력용 Evidence 요약.

    EvidenceRecord를 그대로 안 넘긴다. artifact 참조·해시는 계획에 불필요하고,
    보여줄 것을 명시하지 않으면 필드가 늘 때마다 프롬프트가 조용히 커진다.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    question: str
    claim: str
    purpose: str = "explore"
    protocol: AnalysisProtocol = Field(default_factory=AnalysisProtocol)
    dataset_sha256: str | None = None
    transform_signature: str | None = None
    # 현재 데이터에 대한 applicability
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
        """추가 검토 필요 여부 판정."""
        return (
            self.confidence in {"low", "unreviewed"}
            or bool(self.open_questions)
            or bool(self.contradicts)
        )

TaskKind = Literal["follow_up", "resolve_contradiction", "revalidate"]
TaskStatus = Literal["pending", "in_progress", "completed", "data_limited"]

class AgendaTask(BaseModel):
    """미해결 분석 작업.

    Critic follow-up과 Evidence contradiction의 공통 lifecycle 모델.
    Planner selection 기준은 priority와 current goal.
    """

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
        if self.kind in {"follow_up", "revalidate"} and not self.source_evidence_ids:
            raise ValueError(f"{self.kind} 작업에는 source_evidence_ids가 필요하다")
        if self.kind == "resolve_contradiction" and len(set(self.source_evidence_ids)) < 2:
            raise ValueError("resolve_contradiction에는 서로 다른 Evidence 2개 이상이 필요하다")
        if self.status == "in_progress" and not self.selected_intent_id:
            raise ValueError("in_progress task에는 selected_intent_id가 필요하다")
        if self.status == "completed" and not self.completion_evidence_id:
            raise ValueError("completed task에는 completion_evidence_id가 필요하다")
        if self.status == "data_limited":
            if self.attempt_count < 1:
                raise ValueError("data_limited task는 최소 1회 실제 시도 이후에만 닫을 수 있다")
            if not self.resolution_reason.strip():
                raise ValueError("data_limited task에는 resolution_reason이 필요하다")
        return self

RelationKind = Literal[
    "derived_from", "contradicts", "revalidates", "supersedes", "reconciles"
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
            raise ValueError("Evidence는 자기 자신과 관계를 만들 수 없다")
        return self

class RejectedAttempt(BaseModel):
    """거부된 시도 기록. 중복 접근 방지용."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_id: str
    dedupe_key: str
    stage: Literal["code", "result", "manifest", "critic"]
    reasons: list[str] = Field(default_factory=list)
    # 실패 SubGoal
    # 실제 시도 횟수
    goal_id: str | None = None
    question: str = ""

# Coverage: 컬럼 사용 이력

class AnalysisCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    column_usage: dict[str, int] = Field(default_factory=dict)
    tested_dedupe_keys: list[str] = Field(default_factory=list)

    def with_record(self, columns: list[str]) -> AnalysisCoverage:
        """Coverage 불변 갱신. 새 인스턴스 반환."""
        col = dict(self.column_usage)
        for name in columns:
            col[name] = col.get(name, 0) + 1
        return self.model_copy(update={"column_usage": col})

    def unexplored(self, profile: DatasetProfile) -> list[str]:
        return [
            name for name in profile.analyzable_columns()
            if self.column_usage.get(name, 0) == 0
        ]

# ValidatedProgram: 검증 통과 코드 캐시

class ValidatedProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_key: str
    intent_reuse_signature: str
    code_ref: ArtifactRef
    schema_fingerprint: str
    manifest: dict[str, Any]

# Critic 의미 검토 응답 계약

class CriticReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    claim: str = ""
    reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    # 다음 계획용 검토 요약
    confidence: Literal["high", "medium", "low"] = "medium"
    open_questions: list[str] = Field(default_factory=list)
    # 기존 Evidence와의 관계
    contradicts: list[str] = Field(default_factory=list)
    revalidates: list[str] = Field(default_factory=list)
    # 독립 재검증 질문
    revalidation_required: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    # contradiction 해결 대상
    reconciles: list[str] = Field(default_factory=list)
    # 의미 판정은 Critic 결과 사용
    effect_summary: list[EffectEstimate] = Field(default_factory=list)
    uncertainty_summary: dict[str, Any] = Field(default_factory=dict)
    uncertainty_source_paths: dict[str, ResultPath] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _accepted_review_requires_claim(self) -> CriticReview:
        if self.verdict == "accept" and not self.claim.strip():
            raise ValueError("accept 판정에는 실제 결과가 뒷받침하는 claim이 필요하다")
        return self

# HITL 승인

class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class LoopIssue(BaseModel):
    """동일 오류 반복으로 중단된 회차의 운영 이슈.

    Critic reject와 구분되는 실행/생성 정체 신호.
    대시보드 및 prompt/contract 점검에 사용.
    """

    model_config = ConfigDict(extra="forbid")

    label: str          # 반복 오류 code
    count: int          # 연속 반복 횟수
    iteration: int
    intent_id: str | None = None
    question: str = ""
    messages: list[str] = Field(default_factory=list)

class ExecutionStatus(str, Enum):
    """Worker의 한 실행 상태(Control↔Compute 공유 계약).

    응답 유실 != 실행 실패. 나중에 되물으려면 실행에 이름과 상태가 있어야 한다.
    """

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

class StopCode(str, Enum):
    """런 종료 상태 계약.

    없으면 프런트가 rejected[].stage 같은 그래프 내부를 읽게 되어 노드를 바꿀 때마다
    UI가 깨진다. 같은 COMPLETED라도 GOAL_SATISFIED와 BUDGET_EXHAUSTED는 다른 의미다.
    """

    # 정상 종료(status=COMPLETED)
    GOAL_SATISFIED = "GOAL_SATISFIED"
    """모든 SubGoal 및 확장 분석 완료."""

    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    """시간 예산 소진에 따른 부분 완료."""

    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"
    """회차 상한 도달에 따른 부분 완료."""

    DATA_LIMITED = "DATA_LIMITED"
    """데이터 한계로 미해결 SubGoal이 남은 종료."""

    EXPANSION_LIMIT_REACHED = "EXPANSION_LIMIT_REACHED"
    """추가 분석 확장 상한 도달."""

    # 실패 종료(status=FAILED)
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    """필수 LLM 역할 또는 provider/configuration 오류."""

    PLANNER_STUCK = "PLANNER_STUCK"
    """Planner가 연속으로 Intent 생성에 실패한 상태.

    iteration은 의도가 나와야 오르므로 이 경우 회차 상한이 걸리지 않는다 — 막지
    않으면 재귀 한도까지 유료 호출만 반복한다. 한 회차가 막힌 것(issues 라벨)과
    달리 물어볼 질문 자체를 못 만드는 상태라 런 전체의 실패다."""

    RECURSION_LIMIT = "RECURSION_LIMIT"
    """그래프 recursion limit 도달."""

    UNKNOWN = "UNKNOWN"
    """분류되지 않은 종료 상태."""
