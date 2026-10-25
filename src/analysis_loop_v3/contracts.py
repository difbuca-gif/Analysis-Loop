"""분석 루프의 영속 계약과 경계 모델 compatibility facade.

구현은 `contract_models/` 아래 도메인별 모듈로 분리한다. 기존 코드의
`from analysis_loop_v3.contracts import ...` 경로는 그대로 유지한다.
"""

from .contract_models.analysis import (
    AnalysisProtocol,
    DataLineage,
    GeneratedAnalysis,
    SchemaType,
)
from .contract_models.base import (
    DataApplicability,
    DescriptionItem,
    InvalidRunId,
    MAX_RESULT_BYTES,
    ResultPath,
    RUN_ID_PATTERN,
    SCHEMA_VERSION,
    validate_run_id,
)
from .contract_models.data import (
    ArtifactKind,
    ArtifactRef,
    ColumnProfile,
    DatasetProfile,
)
from .contract_models.evidence import (
    AgendaTask,
    AnalysisCoverage,
    CriticReview,
    EffectEstimate,
    EvidenceRecord,
    EvidenceRelation,
    FindingDigest,
    RejectedAttempt,
    RelationKind,
    TaskKind,
    TaskStatus,
    ValidatedProgram,
)
from .contract_models.planning import (
    GoalStatus,
    IntentPurpose,
    ResearchIntent,
    SubGoal,
)
from .contract_models.reporting import (
    REPORT_REVIEW_CATEGORIES,
    ReportReviewCategory,
    ReportReviewFinding,
    ReportSemanticReview,
)
from .contract_models.runtime import (
    ExecutionStatus,
    LoopIssue,
    RunStatus,
    StopCode,
)
from .contract_models.validation import (
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

__all__ = [
    "SCHEMA_VERSION",
    "ResultPath",
    "MAX_RESULT_BYTES",
    "RUN_ID_PATTERN",
    "InvalidRunId",
    "validate_run_id",
    "DescriptionItem",
    "DataApplicability",
    "ArtifactKind",
    "ArtifactRef",
    "ColumnProfile",
    "DatasetProfile",
    "GoalStatus",
    "SubGoal",
    "IntentPurpose",
    "ResearchIntent",
    "SchemaType",
    "AnalysisProtocol",
    "GeneratedAnalysis",
    "DataLineage",
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationReport",
    "EffectEstimate",
    "EvidenceRecord",
    "FindingDigest",
    "TaskKind",
    "TaskStatus",
    "AgendaTask",
    "RelationKind",
    "EvidenceRelation",
    "RejectedAttempt",
    "AnalysisCoverage",
    "ValidatedProgram",
    "CriticReview",
    "ReportReviewCategory",
    "REPORT_REVIEW_CATEGORIES",
    "ReportReviewFinding",
    "ReportSemanticReview",
    "RunStatus",
    "LoopIssue",
    "ExecutionStatus",
    "StopCode",
]
