"""최종 보고서 의미 검토 계약."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReportReviewCategory = Literal[
    "evidence_support",
    "causal_scope",
    "metric_context",
    "uncertainty",
    "recommendations",
    "completion_status",
]
REPORT_REVIEW_CATEGORIES: tuple[ReportReviewCategory, ...] = (
    "evidence_support",
    "causal_scope",
    "metric_context",
    "uncertainty",
    "recommendations",
    "completion_status",
)


class ReportReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: ReportReviewCategory
    quote: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    suggested_revision: str = Field(min_length=1)
    evidence_ids: list[str]


class ReportSemanticReview(BaseModel):
    """보고서 의미 검토 응답 계약."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    checked_categories: list[ReportReviewCategory]
    reviewed_evidence_ids: list[str]
    findings: list[ReportReviewFinding]

    @model_validator(mode="after")
    def _complete_review(self) -> ReportSemanticReview:
        if (
            len(self.checked_categories) != len(REPORT_REVIEW_CATEGORIES)
            or set(self.checked_categories) != set(REPORT_REVIEW_CATEGORIES)
        ):
            raise ValueError(
                "보고서 의미 검토의 여섯 항목을 각각 확인해야 한다"
            )
        if (self.verdict == "accept") != (not self.findings):
            raise ValueError(
                "accept에는 지적이 없어야 하고 reject에는 지적이 필요하다"
            )
        return self
