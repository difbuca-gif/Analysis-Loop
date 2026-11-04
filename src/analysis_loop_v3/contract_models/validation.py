"""결정론적 validation 결과 계약."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ValidationSeverity(str, Enum):
    FATAL = "fatal"
    REPAIRABLE = "repairable"
    WARNING = "warning"


class ValidationCategory(str, Enum):
    """severity는 차단 여부, category는 검증 범위를 나타낸다."""

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
        return any(
            issue.severity is ValidationSeverity.FATAL
            for issue in self.issues
        )

    @property
    def repairable(self) -> bool:
        return not self.fatal and any(
            issue.repairable for issue in self.issues
        )

    def messages(self) -> list[str]:
        return [
            f"[{issue.code}] {issue.message}"
            for issue in self.issues
        ]
