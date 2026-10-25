"""결정론적 validation issue 생성 helper."""

from __future__ import annotations

from typing import Any

from ..contracts import (
    ValidationCategory,
    ValidationIssue,
    ValidationSeverity,
)


def issue(
    code: str,
    message: str,
    severity: ValidationSeverity,
    *,
    category: ValidationCategory = ValidationCategory.INTEGRITY,
    **detail: Any,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=severity,
        category=category,
        detail=detail,
    )


_issue = issue
