"""실행 Result Artifact의 결정론적 계약 검증."""

from __future__ import annotations

import json
import math
from typing import Any

from ..contracts import (
    MAX_RESULT_BYTES,
    DatasetProfile,
    GeneratedAnalysis,
    ResearchIntent,
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from .issues import issue


def schema_type_of(value: Any) -> str:
    if isinstance(value, dict):
        return "mapping"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def non_finite_numbers(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            found.extend(
                non_finite_numbers(
                    sub,
                    f"{path}.{key}" if path else str(key),
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, sub in enumerate(value):
            found.extend(
                non_finite_numbers(sub, f"{path}[{index}]")
            )
    elif isinstance(value, float) and not math.isfinite(value):
        found.append(path or "<root>")
    return found


def check_result(
    result: Any,
    manifest: GeneratedAnalysis,
    intent: ResearchIntent,
    profile: DatasetProfile | None = None,
) -> ValidationReport:
    """Result의 shape, 크기, finite number, 선언 output schema를 검사한다."""
    del intent, profile  # 향후 계약 확장을 위한 시그니처 호환
    issues: list[ValidationIssue] = []

    if not isinstance(result, dict):
        return ValidationReport(
            stage="result",
            issues=[
                issue(
                    "result_not_mapping",
                    (
                        "analyze(df)는 dict를 반환해야 한다"
                        f"(실제 {type(result).__name__})."
                    ),
                    ValidationSeverity.REPAIRABLE,
                )
            ],
        )

    if not result:
        return ValidationReport(
            stage="result",
            issues=[
                issue(
                    "empty_result",
                    "결과가 비어 있다.",
                    ValidationSeverity.REPAIRABLE,
                )
            ],
        )

    encoded = json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        issues.append(
            issue(
                "result_too_large",
                (
                    f"결과가 {len(encoded):,}바이트다 — 상한 "
                    f"{MAX_RESULT_BYTES:,}. 고카디널리티 컬럼을 raw로 "
                    "그룹화했는지 확인하고 집계 수준을 올려라."
                ),
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.RESOURCE,
                size_bytes=len(encoded),
            )
        )

    non_finite = non_finite_numbers(result)
    if non_finite:
        issues.append(
            issue(
                "non_finite_number",
                f"NaN 또는 Infinity가 결과에 있다(예: {non_finite[:3]}).",
                ValidationSeverity.REPAIRABLE,
                paths=non_finite[:10],
            )
        )

    for key, declared_type in manifest.expected_output_schema.items():
        if key not in result:
            issues.append(
                issue(
                    "missing_declared_output",
                    f"Manifest가 선언한 출력 키가 결과에 없다: {key}",
                    ValidationSeverity.REPAIRABLE,
                    key=key,
                )
            )
            continue

        actual_type = schema_type_of(result[key])
        if actual_type != declared_type:
            issues.append(
                issue(
                    "output_type_mismatch",
                    (
                        f"{key}의 타입이 선언({declared_type})과 다르다"
                        f"(실제 {actual_type})."
                    ),
                    ValidationSeverity.REPAIRABLE,
                    key=key,
                )
            )

    return ValidationReport(stage="result", issues=issues)


_schema_type_of = schema_type_of
_non_finite_numbers = non_finite_numbers
