"""생성 코드의 정적 계약 검증."""

from __future__ import annotations

from ...contracts import ValidationReport
from ...runtime import RuntimeDeps
from ...state import AnalysisGraphState, get_intent, get_profile
from ...validation.checks import check_code_matches_manifest, check_manifest
from ...validation.codelint import lint_program
from ..node_updates import ValidationUpdate
from ._shared import _event, _load_generated, bump_error_streak


async def validate_code(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> ValidationUpdate:
    """생성 코드와 Manifest가 Intent/Profile 계약을 지키는지 검사."""
    intent = get_intent(state)
    profile = get_profile(state)
    assert intent is not None and profile is not None
    generated = _load_generated(state, deps)

    issues = []
    for report in (
        lint_program(generated.code),
        check_manifest(generated, intent, profile),
        check_code_matches_manifest(generated.code, generated, profile),
    ):
        issues.extend(report.issues)

    merged = ValidationReport(stage="code", issues=issues)
    blocking = [
        issue for issue in merged.issues
        if issue.severity.value != "warning"
    ]
    warnings = [
        issue.model_dump(mode="json")
        for issue in merged.issues
        if issue.severity.value == "warning"
    ]
    _event(
        state,
        deps,
        "CODE_VALIDATED",
        {
            "issues": len(blocking),
            "warnings": len(warnings),
            "warning_codes": [item["code"] for item in warnings],
        },
    )

    if not blocking:
        return {
            "validation_errors": [],
            "validation_warnings": warnings,
        }

    errors = [issue.model_dump(mode="json") for issue in blocking]
    return {
        "validation_errors": errors,
        "error_streak": bump_error_streak(state, errors),
        "validation_warnings": warnings,
        "static_repairs": state.get("static_repairs", 0) + 1,
    }
