"""실행 결과의 결정론적 계약 검증."""

from __future__ import annotations

from ...runtime import RuntimeDeps
from ...state import AnalysisGraphState, get_intent, get_profile, get_ref
from ...validation.checks import check_result
from ..node_updates import ValidationUpdate
from ._shared import _event, _load_generated, bump_error_streak


async def validate_result(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> ValidationUpdate:
    """Result Artifact가 생성 분석 계약을 지키는지 검사."""
    intent = get_intent(state)
    ref = get_ref(state, "candidate_ref")
    assert intent is not None and ref is not None
    generated = _load_generated(state, deps)

    payload = deps.artifacts.read_json(ref)
    report = check_result(
        payload,
        generated,
        intent,
        profile=get_profile(state),
    )
    blocking = [
        issue
        for issue in report.issues
        if issue.severity.value != "warning"
    ]
    warnings = [
        *state.get("validation_warnings", []),
        *(
            issue.model_dump(mode="json")
            for issue in report.issues
            if issue.severity.value == "warning"
        ),
    ]
    _event(
        state,
        deps,
        "RESULT_VALIDATED",
        {
            "issues": len(blocking),
            "warnings": len(warnings),
            "warning_codes": [
                str(item.get("code", ""))
                for item in warnings
            ],
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
        "runtime_repairs": state.get("runtime_repairs", 0) + 1,
    }
