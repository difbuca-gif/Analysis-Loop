"""분석 코드를 확보하는 단계 — 캐시 조회, 생성(유료), 정적 검증."""

from __future__ import annotations

import hashlib
from typing import Any

from ...contracts import ArtifactKind, RunStatus, StopCode, ValidationReport
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...runtime import RuntimeDeps
from ...state import AnalysisGraphState, get_intent, get_profile
from ...validation.checks import check_code_matches_manifest, check_manifest
from ...validation.codelint import lint_program
from ._shared import _event, _load_generated, bump_error_streak

async def lookup_program(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """같은 스키마·연구 의도에서 검증된 프로그램을 API 호출 전에 재사용한다."""
    intent = get_intent(state)
    profile = get_profile(state)
    assert intent is not None and profile is not None
    cached = deps.artifacts.load_validated_program(
        schema_fingerprint=profile.schema_fingerprint,
        intent_reuse_signature=intent.reuse_signature(),
    )
    if cached is None:
        return {"program_ref": None, "manifest": None}
    manifest = {**cached.manifest, "intent_id": intent.intent_id}
    _event(state, deps, "PROGRAM_CACHE_HIT", {"cache_key": cached.cache_key})
    return {
        "program_ref": cached.code_ref.model_dump(mode="json"),
        "manifest": manifest,
        "validation_errors": [],
    }

# 코드 생성. 재개 시 체크포인트로 중복 호출을 피한다.

async def generate_code(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    intent = get_intent(state)
    profile = get_profile(state)
    assert intent is not None and profile is not None

    if deps.codegen is None:
        return {
            "status": RunStatus.FAILED.value,
            "stop_code": StopCode.SERVICE_UNAVAILABLE.value,
            "stop_reason": "codegen LLM이 없다",
            "program_ref": None,
            "manifest": None,
            "candidate_ref": None,
        }

    previous = [str(e.get("message", e)) for e in state.get("validation_errors") or []]
    generated = await deps.codegen.generate(
        intent=intent,
        profile=profile,
        previous_errors=previous,
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )
    if generated is None:
        kind = getattr(deps.codegen, "last_failure_kind", None)
        reason = getattr(deps.codegen, "last_error", None) or "코드 생성 응답이 없다"
        _event(state, deps, "CODEGEN_FAILED", {"reason": reason, "kind": kind})
        # 공급자·설정 오류는 재시도로 해결되지 않는다. 단일 호출이므로 역할 서비스가
        # 남긴 오류 종류를 안전하게 읽을 수 있다.
        if kind in {"provider_unavailable", "configuration"}:
            return {
                "status": RunStatus.FAILED.value,
                "stop_code": StopCode.SERVICE_UNAVAILABLE.value,
                "stop_reason": f"codegen 실패: {reason}",
                "program_ref": None,
                "manifest": None,
                "candidate_ref": None,
            }
        return {
            "program_ref": None,
            "manifest": None,
            "candidate_ref": None,
            "validation_errors": [{"code": "codegen_failed", "message": reason}],
            "error_streak": bump_error_streak(
                state, [{"code": "codegen_failed"}]),
            "static_repairs": state.get("static_repairs", 0) + 1,
        }

    # 코드 본문은 artifact로, State에는 참조만. Manifest는 코드를 뺀 형태로 State에 둔다.
    # 코드 내용 기반 ID는 replay가 다른 코드를 만들더라도 과거 artifact를 덮어쓰지 않는다.
    content_key = hashlib.sha256(generated.code.encode("utf-8")).hexdigest()
    code_ref = deps.artifacts.put_text(
        generated.code, kind=ArtifactKind.PROGRAM, suffix=".py",
        summary={"method": generated.method, "chars": len(generated.code)},
        artifact_id=f"program-{content_key}",
    )
    _event(state, deps, "CODE_GENERATED", {
        "method": generated.method, "artifact": code_ref.artifact_id,
    })
    return {
        "program_ref": code_ref.model_dump(mode="json"),
        "manifest": generated.manifest_without_code(),
        "validation_errors": [],
        # 새 프로그램이 생겼으므로 이전 프로그램의 warning은 남기지 않는다.
        "validation_warnings": [],
    }

# 코드 계약 검증.

async def validate_code(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    intent = get_intent(state)
    profile = get_profile(state)
    assert intent is not None and profile is not None
    generated = _load_generated(state, deps)

    issues = []
    for report in (
        lint_program(generated.code),
        check_manifest(generated, intent, profile),
        # 선언한 변수와 실제 접근 변수가 같은가 — Manifest의 존재 이유다.
        check_code_matches_manifest(generated.code, generated, profile),
    ):
        issues.extend(report.issues)

    merged = ValidationReport(stage="code", issues=issues)
    blocking = [i for i in merged.issues if i.severity.value != "warning"]
    warnings = [i.model_dump(mode="json") for i in merged.issues if i.severity.value == "warning"]
    _event(state, deps, "CODE_VALIDATED", {
        "issues": len(blocking),
        "warnings": len(warnings),
        "warning_codes": [item["code"] for item in warnings],
    })

    if not blocking:
        return {"validation_errors": [], "validation_warnings": warnings}

    errors = [i.model_dump(mode="json") for i in blocking]
    return {
        "validation_errors": errors,
        "error_streak": bump_error_streak(state, errors),
        "validation_warnings": warnings,
        "static_repairs": state.get("static_repairs", 0) + 1,
    }
