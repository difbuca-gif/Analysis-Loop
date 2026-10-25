"""분석 코드 확보: 캐시 조회와 코드 생성.

정적 코드 계약 검증은 code_validation.py가 담당한다.
"""

from __future__ import annotations

import hashlib

from ...contracts import ArtifactKind, RunStatus, StopCode
from ...llm.client import DEFAULT_READ_TIMEOUT
from ...runtime import RuntimeDeps
from ...state import AnalysisGraphState, get_intent, get_profile
from ..failure_policy import is_terminal_service_failure
from ..node_updates import CodegenUpdate, ProgramLookupUpdate
from ._shared import _event, bump_error_streak
from .code_validation import validate_code


async def lookup_program(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> ProgramLookupUpdate:
    """동일 schema/Intent의 검증 프로그램을 생성 호출 전에 재사용."""
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
    _event(
        state,
        deps,
        "PROGRAM_CACHE_HIT",
        {"cache_key": cached.cache_key},
    )
    return {
        "program_ref": cached.code_ref.model_dump(mode="json"),
        "manifest": manifest,
        "validation_errors": [],
    }


async def generate_code(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> CodegenUpdate:
    """Codegen 결과를 Artifact로 저장하고 Manifest만 State에 남긴다."""
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

    previous = [
        str(error.get("message", error))
        for error in state.get("validation_errors") or []
    ]
    generated = await deps.codegen.generate(
        intent=intent,
        profile=profile,
        previous_errors=previous,
        timeout_seconds=DEFAULT_READ_TIMEOUT,
    )

    if generated is None:
        kind = getattr(deps.codegen, "last_failure_kind", None)
        reason = (
            getattr(deps.codegen, "last_error", None)
            or "코드 생성 응답이 없다"
        )
        _event(
            state,
            deps,
            "CODEGEN_FAILED",
            {"reason": reason, "kind": kind},
        )

        if is_terminal_service_failure(kind):
            return {
                "status": RunStatus.FAILED.value,
                "stop_code": StopCode.SERVICE_UNAVAILABLE.value,
                "stop_reason": f"codegen 실패: {reason}",
                "program_ref": None,
                "manifest": None,
                "candidate_ref": None,
            }

        errors = [{"code": "codegen_failed", "message": reason}]
        return {
            "program_ref": None,
            "manifest": None,
            "candidate_ref": None,
            "validation_errors": errors,
            "error_streak": bump_error_streak(state, errors),
            "static_repairs": state.get("static_repairs", 0) + 1,
        }

    content_key = hashlib.sha256(
        generated.code.encode("utf-8")
    ).hexdigest()
    code_ref = deps.artifacts.put_text(
        generated.code,
        kind=ArtifactKind.PROGRAM,
        suffix=".py",
        summary={
            "method": generated.method,
            "chars": len(generated.code),
        },
        artifact_id=f"program-{content_key}",
    )
    _event(
        state,
        deps,
        "CODE_GENERATED",
        {
            "method": generated.method,
            "artifact": code_ref.artifact_id,
        },
    )
    return {
        "program_ref": code_ref.model_dump(mode="json"),
        "manifest": generated.manifest_without_code(),
        "validation_errors": [],
        "validation_warnings": [],
    }


__all__ = [
    "lookup_program",
    "generate_code",
    "validate_code",
]
