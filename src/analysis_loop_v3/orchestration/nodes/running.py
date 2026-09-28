"""생성 코드 실행 및 결정론적 결과 검증.

실행 구현은 `deps.sandbox`에 위임하며 로컬/원격 Worker를 동일 인터페이스로 처리한다.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from ...contracts import (
    ArtifactKind,
    ArtifactRef,
    InvalidRunId,
    validate_run_id,
)
from ...runtime import RuntimeDeps
from ...state import AnalysisGraphState, get_intent, get_profile, get_ref, remaining_seconds
from ...validation.checks import check_result
from ._shared import _event, _load_generated, bump_error_streak

SANDBOX_TIMEOUT_SECONDS = 300.0

def _execution_id(state: AnalysisGraphState, program: ArtifactRef, dataset: ArtifactRef) -> str:
    """State 기반 결정론적 execution_id 생성.

    재개 시 동일 ID를 유지하고 repair 시도는 재생성 카운터로 구분한다.
    """
    seed = "|".join([
        str(state.get("run_id")),
        str(state.get("iteration", 0)),
        str(state.get("static_repairs", 0)),
        str(state.get("runtime_repairs", 0)),
        program.sha256,
        dataset.sha256,
    ])
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]

def _run_workdir(root: Path, run_id: str, iteration: int) -> Path:
    """회차별 작업 디렉터리 생성 및 workspace 경계 검증."""
    validate_run_id(run_id)
    resolved_root = root.resolve()
    workdir = (resolved_root / run_id / f"iter{iteration}").resolve()
    if not workdir.is_relative_to(resolved_root):
        raise InvalidRunId(f"작업 디렉터리가 workspace를 벗어난다: {workdir}")
    return workdir


async def execute(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    dataset = get_ref(state, "dataset_ref")
    program = get_ref(state, "program_ref")
    assert dataset is not None and program is not None

    code = deps.artifacts.read_text(program)
    workdir = _run_workdir(deps.workdir_root, state["run_id"], state.get("iteration", 0))
    execution_id = _execution_id(state, program, dataset)

    # 동일 회차의 repair 시도 간 작업 디렉터리 공유
    # 이전 산출물 혼입 방지를 위해 실행 전 초기화
    shutil.rmtree(workdir, ignore_errors=True)

    result = await deps.sandbox.run(
        code=code,
        dataset_path=dataset.path,
        workdir=workdir,
        timeout_seconds=min(SANDBOX_TIMEOUT_SECONDS, max(0.0, remaining_seconds(state))),
        execution_id=execution_id,
    )

    if result.refused_reason:
        # 실행 불가 시 결과 미생성(fail-closed)
        _event(state, deps, "EXECUTION_REFUSED", {
            "execution_id": execution_id,
            "reason": result.refused_reason,
        })
        return {
            "candidate_ref": None,
            "validation_errors": [
                {"code": "execution_refused", "message": result.refused_reason}
            ],
            "error_streak": bump_error_streak(
                state, [{"code": "execution_refused"}]),
        }

    if not result.ok:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        _event(state, deps, "EXECUTION_FAILED", {
            "execution_id": execution_id, "stderr_tail": tail[:300],
        })
        return {
            "candidate_ref": None,
            "validation_errors": [{"code": "runtime_error", "message": tail}],
            "error_streak": bump_error_streak(state, [{"code": "runtime_error"}]),
            "runtime_repairs": state.get("runtime_repairs", 0) + 1,
        }

    assert result.result_path is not None

    figure_refs = []
    if workdir.exists():
        for path in workdir.glob("*.png"):
            fig_ref = deps.artifacts.put_bytes(
                path.read_bytes(),
                kind=ArtifactKind.FIGURE,
                summary={"name": path.name}
            )
            figure_refs.append(fig_ref.model_dump(mode="json"))

    ref = deps.artifacts.put_json_text(
        result.result_path.read_text(encoding="utf-8"),
        kind=ArtifactKind.RESULT,
        summary={"duration_seconds": round(result.duration_seconds, 3)},
    )
    _event(state, deps, "EXECUTED", {
        "execution_id": execution_id,
        "seconds": round(result.duration_seconds, 2), "figures": len(figure_refs),
    })
    return {
        "candidate_ref": ref.model_dump(mode="json"),
        "figure_refs": figure_refs,
        "validation_errors": [],
    }

# 실행 결과 계약 검증

async def validate_result(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    intent = get_intent(state)
    ref = get_ref(state, "candidate_ref")
    assert intent is not None and ref is not None
    generated = _load_generated(state, deps)

    payload = deps.artifacts.read_json(ref)
    report = check_result(payload, generated, intent, profile=get_profile(state))
    blocking = [i for i in report.issues if i.severity.value != "warning"]
    warnings = [
        *state.get("validation_warnings", []),
        *(i.model_dump(mode="json") for i in report.issues if i.severity.value == "warning"),
    ]
    _event(state, deps, "RESULT_VALIDATED", {
        "issues": len(blocking),
        "warnings": len(warnings),
        "warning_codes": [str(item.get("code", "")) for item in warnings],
    })

    if not blocking:
        return {"validation_errors": [], "validation_warnings": warnings}
    errors = [i.model_dump(mode="json") for i in blocking]
    return {
        "validation_errors": errors,
        "error_streak": bump_error_streak(state, errors),
        "validation_warnings": warnings,
        "runtime_repairs": state.get("runtime_repairs", 0) + 1,
    }
