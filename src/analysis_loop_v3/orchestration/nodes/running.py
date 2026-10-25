"""생성 코드를 Sandbox에서 실행하고 Result Artifact를 만든다.

결과 계약 검증은 result_validation.py가 담당한다.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from ...contracts import ArtifactKind, ArtifactRef, InvalidRunId, validate_run_id
from ...runtime import RuntimeDeps
from ...state import AnalysisGraphState, get_ref, remaining_seconds
from ..node_updates import ExecutionUpdate
from ._shared import _event, bump_error_streak
from .result_validation import validate_result

SANDBOX_TIMEOUT_SECONDS = 300.0


def _execution_id(
    state: AnalysisGraphState,
    program: ArtifactRef,
    dataset: ArtifactRef,
) -> str:
    """State 기반 결정론적 execution_id 생성."""
    seed = "|".join(
        [
            str(state.get("run_id")),
            str(state.get("iteration", 0)),
            str(state.get("static_repairs", 0)),
            str(state.get("runtime_repairs", 0)),
            program.sha256,
            dataset.sha256,
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _run_workdir(root: Path, run_id: str, iteration: int) -> Path:
    """회차별 작업 디렉터리 생성 및 workspace 경계 검증."""
    validate_run_id(run_id)
    resolved_root = root.resolve()
    workdir = (resolved_root / run_id / f"iter{iteration}").resolve()
    if not workdir.is_relative_to(resolved_root):
        raise InvalidRunId(
            f"작업 디렉터리가 workspace를 벗어난다: {workdir}"
        )
    return workdir


async def execute(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
) -> ExecutionUpdate:
    """생성 코드를 Sandbox에서 실행하고 결과 참조를 State에 남긴다."""
    dataset = get_ref(state, "dataset_ref")
    program = get_ref(state, "program_ref")
    assert dataset is not None and program is not None

    code = deps.artifacts.read_text(program)
    workdir = _run_workdir(
        deps.workdir_root,
        state["run_id"],
        state.get("iteration", 0),
    )
    execution_id = _execution_id(state, program, dataset)

    shutil.rmtree(workdir, ignore_errors=True)

    result = await deps.sandbox.run(
        code=code,
        dataset_path=dataset.path,
        workdir=workdir,
        timeout_seconds=min(
            SANDBOX_TIMEOUT_SECONDS,
            max(0.0, remaining_seconds(state)),
        ),
        execution_id=execution_id,
    )

    if result.refused_reason:
        errors = [
            {
                "code": "execution_refused",
                "message": result.refused_reason,
            }
        ]
        _event(
            state,
            deps,
            "EXECUTION_REFUSED",
            {
                "execution_id": execution_id,
                "reason": result.refused_reason,
            },
        )
        return {
            "candidate_ref": None,
            "validation_errors": errors,
            "error_streak": bump_error_streak(state, errors),
        }

    if not result.ok:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        errors = [{"code": "runtime_error", "message": tail}]
        _event(
            state,
            deps,
            "EXECUTION_FAILED",
            {
                "execution_id": execution_id,
                "stderr_tail": tail[:300],
            },
        )
        return {
            "candidate_ref": None,
            "validation_errors": errors,
            "error_streak": bump_error_streak(state, errors),
            "runtime_repairs": state.get("runtime_repairs", 0) + 1,
        }

    assert result.result_path is not None

    figure_refs: list[dict[str, object]] = []
    if workdir.exists():
        for path in workdir.glob("*.png"):
            fig_ref = deps.artifacts.put_bytes(
                path.read_bytes(),
                kind=ArtifactKind.FIGURE,
                summary={"name": path.name},
            )
            figure_refs.append(fig_ref.model_dump(mode="json"))

    ref = deps.artifacts.put_json_text(
        result.result_path.read_text(encoding="utf-8"),
        kind=ArtifactKind.RESULT,
        summary={
            "duration_seconds": round(
                result.duration_seconds,
                3,
            )
        },
    )
    _event(
        state,
        deps,
        "EXECUTED",
        {
            "execution_id": execution_id,
            "seconds": round(result.duration_seconds, 2),
            "figures": len(figure_refs),
        },
    )
    return {
        "candidate_ref": ref.model_dump(mode="json"),
        "figure_refs": figure_refs,
        "validation_errors": [],
    }


__all__ = [
    "SANDBOX_TIMEOUT_SECONDS",
    "execute",
    "validate_result",
]
