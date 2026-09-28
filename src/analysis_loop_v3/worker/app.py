"""Compute Worker FastAPI: upload, local execution, result retrieval.

    POST /executions        같은 execution_id면 두 번 실행하지 않는다
    GET  /executions/{id}   응답을 놓쳤을 때 결과를 회수한다

이 서비스는 신뢰할 수 없는 코드를 실행한다. 그래서 경계가 두 겹이다.
`require_token`이 호출자를, `validate_execution_id`가 경로를 막는다.
자원·크기 상한은 SubprocessSandbox/MAX_RESULT_BYTES가 이미 처리한다.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)

from ..contracts import ExecutionStatus
from ..execution.sandbox import SubprocessSandbox
from ..net import is_loopback
from .store import ExecutionStore, InvalidExecutionId, validate_execution_id

app = FastAPI(title="Analysis Loop v3 Worker", version="2.0.0")

_sandbox = SubprocessSandbox()
_store = ExecutionStore(os.environ.get("WORKER_STATE_DIR", "worker-state/executions"))


def require_token(
    request: Request, x_worker_token: str | None = Header(default=None),
) -> None:
    """token 설정 시 Bearer 인증, 미설정 시 loopback만 허용.

    전에는 `__main__`이 기동 시점에만 강제했다. 그러면
    `uvicorn analysis_loop_v3.worker.app:app --host 0.0.0.0`으로 띄웠을 때
    인증 없이 임의 코드 실행이 열린다. 판정을 요청마다 해야 우회되지 않는다.
    """
    expected = os.environ.get("WORKER_TOKEN")
    if expected:
        if not x_worker_token or not hmac.compare_digest(x_worker_token, expected):
            raise HTTPException(status_code=401, detail="worker 토큰이 유효하지 않다")
        return

    client = request.client.host if request.client else None
    if not is_loopback(client):
        raise HTTPException(
            status_code=401,
            detail="WORKER_TOKEN이 설정되지 않아 원격 요청을 받지 않는다",
        )


def _checked_id(execution_id: str) -> str:
    try:
        return validate_execution_id(execution_id)
    except InvalidExecutionId as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"worker": "ok"}


async def _execute_now(
    execution_id: str, code: str, timeout_seconds: float, dataset: UploadFile,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="analysis-worker-") as tmp:
        tmp_dir = Path(tmp)
        # 업로드 파일명 정규화
        dataset_path = tmp_dir / (Path(dataset.filename or "dataset").name or "dataset")
        workdir = tmp_dir / "workdir"
        workdir.mkdir(parents=True, exist_ok=True)

        dataset_path.write_bytes(await dataset.read())

        result = await _sandbox.run(
            code=code,
            dataset_path=str(dataset_path),
            workdir=workdir,
            timeout_seconds=timeout_seconds,
            execution_id=execution_id,
        )

        if not result.ok or result.result_path is None:
            return {
                "status": ExecutionStatus.FAILED.value,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_seconds": result.duration_seconds,
                "refused_reason": result.refused_reason,
            }

        figures = [
            {
                "name": path.name,
                "data_b64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
            for path in workdir.glob("*.png")
        ]

        return {
            "status": ExecutionStatus.SUCCEEDED.value,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": result.duration_seconds,
            "result_json": result.result_path.read_text(encoding="utf-8"),
            "figures": figures,
        }


@app.post("/executions", dependencies=[Depends(require_token)])
async def create_execution(
    execution_id: str = Form(...),
    code: str = Form(...),
    timeout_seconds: float = Form(...),
    dataset: UploadFile = File(...),
) -> dict[str, Any]:
    """execution_id 기준 idempotent 실행.

    끝났으면 보관된 결과를, 도는 중이면 RUNNING을, 처음이면 지금 실행한다.
    """
    execution_id = _checked_id(execution_id)

    if not _store.claim(execution_id):
        existing = _store.read(execution_id)
        if existing is not None:
            return existing
        # claim 실패 + 기록 없음이면 재실행

    try:
        payload = await _execute_now(execution_id, code, timeout_seconds, dataset)
    except BaseException as exc:
        # CancelledError 포함 실패 상태 기록
        _store.finish(execution_id, {
            "status": ExecutionStatus.FAILED.value,
            "stdout": "", "stderr": "",
            "duration_seconds": 0.0,
            "refused_reason": f"worker 내부 오류: {type(exc).__name__}: {exc}",
        })
        raise

    _store.finish(execution_id, payload)
    # 파일 삭제는 event loop 외부에서 비동기 처리
    asyncio.create_task(asyncio.to_thread(_store.sweep))
    return {"execution_id": execution_id, **payload}


@app.get("/executions/{execution_id}", dependencies=[Depends(require_token)])
async def get_execution(execution_id: str) -> dict[str, Any]:
    """execution_id 결과 조회. 404는 재실행 필요 상태."""
    record = _store.read(_checked_id(execution_id))
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"execution {execution_id}의 기록이 없다",
        )
    return record
