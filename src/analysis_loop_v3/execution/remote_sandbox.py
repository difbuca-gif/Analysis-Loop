"""원격 Compute Worker에게 실행을 위임하는 Sandbox 구현.

Worker가 돌려준 바이트를 로컬 workdir에 다시 써서, `execute()`가 원격이라는 사실을
모른 채 `workdir.glob()`/`read_text()`로 결과를 읽게 한다(노드 수정 0줄).

응답 유실 != 실행 실패. 못 받으면 곧바로 접지 않고 execution_id로 되묻는다.
Worker 응답도 신뢰 입력으로 다루지 않는다. 파일명·필드를 그대로 쓰지 않는다.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib.util
import os
from pathlib import Path
from typing import Any

import httpx

from ..contracts import ExecutionStatus
from ..runtime import SandboxResult
from .sandbox import Sandbox, SubprocessSandbox

# 짧게 여러 번보다 Worker가 끝날 시간을 주는 게 목적이다.
_POLL_INTERVAL_SECONDS = 3.0
_CONNECT_RETRIES = 3  # Worker 재시작·순간 끊김을 넘기기 위한 것


class _WorkerUnreachable(Exception):
    """통신 실패. 404(기록 없음)와 반드시 구분한다 — 전자는 더 기다려야 하고
    후자만 "다시 실행해야 함"이다. 둘을 섞으면 멀쩡히 도는 실행을 실패로 접는다."""


class RemoteSandbox(Sandbox):
    def __init__(self, *, worker_url: str, token: str | None = None) -> None:
        self.worker_url = worker_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("WORKER_TOKEN")

    def _headers(self) -> dict[str, str]:
        return {"X-Worker-Token": self.token} if self.token else {}

    async def run(
        self, *, code: str, dataset_path: str, workdir: Path, timeout_seconds: float,
        execution_id: str | None = None,
    ) -> SandboxResult:
        workdir.mkdir(parents=True, exist_ok=True)
        dataset = Path(dataset_path)

        if not execution_id:
            # 이름이 없으면 회수도 idempotency도 불가능하다. 장애 시 중복 실행이 된다.
            return _refused("execution_id 없이 원격 실행을 요청할 수 없다")

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds + 30) as client:
                payload = await self._start(client, execution_id, code, timeout_seconds, dataset)
                if payload is None:
                    return _refused(f"worker에 닿을 수 없다: {self.worker_url}")
                # 실행이 실제로 시작된 뒤에 예산을 센다. 시작 전에 세면 연결 재시도가
                # 예산을 다 먹고, 멀쩡히 도는 worker를 두고 첫 폴링에서 포기한다.
                deadline = asyncio.get_running_loop().time() + timeout_seconds + 60
                payload = await self._await_finish(client, execution_id, payload, deadline)
        except (httpx.HTTPError, _WorkerUnreachable) as exc:
            return _refused(f"worker 호출 실패: {exc}")

        return self._materialize(payload, workdir, execution_id)

    def _materialize(
        self, payload: dict[str, Any], workdir: Path, execution_id: str,
    ) -> SandboxResult:
        status = payload.get("status")
        if status == ExecutionStatus.RUNNING.value:
            return _refused(
                f"worker 실행이 제한 시간 안에 끝나지 않았다(execution_id={execution_id})"
            )
        if status == ExecutionStatus.FAILED.value:
            return SandboxResult(
                ok=False, result_path=None,
                stdout=str(payload.get("stdout") or ""),
                stderr=str(payload.get("stderr") or ""),
                duration_seconds=_as_float(payload.get("duration_seconds")),
                refused_reason=payload.get("refused_reason"),
            )
        if status != ExecutionStatus.SUCCEEDED.value:
            return _refused(f"worker가 알 수 없는 실행 상태를 반환했다: {status!r}")

        result_json = payload.get("result_json")
        if not isinstance(result_json, str):
            return _refused("worker가 SUCCEEDED인데 result_json이 없다")

        result_path = workdir / "result.json"
        result_path.write_text(result_json, encoding="utf-8")
        try:
            self._write_figures(payload.get("figures"), workdir)
        except (binascii.Error, TypeError, ValueError, OSError) as exc:
            return _refused(f"worker figure를 해석할 수 없다: {exc}")

        return SandboxResult(
            ok=True, result_path=result_path,
            stdout=str(payload.get("stdout") or ""),
            stderr=str(payload.get("stderr") or ""),
            duration_seconds=_as_float(payload.get("duration_seconds")),
        )

    @staticmethod
    def _write_figures(figures: Any, workdir: Path) -> None:
        """worker가 준 파일명을 그대로 쓰지 않는다.

        Control은 LAN에서 평문 HTTP로 worker와 통신한다. 파일명을 믿으면
        '../../.ssh/authorized_keys' 같은 값으로 workdir 밖에 쓸 수 있다.
        """
        for figure in figures or []:
            if not isinstance(figure, dict):
                raise TypeError(f"figure가 객체가 아니다: {type(figure).__name__}")
            name = Path(str(figure.get("name") or "")).name
            if not name or not name.endswith(".png"):
                raise ValueError(f"허용되지 않는 figure 이름: {figure.get('name')!r}")
            (workdir / name).write_bytes(base64.b64decode(str(figure.get("data_b64") or "")))

    async def _start(
        self, client: httpx.AsyncClient, execution_id: str, code: str,
        timeout_seconds: float, dataset: Path,
    ) -> dict[str, Any] | None:
        """실행을 요청한다. 연결이 실패하면 이미 시작됐는지 확인하고 재시도한다."""
        for attempt in range(_CONNECT_RETRIES):
            try:
                with dataset.open("rb") as f:
                    response = await client.post(
                        f"{self.worker_url}/executions",
                        headers=self._headers(),
                        data={
                            "execution_id": execution_id,
                            "code": code,
                            "timeout_seconds": timeout_seconds,
                        },
                        files={"dataset": (dataset.name, f)},
                    )
                response.raise_for_status()
                return _decode(response)
            except (httpx.HTTPError, _WorkerUnreachable):
                # POST가 실패해도 Worker가 이미 받아서 돌고 있을 수 있다.
                try:
                    existing = await self._fetch(client, execution_id)
                except _WorkerUnreachable:
                    existing = None
                if existing is not None:
                    return existing
                if attempt == _CONNECT_RETRIES - 1:
                    return None
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        return None

    async def _await_finish(
        self, client: httpx.AsyncClient, execution_id: str,
        payload: dict[str, Any], deadline: float,
    ) -> dict[str, Any]:
        """RUNNING이면 끝날 때까지 되묻는다(Worker는 진행 중이면 즉시 RUNNING을 준다)."""
        loop = asyncio.get_running_loop()
        while payload.get("status") == ExecutionStatus.RUNNING.value:
            if loop.time() >= deadline:
                return payload
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            try:
                fetched = await self._fetch(client, execution_id)
            except _WorkerUnreachable:
                continue  # 일시적 통신 실패. 실행은 계속되고 있을 수 있다.
            if fetched is None:
                # 404 — 기록이 사라졌다. 더 기다릴 근거가 없다.
                return {
                    "status": ExecutionStatus.FAILED.value,
                    "refused_reason": f"worker가 execution {execution_id}의 기록을 잃었다",
                }
            payload = fetched
        return payload

    async def _fetch(
        self, client: httpx.AsyncClient, execution_id: str,
    ) -> dict[str, Any] | None:
        """404면 None. 통신 실패는 _WorkerUnreachable로 구분해 올린다."""
        try:
            response = await client.get(
                f"{self.worker_url}/executions/{execution_id}", headers=self._headers(),
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return _decode(response)
        except httpx.HTTPError as exc:
            raise _WorkerUnreachable(str(exc)) from exc


def _decode(response: httpx.Response) -> dict[str, Any]:
    """JSON이 아닌 본문(게이트웨이 HTML 오류 등)을 통신 실패로 취급한다.

    JSONDecodeError는 ValueError지 httpx.HTTPError가 아니라서, 그냥 두면 호출부의
    except를 통과해 execute() 노드 밖으로 나가고 run이 RUNNING인 채 굳는다.
    """
    try:
        body = response.json()
    except ValueError as exc:
        raise _WorkerUnreachable(f"JSON이 아닌 worker 응답: {response.text[:200]}") from exc
    if not isinstance(body, dict):
        raise _WorkerUnreachable(f"worker 응답이 객체가 아니다: {type(body).__name__}")
    return body


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _refused(reason: str) -> SandboxResult:
    """fail-closed. 애매하면 결과를 만들지 않는다."""
    return SandboxResult(
        ok=False, result_path=None, stdout="", stderr="",
        duration_seconds=0.0, refused_reason=reason,
    )


def build_sandbox() -> Sandbox:
    """`WORKER_URL`이 있으면 원격 Worker에 위임하고, 없으면 로컬 서브프로세스를 쓴다.

    Control 배포 트리에는 sandbox_worker.py가 없다(항상 원격 위임이므로 일부러
    뺐다). 그 상태에서 WORKER_URL이 비면 매 회차 ModuleNotFoundError가 나고,
    그게 설정 오류가 아니라 일반 runtime_error로 codegen에 되먹여진다.
    설정 문제를 코드 문제로 오인하게 만든다. 그래서 여기서 먼저 끊는다.
    """
    if url := os.environ.get("WORKER_URL"):
        return RemoteSandbox(worker_url=url)
    if importlib.util.find_spec("analysis_loop_v3.execution.sandbox_worker") is None:
        raise RuntimeError(
            "WORKER_URL이 비어 있는데 로컬 sandbox_worker도 없다 — "
            "이 배포는 원격 Worker 전용이다. .env의 WORKER_URL을 설정하라."
        )
    return SubprocessSandbox()
