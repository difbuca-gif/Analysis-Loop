"""원격 Compute Worker 실행 위임.

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

# polling timeout은 Worker 완료 대기 기준
_POLL_INTERVAL_SECONDS = 3.0
_CONNECT_RETRIES = 3  # Worker 재시작·순간 끊김을 넘기기 위한 것


class _WorkerUnreachable(Exception):
    """Worker 통신 실패. 404와 구분."""


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
            # execution_id 필수: 결과 회수 및 idempotency 보장
            return _refused("execution_id 없이 원격 실행을 요청할 수 없다")

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds + 30) as client:
                payload = await self._start(client, execution_id, code, timeout_seconds, dataset)
                if payload is None:
                    return _refused(f"worker에 닿을 수 없다: {self.worker_url}")
                # 실행 시작 이후부터 예산 측정
                # 연결 재시도 시간은 실행 예산에서 제외
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
        """Worker 응답 파일명 정규화 및 경로 검증.

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
        """Worker 실행 요청. POST 실패 시 execution_id 조회 후 재시도."""
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
                # POST 실패 시에도 Worker 접수 가능성 확인
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
        """RUNNING 상태를 완료까지 polling."""
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
                # 404: execution record 없음
                return {
                    "status": ExecutionStatus.FAILED.value,
                    "refused_reason": f"worker가 execution {execution_id}의 기록을 잃었다",
                }
            payload = fetched
        return payload

    async def _fetch(
        self, client: httpx.AsyncClient, execution_id: str,
    ) -> dict[str, Any] | None:
        """404는 None, 통신 실패는 _WorkerUnreachable."""
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
    """비JSON 응답을 Worker 통신 실패로 처리.

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
    """불확실한 Worker 결과는 채택하지 않음(fail-closed)."""
    return SandboxResult(
        ok=False, result_path=None, stdout="", stderr="",
        duration_seconds=0.0, refused_reason=reason,
    )


def build_sandbox() -> Sandbox:
    """WORKER_URL 설정 시 RemoteSandbox, 미설정 시 LocalSubprocessSandbox.

    Control-only 배포에서 sandbox_worker 부재 시 configuration error로 즉시 종료.
    """
    if url := os.environ.get("WORKER_URL"):
        return RemoteSandbox(worker_url=url)
    if importlib.util.find_spec("analysis_loop_v3.execution.sandbox_worker") is None:
        raise RuntimeError(
            "WORKER_URL이 비어 있는데 로컬 sandbox_worker도 없다 — "
            "이 배포는 원격 Worker 전용이다. .env의 WORKER_URL을 설정하라."
        )
    return SubprocessSandbox()
