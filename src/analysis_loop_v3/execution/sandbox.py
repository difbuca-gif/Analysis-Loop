"""자원 제한을 설정할 수 없으면 실행을 거부하는 로컬 코드 실행기."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ..contracts import MAX_RESULT_BYTES
from ..runtime import SandboxResult

# 자식 프로세스에는 허용된 환경변수만 전달한다.
ENV_ALLOWLIST = (
    "PATH", "LANG", "LC_ALL", "TZ", "TMPDIR",
    "SystemRoot", "SystemDrive", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HOMEDRIVE", "HOMEPATH"
)

DEFAULT_MEMORY_MB: int | None = None
# CPU 제한은 wall timeout보다 약간 길게 둔다.
CPU_LIMIT_MARGIN_SECONDS = 30
MAX_LOG_BYTES = 64 * 1024

async def _read_limited(stream: asyncio.StreamReader | None) -> bytes:
    if stream is None:
        return b""
    kept = bytearray()
    truncated = False
    while chunk := await stream.read(8192):
        remaining = MAX_LOG_BYTES - len(kept)
        if remaining > 0:
            kept.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    if truncated:
        kept.extend(b"\n<log truncated>")
    return bytes(kept)

def _refused(reason: str, *, duration: float = 0.0) -> SandboxResult:
    """fail-closed. 실행할 수 없거나 끝내지 못했으면 결과를 만들지 않는다."""
    return SandboxResult(
        ok=False, result_path=None, stdout="", stderr="",
        duration_seconds=duration, refused_reason=reason,
    )

def _signal_reason(signum: int) -> str:
    known = {
        24: "CPU 시간 상한(SIGXCPU)을 넘겨 강제 종료됐다. 계산량을 줄이거나 표본을 줄여라",
        9: "SIGKILL로 강제 종료됐다. 메모리 부족(OOM)일 가능성이 크다",
        11: "세그멘테이션 오류(SIGSEGV)로 죽었다",
        25: "파일 크기 상한(SIGXFSZ)을 넘겼다",
    }
    return known.get(signum, f"시그널 {signum}으로 강제 종료됐다")

class Sandbox(ABC):
    @abstractmethod
    async def run(
        self, *, code: str, dataset_path: str, workdir: Path, timeout_seconds: float,
        execution_id: str | None = None,
    ) -> SandboxResult: ...

class SubprocessSandbox(Sandbox):
    """로컬 개발용. 위 표의 ⚠️/❌를 알고 쓴다."""

    def __init__(
        self,
        *,
        memory_mb: int | None = DEFAULT_MEMORY_MB,
        cpu_seconds: int | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.memory_mb = memory_mb
        # None이면 run()의 timeout_seconds에서 파생한다.
        self.cpu_seconds = cpu_seconds
        self.python_executable = python_executable or sys.executable

    # -- 환경 --------------------------------------------------------------
    def _child_env(self) -> dict[str, str]:
        env = {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ}
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # 자식 프로세스에는 src 루트만 PYTHONPATH로 전달한다.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        # BLAS 스레드는 1개로 고정한다.
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[key] = "1"
        # 분석 프로세스의 GPU 접근을 차단한다.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["MPLBACKEND"] = "Agg"  # 헤드리스. 없으면 GUI 백엔드를 고르려다 실패한다.
        return env

    # -- 자원 제한 ---------------------------------------------------------
    def _preexec(self, timeout_seconds: float):
        import resource

        memory_bytes = self.memory_mb * 1024 * 1024 if self.memory_mb else None
        cpu = self.cpu_seconds or int(timeout_seconds) + CPU_LIMIT_MARGIN_SECONDS

        def apply() -> None:
            if memory_bytes is not None:
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            # 코어 덤프를 비활성화한다.
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            # 프로세스 수 제한은 컨테이너 cgroup에 맡긴다.

        return apply

    def _resource_limits_available(self) -> str | None:
        """자원 제한을 걸 수 있는지 미리 확인한다. 불가하면 이유 문자열."""
        if sys.platform == "win32":
            return "win32에서는 rlimit을 쓸 수 없다 — container 백엔드를 사용하라"
        try:
            import resource  # noqa: F401
        except ImportError as exc:
            return f"resource 모듈 없음: {exc}"
        return None

    # -- 실행 --------------------------------------------------------------
    async def _spawn(
        self, *, workdir: Path, dataset_path: str, timeout_seconds: float,
    ) -> asyncio.subprocess.Process:
        preexec = (
            self._preexec(timeout_seconds)
            if not self._resource_limits_available() else None
        )
        return await asyncio.create_subprocess_exec(
            self.python_executable, "-m", "analysis_loop_v3.execution.sandbox_worker",
            "--input", dataset_path,
            "--program", str(workdir / "program.py"),
            "--output", str(workdir / "result.json"),
            cwd=str(workdir),
            env=self._child_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=preexec,
        )

    def _finish(
        self, process: asyncio.subprocess.Process, *, output_path: Path,
        stdout_b: bytes, stderr_b: bytes, duration: float,
    ) -> SandboxResult:
        """자식이 끝난 뒤 결과를 판정한다."""
        stderr = stderr_b.decode("utf-8", "replace")
        if process.returncode is not None and process.returncode < 0:
            # 시그널 종료 원인을 stderr에 보완한다.
            stderr = (_signal_reason(-process.returncode) + "\n" + stderr).strip()

        ok = process.returncode == 0 and output_path.is_file()
        if ok and output_path.stat().st_size > MAX_RESULT_BYTES:
            ok = False
            stderr = (
                f"result.json이 {output_path.stat().st_size:,}바이트다 — "
                f"상한 {MAX_RESULT_BYTES:,}\n{stderr}"
            )
        return SandboxResult(
            ok=ok,
            result_path=output_path if ok else None,
            stdout=stdout_b.decode("utf-8", "replace"),
            stderr=stderr,
            duration_seconds=duration,
        )

    async def run(
        self, *, code: str, dataset_path: str, workdir: Path, timeout_seconds: float,
        execution_id: str | None = None,
    ) -> SandboxResult:
        del execution_id  # 원격 구현용. 같은 프로세스에는 유실될 응답이 없다.
        if timeout_seconds <= 0:
            return _refused("sandbox timeout은 0보다 커야 한다")

        # rlimit이 없으면 로컬 실행은 계속 허용한다.
        workdir.mkdir(parents=True, exist_ok=True)
        # 자식 프로세스에 절대경로를 전달한다.
        workdir = workdir.resolve()
        output_path = workdir / "result.json"
        (workdir / "program.py").write_text(code, encoding="utf-8")

        started = time.monotonic()
        process = await self._spawn(
            workdir=workdir,
            dataset_path=str(Path(dataset_path).resolve()),
            timeout_seconds=timeout_seconds,
        )
        stdout_task = asyncio.create_task(_read_limited(process.stdout))
        stderr_task = asyncio.create_task(_read_limited(process.stderr))
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            stdout_b, stderr_b = await asyncio.shield(asyncio.gather(stdout_task, stderr_task))
        except TimeoutError:
            return _refused(
                f"{timeout_seconds:.0f}초 안에 끝나지 않았다",
                duration=time.monotonic() - started,
            )

        finally:
            # 취소 시 자식 프로세스도 종료한다.
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await process.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        return self._finish(
            process, output_path=output_path,
            stdout_b=stdout_b, stderr_b=stderr_b,
            duration=time.monotonic() - started,
        )
