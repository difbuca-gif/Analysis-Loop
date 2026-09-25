"""실행 레지스트리와 백그라운드 구동.

service.start/resume은 그래프가 끝날 때까지 await로 대기하므로(최대
--budget초) 핸들러에서 직접 부르면 요청이 그만큼 블록된다. asyncio.Task로 던진다.

조회는 이 레지스트리를 안 본다(snapshot이 체크포인터에서 직접 읽는다). 여기는
"지금 이 run_id를 구동하는 task가 있는가"만 알아 동시 ainvoke를 막는다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import validate_run_id
from ..execution.remote_sandbox import build_sandbox
from ..llm.memory import AnalysisMemory, load_bundled_wiki
from ..llm.services import build_services
from ..runtime import ArtifactStore, RuntimeDeps
from ..service import AnalysisService, RunHandle
from . import db

logger = logging.getLogger("analysis_api.runs")


class RunConflictError(Exception):
    """같은 run_id를 이미 다른 task가 구동 중이다."""


@dataclass(slots=True)
class RunRegistry:
    _tasks: dict[str, asyncio.Task] = field(default_factory=dict)

    def is_active(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    def start(self, run_id: str, factory: Callable[[], Awaitable[Any]]) -> asyncio.Task:
        if self.is_active(run_id):
            raise RunConflictError(f"run {run_id!r}는 이미 실행 중이다")
        task = asyncio.create_task(factory(), name=f"run:{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda t: self._log_done(run_id, t))
        return task

    def cancel(self, run_id: str) -> bool:
        """best-effort 취소. 노드마다 체크포인트하므로 진행 중이던 노드 하나만 유실된다."""
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    @staticmethod
    def _log_done(run_id: str, task: asyncio.Task) -> None:
        if task.cancelled():
            logger.info("run %s cancelled", run_id)
            return
        exc = task.exception()
        if exc is not None:
            logger.error("run %s failed in background", run_id, exc_info=exc)


@dataclass(slots=True)
class ServerContext:
    workspace: Path
    checkpointer: Any
    artifacts: ArtifactStore
    registry: RunRegistry
    reader: AnalysisService  # planner/codegen/critic 없음. 조회 전용, 모든 run_id가 공유


async def build_reader_service(workspace: Path, checkpointer: Any) -> AnalysisService:
    """조회 전용 서비스 — LLM 없이 그래프만 만든다.

    aget_state()는 노드를 실행하지 않으므로 planner/codegen/critic=None이어도 안전하다.
    thread_id는 매 호출 config로만 가므로 모든 run이 이 인스턴스를 공유해도 된다.
    """
    deps = RuntimeDeps(
        artifacts=ArtifactStore(workspace / "artifacts"),
        sandbox=build_sandbox(),
        workdir_root=workspace / "work",
    )
    return AnalysisService(deps, checkpointer)


async def drive(
    ctx: ServerContext,
    run_id: str,
    action: Callable[[AnalysisService], Awaitable[RunHandle]],
) -> RunHandle:
    """run 전용 RuntimeDeps를 새로 만들어 action을 실행한다(cli._with_service와 같은 조립).

    run마다 memory 경로가 다르므로 공유하면 안 된다. 체크포인터·ArtifactStore만 공유.
    """
    validate_run_id(run_id)
    memory = AnalysisMemory(
        wiki_path=ctx.workspace / "memory" / "ANALYSIS_WIKI.md",
        notebook_path=ctx.workspace / "memory" / "notebooks" / f"{run_id}.md",
        wiki_seed=load_bundled_wiki(),
    )
    planner, codegen, critic, clients = build_services(memory=memory)
    deps = RuntimeDeps(
        artifacts=ctx.artifacts,
        sandbox=build_sandbox(),
        planner=planner, codegen=codegen, critic=critic,
        workdir_root=ctx.workspace / "work",
        events_log_path=ctx.workspace / "events" / f"{run_id}.jsonl",
    )
    for client in clients:
        client.event_sink = lambda name, payload, _rid=run_id: deps.event(
            name, payload, run_id=_rid,
        )
    service = AnalysisService(deps, ctx.checkpointer)
    try:
        handle = await action(service)
    except Exception as exc:
        logger.exception("run %s failed", run_id)
        await db.update_status(run_id, status="FAILED", stop_reason=str(exc))
        raise
    finally:
        for client in clients:
            await client.aclose()
    await db.update_status(
        run_id, status=handle.status, stop_reason=handle.state.get("stop_reason"),
    )
    return handle
