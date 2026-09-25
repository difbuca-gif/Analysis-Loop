"""시작·재개 인터페이스. 체크포인터가 워크플로 상태의 유일한 저장소다.

재개=ainvoke(None), 조회=aget_state. 어디까지 갔는지 다시
계산하지 않고 외부 상태를 먼저 고치지도 않는다. 파일이 필요하면 그건 projection이다.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.errors import GraphRecursionError
from langgraph.graph import START

from .contracts import (
    SCHEMA_VERSION,
    ArtifactKind,
    RunStatus,
    StopCode,
    validate_run_id,
)
from .data.ingestion import prepare_dataset
from .data.postgres import PostgresSource, materialize_postgres_dataset
from .orchestration.graph import RECURSION_LIMIT, build_graph
from .orchestration.nodes import finalize
from .runtime import RuntimeDeps
from .state import new_state, remaining_seconds, terminal_update

@dataclass(slots=True)
class RunHandle:
    run_id: str
    thread_id: str
    status: str
    state: dict[str, Any]

def _thread_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

@contextlib.asynccontextmanager
async def open_checkpointer(db_path: Path | str) -> AsyncIterator[Any]:
    """SQLite 체크포인터. 프로세스가 죽어도 남는다."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver

class AnalysisService:
    def __init__(self, deps: RuntimeDeps, checkpointer: Any) -> None:
        self.deps = deps
        self.graph = build_graph(deps, checkpointer=checkpointer)

    # -- 시작 --------------------------------------------------------------
    async def start(
        self,
        *,
        dataset_path: Path | str | None = None,
        postgres_source: PostgresSource | None = None,
        objective: str = "",
        max_iterations: int = 25,
        time_budget_seconds: float = 3600.0,
        run_id: str | None = None,
    ) -> RunHandle:
        # 파일 경로에 쓰기 전에 run_id를 검증한다.
        run_id = validate_run_id(run_id) if run_id else uuid4().hex[:12]
        existing = await self.graph.aget_state(_thread_config(run_id))
        if existing.values:
            raise ValueError(f"run_id {run_id!r}가 이미 존재한다")
        if (dataset_path is None) == (postgres_source is None):
            raise ValueError("dataset_path 또는 postgres_source 중 정확히 하나를 지정해야 한다")
        if postgres_source is not None:
            prepared = await asyncio.to_thread(
                materialize_postgres_dataset,
                postgres_source,
                output_dir=self.deps.workdir_root / "_prepared",
            )
        else:
            prepared = prepare_dataset(
                dataset_path, output_dir=self.deps.workdir_root / "_prepared",
            )
        dataset_ref = self.deps.artifacts.put_file(
            prepared.path, kind=ArtifactKind.DATASET,
            summary=prepared.summary,
        )
        state = new_state(
            run_id=run_id, objective=objective,
            dataset_ref=dataset_ref, max_iterations=max_iterations,
            time_budget_seconds=time_budget_seconds,
        )
        return await self._drive(state, thread_id=run_id)

    async def _existing_values(self, thread_id: str) -> dict[str, Any]:
        validate_run_id(thread_id)
        snapshot = await self.graph.aget_state(_thread_config(thread_id))
        values = dict(snapshot.values or {})
        if not values:
            raise ValueError(f"run_id {thread_id!r}를 찾을 수 없다")
        if values.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"지원하지 않는 state schema: {values.get('schema_version')} "
                f"(현재 {SCHEMA_VERSION})"
            )
        return values

    # -- 재개 --------------------------------------------------------------
    async def resume(
        self, *, thread_id: str, time_budget_seconds: float | None = None,
    ) -> RunHandle:
        """중단 지점부터 계속한다. 끝난 노드는 다시 실행되지 않는다(유료 호출 중복 방지).

        time_budget_seconds를 주면 그때만 deadline_at을 지금부터 다시 계산한다.
        상태를 미리 고치는 것처럼 보이지만, 예산은 그래프 밖에서만 정할 수 있는
        운영 파라미터라 노드 입력 경로가 없다.
        """
        await self._existing_values(thread_id)
        if time_budget_seconds is not None:
            if time_budget_seconds <= 0:
                raise ValueError("time_budget_seconds는 0보다 커야 한다")
            # deadline_at 갱신은 START 상태로 기록한다.
            await self.graph.aupdate_state(
                _thread_config(thread_id),
                {"deadline_at": time.time() + time_budget_seconds},
                as_node=START,
            )
        return await self._drive(None, thread_id=thread_id)

    # -- 조회 --------------------------------------------------------------
    async def snapshot(self, *, thread_id: str) -> RunHandle:
        values = await self._existing_values(thread_id)
        return RunHandle(
            run_id=values.get("run_id", thread_id),
            thread_id=thread_id,
            status=values.get("status", RunStatus.RUNNING.value),
            state=values,
        )

    # -- 공통 --------------------------------------------------------------
    async def _drive(self, payload: Any, *, thread_id: str) -> RunHandle:
        config = _thread_config(thread_id)
        initial = payload if payload is not None else await self._existing_values(thread_id)
        budget = asyncio.timeout(max(0.0, remaining_seconds(initial)))
        try:
            async with budget:
                try:
                    await self.graph.ainvoke(payload, config)
                except GraphRecursionError:
                    await self._finish_stopped_run(
                        config, initial, status=RunStatus.FAILED,
                        stop_code=StopCode.RECURSION_LIMIT,
                        reason=f"그래프 재귀 한도({RECURSION_LIMIT}) 도달",
                    )
        except TimeoutError:
            # 내부 TimeoutError와 전체 예산 소진을 구분한다.
            if not budget.expired():
                raise
            await self._finish_stopped_run(
                config, initial, status=RunStatus.COMPLETED,
                stop_code=StopCode.BUDGET_EXHAUSTED, reason="시간 예산 소진",
            )
        # 최종 상태는 체크포인트에서 다시 읽는다.
        return await self.snapshot(thread_id=thread_id)

    async def _finish_stopped_run(
        self, config: dict[str, Any], initial: dict[str, Any], *,
        status: RunStatus, stop_code: StopCode, reason: str,
    ) -> None:
        snapshot = await self.graph.aget_state(config)
        values = dict(snapshot.values or {})
        terminal = terminal_update(status=status, stop_code=stop_code, reason=reason)
        update = await finalize({**(values or initial), **terminal}, self.deps)
        await self.graph.aupdate_state(
            config,
            # 기존 체크포인트의 누적 상태는 다시 쓰지 않는다.
            {**({} if values else initial), **update, **terminal},
            as_node="finalize",
        )
