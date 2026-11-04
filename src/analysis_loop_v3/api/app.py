"""FastAPI application 조립점.

lifespan에서 공유 인프라를 준비하고, HTTP endpoint 구현은 run_routes.py와
system_routes.py에 둔다. Application 자체는 workflow state를 소유하지 않는다.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from ..env import load_project_env
from ..service import open_checkpointer
from . import db
from .http_helpers import (
    _checked_dataset_path,
    _checked_run_id,
    _ctx,
    _postgres_source,
    _shape_snapshot,
    _snapshot_or_404,
    _tail_events,
)
from .run_routes import (
    create_run,
    get_code,
    get_report,
    get_run,
    list_runs,
    resume_run,
    stop_run,
)
from .run_routes import router as run_router
from .runs import RunRegistry, ServerContext, build_reader_service
from .system_routes import health
from .system_routes import router as system_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """DB/checkpointer/artifact store/reader service의 application lifecycle."""
    load_project_env()
    await db.init_db()
    workspace = Path(os.environ.get("ANALYSIS_WORKSPACE", "runs"))

    async with open_checkpointer(
        workspace / "checkpoints.sqlite"
    ) as checkpointer:
        from ..runtime import ArtifactStore

        app.state.ctx = ServerContext(
            workspace=workspace,
            checkpointer=checkpointer,
            artifacts=ArtifactStore(workspace / "artifacts"),
            registry=RunRegistry(),
            reader=await build_reader_service(workspace, checkpointer),
        )
        yield

    await db.dispose_engine()


app = FastAPI(
    title="Analysis Loop v3 API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(system_router)
app.include_router(run_router)


# 기존 app.py에서 endpoint/helper를 직접 import하던 호출부와 호환성을 유지한다.
__all__ = [
    "app",
    "lifespan",
    "health",
    "create_run",
    "list_runs",
    "get_run",
    "resume_run",
    "stop_run",
    "get_report",
    "get_code",
    "_ctx",
    "_checked_dataset_path",
    "_postgres_source",
    "_checked_run_id",
    "_snapshot_or_404",
    "_tail_events",
    "_shape_snapshot",
]
