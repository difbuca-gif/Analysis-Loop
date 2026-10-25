"""API 자체 상태 확인 endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import db

logger = logging.getLogger("analysis_api.system")
router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    checks = {"api": "ok"}
    healthy = True
    try:
        await db.ping()
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001 - health endpoint must report dependency failure.
        healthy = False
        checks["postgres"] = "error"
        logger.warning("PostgreSQL healthcheck failed: %s", exc)

    return JSONResponse(
        status_code=200 if healthy else 503,
        content=checks,
    )
