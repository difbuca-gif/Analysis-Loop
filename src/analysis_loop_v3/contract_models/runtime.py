"""Run lifecycle, stop reason, execution status 계약."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class LoopIssue(BaseModel):
    """동일 오류 반복으로 중단된 회차의 운영 이슈."""

    model_config = ConfigDict(extra="forbid")

    label: str
    count: int
    iteration: int
    intent_id: str | None = None
    question: str = ""
    messages: list[str] = Field(default_factory=list)


class ExecutionStatus(str, Enum):
    """Worker의 한 실행 상태."""

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class StopCode(str, Enum):
    """Run 종료 상태 계약."""

    GOAL_SATISFIED = "GOAL_SATISFIED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"
    DATA_LIMITED = "DATA_LIMITED"
    EXPANSION_LIMIT_REACHED = "EXPANSION_LIMIT_REACHED"

    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    PLANNER_STUCK = "PLANNER_STUCK"
    RECURSION_LIMIT = "RECURSION_LIMIT"
    UNKNOWN = "UNKNOWN"
