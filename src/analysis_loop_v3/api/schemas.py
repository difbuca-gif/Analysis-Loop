"""API 요청/응답 Pydantic 모델."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, model_validator

from ..contracts import RUN_ID_PATTERN


class StartRunRequest(BaseModel):
    dataset_path: str | None = Field(default=None, min_length=1, max_length=4096)
    sql_query: str | None = Field(default=None, min_length=1, max_length=100_000)
    objective: str = Field(min_length=1, max_length=8000)
    max_iterations: int = Field(default=25, ge=1, le=500)
    time_budget_seconds: float = Field(default=3600.0, gt=0)
    # run_id는 경로에 사용되므로 traversal 패턴까지 검증
    run_id: str | None = Field(default=None, pattern=RUN_ID_PATTERN.pattern)

    @model_validator(mode="after")
    def require_exactly_one_source(self) -> Self:
        if (self.dataset_path is None) == (self.sql_query is None):
            raise ValueError("dataset_path 또는 sql_query 중 정확히 하나를 지정해야 한다")
        return self


class ResumeRequest(BaseModel):
    # time_budget_seconds 생략 시 기존 deadline_at 유지
    # 지정 시 현재 시점 기준으로 예산 재계산
    time_budget_seconds: float | None = Field(default=None, gt=0)


class RunSummary(BaseModel):
    run_id: str
    objective: str
    dataset_path: str
    status: str
    stop_reason: str | None = None
    created_at: str
    updated_at: str
