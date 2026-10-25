"""Artifact와 Dataset profile 계약."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArtifactKind(str, Enum):
    DATASET = "dataset"
    PROFILE = "profile"
    PROGRAM = "program"
    RESULT = "result"
    REPORT = "report"
    FIGURE = "figure"
    INSPECTION = "inspection"


class ArtifactRef(BaseModel):
    """실제 바이트는 ArtifactStore에, State에는 이 참조만 둔다."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    kind: ArtifactKind
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def _summary_must_stay_small(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 20:
            raise ValueError(
                f"ArtifactRef.summary는 최대 20개 키다(실제 {len(value)}개). "
                "요약이 아니라 데이터를 넣고 있는지 확인하라."
            )
        return value


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    non_null: int = Field(ge=0)
    null_count: int = Field(ge=0)
    unique_count: int = Field(ge=0)
    sample_values: list[str] = Field(default_factory=list, max_length=5)


class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    row_count: int = Field(ge=0)
    columns: list[ColumnProfile]
    schema_fingerprint: str

    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def analyzable_columns(self) -> list[str]:
        """실제 존재하는 모든 컬럼. 의미상 제외 여부는 Planner가 판단한다."""
        return self.column_names()
