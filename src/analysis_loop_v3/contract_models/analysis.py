"""Codegen Manifest, 분석 protocol, data lineage 계약."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import DescriptionItem
from .data import ArtifactRef, DatasetProfile

SchemaType = Literal["mapping", "list", "number", "string", "boolean"]


class AnalysisProtocol(BaseModel):
    """계산 조건 재현성 계약."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sampling_strategy: str | None = None
    exclusion_rules: list[DescriptionItem] = Field(default_factory=list)
    comparison_groups: list[DescriptionItem] = Field(default_factory=list)
    random_seed: int | None = None
    repetitions: int | None = Field(default=None, ge=1)
    evaluation_strategy: str | None = None
    evaluation_design: Literal[
        "holdout",
        "cross_validation",
        "out_of_fold",
        "time_split",
        "group_split",
        "nested_cv",
        "external_validation",
        "custom",
    ] | None = None
    preprocessing_fit_scope: Literal[
        "train_only",
        "full_data_safe",
        "not_applicable",
        "custom",
    ] | None = None
    temporal_order_preserved: bool | None = None
    group_isolation_preserved: bool | None = None
    uncertainty_method: str | None = None
    hypothesis_testing: bool = False
    planned_comparisons: int | None = Field(default=None, ge=1)
    multiple_testing: str | None = None
    class_balance_strategy: str | None = None
    interpretation_scope: Literal[
        "descriptive",
        "associational",
        "predictive",
        "causal",
    ] = "descriptive"


class GeneratedAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str
    selected_columns: list[str] = Field(min_length=1)
    method: str = Field(min_length=1)
    method_reason: str = ""
    assumptions: list[str] = Field(default_factory=list)

    target: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    adjustment_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None
    split_unit_columns: list[str] = Field(default_factory=list)
    prediction_task: Literal[
        "classification",
        "regression",
        "forecasting",
        "ranking",
        "other",
    ] | None = None
    group_by: list[str] = Field(default_factory=list)
    derived_columns: list[str] = Field(default_factory=list)
    transformations: list[DescriptionItem] = Field(default_factory=list)
    split_strategy: str | None = None
    protocol: AnalysisProtocol = Field(default_factory=AnalysisProtocol)
    result_granularity: Literal["aggregate", "model_metric"] = "aggregate"
    expected_outputs: list[str] = Field(default_factory=list)
    expected_figures: list[str] = Field(default_factory=list)
    expected_output_schema: dict[str, SchemaType] = Field(default_factory=dict)
    code: str = Field(min_length=1)

    @model_validator(mode="after")
    def _target_is_not_a_feature(self) -> GeneratedAnalysis:
        if not self.target:
            return self
        leaked = [
            role
            for role, columns in (
                ("feature_columns", self.feature_columns),
                ("adjustment_columns", self.adjustment_columns),
            )
            if self.target in columns
        ]
        if leaked:
            raise ValueError(
                f"target '{self.target}'이 {', '.join(leaked)}에 들어 있다 — 데이터 누수다"
            )
        return self

    def available_columns(self, source_columns: set[str]) -> set[str]:
        return source_columns | set(self.derived_columns)

    def predictor_columns(self) -> list[str]:
        seen: dict[str, None] = {}
        for name in [*self.feature_columns, *self.adjustment_columns]:
            seen.setdefault(name, None)
        return list(seen)

    def is_predictive(self) -> bool:
        return self.result_granularity == "model_metric" and bool(self.target)

    def manifest_without_code(self) -> dict[str, Any]:
        return self.model_dump(exclude={"code"})


class DataLineage(BaseModel):
    """Evidence의 데이터 snapshot과 변환 조건 식별."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_artifact_id: str = Field(min_length=1)
    dataset_sha256: str = Field(min_length=64, max_length=64)
    schema_fingerprint: str = Field(min_length=1)
    source_rows: int = Field(ge=0)
    transform_signature: str = Field(min_length=64, max_length=64)

    @classmethod
    def from_analysis(
        cls,
        *,
        dataset_ref: ArtifactRef,
        profile: DatasetProfile,
        manifest: GeneratedAnalysis,
    ) -> DataLineage:
        payload = {
            "selected_columns": sorted(set(manifest.selected_columns)),
            "derived_columns": sorted(set(manifest.derived_columns)),
            "transformations": manifest.transformations,
            "sampling_strategy": manifest.protocol.sampling_strategy,
            "exclusion_rules": manifest.protocol.exclusion_rules,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            dataset_artifact_id=dataset_ref.artifact_id,
            dataset_sha256=dataset_ref.sha256,
            schema_fingerprint=profile.schema_fingerprint,
            source_rows=profile.row_count,
            transform_signature=hashlib.sha256(encoded).hexdigest(),
        )
