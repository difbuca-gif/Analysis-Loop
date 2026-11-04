"""SubGoal과 Planner ResearchIntent 계약."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import DescriptionItem

GoalStatus = Literal[
    "open",
    "converged",
    "abandoned",
]


class SubGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(pattern=r"^g[0-9]+$")
    question: str = Field(min_length=1)
    rationale: str = ""
    success_criterion: str = ""
    status: GoalStatus = "open"
    evidence_ids: list[str] = Field(default_factory=list)
    closing_note: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == "open"


IntentPurpose = Literal[
    "explore",
    "deepen",
    "challenge",
    "resolve_contradiction",
]


class ResearchIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str = Field(default_factory=lambda: uuid4().hex)
    question: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    rationale: str = ""

    goal_id: str | None = None
    purpose: IntentPurpose = "explore"
    parent_evidence_ids: list[str] = Field(default_factory=list)
    task_id: str | None = None
    execution_mode: Literal[
        "compute",
        "reuse_evidence",
        "inspect_data",
        "lookup_evidence",
        "mark_data_limited",
    ] = "compute"
    reuse_evidence_ids: list[str] = Field(default_factory=list)
    success_criterion: str = ""
    candidate_columns: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    excluded_columns: list[str] = Field(default_factory=list)
    desired_evidence: list[DescriptionItem] = Field(default_factory=list)
    constraints: list[DescriptionItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _purpose_needs_parent(self) -> ResearchIntent:
        if self.purpose != "explore" and not self.parent_evidence_ids:
            raise ValueError(
                f"purpose={self.purpose}이면 parent_evidence_ids가 있어야 한다"
            )
        return self

    @model_validator(mode="after")
    def _execution_mode_is_consistent(self) -> ResearchIntent:
        if self.execution_mode == "reuse_evidence":
            if not self.reuse_evidence_ids:
                raise ValueError(
                    "reuse_evidence에는 reuse_evidence_ids가 필요하다"
                )
            if self.task_id:
                raise ValueError(
                    "Agenda 작업은 새 Evidence로 닫아야 하므로 재사용할 수 없다"
                )
            missing = set(self.reuse_evidence_ids) - set(self.parent_evidence_ids)
            if missing:
                raise ValueError(
                    "reuse_evidence_ids는 parent_evidence_ids에도 있어야 한다: "
                    f"{sorted(missing)}"
                )
        elif self.reuse_evidence_ids:
            raise ValueError(
                "reuse_evidence가 아닌 행동에는 reuse_evidence_ids를 넣지 않는다"
            )

        if self.execution_mode == "inspect_data":
            if self.task_id:
                raise ValueError(
                    "Agenda 작업 수행 중 inspect_data로 우회할 수 없다"
                )
            if not self.candidate_columns:
                raise ValueError(
                    "inspect_data에는 확인할 candidate_columns가 필요하다"
                )

        if self.execution_mode == "lookup_evidence":
            if self.task_id:
                raise ValueError(
                    "Agenda 작업은 source Evidence가 이미 있어 lookup_evidence가 필요 없다"
                )
            if not (
                self.goal_id
                or self.candidate_columns
                or self.parent_evidence_ids
            ):
                raise ValueError(
                    "lookup_evidence에는 goal_id, candidate_columns, "
                    "parent_evidence_ids 중 하나가 필요하다"
                )

        if self.execution_mode == "mark_data_limited" and not self.task_id:
            raise ValueError("mark_data_limited에는 task_id가 필요하다")
        return self

    @model_validator(mode="after")
    def _required_within_candidates(self) -> ResearchIntent:
        missing = set(self.required_columns) - set(self.candidate_columns)
        if missing:
            self.candidate_columns = [
                *self.candidate_columns,
                *sorted(missing),
            ]

        overlap = set(self.required_columns) & set(self.excluded_columns)
        if overlap:
            raise ValueError(
                f"required와 excluded가 겹친다: {sorted(overlap)}"
            )
        return self

    def dedupe_key(self) -> str:
        """동일 목표 내 분석 요청 중복 판정."""
        return self._analysis_signature(include_goal=True)

    def reuse_signature(self) -> str:
        """goal_id와 무관한 코드 재사용용 분석 요청 서명."""
        return self._analysis_signature(include_goal=False)

    def _analysis_signature(self, *, include_goal: bool) -> str:
        excluded = (
            {"intent_id"}
            if include_goal
            else {"intent_id", "goal_id"}
        )
        payload: dict[str, Any] = self.model_dump(
            mode="json",
            exclude=excluded,
        )
        for field in (
            "candidate_columns",
            "required_columns",
            "excluded_columns",
            "parent_evidence_ids",
            "reuse_evidence_ids",
        ):
            payload[field] = sorted(set(payload[field]))
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "intent-numeric-v1:" + hashlib.sha256(encoded).hexdigest()
