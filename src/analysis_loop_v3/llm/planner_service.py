"""Planner 역할 LLM 서비스."""

from __future__ import annotations

from typing import Any

from ..contracts import (
    AgendaTask,
    DatasetProfile,
    FindingDigest,
    ResearchIntent,
    SubGoal,
)
from ..runtime_protocols import PlannerProposal, ServiceResult
from .client import OpenAICompatClient
from .memory import MEMORY_TOOLS, AnalysisMemory
from .service_common import (
    LLMResearchIntent,
    RoleService,
    dumps,
    goal_verdict,
    load_prompt,
    parse_goals,
    profile_digest,
)


class LLMPlanner(RoleService):
    def __init__(
        self,
        client: OpenAICompatClient,
        *,
        memory: AnalysisMemory | None = None,
    ) -> None:
        super().__init__(client)
        self.memory = memory

    def _memory_context(self) -> dict[str, Any]:
        if self.memory is None:
            return {}
        notebook = self.memory.snapshot("notebook")
        return {
            "분석 메모장": {
                "revision": notebook.revision,
                "content": notebook.content,
            },
        }

    async def _generate_json(
        self,
        *,
        prompt: str,
        system_name: str,
        timeout_seconds: float,
        role: str,
    ) -> dict[str, Any]:
        system = load_prompt(system_name)
        if self.memory is None:
            return await self.client.generate_json(
                prompt=prompt,
                system=system,
                timeout_seconds=timeout_seconds,
                role=role,
            )

        wiki = self.memory.snapshot("wiki")
        return await self.client.generate_json_with_tools(
            prompt=prompt,
            system=(
                f"{load_prompt('master_memory_system')}\n\n"
                f"# 현재 분석 Wiki\n\n"
                f"revision: `{wiki.revision}`\n\n{wiki.content}\n\n{system}"
            ),
            tools=MEMORY_TOOLS,
            execute_tool=self.memory.execute_tool,
            timeout_seconds=timeout_seconds,
            role=role,
        )

    async def decompose_objective(
        self,
        *,
        objective: str,
        profile: DatasetProfile,
        timeout_seconds: float,
    ) -> ServiceResult[list[SubGoal]]:
        prompt = dumps(
            {
                "목표": objective,
                "데이터": {
                    "행 수": profile.row_count,
                    "컬럼": profile_digest(profile),
                },
                **self._memory_context(),
            }
        )
        try:
            raw = await self._generate_json(
                prompt=prompt,
                system_name="decompose_system",
                timeout_seconds=timeout_seconds,
                role="decompose",
            )
        except Exception as exc:
            return self._fail(exc)

        goals = parse_goals(raw)
        if not goals:
            return ServiceResult(value=[], error="유효한 목표 축이 없다")
        return ServiceResult(value=goals)

    async def expand_goals(
        self,
        *,
        objective: str,
        profile: DatasetProfile,
        existing_goals: list[SubGoal],
        findings: list[FindingDigest],
        timeout_seconds: float,
        remaining_seconds: float | None = None,
        remaining_iterations: int | None = None,
        expansion_round: int | None = None,
        max_expansions: int | None = None,
    ) -> ServiceResult[list[SubGoal]]:
        prompt = dumps(
            {
                "목표": objective,
                "데이터": {
                    "행 수": profile.row_count,
                    "컬럼": profile_digest(profile),
                },
                "기존 목표 축": [
                    {
                        "goal_id": goal.goal_id,
                        "question": goal.question,
                        "status": goal.status,
                        "closing_note": goal.closing_note,
                    }
                    for goal in existing_goals
                ],
                "지금까지 알아낸 것": [finding.model_dump(mode="json") for finding in findings],
                "남은 실행 예산": {
                    "seconds": remaining_seconds,
                    "iterations": remaining_iterations,
                    "expansion_round": expansion_round,
                    "max_expansions": max_expansions,
                },
                **self._memory_context(),
            }
        )
        try:
            raw = await self._generate_json(
                prompt=prompt,
                system_name="expand_goals_system",
                timeout_seconds=timeout_seconds,
                role="expand_goals",
            )
        except Exception as exc:
            return self._fail(exc)

        decision = str(raw.get("decision") or "").strip().lower()
        expansion_reason = str(raw.get("reason") or "").strip()[:400] or None
        if decision == "stop":
            return ServiceResult(value=[], reason=expansion_reason)
        if decision != "expand":
            return ServiceResult(
                error="확장 응답에 decision=expand|stop이 없다", failure_kind="response_invalid"
            )

        goals = parse_goals(raw, start=len(existing_goals))
        if not goals:
            return ServiceResult(
                error="decision=expand인데 유효한 새 목표 축이 없다",
                failure_kind="response_invalid",
            )
        return ServiceResult(value=goals, reason=expansion_reason)

    async def propose_intent(
        self,
        *,
        objective: str,
        profile: DatasetProfile,
        unexplored: list[str],
        column_usage: dict[str, int],
        prior_findings: list[FindingDigest],
        unresolved_findings: list[FindingDigest],
        sub_goals: list[SubGoal],
        timeout_seconds: float,
        required_followups: list[AgendaTask] | None = None,
        recent_inspection: dict[str, Any] | None = None,
        recent_evidence_lookup: list[dict[str, Any]] | None = None,
        evidence_relations: list[dict[str, Any]] | None = None,
        remaining_seconds: float | None = None,
        remaining_iterations: int | None = None,
        rejected_attempts: list[dict[str, Any]] | None = None,
        previous_generation_error: str | None = None,
    ) -> ServiceResult[PlannerProposal]:
        prompt = dumps(
            {
                "목표": objective,
                **(
                    {"직전 시도가 거부된 이유": previous_generation_error}
                    if previous_generation_error
                    else {}
                ),
                "목표 축": [
                    {
                        "goal_id": goal.goal_id,
                        "question": goal.question,
                        "rationale": goal.rationale,
                        "success_criterion": goal.success_criterion,
                        "status": goal.status,
                        "근거 수": len(goal.evidence_ids),
                    }
                    for goal in sub_goals
                ],
                "데이터": {
                    "행 수": profile.row_count,
                    "컬럼": profile_digest(profile),
                },
                "아직 한 번도 분석되지 않은 변수": unexplored,
                "컬럼별 분석 사용 횟수": column_usage,
                "지금까지 알아낸 것": [
                    finding.model_dump(mode="json") for finding in prior_findings
                ],
                "아직 정리되지 않은 발견": [
                    {
                        "evidence_id": finding.evidence_id,
                        "claim": finding.claim,
                        "confidence": finding.confidence,
                        "caveats": finding.caveats,
                        "open_questions": finding.open_questions,
                        "contradicts": finding.contradicts,
                    }
                    for finding in unresolved_findings
                ],
                "작업 목록": [item.model_dump(mode="json") for item in (required_followups or [])],
                "최근 데이터 점검": recent_inspection,
                "최근 Evidence 조회": recent_evidence_lookup or [],
                "Evidence 관계": evidence_relations or [],
                "남은 실행 예산": {
                    "seconds": remaining_seconds,
                    "iterations": remaining_iterations,
                },
                "거부된 시도": rejected_attempts or [],
                **self._memory_context(),
            }
        )

        try:
            raw = await self._generate_json(
                prompt=prompt,
                system_name="planner_system",
                timeout_seconds=timeout_seconds,
                role="planner",
            )
        except Exception as exc:
            return self._fail(exc)

        verdict = goal_verdict(raw, sub_goals)

        if not str(raw.get("question") or "").strip():
            open_goal_ids = {goal.goal_id for goal in sub_goals if goal.is_open}
            closed_ids = set(verdict["converged"]) | set(verdict["abandoned"])
            if open_goal_ids and open_goal_ids <= closed_ids:
                verdict["finished"] = True
                return ServiceResult(value=PlannerProposal(verdict=verdict))

            return ServiceResult(
                value=PlannerProposal(verdict=verdict),
                error="플래너 응답에 question이 비어 있다 — goal_review만 왔다",
                failure_kind="response_invalid",
            )

        try:
            boundary = LLMResearchIntent.model_validate(raw)
            intent = ResearchIntent.model_validate(
                boundary.model_dump(
                    mode="python",
                    exclude={"intent_id"},
                )
            )
        except Exception as exc:
            failure = self._fail(exc, message=f"연구 의도 응답 계약 위반: {exc}")
            return ServiceResult(
                value=PlannerProposal(verdict=verdict),
                error=failure.error,
                failure_kind=failure.failure_kind,
            )
        return ServiceResult(value=PlannerProposal(intent=intent, verdict=verdict))
