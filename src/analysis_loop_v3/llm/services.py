"""Planner, Codegen, Critic 역할별 LLM 서비스 구현."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, ValidationError

from ..contracts import (
    AgendaTask,
    CriticReview,
    DatasetProfile,
    FindingDigest,
    GeneratedAnalysis,
    ResearchIntent,
    SubGoal,
)
from .client import (
    LLMError,
    LLMProviderError,
    LLMTransportError,
    OpenAICompatClient,
    build_client,
)
from .memory import MEMORY_TOOLS, AnalysisMemory

_PROMPT_DIR = Path(__file__).parent / "prompts"
_LOGGER = logging.getLogger(__name__)

_DEFAULT_PROVIDER = {"PLANNER": "local", "CODEGEN": "nvidia", "CRITIC": "local"}
_DEFAULT_MODEL = {
    "PLANNER": "Qwen/Qwen3.5-9B",
    # NVIDIA 무료 catalog에서 확인한 코드 지향 모델. 일반 Llama 70B를 암묵적
    # 기본값으로 쓰지 않는다.
    "CODEGEN": "deepseek-ai/deepseek-v4-flash-0731",
    "CRITIC": "Qwen/Qwen3.5-9B",
}
# 코드 이스케이프와 넓은 프로필을 고려한 역할별 생성 토큰 상한.
_DEFAULT_MAX_TOKENS = {"PLANNER": 2048, "CODEGEN": 8192, "CRITIC": 1024}
# "agy"는 HTTP가 아니라 로컬 agy CLI(Antigravity)를 서브프로세스로 부르는
# CODEGEN 전용 백엔드다. build_role_client가 아니라 build_services가 직접 분기한다.
VALID_PROVIDERS = ("local", "ollama", "nvidia", "agy", "none")

class _LLMResearchIntent(ResearchIntent):
    """외부 응답용 tolerant DTO. 내부에서는 다시 strict 모델로 검증한다."""

    model_config = ConfigDict(extra="ignore")

class _LLMGeneratedAnalysis(GeneratedAnalysis):
    model_config = ConfigDict(extra="ignore")

class _LLMCriticReview(CriticReview):
    model_config = ConfigDict(extra="ignore")

@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    return (_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")

def _role_env(role: str, suffix: str) -> str | None:
    value = os.environ.get(f"{role}_LLM_{suffix}")
    return value.strip() if value and value.strip() else None

def resolve_provider(role: str) -> str:
    """미설정은 기본값을 쓰고, 명시된 잘못된 값은 설정 오류로 드러낸다."""
    role = role.upper()
    provider = (_role_env(role, "PROVIDER") or _DEFAULT_PROVIDER.get(role, "local")).lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"{role}_LLM_PROVIDER={provider!r}는 지원하지 않는다 "
            f"(허용: {', '.join(VALID_PROVIDERS)})"
        )
    return provider

def build_role_client(role: str) -> OpenAICompatClient | None:
    role = role.upper()
    client = build_client(
        provider=resolve_provider(role),
        model=_role_env(role, "MODEL") or _DEFAULT_MODEL.get(role, ""),
        base_url=_role_env(role, "BASE_URL"),
    )
    if client is not None:
        raw_max_tokens = _role_env(role, "MAX_TOKENS")
        try:
            client.max_tokens = (
                int(raw_max_tokens) if raw_max_tokens
                else _DEFAULT_MAX_TOKENS.get(role, client.max_tokens)
            )
        except ValueError as exc:
            raise ValueError(f"{role}_LLM_MAX_TOKENS는 정수여야 한다") from exc
        if client.max_tokens <= 0:
            raise ValueError(f"{role}_LLM_MAX_TOKENS는 0보다 커야 한다")
    return client

def _profile_digest(profile: DatasetProfile) -> list[dict[str, Any]]:
    """프롬프트에 넣을 factual 컬럼 요약. 컬럼 순서로 일부를 숨기지 않는다."""
    return [
        {
            "name": c.name,
            "dtype": c.dtype,
            "nulls": c.null_count,
            "unique": c.unique_count,
        }
        for c in profile.columns
    ]

def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

def _parse_goals(raw: dict[str, Any], *, start: int = 0) -> list[SubGoal]:
    """응답에서 목표 축을 뽑는다.

    goal_id는 모델 출력 대신 순번으로 다시 부여해 안정적으로 추적한다.
    """
    goals: list[SubGoal] = []
    for offset, item in enumerate(raw.get("sub_goals") or [], start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        goals.append(SubGoal(
            goal_id=f"g{start + offset}",
            question=question,
            rationale=str(item.get("rationale") or ""),
            success_criterion=str(item.get("success_criterion") or ""),
        ))
    return goals

def _goal_verdict(raw: dict[str, Any], sub_goals: list[SubGoal]) -> dict[str, Any]:
    """열린 축에 대한 판정만 골라낸다. Intent 생성과 무관하게 보존해야 한다."""
    open_ids = {g.goal_id for g in sub_goals if g.is_open}
    converged: list[str] = []
    abandoned: list[str] = []
    reasons: list[str] = []
    reviewed: set[str] = set()
    for item in raw.get("goal_review") or []:
        if not isinstance(item, dict):
            continue
        goal_id = item.get("goal_id")
        if goal_id not in open_ids:
            continue
        reviewed.add(goal_id)
        bucket = {"converged": converged, "abandoned": abandoned}.get(
            str(item.get("verdict") or "").lower()
        )
        if bucket is None:
            continue
        bucket.append(goal_id)
        reasons.append(f"{goal_id}: {str(item.get('reason') or '')[:200]}")
    return {
        "converged": converged,
        "abandoned": abandoned,
        "closing_note": " / ".join(reasons)[:300],
        # 판정하지 않은 축. 노드가 이벤트로 남겨 프롬프트 준수 여부를 관측한다.
        "unreviewed": sorted(open_ids - reviewed),
    }

class _RoleService:
    """Planner, Codegen, Critic이 공유하는 실패 상태를 관리한다."""

    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client
        self.last_error: str | None = None
        self.last_failure_kind: str | None = None

    def _begin(self) -> None:
        self.last_error = None
        self.last_failure_kind = None

    def _fail(self, exc: Exception, *, message: str | None = None) -> None:
        """예외를 실패 종류로 옮긴다. 호출부는 곧바로 실패값을 반환한다."""
        if isinstance(exc, LLMTransportError):
            kind = "provider_unavailable"
        elif isinstance(exc, LLMProviderError):
            # 재시도로 풀리는 장애(503/429)와 설정 오류(401/403)는 대응이 다르다.
            kind = "provider_unavailable" if exc.retryable else "configuration"
        elif isinstance(exc, ValidationError):
            kind = "contract"
        elif isinstance(exc, LLMError):
            kind = "response_invalid"
        else:
            kind = "response_invalid"
        self.last_failure_kind = kind
        if message is None:
            message = str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
        self.last_error = message[:400]

class LLMPlanner(_RoleService):
    def __init__(
        self,
        client: OpenAICompatClient,
        *,
        memory: AnalysisMemory | None = None,
    ) -> None:
        super().__init__(client)
        self.memory = memory
        self.last_expansion_reason: str | None = None

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
        self, *, objective: str, profile: DatasetProfile, timeout_seconds: float,
    ) -> list[SubGoal]:
        """목표를 답할 수 있는 축으로 쪼갠다(실행당 1회).

        실패하면 빈 리스트 — 목표 전체를 단일 축으로 두고 계속 진행한다.
        """
        self._begin()
        prompt = _dumps({
            "목표": objective,
            "데이터": {"행 수": profile.row_count, "컬럼": _profile_digest(profile)},
            **self._memory_context(),
        })
        try:
            raw = await self._generate_json(
                prompt=prompt,
                system_name="decompose_system",
                timeout_seconds=timeout_seconds,
                role="decompose",
            )
        except Exception as exc:
            self._fail(exc)
            return []

        goals = _parse_goals(raw)
        if not goals:
            self.last_error = "유효한 목표 축이 없다"
            return []
        return goals

    async def expand_goals(
        self, *, objective: str, profile: DatasetProfile,
        existing_goals: list[SubGoal], findings: list[FindingDigest],
        timeout_seconds: float,
        remaining_seconds: float | None = None,
        remaining_iterations: int | None = None,
        expansion_round: int | None = None,
        max_expansions: int | None = None,
    ) -> list[SubGoal]:
        """근거를 보고 새 목표 축을 찾는다. stop은 정상, 계약 위반은 실패로 구분한다."""
        self._begin()
        self.last_expansion_reason = None
        prompt = _dumps({
            "목표": objective,
            "데이터": {"행 수": profile.row_count, "컬럼": _profile_digest(profile)},
            "기존 목표 축": [
                {
                    "goal_id": g.goal_id, "question": g.question,
                    "status": g.status, "closing_note": g.closing_note,
                }
                for g in existing_goals
            ],
            "지금까지 알아낸 것": [f.model_dump(mode="json") for f in findings],
            "남은 실행 예산": {
                "seconds": remaining_seconds,
                "iterations": remaining_iterations,
                "expansion_round": expansion_round,
                "max_expansions": max_expansions,
            },
            **self._memory_context(),
        })
        try:
            raw = await self._generate_json(
                prompt=prompt,
                system_name="expand_goals_system",
                timeout_seconds=timeout_seconds,
                role="expand_goals",
            )
        except Exception as exc:
            self._fail(exc)
            return []

        decision = str(raw.get("decision") or "").strip().lower()
        self.last_expansion_reason = str(raw.get("reason") or "").strip()[:400] or None
        if decision == "stop":
            return []
        if decision != "expand":
            self.last_failure_kind = "response_invalid"
            self.last_error = "확장 응답에 decision=expand|stop이 없다"
            return []

        goals = _parse_goals(raw, start=len(existing_goals))
        if not goals:
            self.last_failure_kind = "response_invalid"
            self.last_error = "decision=expand인데 유효한 새 목표 축이 없다"
            return []
        return goals

    async def propose_intent(
        self, *, objective: str, profile: DatasetProfile,
        unexplored: list[str], column_usage: dict[str, int],
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
    ) -> tuple[ResearchIntent | None, dict[str, Any]]:
        """(연구 의도, 목표 축 판정). 축 판정은 노드가 검증 후 반영한다."""
        prompt = _dumps({
            "목표": objective,
            # 직전 거부 사유가 있을 때만 프롬프트에 포함한다.
            **(
                {"직전 시도가 거부된 이유": previous_generation_error}
                if previous_generation_error else {}
            ),
            "목표 축": [
                {
                    "goal_id": g.goal_id, "question": g.question,
                    "rationale": g.rationale,
                    "success_criterion": g.success_criterion, "status": g.status,
                    "근거 수": len(g.evidence_ids),
                }
                for g in sub_goals
            ],
            "데이터": {
                "행 수": profile.row_count,
                "컬럼": _profile_digest(profile),
            },
            "아직 한 번도 분석되지 않은 변수": unexplored,
            "컬럼별 분석 사용 횟수": column_usage,
            # 다음 회차에는 질문이 아니라 검토된 발견을 전달한다.
            "지금까지 알아낸 것": [f.model_dump(mode="json") for f in prior_findings],
            "아직 정리되지 않은 발견": [
                {
                    "evidence_id": f.evidence_id,
                    "claim": f.claim,
                    "confidence": f.confidence,
                    "caveats": f.caveats,
                    "open_questions": f.open_questions,
                    "contradicts": f.contradicts,
                }
                for f in unresolved_findings
            ],
            "작업 목록": [
                item.model_dump(mode="json") for item in (required_followups or [])
            ],
            "최근 데이터 점검": recent_inspection,
            "최근 Evidence 조회": recent_evidence_lookup or [],
            "Evidence 관계": evidence_relations or [],
            "남은 실행 예산": {
                "seconds": remaining_seconds,
                "iterations": remaining_iterations,
            },
            # 거부 사유를 다음 계획에 전달한다.
            "거부된 시도": rejected_attempts or [],
            **self._memory_context(),
        })
        self._begin()
        try:
            raw = await self._generate_json(
                prompt=prompt,
                system_name="planner_system",
                timeout_seconds=timeout_seconds,
                role="planner",
            )
        except Exception as exc:
            self._fail(exc)
            return None, {}

        verdict = _goal_verdict(raw, sub_goals)

        if not str(raw.get("question") or "").strip():
            open_goal_ids = {g.goal_id for g in sub_goals if g.is_open}
            closed_ids = set(verdict["converged"]) | set(verdict["abandoned"])
            if open_goal_ids and open_goal_ids <= closed_ids:
                verdict["finished"] = True
                return None, verdict
            # 빈 question도 명시적 실패로 기록한다.
            self.last_error = "플래너 응답에 question이 비어 있다 — goal_review만 왔다"
            self.last_failure_kind = "response_invalid"
            return None, verdict
        try:
            boundary = _LLMResearchIntent.model_validate(raw)
            intent = ResearchIntent.model_validate(
                boundary.model_dump(mode="python", exclude={"intent_id"})
            )
        except Exception as exc:
            self._fail(exc, message=f"연구 의도 응답 계약 위반: {exc}")
            return None, verdict
        return intent, verdict

class LLMCodegen(_RoleService):

    async def generate(
        self, *, intent: ResearchIntent, profile: DatasetProfile,
        previous_errors: list[str], timeout_seconds: float,
    ) -> GeneratedAnalysis | None:
        payload: dict[str, Any] = {
            "연구 의도": intent.model_dump(mode="json", exclude={"intent_id"}),
            "데이터 컬럼": _profile_digest(profile),
        }
        if previous_errors:
            payload["이전 시도가 거부된 이유"] = previous_errors
        self._begin()
        try:
            raw = await self.client.generate_json(
                prompt=_dumps(payload), system=load_prompt("codegen_system"),
                timeout_seconds=timeout_seconds, role="codegen",
            )
        except Exception as exc:
            self._fail(exc)
            return None

        raw["intent_id"] = intent.intent_id
        try:
            boundary = _LLMGeneratedAnalysis.model_validate(raw)
            return GeneratedAnalysis.model_validate(
                boundary.model_dump(mode="python")
            )
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(x) for x in e['loc']) or '(root)'}={e['type']}"
                for e in exc.errors()[:3]
            )
            self._fail(exc, message=f"응답이 코드 생성 계약을 어겼다 ({detail})")
            return None

class LLMCritic(_RoleService):

    async def review(
        self, *, intent: ResearchIntent, manifest: GeneratedAnalysis,
        result_summary: dict[str, Any], timeout_seconds: float,
        prior_findings: list[FindingDigest] | None = None,
        validation_warnings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {
            "연구 의도": intent.model_dump(mode="json", exclude={"intent_id"}),
            "선언한 분석": manifest.manifest_without_code(),
            "실제 결과": result_summary,
        }
        if prior_findings:
            payload["기존 근거"] = [
                {
                    "evidence_id": f.evidence_id,
                    "question": f.question,
                    "claim": f.claim,
                    "purpose": f.purpose,
                    "protocol": f.protocol.model_dump(mode="json"),
                    "dataset_sha256": f.dataset_sha256,
                    "transform_signature": f.transform_signature,
                    "data_applicability": f.data_applicability,
                    "effect_summary": [
                        effect.model_dump(mode="json") for effect in f.effect_summary
                    ],
                    "uncertainty_summary": f.uncertainty_summary,
                    "confidence": f.confidence,
                    "caveats": f.caveats,
                }
                for f in prior_findings
            ]
        if validation_warnings:
            # 경고는 자동 거부하지 않고 Critic에 전달한다.
            payload["검증 경고"] = [
                {
                    "code": str(item.get("code", "")),
                    "category": str(item.get("category", "quality")),
                    "message": str(item.get("message", "")),
                }
                for item in validation_warnings
            ]
        self._begin()
        try:
            raw = await self.client.generate_json(
                prompt=_dumps(payload), system=load_prompt("critic_system"),
                timeout_seconds=timeout_seconds, role='critic_review',
            )
        except Exception as exc:
            # 호출 실패 사유를 보존한다.
            self._fail(exc)
            return None
        try:
            boundary = _LLMCriticReview.model_validate(raw)
            review = CriticReview.model_validate(
                boundary.model_dump(mode="python")
            )
        except ValidationError as exc:
            self._fail(exc, message=f"critic 응답 계약 위반: {exc}")
            return None
        return review.model_dump(mode="json")

    async def generate_report(
        self, *, run_summary: dict[str, Any], evidence: list[dict[str, Any]],
        rejected: list[dict[str, Any]], timeout_seconds: float,
    ) -> str | None:
        payload = {
            "실행 요약": {
                key: run_summary.get(key)
                for key in (
                    "objective",
                    "status",
                    "stop_code",
                    "stop_reason",
                    "iterations",
                    "evidence_count",
                    "rejected_count",
                    "goals",
                    "incomplete_goal_ids",
                    "pending_task_ids",
                    "unresolved_contradictions",
                    "failure_issues",
                )
            },
            "채택된 근거": [
                {
                    key: record.get(key)
                    for key in (
                        "evidence_id",
                        "goal_id",
                        "question",
                        "claim",
                        "method",
                        "columns_used",
                        "effect_summary",
                        "uncertainty_summary",
                        "confidence",
                        "caveats",
                        "purpose",
                        "parent_evidence_ids",
                        "contradicts",
                    )
                }
                for record in evidence
            ],
            "거부된 시도 요약": [
                {
                    "goal_id": r.get("goal_id"),
                    "question": r.get("question"),
                    "stage": r.get("stage"),
                    "reasons": r.get("reasons") or [],
                }
                for r in rejected
            ],
        }
        try:
            return await self.client.generate_text(
                prompt=_dumps(payload), system=load_prompt("report_system"),
                timeout_seconds=timeout_seconds, role='report',
            )
        except Exception as exc:
            # 보고서 생성 실패 사유를 남기고 finalize의 폴백을 사용한다.
            self.last_failure_kind = "response_invalid"
            self.last_error = f"{type(exc).__name__}: {exc}"[:400]
            return None

def build_services(
    *, memory: AnalysisMemory | None = None,
) -> tuple[Any, Any, Any, list[OpenAICompatClient]]:
    """(planner, codegen, critic, 열린 클라이언트들). 마지막은 종료용이다."""
    clients: list[OpenAICompatClient] = []

    def wrap(role: str, factory):
        client = build_role_client(role)
        if client is None:
            return None
        clients.append(client)
        return factory(client)

    if resolve_provider("CODEGEN") == "agy":
        # agy는 로컬 CLI 경로를 사용한다.
        from .agy_codegen import AgyCodegen

        codegen = AgyCodegen()
    else:
        codegen = wrap("CODEGEN", LLMCodegen)

    services = (
        wrap("PLANNER", lambda client: LLMPlanner(client, memory=memory)),
        codegen,
        wrap("CRITIC", LLMCritic),
    )
    if _same_endpoint_model(services[1], services[2]):
        _LOGGER.warning(
            "CODEGEN과 CRITIC이 같은 endpoint/model이다. "
            "실행은 허용하지만 독립 검토의 오류 상관 위험이 커진다."
        )
    return (*services, clients)

def _same_endpoint_model(codegen: Any, critic: Any) -> bool:
    return bool(
        codegen is not None
        and critic is not None
        and codegen.client.model == critic.client.model
        and codegen.client.base_url == critic.client.base_url
    )

def describe_services(planner: Any, codegen: Any, critic: Any) -> dict[str, str]:
    def label(service: Any) -> str:
        if service is None:
            return "none"
        return f"{type(service).__name__}:{service.client.model}"

    labels = {"planner": label(planner), "codegen": label(codegen), "critic": label(critic)}
    # 같은 endpoint/model 사용 여부를 관측 가능하게 남긴다.
    if _same_endpoint_model(codegen, critic):
        labels["warning"] = (
            f"CODEGEN과 CRITIC이 같은 endpoint/model({codegen.client.model})이다 — "
            "이중 리뷰가 자기검증이 되어 correlated error를 거르지 못한다"
        )
    return labels
