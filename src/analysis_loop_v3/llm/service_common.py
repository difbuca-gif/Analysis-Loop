"""역할별 LLM 서비스가 공유하는 provider 설정, DTO, prompt helper."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, ValidationError

from ..contracts import (
    CriticReview,
    DatasetProfile,
    GeneratedAnalysis,
    ResearchIntent,
    SubGoal,
)
from ..runtime_protocols import FailureKind, ServiceResult
from .client import (
    LLMError,
    LLMProviderError,
    LLMTransportError,
    OpenAICompatClient,
    build_client,
)

_PROMPT_DIR = Path(__file__).parent / "prompts"

_DEFAULT_PROVIDER = {
    "PLANNER": "local",
    "CODEGEN": "nvidia",
    "CRITIC": "local",
}
_DEFAULT_MODEL = {
    "PLANNER": "Qwen/Qwen3.5-9B",
    "CODEGEN": "deepseek-ai/deepseek-v4-flash-0731",
    "CRITIC": "Qwen/Qwen3.5-9B",
}
_DEFAULT_MAX_TOKENS = {
    "PLANNER": 2048,
    "CODEGEN": 8192,
    "CRITIC": 1024,
}
VALID_PROVIDERS = ("local", "ollama", "nvidia", "agy", "none")


class LLMResearchIntent(ResearchIntent):
    """외부 응답용 tolerant DTO. 내부에서는 strict 모델로 다시 검증한다."""

    model_config = ConfigDict(extra="ignore")


class LLMGeneratedAnalysis(GeneratedAnalysis):
    model_config = ConfigDict(extra="ignore")


class LLMCriticReview(CriticReview):
    model_config = ConfigDict(extra="ignore")


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    return (_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def role_env(role: str, suffix: str) -> str | None:
    value = os.environ.get(f"{role}_LLM_{suffix}")
    return value.strip() if value and value.strip() else None


def resolve_provider(role: str) -> str:
    """미설정은 기본값을 사용하고 잘못된 명시값은 즉시 거부한다."""
    role = role.upper()
    provider = (role_env(role, "PROVIDER") or _DEFAULT_PROVIDER.get(role, "local")).lower()
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
        model=role_env(role, "MODEL") or _DEFAULT_MODEL.get(role, ""),
        base_url=role_env(role, "BASE_URL"),
    )
    if client is None:
        return None

    raw_max_tokens = role_env(role, "MAX_TOKENS")
    try:
        client.max_tokens = (
            int(raw_max_tokens)
            if raw_max_tokens
            else _DEFAULT_MAX_TOKENS.get(role, client.max_tokens)
        )
    except ValueError as exc:
        raise ValueError(f"{role}_LLM_MAX_TOKENS는 정수여야 한다") from exc
    if client.max_tokens <= 0:
        raise ValueError(f"{role}_LLM_MAX_TOKENS는 0보다 커야 한다")
    return client


def profile_digest(profile: DatasetProfile) -> list[dict[str, Any]]:
    """프롬프트용 전체 컬럼 요약."""
    return [
        {
            "name": column.name,
            "dtype": column.dtype,
            "nulls": column.null_count,
            "unique": column.unique_count,
        }
        for column in profile.columns
    ]


def dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def parse_goals(
    raw: dict[str, Any],
    *,
    start: int = 0,
) -> list[SubGoal]:
    """모델 출력에서 SubGoal을 추출하고 안정적인 goal_id를 다시 부여한다."""
    goals: list[SubGoal] = []
    for offset, item in enumerate(raw.get("sub_goals") or [], start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        goals.append(
            SubGoal(
                goal_id=f"g{start + offset}",
                question=question,
                rationale=str(item.get("rationale") or ""),
                success_criterion=str(item.get("success_criterion") or ""),
            )
        )
    return goals


def goal_verdict(
    raw: dict[str, Any],
    sub_goals: list[SubGoal],
) -> dict[str, Any]:
    """열린 SubGoal에 대한 모델 verdict만 추출한다."""
    open_ids = {goal.goal_id for goal in sub_goals if goal.is_open}
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
        bucket = {
            "converged": converged,
            "abandoned": abandoned,
        }.get(str(item.get("verdict") or "").lower())
        if bucket is None:
            continue
        bucket.append(goal_id)
        reasons.append(f"{goal_id}: {str(item.get('reason') or '')[:200]}")

    return {
        "converged": converged,
        "abandoned": abandoned,
        "closing_note": " / ".join(reasons)[:300],
        "unreviewed": sorted(open_ids - reviewed),
    }


class RoleService:
    """역할별 LLM client와 호출별 오류 분류."""

    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def _fail(
        self,
        exc: Exception,
        *,
        message: str | None = None,
    ) -> ServiceResult[Any]:
        kind: FailureKind
        if isinstance(exc, LLMTransportError):
            kind = "provider_unavailable"
        elif isinstance(exc, LLMProviderError):
            kind = "provider_unavailable" if exc.retryable else "configuration"
        elif isinstance(exc, ValidationError):
            kind = "contract"
        elif isinstance(exc, LLMError):
            kind = "response_invalid"
        else:
            kind = "response_invalid"

        if message is None:
            message = str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
        return ServiceResult(error=message[:400], failure_kind=kind)


# 기존 services.py private helper 이름과 호환용 alias.
_LLMResearchIntent = LLMResearchIntent
_LLMGeneratedAnalysis = LLMGeneratedAnalysis
_LLMCriticReview = LLMCriticReview
_role_env = role_env
_profile_digest = profile_digest
_dumps = dumps
_parse_goals = parse_goals
_goal_verdict = goal_verdict
_RoleService = RoleService
