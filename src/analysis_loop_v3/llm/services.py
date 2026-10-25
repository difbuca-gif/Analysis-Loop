"""역할별 LLM service factory와 compatibility exports.

Planner/Codegen/Critic 구현은 역할별 모듈에 분리한다. 기존
`analysis_loop_v3.llm.services` import 경로는 그대로 유지한다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from ..runtime_protocols import CodegenService, CriticService, PlannerService
from .client import OpenAICompatClient
from .codegen_service import LLMCodegen
from .critic_service import LLMCritic
from .memory import AnalysisMemory
from .planner_service import LLMPlanner
from .service_common import (
    VALID_PROVIDERS,
    _dumps,
    _goal_verdict,
    _LLMCriticReview,
    _LLMGeneratedAnalysis,
    _LLMResearchIntent,
    _parse_goals,
    _profile_digest,
    _role_env,
    _RoleService,
    build_role_client,
    load_prompt,
    resolve_provider,
)

_LOGGER = logging.getLogger(__name__)
_ServiceT = TypeVar("_ServiceT")


def build_services(
    *,
    memory: AnalysisMemory | None = None,
) -> tuple[
    PlannerService | None, CodegenService | None, CriticService | None, list[OpenAICompatClient]
]:
    """Planner/Codegen/Critic 서비스와 종료 대상 client를 구성한다."""
    clients: list[OpenAICompatClient] = []

    def wrap(role: str, factory: Callable[[OpenAICompatClient], _ServiceT]) -> _ServiceT | None:
        client = build_role_client(role)
        if client is None:
            return None
        clients.append(client)
        return factory(client)

    codegen: CodegenService | None
    if resolve_provider("CODEGEN") == "agy":
        from .agy_codegen import AgyCodegen

        codegen = AgyCodegen()
    else:
        codegen = wrap("CODEGEN", LLMCodegen)

    services = (
        wrap(
            "PLANNER",
            lambda client: LLMPlanner(client, memory=memory),
        ),
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
        and hasattr(codegen, "client")
        and hasattr(critic, "client")
        and codegen.client.model == critic.client.model
        and codegen.client.base_url == critic.client.base_url
    )


def describe_services(
    planner: Any,
    codegen: Any,
    critic: Any,
) -> dict[str, str]:
    def label(service: Any) -> str:
        if service is None:
            return "none"
        client = getattr(service, "client", None)
        model = getattr(client, "model", "local")
        return f"{type(service).__name__}:{model}"

    labels = {
        "planner": label(planner),
        "codegen": label(codegen),
        "critic": label(critic),
    }
    if _same_endpoint_model(codegen, critic):
        labels["warning"] = (
            f"CODEGEN과 CRITIC이 같은 endpoint/model({codegen.client.model})이다 — "
            "이중 리뷰가 자기검증이 되어 correlated error를 거르지 못한다"
        )
    return labels


__all__ = [
    "VALID_PROVIDERS",
    "LLMPlanner",
    "LLMCodegen",
    "LLMCritic",
    "load_prompt",
    "resolve_provider",
    "build_role_client",
    "build_services",
    "describe_services",
    # 기존 private helper import도 refactor 기간에는 유지한다.
    "_LLMResearchIntent",
    "_LLMGeneratedAnalysis",
    "_LLMCriticReview",
    "_RoleService",
    "_role_env",
    "_profile_digest",
    "_dumps",
    "_parse_goals",
    "_goal_verdict",
    "_same_endpoint_model",
]
