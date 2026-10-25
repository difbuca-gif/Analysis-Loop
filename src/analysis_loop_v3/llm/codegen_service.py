"""Codegen 역할 LLM 서비스."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..contracts import DatasetProfile, GeneratedAnalysis, ResearchIntent
from ..runtime_protocols import ServiceResult
from .service_common import (
    LLMGeneratedAnalysis,
    RoleService,
    dumps,
    load_prompt,
    profile_digest,
)


class LLMCodegen(RoleService):
    async def generate(
        self,
        *,
        intent: ResearchIntent,
        profile: DatasetProfile,
        previous_errors: list[str],
        timeout_seconds: float,
    ) -> ServiceResult[GeneratedAnalysis]:
        payload: dict[str, Any] = {
            "연구 의도": intent.model_dump(mode="json", exclude={"intent_id"}),
            "데이터 컬럼": profile_digest(profile),
        }
        if previous_errors:
            payload["이전 시도가 거부된 이유"] = previous_errors

        try:
            raw = await self.client.generate_json(
                prompt=dumps(payload),
                system=load_prompt("codegen_system"),
                timeout_seconds=timeout_seconds,
                role="codegen",
            )
        except Exception as exc:
            return self._fail(exc)

        raw["intent_id"] = intent.intent_id
        try:
            boundary = LLMGeneratedAnalysis.model_validate(raw)
            generated = GeneratedAnalysis.model_validate(boundary.model_dump(mode="python"))
            return ServiceResult(value=generated)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(x) for x in error['loc']) or '(root)'}={error['type']}"
                for error in exc.errors()[:3]
            )
            return self._fail(
                exc,
                message=f"응답이 코드 생성 계약을 어겼다 ({detail})",
            )
