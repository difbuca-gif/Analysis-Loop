"""Antigravity(agy) CLI Codegen backend.

HTTP 대신 `agy` CLI를 서브프로세스로 부른다(--print + --json-schema로 계약 강제).
인스턴스에 last_error를 남기면 CODEGEN_CONCURRENCY 동시 호출에서 레이스가 나므로,
실패 정보는 호출별 ServiceResult로 반환한다.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import ConfigDict, ValidationError

from ..contracts import DatasetProfile, GeneratedAnalysis, ResearchIntent
from ..runtime_protocols import ServiceResult
from .services import _dumps, _profile_digest, load_prompt

DEFAULT_AGY_EXECUTABLE = str(Path.home() / "AppData" / "Local" / "agy" / "bin" / "agy.exe")


class _AgyGeneratedAnalysis(GeneratedAnalysis):
    """외부 CLI 응답용 tolerant DTO. 내부에서는 다시 strict 모델로 검증한다."""

    model_config = ConfigDict(extra="ignore")


def _codegen_schema() -> dict[str, Any]:
    """intent_id를 제외한 GeneratedAnalysis JSON schema."""
    schema = GeneratedAnalysis.model_json_schema()
    schema.get("properties", {}).pop("intent_id", None)
    schema["required"] = [name for name in schema.get("required", []) if name != "intent_id"]
    return schema


_SCHEMA = _codegen_schema()


class AgyCodegen:
    """agy CLI 기반 CodegenService 구현."""

    def __init__(self, *, executable: str | None = None) -> None:
        self.executable = executable or DEFAULT_AGY_EXECUTABLE
        # service.client.model/base_url 호환 인터페이스 제공
        self.client = SimpleNamespace(model="agy-cli", base_url="local-cli://agy")

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
            "데이터 컬럼": _profile_digest(profile),
        }
        if previous_errors:
            payload["이전 시도가 거부된 이유"] = previous_errors
        prompt = f"{load_prompt('codegen_system')}\n\n{_dumps(payload)}"

        schema_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        try:
            json.dump(_SCHEMA, schema_path, ensure_ascii=False)
            schema_path.close()

            try:
                proc = await asyncio.create_subprocess_exec(
                    self.executable,
                    "--dangerously-skip-permissions",
                    "--output-format",
                    "json",
                    "--json-schema",
                    schema_path.name,
                    "--print",
                    prompt,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as exc:
                return ServiceResult(
                    error=f"agy CLI 실행 실패: {exc}"[:400], failure_kind="configuration"
                )

            try:
                stdout, _stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ServiceResult(error="agy CLI 시간 초과", failure_kind="response_invalid")
        finally:
            Path(schema_path.name).unlink(missing_ok=True)

        if proc.returncode != 0:
            return ServiceResult(
                error=f"agy CLI 종료 코드: {proc.returncode}", failure_kind="response_invalid"
            )

        try:
            raw = json.loads(stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return ServiceResult(
                error="agy CLI 실행 또는 출력 계약 위반", failure_kind="response_invalid"
            )
        structured = raw.get("structured_output")
        if not isinstance(structured, dict):
            return ServiceResult(
                error="agy CLI 실행 또는 출력 계약 위반", failure_kind="response_invalid"
            )
        structured["intent_id"] = intent.intent_id
        try:
            boundary = _AgyGeneratedAnalysis.model_validate(structured)
            return ServiceResult(
                value=GeneratedAnalysis.model_validate(boundary.model_dump(mode="python"))
            )
        except ValidationError:
            return ServiceResult(
                error="agy CLI 실행 또는 출력 계약 위반", failure_kind="response_invalid"
            )
