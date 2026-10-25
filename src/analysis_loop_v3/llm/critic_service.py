"""Critic, report generation, report review LLM service."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..contracts import (
    CriticReview,
    FindingDigest,
    GeneratedAnalysis,
    ReportSemanticReview,
    ResearchIntent,
)
from .service_common import (
    LLMCriticReview,
    RoleService,
    dumps,
    load_prompt,
)


class LLMCritic(RoleService):
    async def review(
        self,
        *,
        intent: ResearchIntent,
        manifest: GeneratedAnalysis,
        result_summary: dict[str, Any],
        timeout_seconds: float,
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
                    "evidence_id": finding.evidence_id,
                    "question": finding.question,
                    "claim": finding.claim,
                    "purpose": finding.purpose,
                    "protocol": finding.protocol.model_dump(mode="json"),
                    "dataset_sha256": finding.dataset_sha256,
                    "transform_signature": finding.transform_signature,
                    "data_applicability": finding.data_applicability,
                    "effect_summary": [
                        effect.model_dump(mode="json")
                        for effect in finding.effect_summary
                    ],
                    "uncertainty_summary": finding.uncertainty_summary,
                    "confidence": finding.confidence,
                    "caveats": finding.caveats,
                }
                for finding in prior_findings
            ]
        if validation_warnings:
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
                prompt=dumps(payload),
                system=load_prompt("critic_system"),
                timeout_seconds=timeout_seconds,
                role="critic_review",
            )
        except Exception as exc:
            self._fail(exc)
            return None

        try:
            boundary = LLMCriticReview.model_validate(raw)
            review = CriticReview.model_validate(
                boundary.model_dump(mode="python")
            )
        except ValidationError as exc:
            self._fail(
                exc,
                message=f"critic 응답 계약 위반: {exc}",
            )
            return None
        return review.model_dump(mode="json")

    async def generate_report(
        self,
        *,
        run_summary: dict[str, Any],
        evidence: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        timeout_seconds: float,
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
                        "protocol",
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
                    "goal_id": record.get("goal_id"),
                    "question": record.get("question"),
                    "stage": record.get("stage"),
                    "reasons": record.get("reasons") or [],
                }
                for record in rejected
            ],
        }
        try:
            return await self.client.generate_text(
                prompt=dumps(payload),
                system=load_prompt("report_system"),
                timeout_seconds=timeout_seconds,
                role="report",
            )
        except Exception as exc:
            self.last_failure_kind = "response_invalid"
            self.last_error = f"{type(exc).__name__}: {exc}"[:400]
            return None

    async def review_report(
        self,
        *,
        markdown: str,
        run_summary: dict[str, Any],
        evidence: list[dict[str, Any]],
        results: dict[str, Any],
        dataset_profile: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> dict[str, Any] | None:
        """완성된 보고서를 별도 요청으로 의미 검토한다."""
        payload = {
            "검토할 보고서": markdown,
            "실행 상태": {
                key: run_summary.get(key)
                for key in (
                    "objective",
                    "status",
                    "stop_code",
                    "stop_reason",
                    "goals",
                    "incomplete_goal_ids",
                    "pending_task_ids",
                    "unresolved_contradictions",
                )
            },
            "데이터 구조": dataset_profile,
            "인용 근거": [
                {
                    key: record.get(key)
                    for key in (
                        "evidence_id",
                        "question",
                        "claim",
                        "method",
                        "protocol",
                        "columns_used",
                        "effect_summary",
                        "uncertainty_summary",
                        "confidence",
                        "caveats",
                        "open_questions",
                        "contradicts",
                    )
                }
                for record in evidence
            ],
            "인용 근거의 원본 결과": results,
        }
        self._begin()
        try:
            raw = await self.client.generate_json(
                prompt=dumps(payload),
                system=load_prompt("report_review_system"),
                timeout_seconds=timeout_seconds,
                role="report_review",
            )
            return ReportSemanticReview.model_validate(raw).model_dump(mode="json")
        except Exception as exc:
            self._fail(
                exc,
                message=(
                    f"보고서 의미 검토 실패: {type(exc).__name__}: {exc}"[:400]
                ),
            )
            return None
