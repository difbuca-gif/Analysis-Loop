"""외부 서비스 실패 분류에 대한 공통 정책."""

from __future__ import annotations

from typing import Literal

TERMINAL_SERVICE_FAILURES = frozenset(
    {
        "provider_unavailable",
        "configuration",
    }
)


def is_terminal_service_failure(kind: str | None) -> bool:
    """같은 입력으로 재시도해도 해결되지 않는 서비스 실패인지 판정한다."""
    return kind in TERMINAL_SERVICE_FAILURES


def service_failure_action(
    kind: str | None,
    *,
    stage: Literal["analysis", "report"] = "analysis",
) -> Literal["stop_run", "reject_attempt", "fallback_report"]:
    """분석의 필수 서비스 장애는 종료하고, 보고서 실패는 JSON으로 대체한다.

    응답 형식/계약 오류는 해당 시도만 거부한다. 이미 확보한 근거는 보존한다.
    """
    if stage == "report":
        return "fallback_report"
    return "stop_run" if is_terminal_service_failure(kind) else "reject_attempt"
