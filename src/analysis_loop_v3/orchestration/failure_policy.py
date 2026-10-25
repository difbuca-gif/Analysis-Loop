"""외부 서비스 실패 분류에 대한 공통 정책."""

from __future__ import annotations

TERMINAL_SERVICE_FAILURES = frozenset({
    "provider_unavailable",
    "configuration",
})


def is_terminal_service_failure(kind: str | None) -> bool:
    """같은 입력으로 재시도해도 해결되지 않는 서비스 실패인지 판정한다."""
    return kind in TERMINAL_SERVICE_FAILURES
