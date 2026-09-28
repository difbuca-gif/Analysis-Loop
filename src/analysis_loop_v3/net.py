"""loopback 기반 주소 판정 유틸리티."""

from __future__ import annotations

import ipaddress

_LOOPBACK_NAMES = {"localhost", "::1"}


def is_loopback(host: str | None) -> bool:
    """loopback 주소 여부. 미해석 host는 False.

    판정이 애매하면 원격으로 취급해야 인증이 fail-closed가 된다.
    """
    if not host:
        return False
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
