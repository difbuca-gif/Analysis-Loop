"""주소 판정. 인증 정책이 "루프백인가"로 갈리므로 한 곳에 둔다."""

from __future__ import annotations

import ipaddress

_LOOPBACK_NAMES = {"localhost", "::1"}


def is_loopback(host: str | None) -> bool:
    """루프백 주소인가. 이름을 모르면(None, 빈 문자열) 아니라고 본다.

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
