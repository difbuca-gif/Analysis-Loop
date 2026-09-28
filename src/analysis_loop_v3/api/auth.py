"""Control API 인증.

판정을 기동 스크립트가 아니라 **요청마다** 한다. 기동 인자에만 두면
`uvicorn analysis_loop_v3.api.app:app --host 0.0.0.0`으로 띄웠을 때 통째로
우회된다. 이 API는 서버가 읽을 수 있는 파일로 분석을 시작하고 생성 코드와
결과를 돌려주므로, 열린 채로 LAN에 나가면 안 된다.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, Request

from ..net import is_loopback


def _supplied_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else authorization.strip()


def require_api_token(
    request: Request, authorization: str | None = Header(default=None),
) -> None:
    """`Authorization: Bearer <API_TOKEN>`.

    토큰이 설정돼 있으면 무조건 맞아야 한다. 설정돼 있지 않으면 루프백에서 온
    요청만 받는다. 토큰 없이 개발하는 경우를 막지 않으면서, 설정을 빠뜨린 채
    LAN에 노출되는 것은 막는다.
    """
    expected = os.environ.get("API_TOKEN")
    if expected:
        supplied = _supplied_token(authorization)
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="API 토큰이 유효하지 않다")
        return

    client = request.client.host if request.client else None
    if not is_loopback(client):
        raise HTTPException(
            status_code=401,
            detail=(
                "API_TOKEN이 설정되지 않아 원격 요청을 받지 않는다. "
                ".env에 API_TOKEN을 넣거나 루프백에서만 호출하라."
            ),
        )
