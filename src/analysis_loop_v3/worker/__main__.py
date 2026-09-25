"""`python -m analysis_loop_v3.worker`로 Compute Worker uvicorn을 구동한다."""

from __future__ import annotations

import ipaddress
import os

from ..env import load_project_env

_LOOPBACK_NAMES = {"localhost"}

def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

def main() -> None:
    load_project_env()
    host = os.environ.get("WORKER_HOST", "127.0.0.1")

    # 외부 인터페이스로 열 때는 인증 토큰을 강제한다.
    if not _is_loopback(host) and not os.environ.get("WORKER_TOKEN"):
        raise SystemExit(
            f"WORKER_HOST={host}로 노출하려면 WORKER_TOKEN이 필요하다. "
            "토큰을 설정하거나 WORKER_HOST=127.0.0.1로 두어라."
        )

    import uvicorn

    uvicorn.run(
        "analysis_loop_v3.worker.app:app",
        host=host,
        port=int(os.environ.get("WORKER_PORT", "8200")),
        reload=False,
    )

if __name__ == "__main__":
    main()
