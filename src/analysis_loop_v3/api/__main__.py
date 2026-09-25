"""`python -m analysis_loop_v3.api`로 uvicorn을 구동한다."""

from __future__ import annotations

import os

from ..env import load_project_env


def main() -> None:
    load_project_env()
    import uvicorn

    uvicorn.run(
        "analysis_loop_v3.api.app:app",
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8100")),
        reload=False,
    )


if __name__ == "__main__":
    main()
