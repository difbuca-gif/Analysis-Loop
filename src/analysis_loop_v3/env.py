"""프로젝트 루트의 .env를 환경변수로 올린다(override=False, 부재는 정상, 명시 호출).

컨테이너·CI 주입값이 로컬 파일보다 우선이어야 배포 환경에 로컬 값이 새지 않는다.
여기 올린 값은 ENV_ALLOWLIST 때문에 샌드박스 자식에게 전달되지 않는다.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_env(*, root: Path | None = None) -> bool:
    env_path = (root or _PROJECT_ROOT) / ".env"
    if not env_path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(env_path, override=False))
