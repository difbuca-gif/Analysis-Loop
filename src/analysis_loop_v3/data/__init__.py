"""파일과 PostgreSQL 입력이 공유하는 준비된 데이터셋."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    path: Path
    summary: dict[str, Any]
