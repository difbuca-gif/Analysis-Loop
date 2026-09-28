"""execution_id 기준 실행 상태/결과 저장소.

DB 대신 디렉터리({root}/{id}/status.json). 여기 남는 건 "방금 한 계산의 영수증"일
뿐이라 프로세스 재시작만 넘기면 충분하다.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ..contracts import ExecutionStatus

# 결과 회수 TTL
DEFAULT_RETENTION_SECONDS = 6 * 60 * 60
# RUNNING stale timeout
STALE_RUNNING_SECONDS = 2 * 60 * 60

# execution_id 경로 구분자 금지
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

class InvalidExecutionId(ValueError):
    """디렉터리 이름으로 쓸 수 없는 execution_id."""

def validate_execution_id(execution_id: str) -> str:
    if not _ID_PATTERN.fullmatch(execution_id or ""):
        raise InvalidExecutionId(
            f"execution_id는 {_ID_PATTERN.pattern} 형식이어야 한다: {execution_id!r}"
        )
    return execution_id

class ExecutionStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, execution_id: str) -> Path:
        # 경로 조립 지점에서 execution_id 재검증
        return self.root / validate_execution_id(execution_id)

    def claim(self, execution_id: str) -> bool:
        """execution_id 실행 소유권 획득. 기존 소유 시 False.

        mkdir의 원자성이 잠금이다. 프로세스 내 뮤텍스는 워커가 여러 개면 깨진다.
        """
        directory = self._dir(execution_id)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if self._is_stale(directory):
                # stale RUNNING record는 재실행 가능하도록 회수
                shutil.rmtree(directory, ignore_errors=True)
                try:
                    directory.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    return False
            else:
                return False
        self._write(execution_id, {
            "execution_id": execution_id,
            "status": ExecutionStatus.RUNNING.value,
            "started_at": time.time(),
        })
        return True

    def read(self, execution_id: str) -> dict[str, Any] | None:
        """execution record 조회. None은 미존재만 의미.

        claim의 mkdir과 _write 사이에는 status.json이 없다. 그때 None을 돌려주면
        호출부가 "기록 없음 = 다시 실행"으로 읽어 같은 분석을 두 번 돌린다.
        디렉터리 존재로 "누군가 맡았다"를 구분한다.
        """
        directory = self._dir(execution_id)
        try:
            return json.loads((directory / "status.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            if directory.is_dir():
                return {
                    "execution_id": execution_id,
                    "status": ExecutionStatus.RUNNING.value,
                }
            return None
        except NotADirectoryError:
            return None
        except json.JSONDecodeError:
            return {
                "execution_id": execution_id,
                "status": ExecutionStatus.RUNNING.value,
            }

    def finish(self, execution_id: str, payload: dict[str, Any]) -> None:
        self._write(execution_id, {
            "execution_id": execution_id,
            "finished_at": time.time(),
            **payload,
        })

    def _is_stale(self, directory: Path) -> bool:
        """RUNNING record stale 여부."""
        try:
            record = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # status.json 부재 시 디렉터리 mtime 기준 stale 판정
            try:
                return time.time() - directory.stat().st_mtime > STALE_RUNNING_SECONDS
            except OSError:
                return False
        if record.get("status") != ExecutionStatus.RUNNING.value:
            return False
        started = record.get("started_at")
        if not isinstance(started, (int, float)):
            return False
        return time.time() - started > STALE_RUNNING_SECONDS

    def _write(self, execution_id: str, payload: dict[str, Any]) -> None:
        """temp file + os.replace 기반 atomic JSON write."""
        directory = self._dir(execution_id)
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, directory / "status.json")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def sweep(self, *, retention_seconds: float = DEFAULT_RETENTION_SECONDS) -> int:
        """보존 기간 초과 execution record 정리."""
        cutoff = time.time() - retention_seconds
        removed = 0
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            try:
                if directory.stat().st_mtime >= cutoff:
                    continue
                shutil.rmtree(directory)
                removed += 1
            except OSError:
                continue  # 다른 요청이 쓰는 중일 수 있다. 청소 실패로 막지 않는다.
        return removed
