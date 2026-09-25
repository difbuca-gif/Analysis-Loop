"""실행 상태·결과를 execution_id로 보관한다. 응답이 유실돼도 되물을 수 있게.

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

# 회수는 길어야 몇 분 안에 일어난다. 이 정도면 넉넉하다.
DEFAULT_RETENTION_SECONDS = 6 * 60 * 60
# 실행 중이라던 기록이 이 시간을 넘기면 죽은 요청이 남긴 것으로 본다.
STALE_RUNNING_SECONDS = 2 * 60 * 60

# execution_id는 디렉터리 이름에 사용되므로 경로 구분자를 허용하지 않는다.
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
        # 호출부가 걸렀더라도 여기서 한 번 더 본다. 경로를 실제로 조립하는 곳이
        # 여기뿐이라, 새 호출부가 생겨도 이 검사는 우회되지 않는다.
        return self.root / validate_execution_id(execution_id)

    def claim(self, execution_id: str) -> bool:
        """이 실행을 맡는다고 선언한다(이미 맡았으면 False).

        mkdir의 원자성이 잠금이다. 프로세스 내 뮤텍스는 워커가 여러 개면 깨진다.
        """
        directory = self._dir(execution_id)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if self._is_stale(directory):
                # 오래된 RUNNING 기록은 재실행할 수 있도록 회수한다.
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
        """기록을 읽는다. None은 "그런 실행 없음"만 뜻한다.

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
        """끝나지 않은 채 너무 오래된 기록인가."""
        try:
            record = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # status.json이 없으면 디렉터리 시각으로 stale 여부를 판단한다.
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
        """임시 파일에 쓴 뒤 os.replace로 교체한다(읽는 쪽이 잘린 JSON을 안 보게)."""
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
        """보존 기간이 지난 실행 기록을 제거한다."""
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
