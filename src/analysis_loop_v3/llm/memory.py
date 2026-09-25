"""마스터 LLM이 사용하는 Wiki와 실행별 메모장."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

MemoryScope = Literal["wiki", "notebook"]
BUNDLED_WIKI_PATH = Path(__file__).parent / "wiki" / "ANALYSIS_WIKI.md"


MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "분석 Wiki 또는 현재 실행의 메모장 전체를 읽습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["wiki", "notebook"]},
                },
                "required": ["scope"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": "분석 Wiki 또는 현재 실행의 메모장 전체를 새 내용으로 저장합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["wiki", "notebook"]},
                    "content": {"type": "string"},
                    "expected_revision": {
                        "type": "string",
                        "description": "마지막으로 읽은 문서의 revision",
                    },
                },
                "required": ["scope", "content", "expected_revision"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    content: str
    revision: str


class MemoryConflictError(RuntimeError):
    """읽은 뒤 다른 실행이 같은 문서를 먼저 수정했다."""


def _revision(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """문서별 프로세스 잠금. 잠금 파일은 재사용한다."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


class AnalysisMemory:
    """경로 선택은 애플리케이션이, 문서 내용은 마스터 LLM이 소유한다."""

    def __init__(
        self,
        *,
        wiki_path: Path | str,
        notebook_path: Path | str,
        wiki_seed: str = "",
    ) -> None:
        self.wiki_path = Path(wiki_path).resolve()
        self.notebook_path = Path(notebook_path).resolve()
        self.wiki_seed = wiki_seed

    def _path(self, scope: MemoryScope) -> Path:
        if scope == "wiki":
            return self.wiki_path
        if scope == "notebook":
            return self.notebook_path
        raise ValueError(f"알 수 없는 memory scope: {scope!r}")

    @staticmethod
    def _write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    def read(self, scope: MemoryScope) -> str:
        return self.snapshot(scope).content

    def snapshot(self, scope: MemoryScope) -> MemorySnapshot:
        path = self._path(scope)
        if not path.is_file():
            with _exclusive_lock(path):
                if not path.is_file():
                    seed = self.wiki_seed if scope == "wiki" else "# 분석 메모장\n"
                    self._write(path, seed)
        content = path.read_text(encoding="utf-8")
        return MemorySnapshot(content=content, revision=_revision(content))

    def write(
        self, scope: MemoryScope, content: str, *, expected_revision: str,
    ) -> MemorySnapshot:
        path = self._path(scope)
        with _exclusive_lock(path):
            if not path.is_file():
                seed = self.wiki_seed if scope == "wiki" else "# 분석 메모장\n"
                self._write(path, seed)
            current_content = path.read_text(encoding="utf-8")
            current = MemorySnapshot(
                content=current_content,
                revision=_revision(current_content),
            )
            if current.revision != expected_revision:
                raise MemoryConflictError(
                    f"{scope}가 다른 실행에서 수정되었습니다. "
                    "read_memory로 최신 내용을 읽고 변경을 다시 반영하십시오."
                )
            self._write(path, content)
        return MemorySnapshot(content=content, revision=_revision(content))

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        scope = arguments.get("scope")
        if scope not in {"wiki", "notebook"}:
            raise ValueError(f"알 수 없는 memory scope: {scope!r}")
        if name == "read_memory":
            snapshot = self.snapshot(scope)
            return json.dumps(
                {"scope": scope, "revision": snapshot.revision, "content": snapshot.content},
                ensure_ascii=False,
            )
        if name == "write_memory":
            content = arguments.get("content")
            if not isinstance(content, str):
                raise ValueError("write_memory.content는 문자열이어야 한다")
            expected_revision = arguments.get("expected_revision")
            if not isinstance(expected_revision, str):
                raise ValueError("write_memory.expected_revision은 문자열이어야 한다")
            saved = self.write(
                scope, content, expected_revision=expected_revision,
            )
            return json.dumps(
                {"scope": scope, "revision": saved.revision, "status": "saved"},
                ensure_ascii=False,
            )
        raise ValueError(f"알 수 없는 memory tool: {name!r}")


def load_bundled_wiki() -> str:
    return BUNDLED_WIKI_PATH.read_text(encoding="utf-8")
