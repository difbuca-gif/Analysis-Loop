"""Artifact 파일 저장과 검증된 프로그램 cache."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import (
    MAX_RESULT_BYTES,
    ArtifactKind,
    ArtifactRef,
    ValidatedProgram,
)


class ArtifactStore:
    """대용량 값을 파일에 저장하고 무결성 정보가 포함된 ArtifactRef를 반환한다."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.program_cache_dir = self.root / "program_cache"
        self.program_cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str, suffix: str) -> Path:
        path = (self.root / f"{artifact_id}{suffix}").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(
                f"artifact 경로가 저장소 밖을 가리킨다: {artifact_id!r}"
            )
        return path

    def _checked_ref_path(self, ref: ArtifactRef) -> Path:
        path = Path(ref.path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(
                f"artifact {ref.artifact_id}가 저장소 밖을 가리킨다: {path}"
            )
        return path

    def put_bytes(
        self,
        data: bytes,
        *,
        kind: ArtifactKind,
        suffix: str = ".bin",
        summary: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        artifact_id = artifact_id or uuid4().hex
        path = self._path(artifact_id, suffix)
        if kind is ArtifactKind.RESULT and len(data) > MAX_RESULT_BYTES:
            raise ValueError(
                f"result artifact가 {len(data):,}바이트다 — 상한 {MAX_RESULT_BYTES:,}"
            )

        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            path=str(path),
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            summary=summary or {},
        )

    def put_text(
        self,
        text: str,
        *,
        kind: ArtifactKind,
        suffix: str = ".txt",
        summary: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        return self.put_bytes(
            text.encode("utf-8"),
            kind=kind,
            suffix=suffix,
            summary=summary,
            artifact_id=artifact_id,
        )

    def put_json(
        self,
        payload: Any,
        *,
        kind: ArtifactKind,
        summary: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        )
        return self.put_json_text(
            text,
            kind=kind,
            summary=summary,
            artifact_id=artifact_id,
        )

    def put_json_text(
        self,
        text: str,
        *,
        kind: ArtifactKind,
        summary: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        return self.put_text(
            text,
            kind=kind,
            suffix=".json",
            summary=summary,
            artifact_id=artifact_id,
        )

    def put_file(
        self,
        source: Path | str,
        *,
        kind: ArtifactKind,
        summary: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        source = Path(source)
        artifact_id = uuid4().hex
        path = self._path(artifact_id, source.suffix)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(path)
        data = path.read_bytes()
        return ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            path=str(path),
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            summary=summary or {},
        )

    def read_text(self, ref: ArtifactRef) -> str:
        path = self._checked_ref_path(ref)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != ref.sha256:
            raise ValueError(
                f"artifact {ref.artifact_id} 내용이 바뀌었다 "
                f"(기대 {ref.sha256[:12]}, 실제 {actual[:12]})"
            )
        return data.decode("utf-8")

    def read_json(self, ref: ArtifactRef) -> Any:
        return json.loads(self.read_text(ref))

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            return self._checked_ref_path(ref).is_file()
        except ValueError:
            return False

    @staticmethod
    def program_cache_key(
        schema_fingerprint: str,
        intent_reuse_signature: str,
    ) -> str:
        payload = f"{schema_fingerprint}\n{intent_reuse_signature}".encode()
        return hashlib.sha256(payload).hexdigest()

    def load_validated_program(
        self,
        *,
        schema_fingerprint: str,
        intent_reuse_signature: str,
    ) -> ValidatedProgram | None:
        cache_key = self.program_cache_key(
            schema_fingerprint,
            intent_reuse_signature,
        )
        path = self.program_cache_dir / f"{cache_key}.json"
        if not path.is_file():
            return None

        try:
            cached = ValidatedProgram.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if (
                cached.cache_key != cache_key
                or cached.schema_fingerprint != schema_fingerprint
                or cached.intent_reuse_signature != intent_reuse_signature
                or cached.code_ref.kind is not ArtifactKind.PROGRAM
            ):
                return None
            self.read_text(cached.code_ref)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return cached

    def save_validated_program(
        self,
        *,
        schema_fingerprint: str,
        intent_reuse_signature: str,
        code_ref: ArtifactRef,
        manifest: dict[str, Any],
    ) -> ValidatedProgram:
        cache_key = self.program_cache_key(
            schema_fingerprint,
            intent_reuse_signature,
        )
        cached = ValidatedProgram(
            cache_key=cache_key,
            intent_reuse_signature=intent_reuse_signature,
            code_ref=code_ref,
            schema_fingerprint=schema_fingerprint,
            manifest=manifest,
        )
        path = self.program_cache_dir / f"{cache_key}.json"
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                cached.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
        return cached
