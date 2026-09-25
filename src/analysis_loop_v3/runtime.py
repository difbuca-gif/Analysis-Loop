"""상태를 소유하지 않는 런타임 서비스와 Artifact 저장소."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .contracts import (
    MAX_RESULT_BYTES,
    AgendaTask,
    ArtifactKind,
    ArtifactRef,
    DatasetProfile,
    FindingDigest,
    GeneratedAnalysis,
    ResearchIntent,
    SubGoal,
    ValidatedProgram,
)

# ArtifactStore

class ArtifactStore:
    """큰 값을 파일로 보관하고 ArtifactRef만 돌려준다.

    sha256을 항상 계산한다. 재개 시 "체크포인트가 가리키는 그 파일이 맞는가"를
    확인할 수 있어야 하기 때문이다. 경로만으로는 알 수 없다.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.program_cache_dir = self.root / "program_cache"
        self.program_cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str, suffix: str) -> Path:
        path = (self.root / f"{artifact_id}{suffix}").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"artifact 경로가 저장소 밖을 가리킨다: {artifact_id!r}")
        return path

    def _checked_ref_path(self, ref: ArtifactRef) -> Path:
        path = Path(ref.path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(
                f"artifact {ref.artifact_id}가 저장소 밖을 가리킨다: {path}"
            )
        return path

    def put_bytes(
        self, data: bytes, *, kind: ArtifactKind, suffix: str = ".bin",
        summary: dict[str, Any] | None = None, artifact_id: str | None = None,
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
            artifact_id=artifact_id, kind=kind, path=str(path),
            sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
            summary=summary or {},
        )

    def put_text(
        self, text: str, *, kind: ArtifactKind, suffix: str = ".txt",
        summary: dict[str, Any] | None = None, artifact_id: str | None = None,
    ) -> ArtifactRef:
        return self.put_bytes(
            text.encode("utf-8"), kind=kind, suffix=suffix,
            summary=summary, artifact_id=artifact_id
        )

    def put_json(
        self, payload: Any, *, kind: ArtifactKind,
        summary: dict[str, Any] | None = None, artifact_id: str | None = None,
    ) -> ArtifactRef:
        text = json.dumps(
            payload, ensure_ascii=False, indent=2, default=str, allow_nan=False,
        )
        return self.put_json_text(text, kind=kind, summary=summary, artifact_id=artifact_id)

    def put_json_text(
        self, text: str, *, kind: ArtifactKind,
        summary: dict[str, Any] | None = None, artifact_id: str | None = None,
    ) -> ArtifactRef:
        return self.put_text(
            text, kind=kind, suffix=".json", summary=summary, artifact_id=artifact_id
        )

    def put_file(
        self, source: Path | str, *, kind: ArtifactKind,
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
            artifact_id=artifact_id, kind=kind, path=str(path),
            sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
            summary=summary or {},
        )

    def read_text(self, ref: ArtifactRef) -> str:
        path = self._checked_ref_path(ref)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != ref.sha256:
            # artifact 변조 또는 불일치를 즉시 거부한다.
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
    def program_cache_key(schema_fingerprint: str, intent_reuse_signature: str) -> str:
        """데이터 스키마와 분석 요청 전체의 서명으로 프로그램을 찾는다."""
        payload = f"{schema_fingerprint}\n{intent_reuse_signature}".encode()
        return hashlib.sha256(payload).hexdigest()

    def load_validated_program(
        self, *, schema_fingerprint: str, intent_reuse_signature: str,
    ) -> ValidatedProgram | None:
        cache_key = self.program_cache_key(schema_fingerprint, intent_reuse_signature)
        path = self.program_cache_dir / f"{cache_key}.json"
        if not path.is_file():
            return None
        try:
            cached = ValidatedProgram.model_validate(json.loads(path.read_text(encoding="utf-8")))
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
        cache_key = self.program_cache_key(schema_fingerprint, intent_reuse_signature)
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
            json.dumps(cached.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return cached

# 서비스 프로토콜

class PlannerService(Protocol):
    """메인 모델. 연구 질문과 다음 행동을 고르되 계산 방법 자체는 정하지 않는다."""

    async def propose_intent(
        self,
        *,
        objective: str,
        profile: DatasetProfile,
        unexplored: list[str],
        column_usage: dict[str, int],
        prior_findings: list[FindingDigest],
        unresolved_findings: list[FindingDigest],
        sub_goals: list[SubGoal],
        timeout_seconds: float,
        required_followups: list[AgendaTask] | None = None,
        recent_inspection: dict[str, Any] | None = None,
        recent_evidence_lookup: list[dict[str, Any]] | None = None,
        evidence_relations: list[dict[str, Any]] | None = None,
        remaining_seconds: float | None = None,
        remaining_iterations: int | None = None,
        rejected_attempts: list[dict[str, Any]] | None = None,
        previous_generation_error: str | None = None,
    ) -> tuple[ResearchIntent | None, dict[str, Any]]: ...

class CodegenService(Protocol):
    """코드생성 API. 코드와 Manifest를 함께 만든다."""

    async def generate(
        self,
        *,
        intent: ResearchIntent,
        profile: DatasetProfile,
        previous_errors: list[str],
        timeout_seconds: float,
    ) -> GeneratedAnalysis | None: ...

class CriticService(Protocol):
    """의미 검증. codegen과 다른 endpoint/model을 권장한다.

    결정론적 검증이 이미 실패한 결과를 Critic이 뒤집어 채택할 권한은 없다.
    호출 자체가 결정론적 검증 통과 이후에만 일어난다(nodes/ 참조).
    """

    async def review(
        self,
        *,
        intent: ResearchIntent,
        manifest: GeneratedAnalysis,
        result_summary: dict[str, Any],
        timeout_seconds: float,
        prior_findings: list[FindingDigest] | None = None,
        validation_warnings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None: ...

class SandboxService(Protocol):
    """생성 코드 실행기(execution/sandbox.py, execution/remote_sandbox.py).

    execution_id는 원격 구현이 재시도·회수에 쓴다. 로컬 구현은 무시한다.
    """

    async def run(
        self,
        *,
        code: str,
        dataset_path: str,
        workdir: Path,
        timeout_seconds: float,
        execution_id: str | None = None,
    ) -> SandboxResult: ...

@dataclass(slots=True)
class SandboxResult:
    ok: bool
    result_path: Path | None
    stdout: str
    stderr: str
    duration_seconds: float
    # 실행 거부 사유가 있으면 결과를 채택하지 않는다.
    refused_reason: str | None = None

# 묶음

@dataclass(slots=True)
class RuntimeDeps:
    artifacts: ArtifactStore
    sandbox: SandboxService
    planner: PlannerService | None = None
    codegen: CodegenService | None = None
    critic: CriticService | None = None
    # 실행별 작업 디렉터리 루트.
    workdir_root: Path = field(default_factory=lambda: Path("runs"))
    # 관측용 인메모리 이벤트.
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))
    # 영속 이벤트 로그.
    events_log_path: Path | None = None

    def event(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        iteration: int | None = None,
    ) -> None:
        record = {
            **(payload or {}),
            "event": name,
            "emitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "run_id": run_id,
            "iteration": iteration,
        }
        self.events.append(record)
        if self.events_log_path is not None:
            try:
                self.events_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.events_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                # 로그 실패는 분석 실행을 막지 않는다.
                pass
