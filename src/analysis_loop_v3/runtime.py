"""한 graph run에 주입되는 RuntimeDeps와 compatibility exports.

Artifact 저장은 artifacts.py, 서비스 인터페이스는 runtime_protocols.py가 담당한다.
기존 `from analysis_loop_v3.runtime import ArtifactStore, SandboxResult, ...` 경로는
이 모듈에서 re-export해 유지한다.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .runtime_protocols import (
    CodegenService,
    CriticService,
    PlannerService,
    SandboxResult,
    SandboxService,
)


@dataclass(slots=True)
class RuntimeDeps:
    """Workflow State와 분리해서 Node에 주입하는 실행 의존성."""

    artifacts: ArtifactStore
    sandbox: SandboxService
    planner: PlannerService | None = None
    codegen: CodegenService | None = None
    critic: CriticService | None = None
    workdir_root: Path = field(default_factory=lambda: Path("runs"))
    events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=500)
    )
    events_log_path: Path | None = None

    def event(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        iteration: int | None = None,
    ) -> None:
        """관측 이벤트를 메모리와 선택적 JSONL log에 기록한다.

        파일 로그 실패는 분석 실행 실패로 전파하지 않는다.
        """
        record = {
            **(payload or {}),
            "event": name,
            "emitted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "run_id": run_id,
            "iteration": iteration,
        }
        self.events.append(record)

        if self.events_log_path is None:
            return

        try:
            self.events_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass


__all__ = [
    "ArtifactStore",
    "PlannerService",
    "CodegenService",
    "CriticService",
    "SandboxService",
    "SandboxResult",
    "RuntimeDeps",
]
