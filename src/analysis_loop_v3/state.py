"""체크포인트에 저장되는 분석 워크플로의 단일 상태."""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from .contracts import (
    SCHEMA_VERSION,
    AgendaTask,
    AnalysisCoverage,
    ArtifactRef,
    DataApplicability,
    DatasetProfile,
    EvidenceRecord,
    EvidenceRelation,
    ResearchIntent,
    RunStatus,
    StopCode,
    SubGoal,
)

def _append(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """append-only reducer. Evidence 누적용."""
    return (left or []) + (right or [])

class AnalysisGraphState(TypedDict, total=False):
    # --- 실행 식별 ---
    run_id: str
    schema_version: int
    objective: str

    # --- 입력 (참조만) ---
    dataset_ref: dict[str, Any] | None
    profile_ref: dict[str, Any] | None
    profile: dict[str, Any] | None  # DatasetProfile — 프롬프트에 매번 필요해 State에 둔다

    # SubGoal 진행 상태
    sub_goals: list[dict[str, Any]]
    # SubGoal 확장 시도 횟수
    goal_expansions: int
    # 실행 절대 deadline
    deadline_at: float

    # --- 현재 분석 사이클 ---
    intent: dict[str, Any] | None            # ResearchIntent
    # 다음 계획용 직전 거부 사유
    last_planner_error: str | None
    program_ref: dict[str, Any] | None       # ArtifactRef -> 코드 본문
    manifest: dict[str, Any] | None          # GeneratedAnalysis - code
    candidate_ref: dict[str, Any] | None     # ArtifactRef -> result.json
    figure_refs: list[dict[str, Any]] | None # list[ArtifactRef] -> figures

    # 정적/런타임 재생성 횟수
    static_repairs: int
    runtime_repairs: int
    validation_errors: list[dict[str, Any]]
    # 동일 오류 연속 발생 횟수
    error_streak: dict[str, Any]
    # non-blocking validation warning
    validation_warnings: list[dict[str, Any]]

    review: dict[str, Any] | None            # Critic 판정
    # 미해결 Agenda
    agenda: list[dict[str, Any]]
    # data inspection summary/history
    inspection_ref: dict[str, Any] | None
    inspection_summary: dict[str, Any] | None
    inspection_history: Annotated[list[dict[str, Any]], _append]
    # Evidence lookup summary
    evidence_lookup_summary: list[dict[str, Any]]
    evidence_lookup_history: Annotated[list[dict[str, Any]], _append]
    # --- 누적 지식 (추가 전용) ---
    evidence: Annotated[list[dict[str, Any]], _append]
    evidence_relations: Annotated[list[dict[str, Any]], _append]
    rejected: Annotated[list[dict[str, Any]], _append]
    # 반복 실패로 중단된 회차
    issues: Annotated[list[dict[str, Any]], _append]
    coverage: dict[str, Any]                 # AnalysisCoverage

    # --- 진행 제어 ---
    iteration: int
    max_iterations: int

    # --- 종료 ---
    status: str
    # stop_code와 stop_reason은 함께 설정
    stop_code: str | None
    stop_reason: str | None
    report_ref: dict[str, Any] | None

# 생성

def new_state(
    *,
    run_id: str,
    objective: str,
    dataset_ref: ArtifactRef,
    max_iterations: int = 25,
    time_budget_seconds: float = 3600.0,
) -> AnalysisGraphState:
    """새 run의 checkpoint-compatible 초기 State 생성.

    누적 컬렉션은 빈 값으로, 회차 임시 필드는 None/0으로 초기화해
    resume 시 누락 key에 의존하지 않도록 한다.
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations는 0보다 커야 한다")
    if time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds는 0보다 커야 한다")
    return AnalysisGraphState(
        run_id=run_id,
        schema_version=SCHEMA_VERSION,
        objective=objective,
        dataset_ref=dataset_ref.model_dump(mode="json"),
        profile_ref=None,
        profile=None,
        sub_goals=[],
        goal_expansions=0,
        deadline_at=time.time() + time_budget_seconds,
        intent=None,
        last_planner_error=None,
        program_ref=None,
        manifest=None,
        candidate_ref=None,
        figure_refs=None,
        static_repairs=0,
        runtime_repairs=0,
        validation_errors=[],
        error_streak={},
        validation_warnings=[],
        review=None,
        agenda=[],
        inspection_ref=None,
        inspection_summary=None,
        inspection_history=[],
        evidence_lookup_summary=[],
        evidence_lookup_history=[],
        evidence=[],
        evidence_relations=[],
        rejected=[],
        issues=[],
        coverage=AnalysisCoverage().model_dump(mode="json"),
        iteration=0,
        max_iterations=max_iterations,
        status=RunStatus.RUNNING.value,
        stop_code=None,
        stop_reason=None,
        report_ref=None,
    )

# 타입 있는 접근자 — State 값을 계약 모델로 복원한다.

def terminal_update(
    *, status: RunStatus, stop_code: StopCode, reason: str,
) -> dict[str, Any]:
    """status/stop_code/stop_reason 일괄 갱신."""
    reason = reason.strip()
    if not reason:
        raise ValueError("종료 사유는 비어 있을 수 없다")
    return {
        "status": status.value,
        "stop_code": stop_code.value,
        "stop_reason": reason,
    }

def cycle_reset_update() -> dict[str, Any]:
    """새 Intent 시작 전 회차 임시 state 초기화."""
    return {
        "last_planner_error": None,
        "program_ref": None,
        "manifest": None,
        "candidate_ref": None,
        "figure_refs": None,
        "review": None,
        "validation_errors": [],
        "error_streak": {},
        "validation_warnings": [],
        "static_repairs": 0,
        "runtime_repairs": 0,
    }

def get_intent(state: AnalysisGraphState) -> ResearchIntent | None:
    raw = state.get("intent")
    return ResearchIntent.model_validate(raw) if raw else None

def get_profile(state: AnalysisGraphState) -> DatasetProfile | None:
    raw = state.get("profile")
    return DatasetProfile.model_validate(raw) if raw else None

def get_sub_goals(state: AnalysisGraphState) -> list[SubGoal]:
    return [SubGoal.model_validate(g) for g in state.get("sub_goals") or []]

def all_goals_closed(state: AnalysisGraphState) -> bool:
    """SubGoal이 존재하고 모두 종료됐는지 판정."""
    goals = get_sub_goals(state)
    return bool(goals) and not any(g.is_open for g in goals)

def get_coverage(state: AnalysisGraphState) -> AnalysisCoverage:
    return AnalysisCoverage.model_validate(state.get("coverage") or {})

def get_ref(state: AnalysisGraphState, key: str) -> ArtifactRef | None:
    raw = state.get(key)
    return ArtifactRef.model_validate(raw) if raw else None

def get_evidence(state: AnalysisGraphState) -> list[EvidenceRecord]:
    return [EvidenceRecord.model_validate(e) for e in state.get("evidence") or []]

def evidence_data_applicability(
    state: AnalysisGraphState,
    evidence: EvidenceRecord,
) -> DataApplicability:
    """현재 dataset snapshot 기준 Evidence 적용 가능성 판정.

    같은 SHA-256이면 artifact_id가 달라도 동일 스냅샷이다. 스키마만 같고 내용이
    달라졌다면 과거 근거는 가설/비교 대상으로는 쓸 수 있지만 현재 결론 근거로
    재사용하지 않고 재검증해야 한다.
    """
    dataset = get_ref(state, "dataset_ref")
    profile = get_profile(state)
    if dataset is None:
        return "unknown"

    lineage = evidence.data_lineage
    if profile is not None and lineage.schema_fingerprint != profile.schema_fingerprint:
        return "incompatible_schema"
    if lineage.dataset_sha256 == dataset.sha256:
        return "exact_snapshot"
    if profile is not None and lineage.schema_fingerprint == profile.schema_fingerprint:
        return "revalidation_required"
    return "unknown"

def superseded_evidence_ids(state: AnalysisGraphState) -> set[str]:
    """관계 graph 기준 inactive Evidence ID."""
    return {
        relation.target_evidence_id
        for relation in get_evidence_relations(state)
        if relation.kind == "supersedes"
    }

def active_evidence(state: AnalysisGraphState) -> list[EvidenceRecord]:
    """superseded되지 않은 Evidence."""
    superseded = superseded_evidence_ids(state)
    return [item for item in get_evidence(state) if item.evidence_id not in superseded]

def current_evidence(state: AnalysisGraphState) -> list[EvidenceRecord]:
    """현재 snapshot에 직접 적용 가능한 active Evidence."""
    return [
        item for item in active_evidence(state)
        if evidence_data_applicability(state, item) == "exact_snapshot"
    ]

def get_agenda(state: AnalysisGraphState) -> list[AgendaTask]:
    return [AgendaTask.model_validate(item) for item in state.get("agenda") or []]

def pending_agenda_tasks(state: AnalysisGraphState) -> list[AgendaTask]:
    return [
        item for item in get_agenda(state)
        if item.status in {"pending", "in_progress"}
    ]

def get_evidence_relations(state: AnalysisGraphState) -> list[EvidenceRelation]:
    return [
        EvidenceRelation.model_validate(item)
        for item in state.get("evidence_relations") or []
    ]

def unresolved_contradictions(
    state: AnalysisGraphState,
) -> list[tuple[str, str]]:
    """미해결 contradiction pair."""
    relations = get_evidence_relations(state)
    superseded = superseded_evidence_ids(state)
    reconciles_by_source: dict[str, set[str]] = {}
    for relation in relations:
        if relation.kind == "reconciles":
            reconciles_by_source.setdefault(relation.source_evidence_id, set()).add(
                relation.target_evidence_id
            )

    unresolved: list[tuple[str, str]] = []
    for relation in relations:
        if relation.kind != "contradicts":
            continue
        pair = {relation.source_evidence_id, relation.target_evidence_id}
        if pair & superseded:
            continue
        if any(pair <= targets for targets in reconciles_by_source.values()):
            continue
        key = tuple(sorted(pair))
        if key not in unresolved:
            unresolved.append(key)
    return unresolved

def remaining_seconds(state: AnalysisGraphState) -> float:
    """남은 실행 시간(초). 음수는 예산 초과."""
    return float(state.get("deadline_at", 0.0)) - time.time()
