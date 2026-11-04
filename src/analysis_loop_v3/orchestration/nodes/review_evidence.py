"""Evidence 생성, 캐시, relation 갱신 helper."""

from __future__ import annotations

from typing import Any

from ...contracts import (
    ArtifactRef,
    CriticReview,
    DataLineage,
    EvidenceRecord,
    EvidenceRelation,
    GeneratedAnalysis,
    ResearchIntent,
)
from ...runtime import RuntimeDeps
from ...state import (
    AnalysisGraphState,
    get_agenda,
    get_evidence,
    get_evidence_relations,
    get_profile,
    get_sub_goals,
)
from ._shared import _event


def build_evidence(
    state: AnalysisGraphState,
    *,
    intent: ResearchIntent,
    review: CriticReview,
    generated: GeneratedAnalysis,
    result_ref: ArtifactRef,
    program_ref: ArtifactRef | None,
    dataset_ref: ArtifactRef,
    profile: Any,
) -> EvidenceRecord:
    """검증된 실행과 CriticReview를 불변 EvidenceRecord로 변환한다."""
    warning_caveats = [
        f"[검증 경고:{item.get('code', 'unknown')!s}] {item.get('message', '')!s}"
        for item in state.get("validation_warnings", [])
    ]
    return EvidenceRecord(
        intent_id=intent.intent_id,
        question=intent.question,
        claim=review.claim.strip(),
        result_ref=result_ref,
        program_ref=program_ref,
        figure_refs=state.get("figure_refs") or [],
        method=generated.method,
        protocol=generated.protocol,
        data_lineage=DataLineage.from_analysis(
            dataset_ref=dataset_ref,
            profile=profile,
            manifest=generated,
        ),
        columns_used=generated.selected_columns,
        caveats=list(dict.fromkeys([*review.caveats, *warning_caveats])),
        effect_summary=review.effect_summary,
        uncertainty_summary=review.uncertainty_summary,
        uncertainty_source_paths=review.uncertainty_source_paths,
        confidence=review.confidence,
        open_questions=review.open_questions,
        contradicts=review.contradicts,
        purpose=intent.purpose,
        parent_evidence_ids=intent.parent_evidence_ids,
        goal_id=intent.goal_id,
    )


def cache_program(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    *,
    intent: ResearchIntent,
    generated: GeneratedAnalysis,
    program_ref: ArtifactRef,
) -> None:
    """채택된 Evidence의 프로그램만 재사용 캐시에 저장한다."""
    profile = get_profile(state)
    assert profile is not None
    cached = deps.artifacts.save_validated_program(
        schema_fingerprint=profile.schema_fingerprint,
        intent_reuse_signature=intent.reuse_signature(),
        code_ref=program_ref,
        manifest=generated.manifest_without_code(),
    )
    _event(state, deps, "PROGRAM_CACHED", {"cache_key": cached.cache_key})


def goals_with_evidence(
    state: AnalysisGraphState,
    goal_id: str,
    evidence_id: str,
) -> list[dict[str, Any]] | None:
    """Evidence를 SubGoal에 연결한다."""
    goals = [
        g.model_copy(update={"evidence_ids": [*g.evidence_ids, evidence_id]})
        if g.goal_id == goal_id
        else g
        for g in get_sub_goals(state)
    ]
    return [g.model_dump(mode="json") for g in goals] if goals else None


def revalidation_changed_conditions(
    state: AnalysisGraphState,
    *,
    source_ids: list[str],
    evidence: EvidenceRecord,
) -> list[str]:
    """독립 재검증으로 인정할 실행 조건 변경 여부를 확인한다."""
    by_id = {item.evidence_id: item for item in get_evidence(state)}
    changed: set[str] = set()
    for source_id in source_ids:
        source = by_id.get(source_id)
        if source is None:
            continue
        if source.method != evidence.method:
            changed.add("method")
        if source.protocol != evidence.protocol:
            changed.add("protocol")
        if source.data_lineage.dataset_sha256 != evidence.data_lineage.dataset_sha256:
            changed.add("dataset")
        if (
            source.data_lineage.transform_signature
            != evidence.data_lineage.transform_signature
        ):
            changed.add("transform")
    return sorted(changed)


def relations_after_commit(
    state: AnalysisGraphState,
    *,
    intent: ResearchIntent,
    review: CriticReview,
    evidence: EvidenceRecord,
) -> list[EvidenceRelation]:
    """Critic/Intent가 명시한 Evidence 관계만 추가한다."""
    existing = {
        (item.source_evidence_id, item.target_evidence_id, item.kind)
        for item in get_evidence_relations(state)
    }
    candidates: list[tuple[str, str]] = []
    candidates.extend(("derived_from", target) for target in intent.parent_evidence_ids)
    candidates.extend(("contradicts", target) for target in review.contradicts)

    allowed_revalidates = set(review.revalidates)
    if intent.task_id:
        task = next(
            (item for item in get_agenda(state) if item.task_id == intent.task_id),
            None,
        )
        if (
            task is not None
            and task.kind == "revalidate"
            and not revalidation_changed_conditions(
                state,
                source_ids=task.source_evidence_ids,
                evidence=evidence,
            )
        ):
            allowed_revalidates.difference_update(task.source_evidence_ids)

    candidates.extend(("revalidates", target) for target in allowed_revalidates)
    candidates.extend(("supersedes", target) for target in review.supersedes)
    candidates.extend(("reconciles", target) for target in review.reconciles)

    created: list[EvidenceRelation] = []
    for kind, target in candidates:
        key = (evidence.evidence_id, target, kind)
        if target == evidence.evidence_id or key in existing:
            continue
        relation = EvidenceRelation(
            source_evidence_id=evidence.evidence_id,
            target_evidence_id=target,
            kind=kind,
        )
        created.append(relation)
        existing.add(key)
    return created


# 기존 review.py 내부 helper 이름 호환.
_build_evidence = build_evidence
_cache_program = cache_program
_goals_with_evidence = goals_with_evidence
_revalidation_changed_conditions = revalidation_changed_conditions
_relations_after_commit = relations_after_commit
