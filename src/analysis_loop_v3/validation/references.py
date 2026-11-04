"""Intent/Critic이 State와 schema를 올바르게 참조하는지 검사."""

from __future__ import annotations

from ..contracts import (
    CriticReview,
    DatasetProfile,
    GeneratedAnalysis,
    ResearchIntent,
    SubGoal,
)


def validate_intent_references(
    intent: ResearchIntent,
    profile: DatasetProfile,
    *,
    goals: list[SubGoal],
    evidence_ids: set[str],
) -> None:
    known_columns = set(profile.column_names())
    unknown_columns = sorted(
        (
            set(intent.candidate_columns)
            | set(intent.required_columns)
            | set(intent.excluded_columns)
        )
        - known_columns
    )
    if unknown_columns:
        raise ValueError(f"존재하지 않는 데이터 컬럼: {unknown_columns}")

    referenced_evidence = (
        set(intent.parent_evidence_ids)
        | set(intent.reuse_evidence_ids)
    )
    unknown_evidence = sorted(referenced_evidence - evidence_ids)
    if unknown_evidence:
        raise ValueError(f"존재하지 않는 evidence 참조: {unknown_evidence}")

    open_goal_ids = {goal.goal_id for goal in goals if goal.is_open}
    if intent.goal_id is not None and intent.goal_id not in open_goal_ids:
        raise ValueError(f"열려 있지 않은 goal_id: {intent.goal_id}")


def validate_critic_references(
    review: CriticReview,
    manifest: GeneratedAnalysis,
    *,
    evidence_ids: set[str],
) -> None:
    referenced = (
        set(review.contradicts)
        | set(review.revalidates)
        | set(review.supersedes)
        | set(review.reconciles)
    )
    unknown_evidence = sorted(referenced - evidence_ids)
    if unknown_evidence:
        raise ValueError(
            f"존재하지 않는 Critic evidence 참조: {unknown_evidence}"
        )

    selected_columns = set(manifest.selected_columns)
    unknown_columns = sorted({
        effect.column
        for effect in review.effect_summary
        if effect.column not in selected_columns
    })
    if unknown_columns:
        raise ValueError(
            f"분석에 사용하지 않은 효과 컬럼: {unknown_columns}"
        )
