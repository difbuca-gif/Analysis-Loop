# 이 모듈은 계약과 무결성만 검사한다. 분석 해석의 타당성은 Critic이 판정한다.

"""State 참조, Manifest, 실행 결과의 결정론적 검증."""

from __future__ import annotations

import ast
import json
import math
from typing import Any

from ..contracts import (
    MAX_RESULT_BYTES,
    CriticReview,
    DatasetProfile,
    GeneratedAnalysis,
    ResearchIntent,
    SubGoal,
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

def _issue(
    code: str,
    message: str,
    severity: ValidationSeverity,
    *,
    category: ValidationCategory = ValidationCategory.INTEGRITY,
    **detail: Any,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        severity=severity,
        category=category,
        detail=detail,
    )

def validate_intent_references(
    intent: ResearchIntent,
    profile: DatasetProfile,
    *,
    goals: list[SubGoal],
    evidence_ids: set[str],
) -> None:
    """Planner 구현과 무관하게 Intent의 State·스키마 참조를 검증한다."""
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

    referenced_evidence = set(intent.parent_evidence_ids) | set(intent.reuse_evidence_ids)
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
    """Critic 구현과 무관하게 검토 결과의 Manifest·Evidence 참조를 검증한다."""
    referenced = (
        set(review.contradicts)
        | set(review.revalidates)
        | set(review.supersedes)
        | set(review.reconciles)
    )
    unknown_evidence = sorted(referenced - evidence_ids)
    if unknown_evidence:
        raise ValueError(f"존재하지 않는 Critic evidence 참조: {unknown_evidence}")

    selected_columns = set(manifest.selected_columns)
    unknown_columns = sorted({
        effect.column for effect in review.effect_summary
        if effect.column not in selected_columns
    })
    if unknown_columns:
        raise ValueError(f"분석에 사용하지 않은 효과 컬럼: {unknown_columns}")

# 1) Manifest 자체 검증 — 선언이 Intent·스키마와 맞는가

def check_manifest(
    manifest: GeneratedAnalysis,
    intent: ResearchIntent,
    profile: DatasetProfile,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    source = set(profile.column_names())
    # 파생 컬럼이 원본 컬럼을 가장하지 못하게 한다.
    shadowed = set(manifest.derived_columns) & source
    if shadowed:
        issues.append(_issue(
            "derived_shadows_source",
            f"원본 컬럼을 derived_columns에 선언했다: {sorted(shadowed)}",
            ValidationSeverity.REPAIRABLE, columns=sorted(shadowed),
        ))
    known = manifest.available_columns(source)
    selected = set(manifest.selected_columns)

    unknown = selected - known
    if unknown:
        # 데이터에 없는 컬럼 선언은 실행 전에 거부한다.
        issues.append(_issue(
            "unknown_column",
            f"데이터에 없고 derived_columns에도 없는 컬럼을 선언했다: {sorted(unknown)} — "
            "코드가 만들어낸 컬럼이면 derived_columns에 적어라.",
            ValidationSeverity.REPAIRABLE, columns=sorted(unknown),
        ))

    excluded = selected & set(intent.excluded_columns)
    if excluded:
        issues.append(_issue(
            "excluded_column_used",
            f"Intent가 제외한 컬럼을 선언했다: {sorted(excluded)}",
            ValidationSeverity.REPAIRABLE, columns=sorted(excluded),
        ))

    missing_required = set(intent.required_columns) - selected
    if missing_required:
        issues.append(_issue(
            "missing_required_column",
            f"Intent가 필수로 지정한 컬럼이 빠졌다: {sorted(missing_required)}",
            ValidationSeverity.REPAIRABLE, columns=sorted(missing_required),
        ))

    structural_columns = (
        set(manifest.group_by)
        | set(manifest.predictor_columns())
        | set(manifest.split_unit_columns)
    )
    if manifest.target:
        structural_columns.add(manifest.target)
    if manifest.time_column:
        structural_columns.add(manifest.time_column)
    undeclared_structure = structural_columns - selected
    if undeclared_structure:
        issues.append(_issue(
            "structural_column_not_selected",
            f"target/group_by 컬럼이 selected_columns에 없다: {sorted(undeclared_structure)}",
            ValidationSeverity.REPAIRABLE,
            columns=sorted(undeclared_structure),
        ))

    if not manifest.expected_output_schema:
        issues.append(_issue(
            "no_output_schema",
            "expected_output_schema가 비어 있다 — 결과를 대조할 기준이 없다.",
            ValidationSeverity.REPAIRABLE,
        ))

    # 특정 분석법을 강제하지 않되, 재현·재검증에 필요한 실행 조건은 남겨야 한다.
    if not (manifest.protocol.sampling_strategy or "").strip():
        issues.append(_issue(
            "missing_sampling_strategy",
            "protocol.sampling_strategy가 비어 있다 — 전체 행 사용도 명시해야 한다.",
            ValidationSeverity.REPAIRABLE,
            category=ValidationCategory.INTEGRITY,
        ))

    if manifest.is_predictive():
        if not (manifest.split_strategy or "").strip():
            issues.append(_issue(
                "missing_split_strategy",
                "예측 성능을 보고하는 분석은 split_strategy를 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if not (manifest.protocol.evaluation_strategy or "").strip():
            issues.append(_issue(
                "missing_evaluation_strategy",
                "예측 성능을 보고하는 분석은 protocol.evaluation_strategy를 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if manifest.protocol.evaluation_design is None:
            issues.append(_issue(
                "missing_evaluation_design",
                "예측 성능을 보고하는 분석은 holdout/CV/time/group 등 평가 구조를 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if manifest.prediction_task is None:
            issues.append(_issue(
                "missing_prediction_task",
                "예측 분석은 classification/regression/forecasting 등 task 유형을 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if manifest.time_column and manifest.protocol.temporal_order_preserved is not True:
            issues.append(_issue(
                "temporal_leakage_guard_missing",
                "time_column이 있는 예측 분석은 temporal_order_preserved=true를 명시해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
                time_column=manifest.time_column,
            ))
        if (
            manifest.split_unit_columns
            and manifest.protocol.group_isolation_preserved is not True
        ):
            issues.append(_issue(
                "group_leakage_guard_missing",
                "split_unit_columns가 있으면 train/validation 그룹 격리를 명시해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
                split_unit_columns=manifest.split_unit_columns,
            ))
        if (
            manifest.transformations
            and manifest.protocol.preprocessing_fit_scope is None
        ):
            issues.append(_issue(
                "missing_preprocessing_fit_scope",
                "예측 분석의 변환/전처리는 fit 범위를 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if (
            manifest.prediction_task == "classification"
            and not (manifest.protocol.class_balance_strategy or "").strip()
        ):
            issues.append(_issue(
                "class_balance_not_recorded",
                "분류 분석은 불균형 처리 여부를 기록해야 한다(보정 없음도 명시).",
                ValidationSeverity.WARNING,
                category=ValidationCategory.QUALITY,
            ))
        if manifest.protocol.interpretation_scope != "predictive":
            issues.append(_issue(
                "predictive_scope_mismatch",
                "model_metric 분석인데 interpretation_scope가 predictive가 아니다.",
                ValidationSeverity.WARNING,
                category=ValidationCategory.QUALITY,
                interpretation_scope=manifest.protocol.interpretation_scope,
            ))

    if (
        manifest.protocol.hypothesis_testing
        and manifest.protocol.planned_comparisons is None
    ):
        issues.append(_issue(
            "comparison_count_not_recorded",
            "가설 검정을 수행한다면 planned_comparisons를 기록해야 한다.",
            ValidationSeverity.WARNING,
            category=ValidationCategory.QUALITY,
        ))

    if (
        manifest.protocol.planned_comparisons is not None
        and manifest.protocol.planned_comparisons > 1
        and not (manifest.protocol.multiple_testing or "").strip()
    ):
        issues.append(_issue(
            "multiple_testing_decision_missing",
            "복수 가설 비교에서는 보정 여부와 방법을 multiple_testing에 명시해야 한다.",
            ValidationSeverity.REPAIRABLE,
            category=ValidationCategory.INTEGRITY,
            planned_comparisons=manifest.protocol.planned_comparisons,
        ))

    if (
        manifest.protocol.repetitions is not None
        and manifest.protocol.repetitions > 1
        and manifest.protocol.random_seed is None
    ):
        issues.append(_issue(
            "repeated_analysis_without_seed",
            "반복 계산을 선언했지만 random_seed가 없다 — 재현 가능한 경우 seed를 기록하라.",
            ValidationSeverity.WARNING,
            category=ValidationCategory.QUALITY,
            repetitions=manifest.protocol.repetitions,
        ))

    return ValidationReport(stage="manifest", issues=issues)

# 2) 코드 vs Manifest 대조 — 선언한 변수만 실제로 접근했는가

def extract_referenced_columns(code: str, known_columns: set[str]) -> set[str]:
    """선언되지 않은 컬럼 접근을 놓치지 않도록 보수적으로 추출한다."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in known_columns:
                found.add(node.value)
        elif isinstance(node, ast.Attribute) and node.attr in known_columns:
            found.add(node.attr)
    return found

def check_code_matches_manifest(
    code: str,
    manifest: GeneratedAnalysis,
    profile: DatasetProfile,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    known = manifest.available_columns(set(profile.column_names()))
    declared = set(manifest.selected_columns) | set(manifest.derived_columns)
    actual = extract_referenced_columns(code, known)

    undeclared = actual - declared
    if undeclared:
        issues.append(_issue(
            "undeclared_column_access",
            f"Manifest에 없는 컬럼을 코드가 참조한다: {sorted(undeclared)} — "
            "선언과 실제가 다르면 결과를 신뢰할 수 없다.",
            ValidationSeverity.REPAIRABLE, columns=sorted(undeclared),
        ))

    # 선언했는데 안 쓴 것은 경고에 그친다. 파생변수를 만들다 보면 생길 수 있고,
    # 이것 때문에 재생성을 돌리면 비용만 든다.
    unused = declared - actual
    if unused:
        issues.append(_issue(
            "declared_but_unused",
            f"선언했지만 코드에서 보이지 않는 컬럼: {sorted(unused)}",
            ValidationSeverity.WARNING,
            category=ValidationCategory.QUALITY,
            columns=sorted(unused),
        ))

    return ValidationReport(stage="code", issues=issues)

# 3) 결과 계약 — 직렬화·크기·스키마

def _schema_type_of(value: Any) -> str:
    if isinstance(value, dict):
        return "mapping"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"

def _non_finite_numbers(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            found.extend(_non_finite_numbers(sub, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for i, sub in enumerate(value):
            found.extend(_non_finite_numbers(sub, f"{path}[{i}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        found.append(path or "<root>")
    return found

def check_result(
    result: Any,
    manifest: GeneratedAnalysis,
    intent: ResearchIntent,
    profile: DatasetProfile | None = None,
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if not isinstance(result, dict):
        return ValidationReport(stage="result", issues=[_issue(
            "result_not_mapping",
            f"analyze(df)는 dict를 반환해야 한다(실제 {type(result).__name__}).",
            ValidationSeverity.REPAIRABLE,
        )])

    if not result:
        return ValidationReport(stage="result", issues=[_issue(
            "empty_result", "결과가 비어 있다.", ValidationSeverity.REPAIRABLE,
        )])

    # --- 크기 ---
    encoded = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        # 과도한 결과 크기는 상태와 artifact 크기를 함께 늘린다.
        issues.append(_issue(
            "result_too_large",
            f"결과가 {len(encoded):,}바이트다 — 상한 {MAX_RESULT_BYTES:,}. "
            "고카디널리티 컬럼을 raw로 그룹화했는지 확인하고 집계 수준을 올려라.",
            ValidationSeverity.REPAIRABLE,
            category=ValidationCategory.RESOURCE,
            size_bytes=len(encoded),
        ))

    non_finite = _non_finite_numbers(result)
    if non_finite:
        issues.append(_issue(
            "non_finite_number",
            f"NaN 또는 Infinity가 결과에 있다(예: {non_finite[:3]}).",
            ValidationSeverity.REPAIRABLE,
            paths=non_finite[:10],
        ))

    # --- 선언한 스키마와 대조 ---
    for key, declared_type in manifest.expected_output_schema.items():
        if key not in result:
            issues.append(_issue(
                "missing_declared_output",
                f"Manifest가 선언한 출력 키가 결과에 없다: {key}",
                ValidationSeverity.REPAIRABLE, key=key,
            ))
            continue
        actual_type = _schema_type_of(result[key])
        if actual_type != declared_type:
            issues.append(_issue(
                "output_type_mismatch",
                f"{key}의 타입이 선언({declared_type})과 다르다(실제 {actual_type}).",
                ValidationSeverity.REPAIRABLE, key=key,
            ))

    return ValidationReport(stage="result", issues=issues)
