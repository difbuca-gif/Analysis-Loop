"""GeneratedAnalysis Manifest와 실제 코드 접근의 결정론적 검증."""

from __future__ import annotations

import ast

from ..contracts import (
    DatasetProfile,
    GeneratedAnalysis,
    ResearchIntent,
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from .issues import issue


def check_manifest(
    manifest: GeneratedAnalysis,
    intent: ResearchIntent,
    profile: DatasetProfile,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    source = set(profile.column_names())

    shadowed = set(manifest.derived_columns) & source
    if shadowed:
        issues.append(issue(
            "derived_shadows_source",
            f"원본 컬럼을 derived_columns에 선언했다: {sorted(shadowed)}",
            ValidationSeverity.REPAIRABLE,
            columns=sorted(shadowed),
        ))

    known = manifest.available_columns(source)
    selected = set(manifest.selected_columns)

    unknown = selected - known
    if unknown:
        issues.append(issue(
            "unknown_column",
            (
                "데이터에 없고 derived_columns에도 없는 컬럼을 선언했다: "
                f"{sorted(unknown)} — 코드가 만들어낸 컬럼이면 "
                "derived_columns에 적어라."
            ),
            ValidationSeverity.REPAIRABLE,
            columns=sorted(unknown),
        ))

    excluded = selected & set(intent.excluded_columns)
    if excluded:
        issues.append(issue(
            "excluded_column_used",
            f"Intent가 제외한 컬럼을 선언했다: {sorted(excluded)}",
            ValidationSeverity.REPAIRABLE,
            columns=sorted(excluded),
        ))

    missing_required = set(intent.required_columns) - selected
    if missing_required:
        issues.append(issue(
            "missing_required_column",
            f"Intent가 필수로 지정한 컬럼이 빠졌다: {sorted(missing_required)}",
            ValidationSeverity.REPAIRABLE,
            columns=sorted(missing_required),
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
        issues.append(issue(
            "structural_column_not_selected",
            (
                "target/group_by 컬럼이 selected_columns에 없다: "
                f"{sorted(undeclared_structure)}"
            ),
            ValidationSeverity.REPAIRABLE,
            columns=sorted(undeclared_structure),
        ))

    if not manifest.expected_output_schema:
        issues.append(issue(
            "no_output_schema",
            "expected_output_schema가 비어 있다 — 결과를 대조할 기준이 없다.",
            ValidationSeverity.REPAIRABLE,
        ))

    if not (manifest.protocol.sampling_strategy or "").strip():
        issues.append(issue(
            "missing_sampling_strategy",
            "protocol.sampling_strategy가 비어 있다 — 전체 행 사용도 명시해야 한다.",
            ValidationSeverity.REPAIRABLE,
            category=ValidationCategory.INTEGRITY,
        ))

    if manifest.is_predictive():
        if not (manifest.split_strategy or "").strip():
            issues.append(issue(
                "missing_split_strategy",
                "예측 성능을 보고하는 분석은 split_strategy를 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if not (manifest.protocol.evaluation_strategy or "").strip():
            issues.append(issue(
                "missing_evaluation_strategy",
                "예측 성능을 보고하는 분석은 protocol.evaluation_strategy를 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if manifest.protocol.evaluation_design is None:
            issues.append(issue(
                "missing_evaluation_design",
                "예측 성능을 보고하는 분석은 holdout/CV/time/group 등 평가 구조를 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if manifest.prediction_task is None:
            issues.append(issue(
                "missing_prediction_task",
                "예측 분석은 classification/regression/forecasting 등 task 유형을 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if (
            manifest.time_column
            and manifest.protocol.temporal_order_preserved is not True
        ):
            issues.append(issue(
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
            issues.append(issue(
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
            issues.append(issue(
                "missing_preprocessing_fit_scope",
                "예측 분석의 변환/전처리는 fit 범위를 선언해야 한다.",
                ValidationSeverity.REPAIRABLE,
                category=ValidationCategory.INTEGRITY,
            ))
        if (
            manifest.prediction_task == "classification"
            and not (manifest.protocol.class_balance_strategy or "").strip()
        ):
            issues.append(issue(
                "class_balance_not_recorded",
                "분류 분석은 불균형 처리 여부를 기록해야 한다(보정 없음도 명시).",
                ValidationSeverity.WARNING,
                category=ValidationCategory.QUALITY,
            ))
        if manifest.protocol.interpretation_scope != "predictive":
            issues.append(issue(
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
        issues.append(issue(
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
        issues.append(issue(
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
        issues.append(issue(
            "repeated_analysis_without_seed",
            "반복 계산을 선언했지만 random_seed가 없다 — 재현 가능한 경우 seed를 기록하라.",
            ValidationSeverity.WARNING,
            category=ValidationCategory.QUALITY,
            repetitions=manifest.protocol.repetitions,
        ))

    return ValidationReport(stage="manifest", issues=issues)


def extract_referenced_columns(
    code: str,
    known_columns: set[str],
) -> set[str]:
    """코드의 문자열/attribute에서 알려진 컬럼 접근을 보수적으로 추출한다."""
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
        issues.append(issue(
            "undeclared_column_access",
            (
                "Manifest에 없는 컬럼을 코드가 참조한다: "
                f"{sorted(undeclared)} — 선언과 실제가 다르면 결과를 신뢰할 수 없다."
            ),
            ValidationSeverity.REPAIRABLE,
            columns=sorted(undeclared),
        ))

    unused = declared - actual
    if unused:
        issues.append(issue(
            "declared_but_unused",
            f"선언했지만 코드에서 보이지 않는 컬럼: {sorted(unused)}",
            ValidationSeverity.WARNING,
            category=ValidationCategory.QUALITY,
            columns=sorted(unused),
        ))

    return ValidationReport(stage="code", issues=issues)
