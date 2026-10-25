"""결정론적 validation compatibility facade.

실제 구현은 책임별 모듈로 분리한다.

- references.py: Intent/Critic 참조 무결성
- manifest_checks.py: Manifest 및 코드 접근 검증
- result_checks.py: 실행 Result 계약 검증
- issues.py: ValidationIssue 생성

기존 `validation.checks` import 경로는 그대로 유지한다.
"""

from .issues import _issue, issue
from .manifest_checks import (
    check_code_matches_manifest,
    check_manifest,
    extract_referenced_columns,
)
from .references import (
    validate_critic_references,
    validate_intent_references,
)
from .result_checks import (
    _non_finite_numbers,
    _schema_type_of,
    check_result,
    non_finite_numbers,
    schema_type_of,
)

__all__ = [
    "validate_intent_references",
    "validate_critic_references",
    "check_manifest",
    "extract_referenced_columns",
    "check_code_matches_manifest",
    "check_result",
    # 기존 private/helper import surface 유지.
    "issue",
    "_issue",
    "schema_type_of",
    "_schema_type_of",
    "non_finite_numbers",
    "_non_finite_numbers",
]
