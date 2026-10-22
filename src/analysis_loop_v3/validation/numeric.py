"""Critic 요약을 실행 결과의 특정 항목에 대조한다. 계산의 통계적 타당성 검사는 아니다."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from ..contracts import CriticReview, EffectEstimate, ResultPath


def resolve_result_path(result: Any, path: ResultPath | None) -> Any:
    """문자열은 dict 키, 정수는 list 인덱스다. 변환·음수 인덱스·빈 경로는 허용하지 않는다."""
    if not path:
        raise ValueError("원본 결과 source_path가 필요하다")
    current = result
    for part in path:
        if (isinstance(current, dict) and type(part) is str and part in current) or (
            isinstance(current, list) and type(part) is int
            and 0 <= part < len(current)
        ):
            current = current[part]
        else:
            raise ValueError(f"원본 결과 경로가 없거나 타입이 다르다: {path!r}")
    return current


def _same_value(claimed: Any, source: Any) -> bool:
    """JSON 값을 재귀 비교한다. bool=1, 숫자 문자열, NaN을 같은 수치로 인정하지 않는다."""
    if type(source) in (int, float):
        return (
            type(claimed) in (int, float)
            and (type(source) is int or math.isfinite(source))
            and (type(claimed) is int or math.isfinite(claimed))
            and claimed == source
        )
    if type(claimed) is not type(source):
        return False
    if isinstance(source, dict):
        return claimed.keys() == source.keys() and all(
            _same_value(claimed[key], value) for key, value in source.items()
        )
    if isinstance(source, list):
        return len(claimed) == len(source) and all(
            _same_value(left, right) for left, right in zip(claimed, source)
        )
    return claimed == source


def bind_review_to_result(review: CriticReview, result: Any) -> CriticReview:
    """모든 항목이 일치할 때만 원본에서 읽은 값으로 요약을 만들어 반환한다."""
    effects: list[EffectEstimate] = []
    for effect in review.effect_summary:
        source = resolve_result_path(result, effect.source_path)
        if not isinstance(source, dict):
            # 응답 계약 위반은 Critic 경계가 처리하는 ValueError로 통일한다.
            raise ValueError(  # noqa: TRY004
                f"효과 source_path는 효과 객체를 가리켜야 한다: {effect.source_path}"
            )
        for field in ("column", "metric", "value", "directional"):
            if field not in source or not _same_value(getattr(effect, field), source[field]):
                raise ValueError(f"효과 {field}가 원본 결과와 다르다: {effect.source_path}")
        if not _same_value(effect.unit, source.get("unit")):
            raise ValueError(f"효과 unit이 원본 결과와 다르다: {effect.source_path}")
        effects.append(EffectEstimate(
            **{key: source[key] for key in ("column", "metric", "value", "directional")},
            unit=source.get("unit"), source_path=list(effect.source_path),
        ))

    if review.uncertainty_summary.keys() != review.uncertainty_source_paths.keys():
        raise ValueError("uncertainty_summary의 모든 항목에 원본 경로가 필요하다")
    uncertainty: dict[str, Any] = {}
    for key, claimed in review.uncertainty_summary.items():
        source = resolve_result_path(result, review.uncertainty_source_paths[key])
        if not _same_value(claimed, source):
            raise ValueError(f"불확실성 {key}가 원본 결과와 다르다")
        uncertainty[key] = deepcopy(source)
    return review.model_copy(update={
        "effect_summary": effects, "uncertainty_summary": uncertainty,
    }, deep=True)
