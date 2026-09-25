"""본 분석 코드를 만들기 전에 데이터 상태를 결정론적으로 점검하는 행동."""

from __future__ import annotations

import math
from typing import Any

from pandas.api.types import is_numeric_dtype

from ...contracts import ArtifactKind
from ...data.profiling import load_dataframe
from ...runtime import RuntimeDeps
from ...state import AnalysisGraphState, get_intent, get_ref
from ._shared import _event


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


async def inspect_data(state: AnalysisGraphState, deps: RuntimeDeps) -> dict[str, Any]:
    """선택 컬럼의 결측·고유값·수치 분포만 본다. 원시 행/범주 값은 State에 넣지 않는다."""
    intent = get_intent(state)
    dataset = get_ref(state, "dataset_ref")
    assert intent is not None and intent.execution_mode == "inspect_data"
    assert dataset is not None

    frame = load_dataframe(dataset.path)
    columns = list(dict.fromkeys(intent.candidate_columns))
    summary: dict[str, Any] = {
        "question": intent.question,
        "row_count": len(frame),
        "columns": {},
    }
    for name in columns:
        series = frame[name]
        item: dict[str, Any] = {
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if is_numeric_dtype(series.dtype):
            numeric = series.dropna()
            if not numeric.empty:
                item["numeric"] = {
                    "min": _finite_number(numeric.min()),
                    "max": _finite_number(numeric.max()),
                    "mean": _finite_number(numeric.mean()),
                    "std": _finite_number(numeric.std()),
                    "q25": _finite_number(numeric.quantile(0.25)),
                    "median": _finite_number(numeric.quantile(0.5)),
                    "q75": _finite_number(numeric.quantile(0.75)),
                }
        summary["columns"][name] = item

    if columns:
        summary["complete_case_rows"] = int(frame[columns].notna().all(axis=1).sum())

    ref = deps.artifacts.put_json(
        summary,
        kind=ArtifactKind.INSPECTION,
        summary={"columns": len(columns), "rows": len(frame)},
    )
    history = {
        "intent_id": intent.intent_id,
        "question": intent.question,
        "columns": columns,
        "inspection_ref": ref.model_dump(mode="json"),
    }
    _event(state, deps, "DATA_INSPECTED", {
        "intent_id": intent.intent_id,
        "columns": columns,
        "inspection_id": ref.artifact_id,
    })
    return {
        "inspection_ref": ref.model_dump(mode="json"),
        "inspection_summary": summary,
        "inspection_history": [history],
        "intent": None,
    }
