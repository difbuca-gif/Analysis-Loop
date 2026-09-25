"""데이터 프로파일 — 분석 전에 데이터가 무엇인지 결정론적으로 파악한다.

두 사고를 막는다: (1) 결측 소실 — astype(str)은 None을 "None"으로 바꾸므로
astype("string")으로 <NA>를 보존한다. (2) 사실과 해석 분리 — 행/고유값 수는
기록하되 컬럼 이름이나 임의 비율로 식별자 의미를 확정하지 않는다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ..contracts import ColumnProfile, DatasetProfile


def load_dataframe(path: Path | str) -> pd.DataFrame:
    """CSV/Parquet을 같은 의미로 읽는다. 입력 형식에 따라 결과가 달라지면 안 된다."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    return normalize_dtypes(frame)


def normalize_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """object 컬럼을 nullable string으로. 결측을 문자열로 바꾸지 않는다."""
    out = frame.copy()
    for name in out.columns:
        if out[name].dtype == object:
            out[name] = out[name].astype("string")
    return out


def schema_fingerprint(frame: pd.DataFrame) -> str:
    """컬럼 이름+dtype의 지문(용도는 contracts.py의 schema_fingerprint 참고)."""
    parts = [f"{name}:{frame[name].dtype}" for name in frame.columns]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def profile_dataframe(frame: pd.DataFrame, *, dataset_id: str) -> DatasetProfile:
    rows = len(frame)
    columns: list[ColumnProfile] = []

    for name in frame.columns:
        series = frame[name]
        unique = int(series.nunique(dropna=True))
        # 샘플은 결측을 뺀 실제 값 5개까지. str()은 여기서만 쓴다. 표시용이라
        # 결측 통계에 영향을 주지 않는다.
        sample = [str(v) for v in series.dropna().unique()[:5]]

        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                non_null=int(series.notna().sum()),
                null_count=int(series.isna().sum()),
                unique_count=unique,
                sample_values=sample,
            )
        )

    return DatasetProfile(
        dataset_id=dataset_id,
        row_count=rows,
        columns=columns,
        schema_fingerprint=schema_fingerprint(frame),
    )
