"""분석 전 결정론적 dataset profiling.

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
    """CSV/Parquet 공통 dataframe 로드."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    return normalize_dtypes(frame)


def normalize_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """object column을 nullable string으로 정규화."""
    out = frame.copy()
    for name in out.columns:
        if out[name].dtype == object:
            out[name] = out[name].astype("string")
    return out


def schema_fingerprint(frame: pd.DataFrame) -> str:
    """column name/dtype 기반 schema fingerprint."""
    parts = [f"{name}:{frame[name].dtype}" for name in frame.columns]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def profile_dataframe(frame: pd.DataFrame, *, dataset_id: str) -> DatasetProfile:
    rows = len(frame)
    columns: list[ColumnProfile] = []

    for name in frame.columns:
        series = frame[name]
        unique = int(series.nunique(dropna=True))
        # non-null sample 최대 5개; 표시용 str 변환
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
