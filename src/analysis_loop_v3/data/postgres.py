"""PostgreSQL 결과를 분석 루프용 불변 스냅샷으로 물질화한다.

생성된 분석 코드는 DB 자격 증명이나 네트워크에 접근하지 않는다. 이 경계에서
읽기 전용 쿼리를 실행해 Parquet로 고정한 뒤, 이후 단계는 기존 파일 분석과 같은
경로를 사용한다.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd

from . import PreparedDataset

_FORBIDDEN_SQL = re.compile(
    r"\b(?:alter|call|copy|create|delete|do|drop|execute|grant|insert|merge|revoke|truncate|update)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PostgresSource:
    """명시적으로 승인된 읽기 전용 PostgreSQL 조회."""

    database_url: str
    query: str
    params: Mapping[str, Any] = field(default_factory=dict)
    row_limit: int = 100_000
    statement_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        _source_identity(self.database_url)
        validate_read_query(self.query)
        if self.row_limit <= 0:
            raise ValueError("row_limit은 0보다 커야 한다")
        if self.statement_timeout_seconds <= 0:
            raise ValueError("statement_timeout_seconds는 0보다 커야 한다")


def validate_read_query(query: str) -> None:
    """SELECT/CTE만 허용한다. 권한은 DB의 read-only 계정이 최종 경계다."""
    normalized = query.strip()
    if not normalized:
        raise ValueError("PostgreSQL 쿼리가 비어 있다")
    if not re.match(r"^(select|with)\b", normalized, re.IGNORECASE):
        raise ValueError("PostgreSQL 입력은 SELECT 또는 WITH 쿼리만 허용한다")
    if ";" in normalized or "--" in normalized or "/*" in normalized:
        raise ValueError("PostgreSQL 입력에 다중 명령 또는 SQL 주석은 허용하지 않는다")
    if _FORBIDDEN_SQL.search(normalized):
        raise ValueError("PostgreSQL 입력에 변경 가능한 SQL 키워드가 포함되어 있다")


def _source_identity(database_url: str) -> str:
    parts = urlsplit(database_url)
    if parts.scheme.split("+", 1)[0] not in {"postgres", "postgresql"} or not parts.hostname:
        raise ValueError("database_url은 PostgreSQL 연결 URL이어야 한다")
    port = f":{parts.port}" if parts.port else ""
    database = parts.path.lstrip("/")
    return f"{parts.scheme}://{parts.hostname}{port}/{database}"


def _fingerprint(source: PostgresSource) -> str:
    # 실제 연결 URL은 사용자를 포함해 스냅샷 출처를 구분하지만 해시 외에는 남기지 않는다.
    digest = hashlib.sha256()
    digest.update(source.database_url.encode())
    digest.update(source.query.strip().encode())
    digest.update(repr(sorted(source.params.items())).encode())
    digest.update(str(source.row_limit).encode())
    return digest.hexdigest()[:20]


def materialize_postgres_dataset(
    source: PostgresSource, *, output_dir: Path | str,
) -> PreparedDataset:
    """읽기 전용 트랜잭션으로 조회한 뒤 원자적으로 고유 Parquet 스냅샷을 만든다."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL 입력에는 analysis-loop-v3[postgres] 의존성이 필요하다"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(source)
    snapshot_id = uuid4().hex
    output = output_dir / f"postgres-{fingerprint}-{snapshot_id}.parquet"

    wrapped_query = f"SELECT * FROM ({source.query.strip()}) AS aia_source LIMIT :_aia_limit"
    engine = create_engine(source.database_url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            connection.execute(
                text("SELECT set_config('statement_timeout', CAST(:timeout_ms AS text), true)"),
                {"timeout_ms": str(round(source.statement_timeout_seconds * 1000))},
            )
            frame = pd.read_sql_query(
                text(wrapped_query),
                connection,
                params={**source.params, "_aia_limit": source.row_limit},
            )
    finally:
        engine.dispose()

    temporary = output.with_name(f".{output.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    with output.open("rb") as stream:
        snapshot_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    return PreparedDataset(
        path=output,
        summary={
            "source_kind": "postgresql",
            "source": _source_identity(source.database_url),
            "source_fingerprint": fingerprint,
            "snapshot_id": snapshot_id,
            "snapshot_sha256": snapshot_sha256,
            "rows": len(frame),
            "columns": len(frame.columns),
            "row_limit": source.row_limit,
        },
    )
