"""공통 상수, 식별자, 경계 타입."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import StrictInt, StrictStr

# 수치 출처를 검사하지 않은 이전 체크포인트는 새 실행에서 재사용하지 않는다.
SCHEMA_VERSION = 9

ResultPath = list[StrictStr | StrictInt]

MAX_RESULT_BYTES = 64 * 1024

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class InvalidRunId(ValueError):
    """디렉터리 이름으로 쓸 수 없는 run_id."""


def validate_run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value or ""):
        raise InvalidRunId(
            "run_id는 영문·숫자로 시작하고 영문·숫자·밑줄·하이픈만 쓸 수 있다"
            f"(최대 64자): {value!r}"
        )
    return value


DescriptionItem = str | dict[str, Any]
DataApplicability = Literal[
    "exact_snapshot",
    "revalidation_required",
    "incompatible_schema",
    "unknown",
]
