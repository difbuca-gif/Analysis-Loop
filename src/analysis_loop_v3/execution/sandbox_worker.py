"""생성 코드를 실행하는 샌드박스 서브프로세스 진입점.

import 화이트리스트와 socket 차단을 적용하며, 주 격리는 컨테이너 계층이 담당한다.
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import statistics
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..contracts import MAX_RESULT_BYTES

ALLOWED_IMPORT_ROOTS = frozenset({
    "pandas", "numpy", "scipy", "sklearn", "statsmodels", "math", "statistics",
    "matplotlib", "tqdm", "json",
    # 추가 허용 분석 라이브러리.
    "xgboost", "lightgbm", "linearmodels", "mlxtend", "ruptures",
})

_real_import = builtins.__import__

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level != 0 or root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"샌드박스에서 '{name}' import는 허용되지 않는다")
    return _real_import(name, globals, locals, fromlist, level)

def _block_network() -> None:
    """socket 생성을 막는다(차단의 흉내이지 보장은 아니다).

    socket 타입 상속은 유지하고 실제 생성만 차단한다.
    """
    import socket

    _real_socket = socket.socket

    class _BlockedSocket(_real_socket):  # type: ignore[misc, valid-type]
        def __init__(self, *_args, **_kwargs):
            raise PermissionError("샌드박스에서 네트워크 접근은 금지된다")

    def _refuse(*_args, **_kwargs):
        raise PermissionError("샌드박스에서 네트워크 접근은 금지된다")

    socket.socket = _BlockedSocket      # type: ignore[assignment]
    socket.create_connection = _refuse  # type: ignore[assignment]

ANALYSIS_NAMESPACE = {
    "pd": pd,
    "np": np,
    "stats": scipy_stats,
    "math": math,
    "statistics": statistics,
}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ANALYSIS_NAMESPACE["plt"] = plt
except ImportError:
    pass

SAFE_BUILTINS = {
    "__import__": _restricted_import,
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "pow": pow, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError, "ArithmeticError": ArithmeticError,
    "RuntimeError": RuntimeError, "OverflowError": OverflowError,
    "NotImplementedError": NotImplementedError, "StopIteration": StopIteration,
}

def _stringify_keys(value, _depth: int = 0):
    """중첩 dict의 키를 전부 str로(groupby의 numpy.int64/tuple 키는 default=로도 안 된다).

    깊이 제한은 병적으로 깊은 결과의 RecursionError 방지용.
    """
    if _depth > 10:
        return str(value)
    if isinstance(value, dict):
        return {str(k): _stringify_keys(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_keys(v, _depth + 1) for v in value]
    if isinstance(value, tuple):
        return [_stringify_keys(v, _depth + 1) for v in value]
    return value

def main() -> int:
    warnings.simplefilter("ignore")
    _block_network()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    frame = (
        pd.read_parquet(source)
        if source.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(source)
    )

    code = Path(args.program).read_text(encoding="utf-8")
    scope = {"__builtins__": SAFE_BUILTINS, **ANALYSIS_NAMESPACE}
    exec(code, scope)  # noqa: S102 - 이 프로세스의 존재 이유

    analyze = scope.get("analyze")
    if not callable(analyze):
        raise TypeError("analyze(df) 함수를 찾을 수 없다")

    result = analyze(frame)
    encoded = json.dumps(
        _stringify_keys(result),
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        raise ValueError(
            f"분석 결과가 {len(encoded):,}바이트다 — 상한 {MAX_RESULT_BYTES:,}"
        )
    Path(args.output).write_bytes(encoded)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
