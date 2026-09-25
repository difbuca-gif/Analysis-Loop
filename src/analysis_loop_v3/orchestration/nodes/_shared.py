"""노드들이 공유하는 최소 도구.

여기에는 여러 단계가 함께 쓰는 것만 둔다. 한 단계에서만 쓰는 상수와 헬퍼는
그 단계의 모듈에 둬야, 나중에 그 파일만 읽어도 동작을 다 알 수 있다.
"""

from __future__ import annotations

from typing import Any

from ...contracts import GeneratedAnalysis
from ...runtime import RuntimeDeps
from ...state import AnalysisGraphState, get_ref

# 오류 종류가 바뀌어도 한 의도에 쓸 재생성 횟수는 제한한다.
MAX_STATIC_REPAIRS = 5
MAX_RUNTIME_REPAIRS = 1

# 같은 오류가 이만큼 연속되면 그 회차를 포기하고 다음 질문으로 넘어간다.
# 무한 재시도를 막되 런은 계속한다. 한 질문이 막힌 것이 분석 전체의 실패는 아니다.
STUCK_THRESHOLD = 5


def streak_key(errors: list[dict[str, Any]] | None) -> str:
    """이번 오류 묶음의 안정된 키. 코드 집합이 같으면 같은 키다.

    첫 코드만 쓰면 두 오류가 번갈아 나올 때 연속으로 안 세지고, 전체 메시지를
    쓰면 컬럼명 같은 게 섞여 매번 달라진다.
    """
    codes = sorted({str(e.get("code", "")) for e in errors or []} - {""})
    return ",".join(codes)


def bump_error_streak(
    state: AnalysisGraphState, errors: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """연속 카운터를 갱신한다.

    초기화는 plan_intent(새 의도) 한 곳에서만 한다. 중간 단계의 성공으로
    지우면 안 된다. 코드 생성에 성공하고 검증에서 같은 오류로 걸리는 회차가
    반복되면 카운터가 매번 1로 돌아가 임계에 영영 못 닿는다(실측: 정적 오류
    248회 반복, 라벨 0개).
    """
    key = streak_key(errors)
    if not key:
        return {}
    previous = state.get("error_streak") or {}
    count = previous.get("count", 0) + 1 if previous.get("key") == key else 1
    return {"key": key, "count": count}


def is_stuck(state: AnalysisGraphState) -> bool:
    """같은 오류로 임계만큼 제자리걸음인가 — graph.py 라우팅이 읽는다."""
    return int((state.get("error_streak") or {}).get("count", 0)) >= STUCK_THRESHOLD


def _event(
    state: AnalysisGraphState,
    deps: RuntimeDeps,
    name: str,
    payload: dict[str, Any] | None = None,
    *,
    iteration: int | None = None,
) -> None:
    """State를 아는 노드가 run/iteration을 명시하는 공통 이벤트 경계."""

    deps.event(
        name,
        payload,
        run_id=state.get("run_id"),
        iteration=state.get("iteration", 0) if iteration is None else iteration,
    )


def _load_generated(state: AnalysisGraphState, deps: RuntimeDeps) -> GeneratedAnalysis:
    ref = get_ref(state, "program_ref")
    assert ref is not None
    code = deps.artifacts.read_text(ref)
    return GeneratedAnalysis.model_validate({**(state.get("manifest") or {}), "code": code})
