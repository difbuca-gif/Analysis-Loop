"""State 업데이트만 반환하는 체크포인트 단위 그래프 노드. 분할 순서 = graph.py 흐름.

    planning   무엇을 물을지    profile_data / decompose_objective / plan_intent
    program    어떻게 계산할지  lookup_program / generate_code / validate_code
    running    실제로 돌리기    execute / validate_result
    review     의미 판정        critique / commit_evidence / reject_attempt
    progress   계속/종료        assess_progress / expand_goals / finalize

여기서 전부 re-export하므로 `nodes.execute` 같은 기존 호출부는 그대로 동작한다.
"""

from ._shared import (
    MAX_RUNTIME_REPAIRS,
    MAX_STATIC_REPAIRS,
    STUCK_THRESHOLD,
    _event,
    _load_generated,
    bump_error_streak,
    is_stuck,
)
from .evidence_lookup import lookup_evidence
from .inspection import inspect_data
from .planning import (
    _apply_goal_verdict,
    decompose_objective,
    mark_data_limited,
    plan_intent,
    profile_data,
    reuse_evidence,
)
from .program import generate_code, lookup_program, validate_code
from .progress import (
    MAX_GOAL_EXPANSIONS,
    MIN_SECONDS_FOR_EXPANSION,
    assess_progress,
    expand_goals,
    finalize,
)
from .review import commit_evidence, critique, reject_attempt
from .running import SANDBOX_TIMEOUT_SECONDS, execute, validate_result

__all__ = [
    # 상수 — graph.py 라우팅과 테스트가 참조한다.
    "MAX_RUNTIME_REPAIRS",
    "MAX_STATIC_REPAIRS",
    "STUCK_THRESHOLD",
    "is_stuck",
    "bump_error_streak",
    "SANDBOX_TIMEOUT_SECONDS",
    "MAX_GOAL_EXPANSIONS",
    "MIN_SECONDS_FOR_EXPANSION",
    # 노드 — graph.py가 이 이름으로 등록한다.
    "profile_data",
    "decompose_objective",
    "plan_intent",
    "reuse_evidence",
    "mark_data_limited",
    "inspect_data",
    "lookup_evidence",
    "lookup_program",
    "generate_code",
    "validate_code",
    "execute",
    "validate_result",
    "critique",
    "commit_evidence",
    "reject_attempt",
    "assess_progress",
    "expand_goals",
    "finalize",
    # 내부 헬퍼 — 테스트가 직접 검사한다.
    "_apply_goal_verdict",
    "_event",
    "_load_generated",
]
