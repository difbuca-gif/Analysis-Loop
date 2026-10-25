# Orchestration 구조

이 디렉터리는 **분석 실행 흐름**만 관리한다. 기능을 추가할 때 Node 하나에 정책·검증·저장 로직을 다시 몰아넣지 않는다.

## 실행 흐름

```text
START
  ↓
profile_data
  ↓
decompose_objective
  ↓
plan_intent
  ↓
lookup/reuse/inspect/compute
  ↓
generate_code
  ↓
validate_code
  ↓
execute
  ↓
validate_result
  ↓
critique
  ↓
commit_evidence / reject_attempt
  ↓
assess_progress
  ↓
plan_intent / expand_goals / finalize
  ↓
END
```

## 파일 책임

### 그래프

| 파일 | 책임 |
|---|---|
| `graph.py` | Node 등록과 Edge 연결만 담당 |
| `routes.py` | State를 읽고 다음 Node를 선택하는 라우팅 정책 |
| `node_updates.py` | Node별 partial State update 타입 |
| `failure_policy.py` | 공통 외부 서비스 실패 분류 |

### Node

| 파일 | 책임 |
|---|---|
| `nodes/planning.py` | Profile, SubGoal 분해, Planner 호출 orchestration |
| `nodes/planner_context.py` | Planner에 전달할 Evidence 선택 |
| `nodes/goal_policy.py` | SubGoal converge/abandon 적용 규칙 |
| `nodes/intent_policy.py` | Agenda/중복/재사용 Intent 정책 |
| `nodes/program.py` | 프로그램 cache 조회와 Codegen 호출 |
| `nodes/code_validation.py` | 코드/Manifest 정적 검증 |
| `nodes/running.py` | Sandbox 실행과 Result Artifact 생성 |
| `nodes/result_validation.py` | Result 계약 검증 |
| `nodes/review.py` | Critic 호출, Evidence commit/reject orchestration |
| `nodes/review_evidence.py` | Evidence 생성, cache, relation 규칙 |
| `nodes/review_agenda.py` | Evidence 채택 후 Agenda lifecycle |
| `nodes/rejection_policy.py` | 거부 사유와 반복 오류 분류 |
| `nodes/progress.py` | 계속/종료 및 Goal 확장 orchestration |
| `nodes/progress_policy.py` | 종료/contradiction 복구 정책 |
| `nodes/reporting.py` | 보고서 생성, 구조·수치 검증, semantic review |

## 변경 규칙

### 1. Node는 orchestration만 한다

Node 함수에서는 다음 순서가 바로 보여야 한다.

```text
State 읽기
→ 외부 서비스 또는 helper 호출
→ 이벤트 기록
→ partial State update 반환
```

복잡한 순수 규칙은 `*_policy.py` 또는 목적별 helper로 이동한다.

### 2. 라우팅은 `routes.py`에 둔다

Node 내부에서 다음 Node 이름을 결정하지 않는다. Graph 변경은 `graph.py`, 조건 변경은 `routes.py`에서 찾을 수 있어야 한다.

### 3. 결정론적 검증과 LLM 검토를 섞지 않는다

```text
code validation
→ execution
→ result validation
→ Critic
→ commit boundary validation
```

Critic은 deterministic validation 실패를 승인으로 뒤집을 수 없다.

### 4. State에는 큰 원문을 넣지 않는다

Dataset, 생성 코드, 실행 결과, Figure, Report는 ArtifactStore에 저장하고 State에는 ArtifactRef를 둔다.

### 5. 누적 지식과 현재 cycle 값을 구분한다

- 현재 cycle: `intent`, `program_ref`, `candidate_ref`, `review`, validation fields
- 누적: `evidence`, `evidence_relations`, `rejected`, `issues`

새 Intent 시작 시 `cycle_reset_update()`의 대상인지 먼저 확인한다.

### 6. Node 반환 타입을 명시한다

가능하면 `node_updates.py`의 TypedDict를 사용한다. `dict[str, Any]`를 새 Node의 기본 반환 타입으로 추가하지 않는다.

### 7. 기존 import 경로를 무심코 깨지 않는다

현재 일부 테스트와 호출부는 `graph.route_after_*`, `program.validate_code`, `running.validate_result`, `progress._write_report` 같은 기존 경로를 사용한다. 파일 이동 시 re-export 또는 compatibility wrapper를 먼저 제공한다.

## 코드 리뷰 체크리스트

- 이 함수의 책임을 한 문장으로 설명할 수 있는가?
- 함수가 `AnalysisGraphState`에서 읽는 key가 지나치게 많은가?
- 순수 정책을 별도 함수/모듈로 테스트할 수 있는가?
- LLM 응답을 코드 검증 없이 State에 확정하고 있지 않은가?
- Artifact 원문을 State에 복사하고 있지 않은가?
- 실패 시 다음 경로가 `routes.py`에서 명확하게 보이는가?
- 새 update key가 `AnalysisGraphState`와 적절한 Node update 타입에 반영됐는가?
- Resume/Checkpoint 경로에서도 같은 검증 경계가 유지되는가?
