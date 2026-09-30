<div align="center">

# Analysis Loop V2

**계약 검사와 모델 검토를 통과한 근거·후속 작업을 관리하는 데이터 분석 에이전트**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-checkpointed-1C3C3C)](https://langchain-ai.github.io/langgraph/)

</div>

---

## 문제

반복 분석을 자동화하는 것만으로는 충분하지 않았습니다.

장시간 실행에서는 **어떤 결과를 다시 써도 되는지, 무엇을 재검증해야 하는지, 왜 다음 분석을 선택했는지**를 관리해야 합니다.

Analysis Loop V2는 이를 다음처럼 나눕니다.

- **Planner** — 현재 목표·근거·후속 작업을 보고 다음 행동 선택
- **Codegen** — 분석 방법과 Python 코드 생성
- **규칙 기반 검증** — 컬럼 참조·출력 구조·요약 수치의 원본 일치 검사, 실행 조건 선언 확인
- **Critic** — 분석 타당성과 해석 범위 검토
- **Evidence / Agenda** — 채택 근거와 후속 검증 작업을 다음 회차에 전달

## 구조

```text
사용자
  │
  ▼
AnalysisService
  │
  ▼
┌──────────── LangGraph Analysis Loop ────────────┐
│ profile → objective decomposition → plan        │
│                         │                       │
│      ┌──────────┬───────┼──────────┐            │
│   compute     reuse   inspect    lookup          │
│      │                                          │
│ codegen → code validation → execute             │
│                    → result validation → critic │
│                                  │              │
│                         commit / reject          │
│                                  │              │
│                         assess progress          │
│                                  │              │
│                  replan / expand / finalize      │
└─────────────────────────────────────────────────┘
  │
  ▼
Evidence + Agenda + Artifacts
  │
  ▼
Report validation / Audit / Benchmark
```

## 핵심 설계

### LLM 판단과 규칙 검사를 분리

실행 전에는 코드·컬럼·분석 계약을, 실행 후에는 출력 구조·크기·누락을 규칙으로 검사합니다. 그 이후 별도 Critic이 질문 적합성, 효과 크기, 불확실성, 과장 해석을 검토합니다.

규칙 검사에서 실패한 결과를 Critic이 뒤집어 채택할 수 없습니다.

Critic의 효과 요약은 `source_path`로 원본 결과의 효과 객체에 연결합니다. 컬럼·지표·값·단위·방향성이 일치해야 하며, 저장할 값은 코드가 원본에서 다시 읽습니다. 불확실성 요약도 항목마다 `uncertainty_source_paths`로 원본과 대조합니다. 누락이나 불일치는 미검토로 처리하고 Evidence로 채택하지 않습니다. 저장 직전에도 같은 검사를 수행합니다.

최종 보고서의 수치 검사는 원본 결과 Artifact만 허용값으로 사용합니다. Critic 요약에만 있는 숫자는 허용하지 않으며, 정수·소수·지수 표기·백분율을 검사합니다. 원본 파일을 읽지 못하면 요약 수치로 대신 통과시키지 않습니다.

### 검증 범위와 한계

- 이 검사는 **원본 출력과 요약의 일치**를 확인합니다. 생성 코드의 계산식·통계 방법 자체가 옳다는 보장은 아닙니다.
- 시간 순서·그룹 격리 검사는 Manifest의 선언을 확인합니다. 실제 학습·검증 행을 대조하는 누수 검사는 아직 없습니다. 컬럼 정적 검사도 위치 기반 접근 등을 모두 추적하지 못합니다.
- 보고서는 인용한 결과에 해당 숫자가 있는지 검사합니다. 문장 속 숫자가 어떤 지표를 뜻하는지, 인과 해석이나 권고가 적절한지는 별도 검토가 필요합니다.
- 기본 subprocess 실행은 파일·네트워크 접근을 완전히 격리하지 않으며, 기본 메모리 제한도 없습니다. 저장소에 컨테이너 격리 구성이 포함돼 있지 않습니다.
- 현재 공개 테스트는 아래 수치 전달 회귀 검사입니다. 실제 LLM의 분석 성능이나 baseline 대비 개선을 입증하는 벤치 결과는 별도로 측정해야 합니다.

### 수치 검증 패치 적용

상태 스키마를 9로 올렸습니다. 패치 전 체크포인트는 수치 출처가 검증되지 않았으므로 `resume`이 거부됩니다. 기존 파일은 보존하고 같은 데이터로 **새 run**을 시작하십시오. 코드 캐시 키도 변경해 이전 출력 형식의 프로그램을 재사용하지 않습니다. 과거 결과는 이번 패치로 소급 검증되지 않습니다.

```bash
pip install -e '.[dev]'
python -m pytest -q
```

테스트는 외부 LLM 없이 원본 `1.25`를 `999`로 바꾸는 응답, 잘못된 지표·단위·경로, 불확실성 변조, 원본을 잃은 보고서 및 정상 채택 경로를 확인합니다.

### 결과가 아니라 Evidence를 저장

채택된 결과에는 데이터 스냅샷, schema, transform signature, 실행 코드, 결과 Artifact를 연결합니다. 데이터가 바뀌면 기존 Evidence를 그대로 재사용하지 않고 재검증 필요 여부를 판정합니다.

### 후속 검증을 Agenda로 관리

필수 후속 질문이나 Evidence 간 모순은 프롬프트 메모가 아니라 Agenda 작업으로 남깁니다. 처리되기 전에는 분석 완료로 종료하지 않습니다.

## 핵심 구현

- [LangGraph 전체 실행 흐름](src/analysis_loop_v3/orchestration/graph.py) — 상태 기반 분기와 재시도·종료 경로
- [Planner 의사결정](src/analysis_loop_v3/orchestration/nodes/planning.py) — 목표·근거·Agenda를 바탕으로 다음 분석 선택
- [Critic → Evidence 반영](src/analysis_loop_v3/orchestration/nodes/review.py) — 결과 채택·거부, Evidence 관계와 후속 작업 갱신
- [결정론적 검증](src/analysis_loop_v3/validation/checks.py) — 선언한 분석 계약과 실제 코드·결과 대조
- [Evidence / Agenda 상태 계약](src/analysis_loop_v3/contracts.py) · [그래프 상태](src/analysis_loop_v3/state.py) — 재사용·재검증·진행 상태 정의

## 실행

```bash
pip install -e .
cp .env.example .env

python -m analysis_loop_v3 run data/churn.csv \
  --objective "이탈에 영향을 주는 요인을 찾고 효과 크기와 방향을 정량화한다"
```

```bash
python -m analysis_loop_v3 status <run_id>
python -m analysis_loop_v3 resume <run_id>
python -m analysis_loop_v3 audit <run_id>
python -m analysis_loop_v3 bench <dataset>
```

## 기술

Python · LangGraph · FastAPI · PostgreSQL · vLLM
