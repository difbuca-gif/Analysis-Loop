<div align="center">

# Analysis Loop V3

**검증된 근거와 후속 작업을 상태로 관리하는 데이터 분석 에이전트**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-checkpointed-1C3C3C)](https://langchain-ai.github.io/langgraph/)

</div>

---

## 문제

반복 분석을 자동화하는 것만으로는 충분하지 않았습니다.

장시간 실행에서는 **어떤 결과를 다시 써도 되는지, 무엇을 재검증해야 하는지, 왜 다음 분석을 선택했는지**를 관리해야 합니다.

Analysis Loop V3는 이를 다음처럼 나눕니다.

- **Planner** — 현재 목표·근거·후속 작업을 보고 다음 행동 선택
- **Codegen** — 분석 방법과 Python 코드 생성
- **규칙 기반 검증** — 선언한 컬럼·실행 조건·출력 구조와 실제 코드/결과 대조
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
