# LLM 프롬프트 자산

이 디렉터리의 마크다운은 실행 중 LLM의 `system` 메시지로 전달되는 정적 지시문이다.
`services.py`는 `load_prompt()`로 문서를 읽고, 실행마다 달라지는 목표·데이터·근거·오류는
별도의 JSON `user` 메시지로 전달한다.

| 파일 | 역할 | 호출부 |
|---|---|---|
| [master_memory_system.md](./master_memory_system.md) | 마스터의 Wiki·메모장 사용 방법 | Planner 호출 공통 |
| [decompose_system.md](./decompose_system.md) | 분석 목표를 하위 목표로 분해(실행당 최초 1회) | `LLMPlanner.decompose_objective` |
| [expand_goals_system.md](./expand_goals_system.md) | 목표 축이 전부 닫혔는데 시간이 남으면 근거를 딛고 새 축 제안 | `LLMPlanner.expand_goals` |
| [planner_system.md](./planner_system.md) | 다음 연구 의도 선택과 목표 축 판정 | `LLMPlanner.propose_intent` |
| [codegen_system.md](./codegen_system.md) | 분석 코드와 실행 계약 생성 | `LLMCodegen.generate` |
| [critic_system.md](./critic_system.md) | 실행 결과 검토와 근거 채택 판정 | `LLMCritic.review` |
| [report_system.md](./report_system.md) | 채택 근거 기반 최종 보고서 작성 | `LLMCritic.generate_report` |

## 경계

- 행동 규칙, 역할, 판단 기준, 출력 형식은 이 디렉터리의 문서에 둔다.
- 실제 목표, 데이터 프로필, 발견, 오류 메시지는 코드가 JSON payload로 조립한다.
- Python 코드에는 프롬프트 원문을 중복 작성하지 않는다.
- 프롬프트를 바꾸면 해당 역할 전체의 판단 방식이 바뀌므로 코드 변경과 같은 수준으로 검토한다.
