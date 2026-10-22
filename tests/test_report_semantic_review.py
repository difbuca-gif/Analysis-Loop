import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from analysis_loop_v3.contracts import REPORT_REVIEW_CATEGORIES
from analysis_loop_v3.llm.services import LLMCritic
from analysis_loop_v3.orchestration.nodes import progress
from analysis_loop_v3.orchestration.nodes.progress import _run_summary, _write_report
from analysis_loop_v3.orchestration.nodes.review import commit_evidence, critique


def report(statement, evidence_id):
    return f'''## 원래 목표와 실행 상태
실험 결과를 검토했다.
## 핵심 결론
{statement} [증거: {evidence_id}]
## 목표 축별 근거
관찰 결과를 확인했다. [증거: {evidence_id}]
## 한계와 미완료 사항
원인의 확정과 실제 조치에는 추가 확인이 필요하다.
'''


def assessment(evidence_id, *, category=None, quote=None):
    return {
        "verdict": "reject" if category else "accept",
        "checked_categories": list(REPORT_REVIEW_CATEGORIES),
        "reviewed_evidence_ids": [evidence_id],
        "findings": [] if category is None else [{
            "category": category, "quote": quote,
            "reason": "원본 결과는 기술 통계이며 이 주장을 입증하지 않는다.",
            "suggested_revision": "결과가 보여주는 범위로 표현을 줄이고 추가 확인을 명시한다.",
            "evidence_ids": [evidence_id],
        }],
    }


@pytest.fixture
def report_harness(harness):
    state, deps = harness
    state.update(asyncio.run(critique(state, deps)))
    state.update(asyncio.run(commit_evidence(state, deps)))
    record = state["evidence"][0]
    record["caveats"] = ["스키마 정의가 불분명하며 정산 오류를 확인한 분석이 아니다."]
    return state, deps, record["evidence_id"]


def write(state, deps):
    ref, accepted = asyncio.run(_write_report(state, deps, _run_summary(state)))
    return ref, accepted


@pytest.mark.parametrize(("category", "statement"), [
    ("evidence_support", "정산 오류가 확인됐다."),
    ("causal_scope", "이 변수 때문에 정산 오류가 발생했다."),
    ("metric_context", "평균이 확인됐으므로 인과 효과가 입증됐다."),
    ("uncertainty", "모든 결과는 확실하며 추가 확인이 필요 없다."),
    ("recommendations", "정산 로직 긴급 감사를 최우선으로 시행해야 한다."),
    ("completion_status", "모든 원인과 분석 과제를 완전히 해결했다."),
])
def test_reviewer_finding_blocks_report(report_harness, category, statement):
    # 탐지 성능 측정이 아니라, 각 의미 문제를 지적받았을 때 채택을 막는 경로다.
    state, deps, eid = report_harness
    deps.critic.generate_report.return_value = report(statement, eid)
    deps.critic.review_report.side_effect = None
    deps.critic.review_report.return_value = assessment(eid, category=category, quote=statement)
    ref, accepted = write(state, deps)
    assert not accepted
    saved = deps.artifacts.read_json(ref)
    assert saved["report_validation"]["passed"]
    assert saved["report_review"]["status"] == "rejected"
    assert saved["report_review"]["review"]["findings"][0]["quote"] == statement
    assert not list(deps.artifacts.root.glob("*.md"))


def test_qualified_recommendation_can_be_accepted(report_harness):
    state, deps, eid = report_harness
    markdown = report("정산 오류를 확정할 수 없으므로 데이터 정의와 표본을 먼저 확인한다.", eid)
    deps.critic.generate_report.return_value = markdown
    ref, accepted = write(state, deps)
    assert accepted
    assert deps.artifacts.read_text(ref) == markdown
    assert ref.summary["semantic_review_passed"]
    assert ref.summary["report_sha256"] == hashlib.sha256(markdown.encode()).hexdigest()
    assert any(e["event"] == "REPORT_REVIEWED" and e["passed"] for e in deps.events)


@pytest.mark.parametrize("defect", [
    "missing_category", "duplicate_category", "unknown_evidence", "missing_evidence",
    "unknown_quote", "accept_with_finding", "reject_without_finding", "empty_reason", "none",
])
def test_incomplete_or_invalid_review_cannot_approve(report_harness, defect):
    state, deps, eid = report_harness
    statement = "정산 로직 긴급 감사를 최우선으로 시행해야 한다."
    deps.critic.generate_report.return_value = report(statement, eid)
    response = assessment(eid, category="recommendations", quote=statement)
    if defect == "missing_category":
        response["checked_categories"].pop()
    elif defect == "duplicate_category":
        response["checked_categories"][-1] = response["checked_categories"][0]
    elif defect == "unknown_evidence":
        response["findings"][0]["evidence_ids"] = ["invented"]
    elif defect == "missing_evidence":
        response["reviewed_evidence_ids"] = []
    elif defect == "unknown_quote":
        response["findings"][0]["quote"] = "보고서에 없는 문장"
    elif defect == "accept_with_finding":
        response["verdict"] = "accept"
    elif defect == "reject_without_finding":
        response["findings"] = []
    elif defect == "empty_reason":
        response["findings"][0]["reason"] = "  "
    else:
        response = None
    deps.critic.review_report.side_effect = None
    deps.critic.review_report.return_value = response
    ref, accepted = write(state, deps)
    assert not accepted
    assert deps.artifacts.read_json(ref)["report_review"]["status"] == "unreviewed"


def test_structural_or_numeric_failure_never_calls_semantic_reviewer(report_harness):
    state, deps, eid = report_harness
    deps.critic.generate_report.return_value = report("평균은 999다.", eid)
    _, accepted = write(state, deps)
    assert not accepted
    deps.critic.review_report.assert_not_awaited()


def test_missing_result_blocks_even_non_numeric_claim(report_harness):
    state, deps, eid = report_harness
    deps.critic.generate_report.return_value = report("관찰 결과를 토대로 추가 검토를 제안한다.", eid)
    Path(state["candidate_ref"]["path"]).unlink()
    ref, accepted = write(state, deps)
    assert not accepted
    saved = deps.artifacts.read_json(ref)
    assert saved["report_validation"]["passed"]
    assert "원본 결과" in saved["report_review"]["error"]
    deps.critic.review_report.assert_not_awaited()


@pytest.mark.parametrize("failure", ["missing_service", "transport", "timeout"])
def test_reviewer_unavailable_is_not_approval(report_harness, monkeypatch, failure):
    state, deps, eid = report_harness
    deps.critic.generate_report.return_value = report("추가 검증이 필요하다.", eid)
    if failure == "missing_service":
        del deps.critic.review_report
    elif failure == "transport":
        deps.critic.review_report.side_effect = RuntimeError("connection lost")
    else:
        async def wait_forever(**kwargs):
            await asyncio.Event().wait()
        deps.critic.review_report.side_effect = wait_forever
        monkeypatch.setattr(progress, "DEFAULT_READ_TIMEOUT", 0.01)
    ref, accepted = write(state, deps)
    assert not accepted
    assert deps.artifacts.read_json(ref)["report_review"]["status"] == "unreviewed"


def test_llm_service_receives_original_results_method_caveats_and_profile(report_harness):
    state, deps, eid = report_harness
    markdown = report("정의와 표본을 확인한 후 추가 판단한다.", eid)
    client = SimpleNamespace(
        generate_text=AsyncMock(return_value=markdown),
        generate_json=AsyncMock(return_value=assessment(eid)),
    )
    deps.critic = LLMCritic(client)
    _, accepted = write(state, deps)
    assert accepted
    call = client.generate_json.call_args.kwargs
    assert call["role"] == "report_review"
    payload = json.loads(call["prompt"])
    assert payload["검토할 보고서"] == markdown
    assert payload["인용 근거의 원본 결과"][eid]["effects"][0]["value"] == 1.25
    assert payload["인용 근거"][0]["protocol"]["interpretation_scope"] == "descriptive"
    assert payload["인용 근거"][0]["caveats"]
    assert payload["데이터 구조"]["columns"][0]["name"] == "x"


def test_llm_service_invalid_json_contract_fails_closed(report_harness):
    state, deps, eid = report_harness
    client = SimpleNamespace(
        generate_text=AsyncMock(return_value=report("추가 검증이 필요하다.", eid)),
        generate_json=AsyncMock(return_value={"verdict": "accept"}),
    )
    deps.critic = LLMCritic(client)
    ref, accepted = write(state, deps)
    assert not accepted
    assert "보고서 의미 검토 실패" in deps.artifacts.read_json(ref)["report_review"]["error"]
