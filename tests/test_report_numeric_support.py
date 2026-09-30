import asyncio
from pathlib import Path

import pytest

from analysis_loop_v3.contracts import (
    ArtifactKind,
    ArtifactRef,
    DataLineage,
    EffectEstimate,
    EvidenceRecord,
)
from analysis_loop_v3.execution.sandbox import SubprocessSandbox
from analysis_loop_v3.orchestration.graph import route_after_validate_result
from analysis_loop_v3.orchestration.nodes.progress import _run_summary, _write_report
from analysis_loop_v3.orchestration.nodes.review import commit_evidence, critique
from analysis_loop_v3.orchestration.nodes.running import validate_result
from analysis_loop_v3.report_validation import validate_report


def report(number, evidence_id="e1"):
    return f'''## 원래 목표와 실행 상태
평균을 확인했다.
## 핵심 결론
1. 평균 {number} [증거: {evidence_id}]
## 목표 축별 근거
평균을 계산했다. [증거: {evidence_id}]
## 한계와 미완료 사항
작은 테스트 데이터의 계산이다.
'''


@pytest.fixture
def legacy_evidence():
    # 패치 전 저장된 요약에만 999가 있는 사례를 그대로 재현한다.
    return EvidenceRecord(
        evidence_id="e1", intent_id="i1", question="평균은?", claim="평균은 999",
        method="mean", columns_used=["x"],
        result_ref=ArtifactRef(
            artifact_id="r1", kind=ArtifactKind.RESULT, path="result.json",
            sha256="0" * 64, size_bytes=10,
        ),
        data_lineage=DataLineage(
            dataset_artifact_id="d1", dataset_sha256="0" * 64,
            schema_fingerprint="s1", source_rows=4, transform_signature="0" * 64,
        ),
        effect_summary=[EffectEstimate(column="x", metric="mean", value=999.0)],
        uncertainty_summary={"ci": [888.0, 1000.0]},
    )


@pytest.mark.parametrize("number", ["999.0", "999", "9.99e2", "+999", "999%", "888.0", "1,000"])
def test_summary_only_numbers_cannot_authorize_report(legacy_evidence, number):
    checked = validate_report(
        report(number), evidence=[legacy_evidence], relations=[],
        numeric_support={"e1": [1.25]},
    )
    assert not checked["passed"]
    assert any("수치" in error for error in checked["errors"])


@pytest.mark.parametrize("support", [{}, {"e1": []}, {"e1": [float("nan"), float("inf")]}])
def test_missing_original_does_not_fall_back_to_summary(legacy_evidence, support):
    assert not validate_report(
        report("999.0"), evidence=[legacy_evidence], relations=[], numeric_support=support,
    )["passed"]


@pytest.mark.parametrize(("number", "source"), [
    ("1.25", 1.25), ("4", 4), ("1,250", 1250), ("1.25e-3", 0.00125),
    ("25%", 0.25), (".25", 0.25), ("−1.25", -1.25),
])
def test_actual_source_numbers_are_allowed(legacy_evidence, number, source):
    assert validate_report(
        report(number), evidence=[legacy_evidence], relations=[], numeric_support={"e1": [source]},
    )["passed"]


def test_small_nonzero_number_does_not_support_zero(legacy_evidence):
    assert not validate_report(
        report("0"), evidence=[legacy_evidence], relations=[], numeric_support={"e1": [1e-10]},
    )["passed"]


def test_another_evidences_number_does_not_support_citation(legacy_evidence):
    assert not validate_report(
        report("999"), evidence=[legacy_evidence], relations=[],
        numeric_support={"e1": [1.25], "e2": [999]},
    )["passed"]


def test_citation_digits_and_list_numbers_are_not_measurements(legacy_evidence):
    record = legacy_evidence.model_copy(update={"evidence_id": "123-456"})
    assert validate_report(
        report("1.25", "123-456"), evidence=[record], relations=[],
        numeric_support={"123-456": [1.25]},
    )["passed"]


def test_bad_report_falls_back_to_json(harness):
    state, deps = harness
    state.update(asyncio.run(critique(state, deps)))
    state.update(asyncio.run(commit_evidence(state, deps)))
    evidence_id = state["evidence"][0]["evidence_id"]
    deps.critic.generate_report.return_value = report("999", evidence_id)
    ref, accepted = asyncio.run(_write_report(state, deps, _run_summary(state)))
    assert not accepted
    assert ref.path.endswith(".json")
    assert not deps.artifacts.read_json(ref)["report_validation"]["passed"]


def test_missing_result_artifact_rejects_numeric_report(harness):
    state, deps = harness
    state.update(asyncio.run(critique(state, deps)))
    state.update(asyncio.run(commit_evidence(state, deps)))
    evidence_id = state["evidence"][0]["evidence_id"]
    deps.critic.generate_report.return_value = report("1.25", evidence_id)
    Path(state["candidate_ref"]["path"]).unlink()
    _, accepted = asyncio.run(_write_report(state, deps, _run_summary(state)))
    assert not accepted


def test_calculation_to_evidence_to_report(harness, tmp_path):
    state, deps = harness
    code = deps.artifacts.read_text(ArtifactRef.model_validate(state["program_ref"]))
    workdir = tmp_path / "execution"
    workdir.mkdir()
    executed = asyncio.run(SubprocessSandbox().run(
        code=code, dataset_path=state["dataset_ref"]["path"],
        workdir=workdir, timeout_seconds=30,
    ))
    assert executed.ok, executed.stderr or executed.refused_reason
    result = deps.artifacts.put_file(executed.result_path, kind=ArtifactKind.RESULT)
    state["candidate_ref"] = result.model_dump(mode="json")
    state.update(asyncio.run(validate_result(state, deps)))
    assert route_after_validate_result(state) == "critique"
    state.update(asyncio.run(critique(state, deps)))
    state.update(asyncio.run(commit_evidence(state, deps)))
    evidence = state["evidence"][0]
    assert evidence["effect_summary"][0]["value"] == 1.25
    deps.critic.generate_report.return_value = report("1.25", evidence["evidence_id"])
    ref, accepted = asyncio.run(_write_report(state, deps, _run_summary(state)))
    assert accepted
    assert ref.path.endswith(".md")
    assert "1.25" in deps.artifacts.read_text(ref)
