import asyncio
from copy import deepcopy

import pytest
from pydantic import ValidationError

from analysis_loop_v3.contracts import CriticReview
from analysis_loop_v3.orchestration.graph import route_after_critique
from analysis_loop_v3.orchestration.nodes.review import commit_evidence, critique
from analysis_loop_v3.validation.numeric import bind_review_to_result, resolve_result_path


def test_matching_review_is_copied_from_result(review_payload, result_payload):
    bound = bind_review_to_result(CriticReview.model_validate(review_payload), result_payload)
    assert bound.effect_summary[0].value == 1.25
    assert bound.effect_summary[0].source_path == ["effects", 0]
    result_payload["ci"][0] = 999
    assert bound.uncertainty_summary["ci"] == [1.0, 1.5]


@pytest.mark.parametrize(("field", "wrong"), [
    ("value", 999.0), ("column", "other"), ("metric", "p_value"),
    ("unit", "%"), ("directional", True),
])
def test_effect_must_match_its_source_object(review_payload, result_payload, field, wrong):
    review_payload["effect_summary"][0][field] = wrong
    # 같은 숫자가 다른 곳에 있어도 지정한 효과 항목과 다르면 거부한다.
    result_payload["unrelated"] = wrong
    with pytest.raises(ValueError, match=field):
        bind_review_to_result(CriticReview.model_validate(review_payload), result_payload)


@pytest.mark.parametrize("path", [None, [], ["missing"], ["effects", -1], ["effects", 5],
                                  ["effects", "0"], ["effects", True], ["effects", 0, "value"]])
def test_invalid_effect_paths_are_rejected(review_payload, result_payload, path):
    review_payload["effect_summary"][0]["source_path"] = path
    with pytest.raises(ValueError):
        bind_review_to_result(CriticReview.model_validate(review_payload), result_payload)


@pytest.mark.parametrize("value", [True, "1.25", float("nan"), float("inf"), -float("inf")])
def test_non_numeric_or_non_finite_effect_value_rejected(review_payload, value):
    review_payload["effect_summary"][0]["value"] = value
    with pytest.raises(ValidationError):
        CriticReview.model_validate(review_payload)


@pytest.mark.parametrize("value", [True, "1.25", float("nan"), float("inf")])
def test_bad_original_value_rejected(review_payload, result_payload, value):
    result_payload["effects"][0]["value"] = value
    with pytest.raises(ValueError):
        bind_review_to_result(CriticReview.model_validate(review_payload), result_payload)


def test_missing_source_metadata_rejected(review_payload, result_payload):
    del result_payload["effects"][0]["directional"]
    with pytest.raises(ValueError, match="directional"):
        bind_review_to_result(CriticReview.model_validate(review_payload), result_payload)


@pytest.mark.parametrize("change", ["number", "bool", "missing_path", "extra_path", "wrong_path"])
def test_uncertainty_must_match_exact_source(review_payload, result_payload, change):
    if change == "number":
        review_payload["uncertainty_summary"]["ci"][0] = 999
    elif change == "bool":
        review_payload["uncertainty_summary"]["ci"][0] = True
    elif change == "missing_path":
        del review_payload["uncertainty_source_paths"]["ci"]
    elif change == "extra_path":
        review_payload["uncertainty_source_paths"]["extra"] = ["ci"]
    else:
        review_payload["uncertainty_source_paths"]["ci"] = ["sample", "n"]
    with pytest.raises(ValueError):
        bind_review_to_result(CriticReview.model_validate(review_payload), result_payload)


def test_path_keys_are_literal_not_expressions():
    assert resolve_result_path({"a.b/[]": [{"0": 1.25}]}, ["a.b/[]", 0, "0"]) == 1.25


def test_effectless_review_is_allowed():
    review = CriticReview(verdict="accept", claim="데이터 구조를 확인했다")
    assert bind_review_to_result(review, {"columns": ["x"]}).effect_summary == []


@pytest.mark.parametrize("wrong", [999.0, -1.25])
def test_bad_critic_goes_to_rejection(harness, wrong):
    state, deps = harness
    deps.critic.review.return_value["effect_summary"][0]["value"] = wrong
    update = asyncio.run(critique(state, deps))
    assert update["review"]["verdict"] == "unreviewed"
    assert route_after_critique({**state, **update}) == "reject_attempt"
    assert state["evidence"] == []
    assert any(event["event"] == "CRITIC_RESPONSE_INVALID" for event in deps.events)


def test_missing_path_cannot_be_accepted_from_old_critic(harness):
    state, deps = harness
    del deps.critic.review.return_value["effect_summary"][0]["source_path"]
    update = asyncio.run(critique(state, deps))
    assert route_after_critique({**state, **update}) == "reject_attempt"


def test_valid_review_is_committed_with_provenance(harness):
    state, deps = harness
    state.update(asyncio.run(critique(state, deps)))
    assert route_after_critique(state) == "commit_evidence"
    update = asyncio.run(commit_evidence(state, deps))
    evidence = update["evidence"][0]
    assert evidence["effect_summary"][0]["value"] == 1.25
    assert evidence["uncertainty_source_paths"] == {"ci": ["ci"], "sample": ["sample"]}
    assert len(list(deps.artifacts.program_cache_dir.glob("*.json"))) == 1


def test_commit_rechecks_review_before_any_side_effect(harness):
    state, deps = harness
    state["review"] = deepcopy(deps.critic.review.return_value)
    state["review"]["effect_summary"][0]["value"] = 999
    with pytest.raises(ValueError, match="value"):
        asyncio.run(commit_evidence(state, deps))
    assert state["evidence"] == []
    assert list(deps.artifacts.program_cache_dir.glob("*.json")) == []
    assert not any(event["event"] == "EVIDENCE_COMMITTED" for event in deps.events)
