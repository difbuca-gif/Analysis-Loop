from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from analysis_loop_v3.contracts import (
    ArtifactKind,
    ColumnProfile,
    DatasetProfile,
    GeneratedAnalysis,
    ResearchIntent,
)
from analysis_loop_v3.runtime import ArtifactStore, RuntimeDeps


@pytest.fixture
def result_payload():
    return {
        "effects": [{
            "column": "x", "metric": "mean", "value": 1.25,
            "directional": False, "unit": "score",
        }],
        "ci": [1.0, 1.5],
        "sample": {"n": 4, "method": "all rows"},
    }


@pytest.fixture
def review_payload(result_payload):
    return {
        "verdict": "accept", "claim": "x의 평균은 1.25다.",
        "effect_summary": [{**result_payload["effects"][0], "source_path": ["effects", 0]}],
        "uncertainty_summary": {"ci": [1.0, 1.5], "sample": {"n": 4, "method": "all rows"}},
        "uncertainty_source_paths": {"ci": ["ci"], "sample": ["sample"]},
    }


@pytest.fixture
def harness(tmp_path, result_payload, review_payload):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    intent = ResearchIntent(question="x의 평균은?", hypothesis="평균을 계산할 수 있다")
    code = '''def analyze(df):
    return {"effects": [{"column": "x", "metric": "mean",
                         "value": float(df["x"].mean()),
                         "directional": False, "unit": "score"}],
            "ci": [1.0, 1.5], "sample": {"n": int(len(df)), "method": "all rows"}}
'''
    generated = GeneratedAnalysis(
        intent_id=intent.intent_id, selected_columns=["x"], method="mean", code=code,
        expected_output_schema={"effects": "list", "ci": "list", "sample": "mapping"},
    )
    dataset = artifacts.put_text("x\n0.5\n1.0\n1.5\n2.0\n", kind=ArtifactKind.DATASET, suffix=".csv")
    profile = DatasetProfile(
        dataset_id=dataset.artifact_id, row_count=4, schema_fingerprint="test-schema",
        columns=[ColumnProfile(name="x", dtype="float64", non_null=4, null_count=0, unique_count=4)],
    )
    program = artifacts.put_text(code, kind=ArtifactKind.PROGRAM, suffix=".py")
    result = artifacts.put_json(result_payload, kind=ArtifactKind.RESULT)
    critic = SimpleNamespace(
        review=AsyncMock(return_value=deepcopy(review_payload)),
        generate_report=AsyncMock(return_value=None),
    )
    deps = RuntimeDeps(artifacts=artifacts, sandbox=None, critic=critic, workdir_root=tmp_path)
    state = {
        "run_id": "numeric-hotfix", "intent": intent.model_dump(mode="json"),
        "dataset_ref": dataset.model_dump(mode="json"), "profile": profile.model_dump(),
        "program_ref": program.model_dump(mode="json"),
        "candidate_ref": result.model_dump(mode="json"),
        "manifest": generated.manifest_without_code(), "evidence": [],
        "validation_errors": [], "validation_warnings": [],
    }
    return state, deps
