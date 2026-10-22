import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from analysis_loop_v3.contracts import ArtifactRef, ResearchIntent
from analysis_loop_v3.orchestration.nodes.program import lookup_program
from analysis_loop_v3.service import AnalysisService


def test_pre_patch_checkpoint_cannot_resume(harness):
    state, deps = harness
    service = AnalysisService(deps, InMemorySaver())

    async def check():
        await service.graph.aupdate_state(
            {"configurable": {"thread_id": "legacy-run"}},
            {**state, "run_id": "legacy-run", "schema_version": 8},
            as_node="critique",
        )
        with pytest.raises(ValueError, match="지원하지 않는 state schema"):
            await service.resume(thread_id="legacy-run")

    asyncio.run(check())
    assert not deps.events


def test_pre_patch_program_cache_is_not_reused(harness):
    state, deps = harness
    intent = ResearchIntent.model_validate(state["intent"])
    old_signature = "intent-v2:" + intent.reuse_signature().split(":", 1)[1]
    deps.artifacts.save_validated_program(
        schema_fingerprint=state["profile"]["schema_fingerprint"],
        intent_reuse_signature=old_signature,
        code_ref=ArtifactRef.model_validate(state["program_ref"]), manifest=state["manifest"],
    )
    update = asyncio.run(lookup_program(state, deps))
    assert update["program_ref"] is None
    # 구버전 파일 자체는 삭제하지 않는다.
    assert len(list(deps.artifacts.program_cache_dir.glob("*.json"))) == 1
