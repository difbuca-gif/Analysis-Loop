"""유지보수 리팩터링이 기존 import surface와 factory 동작을 깨지 않는지 확인한다."""

from types import SimpleNamespace

from analysis_loop_v3.llm.services import (
    LLMCritic,
    LLMCodegen,
    LLMPlanner,
    _same_endpoint_model,
    build_services,
    describe_services,
    resolve_provider,
)
from analysis_loop_v3.orchestration.graph import (
    route_after_critique,
    route_after_execute,
    route_after_plan,
    route_after_validate_result,
)
from analysis_loop_v3.orchestration.nodes.planning import _apply_goal_verdict
from analysis_loop_v3.orchestration.nodes.program import validate_code
from analysis_loop_v3.orchestration.nodes.progress import _run_summary, _write_report
from analysis_loop_v3.orchestration.nodes.running import validate_result
from analysis_loop_v3.runtime import (
    ArtifactStore,
    RuntimeDeps,
    SandboxResult,
)
from analysis_loop_v3.validation.checks import (
    check_code_matches_manifest,
    check_manifest,
    check_result,
    validate_critic_references,
    validate_intent_references,
)


def test_legacy_graph_route_imports_remain_available():
    assert callable(route_after_plan)
    assert callable(route_after_execute)
    assert callable(route_after_validate_result)
    assert callable(route_after_critique)


def test_legacy_node_helper_imports_remain_available():
    assert callable(_apply_goal_verdict)
    assert callable(validate_code)
    assert callable(validate_result)
    assert callable(_run_summary)
    assert callable(_write_report)


def test_runtime_compatibility_exports_remain_available():
    assert ArtifactStore is not None
    assert RuntimeDeps is not None
    assert SandboxResult is not None


def test_llm_services_compatibility_exports_remain_available():
    assert LLMPlanner is not None
    assert LLMCodegen is not None
    assert LLMCritic is not None
    assert callable(build_services)
    assert callable(resolve_provider)


def test_validation_checks_facade_remains_available():
    assert callable(validate_intent_references)
    assert callable(validate_critic_references)
    assert callable(check_manifest)
    assert callable(check_code_matches_manifest)
    assert callable(check_result)


def test_same_endpoint_check_accepts_codegen_without_client():
    cli_codegen = SimpleNamespace()
    critic = SimpleNamespace(
        client=SimpleNamespace(model="critic", base_url="http://localhost")
    )

    assert _same_endpoint_model(cli_codegen, critic) is False
    assert describe_services(None, cli_codegen, critic)["codegen"].endswith(":local")
