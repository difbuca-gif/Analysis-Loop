"""유지보수 리팩터링이 기존 import surface를 깨지 않는지 확인한다."""

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
