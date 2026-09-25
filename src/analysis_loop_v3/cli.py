"""체크포인트 기반 분석 루프의 CLI 진입점."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from .contracts import validate_run_id
from .env import load_project_env
from .execution.remote_sandbox import build_sandbox
from .llm.memory import AnalysisMemory, load_bundled_wiki
from .llm.services import build_services, describe_services
from .runtime import ArtifactStore, RuntimeDeps
from .service import AnalysisService, open_checkpointer

DEFAULT_WORKSPACE = Path("runs")


def _workspace(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "workspace", None) or DEFAULT_WORKSPACE)


async def _with_service(args: argparse.Namespace, fn):
    workspace = _workspace(args)
    run_id = getattr(args, "run_id", None) or "unnamed"
    validate_run_id(run_id)
    memory = AnalysisMemory(
        wiki_path=workspace / "memory" / "ANALYSIS_WIKI.md",
        notebook_path=workspace / "memory" / "notebooks" / f"{run_id}.md",
        wiki_seed=load_bundled_wiki(),
    )
    planner, codegen, critic, clients = build_services(memory=memory)
    deps = RuntimeDeps(
        artifacts=ArtifactStore(workspace / "artifacts"),
        sandbox=build_sandbox(),
        planner=planner, codegen=codegen, critic=critic,
        workdir_root=workspace / "work",
        events_log_path=workspace / "events" / f"{run_id}.jsonl",
    )
    for client in clients:
        client.event_sink = lambda name, payload: deps.event(
            name, payload, run_id=run_id,
        )
    try:
        async with open_checkpointer(workspace / "checkpoints.sqlite") as saver:
            return await fn(AnalysisService(deps, saver), deps)
    finally:
        for client in clients:
            await client.aclose()


def _report(handle, deps) -> None:
    print(json.dumps({
        "run_id": handle.run_id,
        "status": handle.status,
        "stop_code": handle.state.get("stop_code"),
        "stop_reason": handle.state.get("stop_reason"),
        "iterations": handle.state.get("iteration"),
        "evidence": len(handle.state.get("evidence") or []),
        "rejected": len(handle.state.get("rejected") or []),
        "issues": handle.state.get("issues") or [],
        "report": (handle.state.get("report_ref") or {}).get("path"),
        "events": [e["event"] for e in deps.events],
    }, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis_loop_v3")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="새 분석 시작")
    run.add_argument("dataset", nargs="?", help="CSV/Parquet 파일 또는 지원 데이터 디렉터리")
    run.add_argument("--sql-file", type=Path, help="PostgreSQL SELECT/WITH 쿼리 파일")
    run.add_argument(
        "--postgres-url-env", default="AIA_POSTGRES_URL",
        help="PostgreSQL URL을 읽을 환경 변수명 (기본: AIA_POSTGRES_URL)",
    )
    run.add_argument("--objective", required=True)
    run.add_argument("--max-iterations", type=int, default=25)
    # 목표 축이 전부 닫혀도 이 시간이 남아있으면 expand_goals가 새 축을 찾아
    # 계속 고도화한다. 회차 상한과 마찬가지로 이것도 "다 채워라"가 아니라
    # "이 안에서 끝내라"는 상한이다.
    run.add_argument("--budget", type=float, default=3600.0)
    run.add_argument("--run-id")

    status = sub.add_parser("status", help="현재 상태")
    status.add_argument("run_id")

    resume = sub.add_parser("resume", help="중단 지점부터 재개")
    resume.add_argument("run_id")
    # 안 주면 deadline_at을 그대로 두고 재개한다(순수 재개). 준 값만큼 지금부터
    # 다시 계산해 예산을 연장하려면 명시적으로 넘긴다.
    resume.add_argument("--budget", type=float, default=None)

    golden = sub.add_parser("golden", help="골든 데이터셋을 생성한다")
    golden.add_argument("--out", default="data/golden_churn.csv")

    score = sub.add_parser("score", help="완료된 실행을 골든 명세로 채점한다")
    score.add_argument("run_id")

    audit = sub.add_parser("audit", help="State·Evidence·Artifact·이벤트 계보를 감사한다")
    audit.add_argument("run_id")
    audit.add_argument("--out", default=None, help="JSON 감사 결과를 저장할 경로")

    bench = sub.add_parser("bench", help="팔별로 반복 실행해 비교표를 만든다")
    bench.add_argument("dataset")
    bench.add_argument("--objective", required=True)
    bench.add_argument("--arms", default="aia,single")
    bench.add_argument("--repeat", type=int, default=3)
    # run의 기본값(25)보다 낮다. 두 팔을 같은 예산으로 여러 번 돌려야 하므로
    # 한 시행이 길면 표 한 장에 몇 시간이 든다.
    bench.add_argument("--max-iterations", type=int, default=8)
    bench.add_argument("--budget", type=float, default=3600.0)
    bench.add_argument("--out", default=None)

    return parser


def _cmd_golden(args: argparse.Namespace) -> int:
    from .evaluation import CHURN_SPEC, write_golden_dataset

    path = write_golden_dataset(args.out)
    print(json.dumps({
        "path": str(path),
        "rows": CHURN_SPEC.row_count,
        "seed": CHURN_SPEC.seed,
        "real_effects": [e.column for e in CHURN_SPEC.real_effects],
        "decoys": [e.column for e in CHURN_SPEC.decoys],
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_bench(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from datetime import UTC, datetime

    from .bench import ARMS
    from .bench import bench as run_bench

    arm_names = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arm_names if name not in ARMS]
    if unknown or not arm_names:
        parser.error(f"--arms는 {', '.join(ARMS)} 중에서 고른다 (잘못된 값: {unknown})")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else _workspace(args) / "bench" / stamp
    report_path = run_bench(
        arm_names=arm_names, repeat=args.repeat, dataset=Path(args.dataset),
        objective=args.objective, max_iterations=args.max_iterations,
        time_budget_seconds=args.budget, out_dir=out_dir,
    )
    print(report_path.read_text(encoding="utf-8"))
    print(f"\n원자료: {out_dir / 'trials.jsonl'}\n보고서: {report_path}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from .evaluation import score_report, score_run

    async def read_state(service: AnalysisService, _deps: RuntimeDeps):
        return await service.snapshot(thread_id=args.run_id)

    handle = asyncio.run(_with_service(args, read_state))
    print(score_report(score_run(handle.state)))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .audit import build_run_audit, read_event_log

    async def read_state(service: AnalysisService, deps: RuntimeDeps):
        return await service.snapshot(thread_id=args.run_id), deps

    handle, deps = asyncio.run(_with_service(args, read_state))
    log_path = deps.events_log_path or (
        _workspace(args) / "events" / f"{args.run_id}.jsonl"
    )
    events, parse_issues = read_event_log(log_path)
    result = build_run_audit(
        handle.state, events, event_parse_issues=parse_issues,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 2


async def _dispatch(args: argparse.Namespace, service: AnalysisService, deps: RuntimeDeps):
    if args.command == "run":
        print(json.dumps(
            describe_services(deps.planner, deps.codegen, deps.critic),
            ensure_ascii=False, indent=2,
        ))
        if args.sql_file:
            if args.dataset:
                raise ValueError("PostgreSQL 입력에는 dataset 경로를 함께 지정할 수 없다")
            database_url = os.environ.get(args.postgres_url_env)
            if not database_url:
                raise ValueError(f"환경 변수 {args.postgres_url_env}에 PostgreSQL URL이 없다")
            from .data.postgres import PostgresSource

            return await service.start(
                postgres_source=PostgresSource(
                    database_url=database_url,
                    query=args.sql_file.read_text(encoding="utf-8"),
                ),
                objective=args.objective,
                max_iterations=args.max_iterations,
                time_budget_seconds=args.budget,
                run_id=args.run_id,
            )
        if not args.dataset:
            raise ValueError("파일 분석에는 dataset 경로가 필요하다")
        return await service.start(
            dataset_path=args.dataset, objective=args.objective,
            max_iterations=args.max_iterations,
            time_budget_seconds=args.budget,
            run_id=args.run_id,
        )
    if args.command == "status":
        return await service.snapshot(thread_id=args.run_id)
    return await service.resume(
        thread_id=args.run_id, time_budget_seconds=args.budget,
    )


def main(argv: list[str] | None = None) -> int:
    load_project_env()

    parser = _build_parser()
    args = parser.parse_args(argv)

    # RuntimeDeps를 만들기 전에 실제 run ID를 확정한다. 그래야 새 실행의 이벤트가
    # unnamed.jsonl에 기록되지 않고 체크포인트 thread와 같은 이름을 쓴다.
    if args.command == "run" and not args.run_id:
        args.run_id = uuid4().hex[:12]

    # golden과 bench는 그래프도 LLM도 필요 없다. 서비스를 세우기 전에 처리한다.
    if args.command == "golden":
        return _cmd_golden(args)
    if args.command == "bench":
        return _cmd_bench(args, parser)
    if args.command == "score":
        return _cmd_score(args)
    if args.command == "audit":
        return _cmd_audit(args)

    async def runner(service: AnalysisService, deps: RuntimeDeps):
        return await _dispatch(args, service, deps), deps

    handle, deps = asyncio.run(_with_service(args, runner))
    _report(handle, deps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
