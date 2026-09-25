"""AIA와 단일 모델 baseline을 같은 평가 함수로 반복 비교한다."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .contracts import EffectEstimate, StopCode
from .data.profiling import load_dataframe, profile_dataframe
from .evaluation import CHURN_SPEC, GoldenSpec, score_run
from .execution.remote_sandbox import build_sandbox
from .llm.memory import AnalysisMemory, load_bundled_wiki
from .llm.services import build_role_client, build_services
from .runtime import ArtifactStore, RuntimeDeps
from .service import AnalysisService, open_checkpointer

# baseline에도 AIA와 같은 수준의 실행 재시도 1회를 허용한다.
MAX_SINGLE_REPAIRS = 1
SANDBOX_TIMEOUT_SECONDS = 120.0
LLM_TIMEOUT_SECONDS = 180.0

# 결과 레코드

@dataclass(frozen=True)
class TrialResult:
    """한 팔의 한 번 실행. 집계 전 원자료이며 trials.jsonl에 그대로 남는다."""

    arm: str
    run_id: str
    seed_index: int
    score: Any = None          # GoldenScore | None
    llm_calls: int = 0
    total_tokens: int = 0
    llm_seconds: float = 0.0
    wall_seconds: float = 0.0
    stop_code: str | None = None
    rejected_by_stage: dict[str, int] = field(default_factory=dict)
    # 형식 오류로 제외한 효과 항목 수.
    dropped_effects: int = 0
    # 실패 시행의 오류.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.score is not None

    def as_record(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "run_id": self.run_id,
            "seed_index": self.seed_index,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.llm_seconds, 2),
            "wall_seconds": round(self.wall_seconds, 2),
            "stop_code": self.stop_code,
            "rejected_by_stage": self.rejected_by_stage,
            "dropped_effects": self.dropped_effects,
            "error": self.error,
            "score": self.score.model_dump(mode="json") if self.score else None,
        }

class Arm(Protocol):
    name: str

    async def run(
        self, *, dataset: Path, objective: str, max_iterations: int,
        time_budget_seconds: float, run_id: str,
    ) -> dict[str, Any]:
        """State 모양(최소 evidence/sub_goals/rejected/iteration/stop_code)을 돌려준다."""
        ...

# 비용 — 이미 남는 이벤트를 읽기만 한다(새 계측 없음)

def events_path(workspace: Path, run_id: str) -> Path:
    return Path(workspace) / "events" / f"{run_id}.jsonl"

def read_llm_cost(path: Path) -> dict[str, Any]:
    calls = tokens = failed = 0
    seconds = 0.0
    if not Path(path).is_file():
        return {"llm_calls": 0, "total_tokens": 0, "llm_seconds": 0.0, "llm_failed": 0}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = record.get("event")
            if name == "LLM_CALL_COMPLETED":
                calls += 1
                tokens += int(record.get("total_tokens") or 0)
                seconds += float(record.get("latency_seconds") or 0.0)
            elif name == "LLM_CALL_FAILED":
                failed += 1
    return {
        "llm_calls": calls, "total_tokens": tokens,
        "llm_seconds": round(seconds, 2), "llm_failed": failed,
    }

def _rejected_by_stage(state: dict[str, Any]) -> dict[str, int]:
    """Critic이 몇 개를 걸렀나 — ablation 팔 없이 State에서 공짜로 읽는다."""
    counts = Counter(
        str(record.get("stage") or "unknown") for record in state.get("rejected") or []
    )
    return dict(sorted(counts.items()))

# aia 팔 — 현행 그대로

class AiaArm:
    name = "aia"

    def __init__(
        self, *, workspace: Path,
        services_factory=build_services, sandbox_factory=build_sandbox,
    ) -> None:
        self.workspace = Path(workspace)
        self.services_factory = services_factory
        self.sandbox_factory = sandbox_factory

    async def run(
        self, *, dataset: Path, objective: str, max_iterations: int,
        time_budget_seconds: float, run_id: str,
    ) -> dict[str, Any]:
        # 반복 시행 간 메모리를 격리한다.
        memory_root = self.workspace / "memory" / run_id
        memory = AnalysisMemory(
            wiki_path=memory_root / "ANALYSIS_WIKI.md",
            notebook_path=memory_root / "notebook.md",
            wiki_seed=load_bundled_wiki(),
        )
        planner, codegen, critic, clients = self.services_factory(memory=memory)
        deps = RuntimeDeps(
            artifacts=ArtifactStore(self.workspace / "artifacts"),
            sandbox=self.sandbox_factory(),
            planner=planner, codegen=codegen, critic=critic,
            workdir_root=self.workspace / "work",
            events_log_path=events_path(self.workspace, run_id),
        )
        for client in clients:
            client.event_sink = lambda name, payload: deps.event(
                name, payload, run_id=run_id,
            )
        try:
            async with open_checkpointer(self.workspace / "checkpoints.sqlite") as saver:
                handle = await AnalysisService(deps, saver).start(
                    dataset_path=dataset, objective=objective,
                    max_iterations=max_iterations,
                    time_budget_seconds=time_budget_seconds,
                    run_id=run_id,
                )
            return handle.state
        finally:
            for client in clients:
                await client.aclose()

# single 팔 — 단일 모델, 검증·게이트 없음

# baseline은 출력 계약과 실행 예산을 맞추고 검증/게이트만 제외한다.
SINGLE_SYSTEM = """# 역할

당신은 데이터 분석가입니다. 목표에 답하는 분석 코드를 직접 작성하고, 그 결과에서
결론까지 스스로 냅니다. 검토자는 없습니다.

## 실행 규칙

- `analyze(df)` 함수 하나를 정의하고 JSON 직렬화 가능한 dict를 반환하십시오.
- 사용 가능한 이름은 `pd`, `np`, `stats`, `math`, `statistics`, `plt`입니다.
- `pandas`, `numpy`, `scipy`, `sklearn`, `statsmodels`, `matplotlib`만
  추가 import할 수 있습니다.
- 파일 읽기, 네트워크, 환경변수 접근은 금지됩니다.
- 결과 전체는 UTF-8 JSON 기준 64KB 이하여야 합니다.
- 실패를 결과 dict로 위장하지 말고 계산할 수 없으면 예외를 내십시오.

## 반환 dict에 반드시 넣을 두 키

- `claim`: 이번 분석 결과가 직접 뒷받침하는 결론 한 문장.
- `effect_summary`: 아래 모양의 객체 배열.

```json
[{"column": "효과를 측정한 실제 컬럼", "metric": "결과에 있는 지표",
  "value": 0.0, "directional": true}]
```

- `effect_summary`에는 실제 결과에 있는 값만 넣습니다.
- `directional`은 값의 부호가 효과 방향인 대비·차이·계수 등에만 `true`입니다.
  그룹별 수준, 표본 수와 단순 건수는 `false`이거나 효과 목록에서 제외합니다.

## 반복

이전 회차의 주장이 주어지면 이미 답한 것을 되풀이하지 말고 목표에서 남은 부분을
분석하십시오. 직전 실행이 실패했다면 같은 방향을 유지한 채 원인을 고치십시오.

## 응답 형식

설명 문장 없이 JSON 객체 하나만 반환하십시오: `{"code": "def analyze(df): ..."}`
"""

def _default_single_client():
    client = build_role_client("CODEGEN")
    if client is None:
        raise RuntimeError("CODEGEN_LLM_PROVIDER가 none이라 single 팔을 돌릴 수 없다")
    return client

class SingleArm:
    """1콜로 코드를 만들고, 나온 값을 그대로 근거로 채택한다.

    빼는 것: Manifest 선언 검사, 결과 스키마 검증, 독립 Critic, 중복 억제, 목표 축.
    주는 것: 같은 회차 예산, 같은 샌드박스, 크래시 재생성 1회.
    """

    name = "single"

    def __init__(
        self, *, workspace: Path,
        client_factory=_default_single_client, sandbox_factory=build_sandbox,
    ) -> None:
        self.workspace = Path(workspace)
        self.client_factory = client_factory
        self.sandbox_factory = sandbox_factory

    async def run(
        self, *, dataset: Path, objective: str, max_iterations: int,
        time_budget_seconds: float, run_id: str,
    ) -> dict[str, Any]:
        frame = load_dataframe(dataset)
        profile = profile_dataframe(frame, dataset_id=Path(dataset).stem)
        columns = [
            {"name": c.name, "dtype": c.dtype, "nulls": c.null_count,
             "unique": c.unique_count}
            for c in profile.columns
        ]

        deps = RuntimeDeps(
            artifacts=ArtifactStore(self.workspace / "artifacts"),
            sandbox=self.sandbox_factory(),
            workdir_root=self.workspace / "work",
            events_log_path=events_path(self.workspace, run_id),
        )
        client = self.client_factory()
        client.event_sink = lambda name, payload: deps.event(
            name, payload, run_id=run_id,
        )

        evidence: list[dict[str, Any]] = []
        dropped = 0
        deadline = time.monotonic() + time_budget_seconds
        stop_code = StopCode.MAX_ITERATIONS_REACHED.value
        iteration = 0
        try:
            for iteration in range(1, max_iterations + 1):
                if time.monotonic() >= deadline:
                    stop_code = StopCode.BUDGET_EXHAUSTED.value
                    break
                record, lost = await self._round(
                    client=client, deps=deps, dataset=dataset, objective=objective,
                    columns=columns, evidence=evidence, run_id=run_id,
                    iteration=iteration,
                )
                dropped += lost
                if record is not None:
                    evidence.append(record)
        finally:
            await client.aclose()

        return {
            "evidence": evidence,
            "sub_goals": [],
            "rejected": [],
            "iteration": iteration,
            "stop_code": stop_code,
            "dropped_effects": dropped,
        }

    async def _round(
        self, *, client, deps: RuntimeDeps, dataset: Path, objective: str,
        columns: list[dict[str, Any]], evidence: list[dict[str, Any]],
        run_id: str, iteration: int,
    ) -> tuple[dict[str, Any] | None, int]:
        previous_error: str | None = None
        for attempt in range(MAX_SINGLE_REPAIRS + 1):
            code = await self._ask_for_code(
                client=client, objective=objective, columns=columns,
                evidence=evidence, previous_error=previous_error,
            )
            if code is None:
                return None, 0
            workdir = self.workspace / "work" / run_id / f"i{iteration}a{attempt}"
            workdir.mkdir(parents=True, exist_ok=True)
            try:
                result = await deps.sandbox.run(
                    code=code, dataset_path=str(dataset), workdir=workdir,
                    timeout_seconds=SANDBOX_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # noqa: BLE001 - 한 시행이 죽어도 나머지는 돌린다
                previous_error = f"{type(exc).__name__}: {exc}"[:2000]
                deps.event("SINGLE_EXECUTION_CRASHED", {"error": previous_error[:400]},
                           run_id=run_id, iteration=iteration)
                continue
            if not result.ok or result.result_path is None:
                previous_error = (result.refused_reason or result.stderr or "")[-2000:]
                deps.event("SINGLE_EXECUTION_FAILED", {"error": previous_error[:400]},
                           run_id=run_id, iteration=iteration)
                continue
            try:
                payload = json.loads(result.result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                previous_error = f"결과 JSON을 읽을 수 없다: {exc}"[:2000]
                continue
            return self._to_evidence(
                payload, iteration=iteration, deps=deps, run_id=run_id,
            )
        return None, 0

    async def _ask_for_code(
        self, *, client, objective: str, columns: list[dict[str, Any]],
        evidence: list[dict[str, Any]], previous_error: str | None,
    ) -> str | None:
        payload: dict[str, Any] = {
            "목표": objective,
            "데이터 컬럼": columns,
            "지금까지의 주장": [
                {"claim": item["claim"], "effect_summary": item["effect_summary"]}
                for item in evidence
            ],
        }
        if previous_error:
            payload["직전 실행 실패"] = previous_error
        try:
            raw = await client.generate_json(
                prompt=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                system=SINGLE_SYSTEM,
                timeout_seconds=LLM_TIMEOUT_SECONDS,
                role="single_codegen",
            )
        except Exception:  # noqa: BLE001 - 실패는 LLM_CALL_FAILED로 이미 남는다
            return None
        code = raw.get("code")
        return code if isinstance(code, str) and code.strip() else None

    def _to_evidence(
        self, payload: Any, *, iteration: int, deps: RuntimeDeps, run_id: str,
    ) -> tuple[dict[str, Any] | None, int]:
        """나온 값을 그대로 근거로 채택한다. 심사가 없다는 것이 이 팔의 정의다."""
        if not isinstance(payload, dict):
            return None, 0
        effects: list[dict[str, Any]] = []
        dropped = 0
        for item in payload.get("effect_summary") or []:
            try:
                # 값은 바꾸지 않고 평가 가능한 형식만 검증한다.
                effects.append(
                    EffectEstimate.model_validate(item).model_dump(mode="json")
                )
            except Exception:  # noqa: BLE001
                dropped += 1
        if dropped:
            deps.event("SINGLE_EFFECT_DROPPED", {"count": dropped},
                       run_id=run_id, iteration=iteration)
        return {
            "evidence_id": uuid4().hex,
            "iteration": iteration,
            "claim": str(payload.get("claim") or ""),
            # Manifest가 없으므로 효과가 보고된 컬럼을 사용 컬럼으로 기록한다.
            "columns_used": sorted({effect["column"] for effect in effects}),
            "effect_summary": effects,
        }, dropped

ARMS: dict[str, Any] = {"aia": AiaArm, "single": SingleArm}

# 실행

async def run_trial(
    arm: Arm, *, dataset: Path, objective: str, max_iterations: int,
    time_budget_seconds: float, workspace: Path, seed_index: int,
    spec: GoldenSpec = CHURN_SPEC,
) -> TrialResult:
    run_id = f"bench-{arm.name}-{seed_index}-{uuid4().hex[:6]}"
    started = time.monotonic()
    try:
        state = await arm.run(
            dataset=dataset, objective=objective, max_iterations=max_iterations,
            time_budget_seconds=time_budget_seconds, run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001 - 한 시행이 죽어도 나머지는 돌린다
        cost = read_llm_cost(events_path(workspace, run_id))
        cost.pop("llm_failed", None)
        return TrialResult(
            arm=arm.name, run_id=run_id, seed_index=seed_index, score=None,
            wall_seconds=round(time.monotonic() - started, 2),
            error=f"{type(exc).__name__}: {exc}"[:500], **cost,
        )
    cost = read_llm_cost(events_path(workspace, run_id))
    cost.pop("llm_failed", None)
    return TrialResult(
        arm=arm.name, run_id=run_id, seed_index=seed_index,
        score=score_run(state, spec),
        wall_seconds=round(time.monotonic() - started, 2),
        stop_code=state.get("stop_code"),
        rejected_by_stage=_rejected_by_stage(state),
        dropped_effects=int(state.get("dropped_effects") or 0),
        **cost,
    )

async def run_bench(
    *, arm_names: list[str], repeat: int, dataset: Path, objective: str,
    max_iterations: int, time_budget_seconds: float, out_dir: Path,
    spec: GoldenSpec = CHURN_SPEC, arm_factory=None,
) -> list[TrialResult]:
    """팔 × 반복을 순차 실행한다. 병렬로 돌리지 않는다. 같은 GPU를 공유하므로
    동시에 돌리면 지연 수치가 서로를 오염시킨다."""
    out_dir = Path(out_dir)
    workspace = out_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"

    def make(name: str) -> Arm:
        if arm_factory is not None:
            return arm_factory(name, workspace)
        return ARMS[name](workspace=workspace)

    trials: list[TrialResult] = []
    for index in range(1, repeat + 1):
        for name in arm_names:
            trial = await run_trial(
                make(name), dataset=dataset, objective=objective,
                max_iterations=max_iterations,
                time_budget_seconds=time_budget_seconds,
                workspace=workspace, seed_index=index, spec=spec,
            )
            trials.append(trial)
            # 매 시행마다 append한다. 도중에 죽어도 앞선 유료 실행을 잃지 않는다.
            with trials_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trial.as_record(), ensure_ascii=False) + "\n")
    return trials

# 보고

def _arm_order(trials: list[TrialResult]) -> list[str]:
    seen: list[str] = []
    for trial in trials:
        if trial.arm not in seen:
            seen.append(trial.arm)
    return seen

def _spread(values: list[float], fmt: str = "{:.2f}") -> str:
    """평균만 쓰지 않는다. 데이터는 시드 고정이라 변동은 전부 LLM 샘플링에서 온다."""
    if not values:
        return "—"
    body = fmt.format(statistics.median(values))
    if len(values) > 1 and min(values) != max(values):
        body += f" ({fmt.format(min(values))}~{fmt.format(max(values))})"
    return body

def _effect_table(trials: list[TrialResult], spec: GoldenSpec) -> list[str]:
    """효과별 판정. 진짜 효과가 4개뿐이라 recall 숫자 하나는 해상도가 너무 거칠다."""
    real = spec.real_effects
    lines = [
        "| arm | " + " | ".join(e.column for e in real) + " | 미끼 오보 |",
        "|" + "---|" * (len(real) + 2),
    ]
    for arm in _arm_order(trials):
        done = [t for t in trials if t.arm == arm and t.ok]
        cells = []
        for effect in real:
            findings = [
                f for t in done
                for f in t.score.findings if f.effect.column == effect.column
            ]
            found = [f for f in findings if f.found]
            signed = [f for f in found if f.direction_matches is not None]
            cell = f"{len(found)}/{len(findings)}"
            if signed:
                right = sum(1 for f in signed if f.direction_matches)
                cell += f" 부호 {right}/{len(signed)}"
            cells.append(cell)
        decoys = Counter(
            claim.split(":")[0] for t in done for claim in t.score.decoy_claims
        )
        cells.append(", ".join(f"{c}×{n}" for c, n in sorted(decoys.items())) or "—")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines

def _trial_rows(trials: list[TrialResult]) -> list[str]:
    """시행 하나에 한 줄. 실패한 시행도 지우지 않고 이유와 함께 남긴다."""
    rows = []
    for trial in trials:
        if not trial.ok:
            rows.append(
                f"| {trial.arm} | {trial.seed_index} | 실패: {trial.error} | | | | "
                f"{trial.llm_calls} | {trial.total_tokens:,} | "
                f"{trial.wall_seconds:.0f}s | — | — |"
            )
            continue
        score = trial.score
        sign = "n/a" if score.sign_accuracy is None else f"{score.sign_accuracy:.0%}"
        rejected = ", ".join(f"{k}:{v}" for k, v in trial.rejected_by_stage.items()) or "—"
        rows.append(
            f"| {trial.arm} | {trial.seed_index} | {score.recall:.0%} | "
            f"{score.directional_rate:.0%} | {sign} | {score.false_positives} | "
            f"{trial.llm_calls} | {trial.total_tokens:,} | {trial.wall_seconds:.0f}s | "
            f"{trial.stop_code or '—'} | {rejected} |"
        )
    return rows

def _arm_rows(trials: list[TrialResult]) -> list[str]:
    """팔 하나에 한 줄. 평균이 아니라 중앙값과 최소~최대를 쓴다."""
    rows = []
    for arm in _arm_order(trials):
        done = [t for t in trials if t.arm == arm and t.ok]
        total = sum(1 for t in trials if t.arm == arm)
        if not done:
            rows.append(f"| {arm} | 0/{total} | — | — | — | — | — | — |")
            continue
        rows.append(
            f"| {arm} | {len(done)}/{total} | "
            f"{_spread([t.score.recall for t in done], '{:.0%}')} | "
            f"{_spread([t.score.directional_rate for t in done], '{:.0%}')} | "
            f"{_spread([float(t.score.false_positives) for t in done], '{:.0f}')} | "
            f"{_spread([float(t.llm_calls) for t in done], '{:.0f}')} | "
            f"{_spread([float(t.total_tokens) for t in done], '{:,.0f}')} | "
            f"{_spread([t.wall_seconds for t in done], '{:.0f}s')} |"
        )
    return rows

def _reading_notes(spec: GoldenSpec) -> list[str]:
    resolution = 1 / len(spec.real_effects) if spec.real_effects else 0.0
    return [
        "",
        "## 읽는 법",
        "",
        (
            f"- 진짜 효과가 {len(spec.real_effects)}개뿐이라 recall 해상도는 "
            f"{resolution:.2f}다. 시행 수가 적으면 경향까지만 말하고 유의성은 주장하지 않는다."
        ),
        "- 비용을 숨기지 않는다. 정확도가 올라도 콜이 몇 배면 그것도 결과다.",
        "- `stop_code`가 다른 시행은 같은 줄에서 비교하지 않는다. 예산 소진 런의 낮은",
        "  recall은 루프의 실패가 아니라 예산의 결과다.",
        "- 미끼 오보(FP)는 recall보다 중요할 수 있다. 없는 효과를 주장하지 않는 것이",
        "  '검증 가능한 근거'라는 이 프로젝트의 정체성에 더 가깝다.",
        "- single이 이기는 항목도 그대로 싣는다. 유리한 것만 고르면 표 전체가 무의미하다.",
    ]

def render_report(
    trials: list[TrialResult], *, objective: str, spec: GoldenSpec = CHURN_SPEC,
    max_iterations: int | None = None,
) -> str:
    stamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    lines = [
        f"# 벤치 결과 — {spec.dataset_id}",
        "",
        f"- 목표: {objective}",
        f"- 회차 상한: {max_iterations if max_iterations is not None else '—'} (팔 공통)",
        f"- 시행: {len(trials)}회 ({', '.join(_arm_order(trials)) or '없음'})",
        f"- 생성: {stamp}",
        "",
        "## 시행별",
        "",
        "| arm | run | recall | 방향 | 부호 | FP | calls | tokens | wall | stop_code | rejected |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *_trial_rows(trials),
        "",
        "## 팔별 요약 (중앙값, 괄호는 최소~최대)",
        "",
        "| arm | 시행 | recall | 방향 | FP | calls | tokens | wall |",
        "|---|---|---|---|---|---|---|---|",
        *_arm_rows(trials),
        "",
        "## 효과별",
        "",
        *_effect_table(trials, spec),
    ]
    if dropped := sum(t.dropped_effects for t in trials):
        lines += ["", f"> single 팔에서 형식을 어겨 버린 효과 항목 {dropped}개."]
    return "\n".join(lines + _reading_notes(spec))
def bench(
    *, arm_names: list[str], repeat: int, dataset: Path, objective: str,
    max_iterations: int, time_budget_seconds: float, out_dir: Path,
    spec: GoldenSpec = CHURN_SPEC,
) -> Path:
    """벤치를 돌리고 report.md 경로를 돌려준다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = asyncio.run(run_bench(
        arm_names=arm_names, repeat=repeat, dataset=dataset, objective=objective,
        max_iterations=max_iterations, time_budget_seconds=time_budget_seconds,
        out_dir=out_dir, spec=spec,
    ))
    report_path = out_dir / "report.md"
    report_path.write_text(
        render_report(
            trials, objective=objective, spec=spec, max_iterations=max_iterations,
        ),
        encoding="utf-8",
    )
    return report_path
