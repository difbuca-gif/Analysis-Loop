"""알려진 효과를 심은 합성 데이터 기반 상대 품질 평가."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .contracts import EffectEstimate

Direction = Literal["positive", "negative", "none"]

GOLDEN_SEED = 20260819
GOLDEN_ROWS = 1500

class PlantedEffect(BaseModel):
    """데이터에 의도적으로 심은(또는 심지 않은) 관계."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    # 이 효과에 관여하는 컬럼. 근거가 이 컬럼을 썼는지로 매칭한다.
    column: str
    direction: Direction
    # 생성식에 쓴 실제 계수. direction이 none이면 0.
    coefficient: float = 0.0
    # 찾기 난이도 — 낮은 효과를 못 찾는 것과 강한 효과를 못 찾는 것은 무게가 다르다.
    strength: Literal["strong", "moderate", "weak", "none"] = "moderate"

    @property
    def is_real(self) -> bool:
        return self.direction != "none"

class GoldenSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    seed: int
    row_count: int
    target: str
    effects: list[PlantedEffect]

    @property
    def real_effects(self) -> list[PlantedEffect]:
        return [e for e in self.effects if e.is_real]

    @property
    def decoys(self) -> list[PlantedEffect]:
        return [e for e in self.effects if not e.is_real]

CHURN_SPEC = GoldenSpec(
    dataset_id="golden_churn_v1",
    seed=GOLDEN_SEED,
    row_count=GOLDEN_ROWS,
    target="churn",
    effects=[
        PlantedEffect(name="장기 계약은 이탈을 낮춘다", column="contract",
                      direction="negative", coefficient=-0.30, strength="strong"),
        PlantedEffect(name="가입 기간이 길수록 이탈이 낮다", column="tenure",
                      direction="negative", coefficient=-0.0040, strength="moderate"),
        PlantedEffect(name="지원 요청이 많으면 이탈이 높다", column="support_calls",
                      direction="positive", coefficient=0.0300, strength="moderate"),
        PlantedEffect(name="월 요금이 높으면 이탈이 높다", column="monthly_charges",
                      direction="positive", coefficient=0.0014, strength="weak"),
        # 미끼 — 이탈과 아무 관계 없이 생성한다.
        PlantedEffect(name="지역은 이탈과 무관하다", column="region",
                      direction="none", strength="none"),
        PlantedEffect(name="기기 유형은 이탈과 무관하다", column="device",
                      direction="none", strength="none"),
    ],
)

def generate_golden_dataset(spec: GoldenSpec = CHURN_SPEC) -> pd.DataFrame:
    """명세대로 데이터를 만든다(시드 고정). 재생성 가능하므로 파일을 커밋하지 않는다."""
    rng = np.random.default_rng(spec.seed)
    n = spec.row_count
    coef = {e.column: e.coefficient for e in spec.effects}

    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    # 계약을 순서형으로 본다: 월 단위 0, 1년 1, 2년 2. 계수가 음수이므로
    # 장기 계약일수록 이탈이 낮아진다.
    contract_rank = np.select(
        [contract == "month-to-month", contract == "one-year"], [0, 1], default=2
    )
    tenure = rng.integers(1, 72, n)
    charges = np.round(rng.normal(70, 25, n).clip(20, 130), 2)
    support = rng.poisson(1.2, n)

    logit = (
        0.30
        + coef["contract"] * contract_rank
        + coef["tenure"] * tenure
        + coef["support_calls"] * support
        + coef["monthly_charges"] * (charges - 70)
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "contract": contract,
        "tenure": tenure,
        "monthly_charges": charges,
        "support_calls": support,
        # 미끼: 이탈과 독립적으로 생성한다.
        "region": rng.choice(["A", "B", "C", "D"], n),
        "device": rng.choice(["ios", "android", "web"], n),
        "churn": churn,
    })

def write_golden_dataset(path: Path | str, spec: GoldenSpec = CHURN_SPEC) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_golden_dataset(spec).to_csv(path, index=False)
    return path

# 채점

class EffectFinding(BaseModel):
    """심은 효과 하나에 대한 판정."""

    model_config = ConfigDict(extra="forbid")

    effect: PlantedEffect
    found: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    reported_values: dict[str, float] = Field(default_factory=dict)
    direction_matches: bool | None = None
    note: str = ""

class GoldenScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    evidence_count: int
    iterations: int

    # 기준 효과 중 포착한 비율.
    recall: float
    # 찾은 것 중 부호가 맞은 비율. 방향을 못 읽으면 None으로 세지 않는다.
    sign_accuracy: float | None
    # 미끼를 "효과 있다"고 주장한 수. 다중검정 보정 필요성의 지표다.
    false_positives: int
    # 축 중 수렴 비율. 목표 대비 진척.
    goal_convergence: float | None
    # 근거 하나당 포착한 기준 효과 수.
    evidence_efficiency: float
    # 다룬 효과 중 방향까지 밝힌 비율. 그룹별 수준만 보고하면 0이다.
    # "정량화했다"와 "방향을 답했다"는 다르며, 목표는 대개 후자다.
    directional_rate: float

    findings: list[EffectFinding]
    decoy_claims: list[str] = Field(default_factory=list)

    def summary_line(self) -> str:
        sign = "n/a" if self.sign_accuracy is None else f"{self.sign_accuracy:.0%}"
        goal = "n/a" if self.goal_convergence is None else f"{self.goal_convergence:.0%}"
        return (
            f"발견율 {self.recall:.0%} | 방향제시 {self.directional_rate:.0%} | "
            f"부호정확 {sign} | 위양성 {self.false_positives} | "
            f"축수렴 {goal} | 효율 {self.evidence_efficiency:.2f} | 근거 {self.evidence_count}"
        )

def _estimates_for(effect_summary: list[Any], column: str) -> list[EffectEstimate]:
    """문자열 추측 없이 명시적으로 같은 컬럼을 가리키는 효과만 반환한다."""
    estimates = [EffectEstimate.model_validate(item) for item in (effect_summary or [])]
    return [item for item in estimates if item.column == column]

def _collect_finding(
    effect: PlantedEffect, evidence: list[dict[str, Any]],
) -> tuple[EffectFinding, list[float]]:
    """이 효과를 다룬 근거들을 모은다. (판정 전 finding, 방향성 수치들)"""
    finding = EffectFinding(effect=effect)
    directional: list[float] = []
    for record in evidence:
        if effect.column not in (record.get("columns_used") or []):
            continue
        estimates = _estimates_for(record.get("effect_summary") or [], effect.column)
        if not estimates:
            # 컬럼은 다뤘지만 수치가 없다. "이 변수를 봤다"와 "이 변수의 효과를
            # 밝혔다"는 다르다.
            continue
        finding.found = True
        finding.evidence_ids.append(record["evidence_id"])
        for estimate in estimates:
            key = f"{estimate.metric}[{len(finding.reported_values)}]"
            finding.reported_values[key] = estimate.value
            if estimate.directional:
                directional.append(estimate.value)
    return finding, directional

def _judge_finding(
    finding: EffectFinding, effect: PlantedEffect, directional: list[float],
) -> str | None:
    """finding에 부호 판정과 비고를 채운다. 미끼를 효과라고 주장했으면 그 문구를 반환."""
    if not finding.found:
        return None

    if effect.is_real:
        if not directional:
            # 그룹별 수준만 보고했다. 정량화는 했지만 방향은 밝히지 않았다.
            # 이탈률 0.56은 양의 효과가 아니라 그냥 수준이다. 방향은 그룹 간
            # 대비에서만 나오며, 그것을 계산하지 않았다면 판정할 수 없다.
            finding.note = (
                f"수준만 보고({len(finding.reported_values)}개). "
                "부호를 읽을 대비 통계량이 없다"
            )
            return None
        mean = sum(directional) / len(directional)
        finding.direction_matches = (mean > 0) == (effect.direction == "positive")
        finding.note = f"부호 통계량 {mean:+.4f} / 실제 {effect.coefficient:+.4f}"
        return None

    # 미끼에 0이 아닌 방향성 효과를 보고했다 = 위양성. 그룹별 수준만 제시한 것은
    # 효과 주장으로 세지 않는다.
    nonzero = [v for v in directional if abs(v) > 1e-9]
    if not nonzero:
        return None
    finding.note = "미끼인데 효과를 보고했다"
    return f"{effect.column}: {nonzero[:3]}"

def score_run(state: dict[str, Any], spec: GoldenSpec = CHURN_SPEC) -> GoldenScore:
    """완료된 실행을 채점한다. State에 남은 것만 본다(artifact를 다시 읽지 않는다).

    되먹임에 실려야 할 정보가 실제로 실렸는지가 함께 검증된다.
    """
    evidence = state.get("evidence") or []
    findings: list[EffectFinding] = []
    decoy_claims: list[str] = []

    for effect in spec.effects:
        finding, directional = _collect_finding(effect, evidence)
        if claim := _judge_finding(finding, effect, directional):
            decoy_claims.append(claim)
        findings.append(finding)

    real = [f for f in findings if f.effect.is_real]
    found_real = [f for f in real if f.found]
    signed = [f for f in found_real if f.direction_matches is not None]

    goals = state.get("sub_goals") or []
    convergence = (
        sum(1 for g in goals if g.get("status") == "converged") / len(goals)
        if goals else None
    )

    return GoldenScore(
        dataset_id=spec.dataset_id,
        evidence_count=len(evidence),
        iterations=int(state.get("iteration", 0)),
        recall=len(found_real) / len(real) if real else 0.0,
        sign_accuracy=(
            sum(1 for f in signed if f.direction_matches) / len(signed) if signed else None
        ),
        false_positives=len(decoy_claims),
        goal_convergence=convergence,
        evidence_efficiency=len(found_real) / len(evidence) if evidence else 0.0,
        directional_rate=len(signed) / len(found_real) if found_real else 0.0,
        findings=findings,
        decoy_claims=decoy_claims,
    )

def _finding_row(finding: EffectFinding) -> str:
    effect = finding.effect
    if not effect.is_real:
        # 미끼는 안 다루는 게 정답이다. 효과를 주장했으면 !, 수치 없이 컬럼만
        # 건드렸으면 ~로 구분한다.
        if not finding.found:
            mark = "O"
        elif finding.note:
            mark = "!"
        else:
            mark = "~"
        return f"| {effect.column} | 미끼 | {mark} | - | {finding.note} |"

    mark = "O" if finding.found else "X"
    if finding.direction_matches is None:
        sign = "-"
    else:
        sign = "O" if finding.direction_matches else "X"
    return f"| {effect.column} | {effect.strength} | {mark} | {sign} | {finding.note} |"

def score_report(score: GoldenScore) -> str:
    """사람이 읽는 채점표."""
    lines = [
        f"# 골든 채점 — {score.dataset_id}",
        "",
        score.summary_line(),
        "",
        "| 효과 | 강도 | 다룸 | 부호 | 비고 |",
        "|---|---|---|---|---|",
        *(_finding_row(f) for f in score.findings),
    ]
    if score.decoy_claims:
        lines += ["", "## 위양성", *(f"- {c}" for c in score.decoy_claims)]
    return "\n".join(lines)
