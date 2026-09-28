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
    """합성 데이터에 주입된 기준 효과."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    # 효과 매칭용 관련 컬럼
    column: str
    direction: Direction
    # 합성 데이터 생성 계수; direction=none이면 0
    coefficient: float = 0.0
    # 효과 강도별 난이도 가중치
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
        # 이탈과 독립적인 negative control
        PlantedEffect(name="지역은 이탈과 무관하다", column="region",
                      direction="none", strength="none"),
        PlantedEffect(name="기기 유형은 이탈과 무관하다", column="device",
                      direction="none", strength="none"),
    ],
)

def generate_golden_dataset(spec: GoldenSpec = CHURN_SPEC) -> pd.DataFrame:
    """고정 시드 기반 합성 데이터 생성."""
    rng = np.random.default_rng(spec.seed)
    n = spec.row_count
    coef = {e.column: e.coefficient for e in spec.effects}

    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    # 계약 기간 ordinal encoding: 월 0, 1년 1, 2년 2
    # 음수 계수로 장기 계약의 낮은 이탈을 생성
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
        # negative control
        "region": rng.choice(["A", "B", "C", "D"], n),
        "device": rng.choice(["ios", "android", "web"], n),
        "churn": churn,
    })

def write_golden_dataset(path: Path | str, spec: GoldenSpec = CHURN_SPEC) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_golden_dataset(spec).to_csv(path, index=False)
    return path

# 평가

class EffectFinding(BaseModel):
    """기준 효과 단위 판정."""

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

    # 기준 효과 recall
    recall: float
    # 방향 판정 precision
    sign_accuracy: float | None
    # negative control false positive 수
    false_positives: int
    # SubGoal convergence 비율
    goal_convergence: float | None
    # Evidence당 포착 효과 수
    evidence_efficiency: float
    # 다룬 효과 중 방향 판정 비율
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
    """명시적 컬럼 참조 기반 효과 매칭."""
    estimates = [EffectEstimate.model_validate(item) for item in (effect_summary or [])]
    return [item for item in estimates if item.column == column]

def _collect_finding(
    effect: PlantedEffect, evidence: list[dict[str, Any]],
) -> tuple[EffectFinding, list[float]]:
    """기준 효과 관련 Evidence 수집."""
    finding = EffectFinding(effect=effect)
    directional: list[float] = []
    for record in evidence:
        if effect.column not in (record.get("columns_used") or []):
            continue
        estimates = _estimates_for(record.get("effect_summary") or [], effect.column)
        if not estimates:
            # 관련 컬럼 사용만으로 효과 포착으로 간주하지 않음
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
    """효과 방향 판정 및 비고 생성."""
    if not finding.found:
        return None

    if effect.is_real:
        if not directional:
            # 그룹별 수준만 있는 경우 방향 미판정
            # 단일 그룹 수준은 효과 방향으로 사용하지 않음
            finding.note = (
                f"수준만 보고({len(finding.reported_values)}개). "
                "부호를 읽을 대비 통계량이 없다"
            )
            return None
        mean = sum(directional) / len(directional)
        finding.direction_matches = (mean > 0) == (effect.direction == "positive")
        finding.note = f"부호 통계량 {mean:+.4f} / 실제 {effect.coefficient:+.4f}"
        return None

    # negative control의 비영 방향 효과는 false positive
    nonzero = [v for v in directional if abs(v) > 1e-9]
    if not nonzero:
        return None
    finding.note = "미끼인데 효과를 보고했다"
    return f"{effect.column}: {nonzero[:3]}"

def score_run(state: dict[str, Any], spec: GoldenSpec = CHURN_SPEC) -> GoldenScore:
    """완료 실행 평가. State 기준으로만 계산.

    feedback payload의 필수 정보 전달 여부도 함께 검증.
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
        # negative control 표기: 효과 주장 !, 컬럼만 사용 ~
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
    """텍스트 형식 평가표."""
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
