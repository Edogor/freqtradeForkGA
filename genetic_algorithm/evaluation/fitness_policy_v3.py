"""Feasibility-first search ordering for the V3 quality experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeasibilityFirstPolicyV3(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: Literal[True] = True
    policy_version: str = Field(min_length=1)
    min_trades: int = Field(default=12, ge=1)
    min_net_profit: float = 0.0
    min_profit_factor: float = Field(default=1.0, ge=0)
    min_profitable_pair_ratio: float = Field(default=2 / 3, gt=0, le=1)
    max_worst_pair_loss: float = Field(default=-5.0, le=0)
    max_drawdown: float = Field(default=0.20, ge=0, le=1)
    max_drawdown_duration_days: float = Field(default=120.0, ge=0)
    median_hold_hours_target: float = Field(default=24.0, gt=0)
    p90_hold_hours_target: float = Field(default=72.0, gt=0)
    holding_soft_weight: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="after")
    def _edge_is_strictly_positive(self) -> "FeasibilityFirstPolicyV3":
        if self.min_profit_factor < 1.0:
            raise ValueError("min_profit_factor must be at least break-even (1.0)")
        if self.p90_hold_hours_target < self.median_hold_hours_target:
            raise ValueError("p90 holding target must not be below median target")
        return self


def fitness_policy_v3_from_config(
    config: Mapping[str, object],
) -> FeasibilityFirstPolicyV3 | None:
    raw = config.get("fitness_policy_v3")
    if not isinstance(raw, Mapping) or not raw.get("enabled", False):
        return None
    return FeasibilityFirstPolicyV3.model_validate(raw)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def apply_feasibility_first_v3(
    base_fitness: float,
    metrics: dict[str, Any],
    policy: FeasibilityFirstPolicyV3,
    *,
    resolved_profit_factor: float | None = None,
) -> float:
    """Map a smooth score into non-overlapping feasibility bands.

    A candidate can only enter a higher band by satisfying every preceding
    evidence/edge/pair/risk condition. Its ordinary fitness still supplies the
    gradient within a band, so evolution can make progress toward the next one.
    """

    trades = _finite_number(metrics.get("num_trades"))
    profit = _finite_number(metrics.get("profit"))
    profit_factor = _finite_number(
        resolved_profit_factor
        if resolved_profit_factor is not None
        else metrics.get("profit_factor")
    )
    drawdown = _finite_number(metrics.get("max_drawdown"))
    drawdown_duration = _finite_number(metrics.get("max_drawdown_duration_days"))
    pair_profit = metrics.get("per_pair_profit")
    usable_pair_profit = (
        {
            str(pair): float(value)
            for pair, value in pair_profit.items()
            if _finite_number(value) is not None
        }
        if isinstance(pair_profit, Mapping)
        else {}
    )

    evidence_ok = trades is not None and trades >= policy.min_trades
    edge_ok = (
        evidence_ok
        and profit is not None
        and profit > policy.min_net_profit
        and profit_factor is not None
        and profit_factor > policy.min_profit_factor
    )
    profitable_ratio = (
        sum(value > 0.0 for value in usable_pair_profit.values())
        / len(usable_pair_profit)
        if usable_pair_profit
        else 0.0
    )
    pair_ok = (
        edge_ok
        and bool(usable_pair_profit)
        and profitable_ratio >= policy.min_profitable_pair_ratio
        and min(usable_pair_profit.values()) >= policy.max_worst_pair_loss
    )
    risk_ok = (
        pair_ok
        and drawdown is not None
        and drawdown <= policy.max_drawdown
        and drawdown_duration is not None
        and drawdown_duration <= policy.max_drawdown_duration_days
    )

    hold_hours: list[float] = []
    raw_trades = metrics.get("trades")
    if isinstance(raw_trades, list):
        for trade in raw_trades:
            duration = trade.get("trade_duration") if isinstance(trade, Mapping) else None
            resolved = _finite_number(duration)
            if resolved is not None and resolved >= 0:
                hold_hours.append(resolved / 60.0)
    median_hold = _quantile(hold_hours, 0.5)
    p90_hold = _quantile(hold_hours, 0.9)
    holding_score = None
    if median_hold is not None and p90_hold is not None:
        holding_score = min(
            1.0,
            policy.median_hold_hours_target / max(median_hold, 1e-12),
            policy.p90_hold_hours_target / max(p90_hold, 1e-12),
        )

    stage = sum((evidence_ok, edge_ok, pair_ok, risk_ok))
    stage_names = ("NO_EVIDENCE", "EVIDENCE", "POSITIVE_EDGE", "PAIR_ROBUST", "RISK_FEASIBLE")
    bounded = max(0.0, float(base_fitness))
    if holding_score is not None:
        bounded *= (
            1.0 - policy.holding_soft_weight
            + policy.holding_soft_weight * holding_score
        )
    within_band = bounded / (1.0 + bounded)
    score = (stage + within_band) / 5.0
    metrics.update(
        {
            "feasibility_policy_version": policy.policy_version,
            "feasibility_stage": stage,
            "feasibility_stage_name": stage_names[stage],
            "feasibility_evidence_ok": evidence_ok,
            "feasibility_edge_ok": edge_ok,
            "feasibility_pair_ok": pair_ok,
            "feasibility_risk_ok": risk_ok,
            "feasibility_profitable_pair_ratio": profitable_ratio,
            "feasibility_median_hold_hours": median_hold,
            "feasibility_p90_hold_hours": p90_hold,
            "feasibility_holding_target_score": holding_score,
            "pre_feasibility_fitness": base_fitness,
        }
    )
    return score
