"""Versioned v5 scoring contracts for the evidence-based hardcore campaign.

V5 deliberately keeps the raw six-pair point-estimate contract from V4, but
makes the selection policy an explicit, hash-bound experiment.  A candidate is
backtested once on a fixed panel and can then be rescored under every policy;
no policy may introduce an eligibility gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from genetic_algorithm.evaluation.raw_multipair_score import (
    COMPONENT_NAMES,
    HARDCORE_DEVELOPMENT_PAIRS,
    HARDCORE_FEE_RATE,
    HARDCORE_SLIPPAGE_RATE,
    HARDCORE_VALIDATION_PAIRS,
    ComponentVector,
    PairScenario,
    PairScoreComponents,
    RawMultiPairPolicyV4,
    RawMultiPairScore,
    RawMultiPairStatus,
    _benefit_group,
    _EvidenceError,
    _risk_group,
    _score_pair,
)


RAW_MULTIPAIR_SCORE_V5_VERSION = "raw-multipair-score-v5"
V5_TIMEFRAMES = frozenset({"15m", "1h", "4h"})

# Exact candle-open boundaries.  The 4h start is the earliest point at which
# PEPE has 75 honest pre-start candles; no synthetic pre-listing data exists.
V5_PANEL_BOUNDS: dict[str, tuple[datetime, datetime]] = {
    "15m": (
        datetime(2023, 5, 9, 0, 0, tzinfo=UTC),
        datetime(2026, 3, 26, 23, 45, tzinfo=UTC),
    ),
    "1h": (
        datetime(2023, 5, 9, 0, 0, tzinfo=UTC),
        datetime(2026, 3, 26, 23, 0, tzinfo=UTC),
    ),
    "4h": (
        datetime(2023, 5, 18, 4, 0, tzinfo=UTC),
        datetime(2026, 3, 26, 20, 0, tzinfo=UTC),
    ),
}


@dataclass(frozen=True)
class RawMultiPairPolicyV5(RawMultiPairPolicyV4):
    """Continuous V5 policy, including an explicit 4h activity reference."""

    activity_target_trades_per_month_4h: float = 6.0
    holding_median_target_hours_4h: float = 24.0
    holding_p90_target_hours_4h: float = 72.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "activity_target_trades_per_month_4h",
            "holding_median_target_hours_4h",
            "holding_p90_target_hours_4h",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if not math.isclose(
            self.score_edge_weight
            + self.score_productive_frequency_weight
            + self.score_holding_weight,
            0.92,
            abs_tol=1e-12,
        ):
            raise ValueError("v5 benefit score weights must total 0.92")

    def activity_target(self, timeframe: str) -> float:
        return {
            "15m": self.activity_target_trades_per_month_15m,
            "1h": self.activity_target_trades_per_month_1h,
            "4h": self.activity_target_trades_per_month_4h,
        }[timeframe]

    def holding_targets(self, timeframe: str) -> tuple[float, float]:
        return {
            "15m": (
                self.holding_median_target_hours_15m,
                self.holding_p90_target_hours_15m,
            ),
            "1h": (
                self.holding_median_target_hours_1h,
                self.holding_p90_target_hours_1h,
            ),
            "4h": (
                self.holding_median_target_hours_4h,
                self.holding_p90_target_hours_4h,
            ),
        }[timeframe]


V5_EDGE_POLICY = RawMultiPairPolicyV5(
    score_edge_weight=0.60,
    score_productive_frequency_weight=0.30,
)
V5_BALANCED_POLICY = RawMultiPairPolicyV5(
    score_edge_weight=0.45,
    score_productive_frequency_weight=0.45,
)
V5_PRODUCTIVE_POLICY = RawMultiPairPolicyV5(
    score_edge_weight=0.30,
    score_productive_frequency_weight=0.60,
)
V5_POLICY_PROFILES: dict[str, RawMultiPairPolicyV5] = {
    "edge": V5_EDGE_POLICY,
    "balanced": V5_BALANCED_POLICY,
    "productive": V5_PRODUCTIVE_POLICY,
}


@dataclass(frozen=True)
class RawMultiPairPanelV5:
    """Immutable exact-candle panel for a V5 lane and policy."""

    timeframe: str
    policy: RawMultiPairPolicyV5 = V5_BALANCED_POLICY
    development_pairs: tuple[str, ...] = HARDCORE_DEVELOPMENT_PAIRS
    validation_pairs: tuple[str, ...] = HARDCORE_VALIDATION_PAIRS
    period_start: datetime | None = None
    period_end: datetime | None = None
    fee_rate: float = HARDCORE_FEE_RATE
    slippage_rate: float = HARDCORE_SLIPPAGE_RATE
    data_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        if self.timeframe not in V5_TIMEFRAMES:
            raise ValueError("v5 supports only 15m, 1h and 4h")
        if not isinstance(self.policy, RawMultiPairPolicyV5):
            raise TypeError("policy must be a RawMultiPairPolicyV5")
        if tuple(self.development_pairs) != HARDCORE_DEVELOPMENT_PAIRS:
            raise ValueError("development_pairs do not match the fixed v5 panel")
        if tuple(self.validation_pairs) != HARDCORE_VALIDATION_PAIRS:
            raise ValueError("validation_pairs do not match the fixed v5 panel")
        expected_start, expected_end = V5_PANEL_BOUNDS[self.timeframe]
        start = self.period_start or expected_start
        end = self.period_end or expected_end
        if start != expected_start or end != expected_end:
            raise ValueError("period bounds do not match the immutable v5 panel")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("period bounds must be timezone aware")
        object.__setattr__(self, "period_start", start.astimezone(UTC))
        object.__setattr__(self, "period_end", end.astimezone(UTC))
        if not math.isclose(float(self.fee_rate), HARDCORE_FEE_RATE, abs_tol=1e-12):
            raise ValueError("fee_rate must be 0.001")
        if not math.isclose(float(self.slippage_rate), HARDCORE_SLIPPAGE_RATE, abs_tol=1e-12):
            raise ValueError("slippage_rate must be 0.0005")
        if self.data_manifest_hash is not None and (
            len(self.data_manifest_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.data_manifest_hash)
        ):
            raise ValueError("data_manifest_hash must be a lowercase SHA-256 digest")

    @property
    def calendar_months(self) -> int:
        return (
            (self.period_end.year - self.period_start.year) * 12
            + self.period_end.month
            - self.period_start.month
            + 1
        )

    @property
    def calendar_days(self) -> float:
        return (self.period_end - self.period_start).total_seconds() / 86400.0

    @property
    def calendar_years(self) -> float:
        return self.calendar_days / 365.2425

    @property
    def policy_hash(self) -> str:
        return self.policy.policy_hash

    @property
    def panel_id(self) -> str:
        payload = json.dumps(self.identity_descriptor(), sort_keys=True, separators=(",", ":"))
        return "raw_multipair_panel_v5_" + hashlib.sha256(payload.encode()).hexdigest()[:24]

    def identity_descriptor(self) -> dict[str, Any]:
        return {
            "score_version": RAW_MULTIPAIR_SCORE_V5_VERSION,
            "timeframe": self.timeframe,
            "development_pairs": list(self.development_pairs),
            "validation_pairs": list(self.validation_pairs),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "data_manifest_hash": self.data_manifest_hash,
            "policy_hash": self.policy_hash,
            "policy": self.policy.to_dict(),
        }


def policy_from_profile(profile: str) -> RawMultiPairPolicyV5:
    """Resolve only the three declared V5 experiment policies."""

    try:
        return V5_POLICY_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown v5 policy profile: {profile}") from exc


def score_raw_multipair_v5(
    panel: RawMultiPairPanelV5,
    scenarios: Iterable[PairScenario],
) -> RawMultiPairScore:
    """Score six raw scenarios under one V5 policy without any gate."""

    try:
        supplied = tuple(scenarios)
    except TypeError as exc:
        return _invalid_v5(panel, "INVALID_EVIDENCE_TYPE", str(exc))
    if len(supplied) != 6:
        return _invalid_v5(panel, "PAIR_SET_MISMATCH", "expected six scenarios")
    by_pair = {
        scenario.pair: scenario for scenario in supplied if isinstance(scenario, PairScenario)
    }
    expected_pairs = set(panel.development_pairs) | set(panel.validation_pairs)
    if len(by_pair) != 6 or set(by_pair) != expected_pairs:
        return _invalid_v5(panel, "PAIR_SET_MISMATCH", "scenarios do not match fixed panel")

    components: list[PairScoreComponents] = []
    try:
        for pair in sorted(expected_pairs):
            scenario = by_pair[pair]
            if scenario.timeframe != panel.timeframe:
                raise _EvidenceError("TIMEFRAME_MISMATCH", f"{pair} has wrong timeframe")
            group = "development" if pair in panel.development_pairs else "validation"
            components.append(_score_pair(panel, scenario, group))
    except _EvidenceError as exc:
        return _invalid_v5(panel, exc.reason_code, exc.detail)

    values: dict[str, float] = {}
    component_map = {item.pair: item.components for item in components}
    for name in COMPONENT_NAMES:
        dev = [getattr(component_map[pair], name) for pair in panel.development_pairs]
        val = [getattr(component_map[pair], name) for pair in panel.validation_pairs]
        group_fn = (
            _benefit_group
            if name
            in {
                "return_score",
                "expectancy_score",
                "profit_factor_score",
                "activity_score",
                "edge_score",
                "productive_frequency_score",
                "holding_score",
            }
            else _risk_group
        )
        values[name] = panel.policy.development_group_weight * group_fn(
            dev, policy=panel.policy
        ) + panel.policy.validation_group_weight * group_fn(val, policy=panel.policy)
    aggregate = ComponentVector(**values)
    p = panel.policy
    score = 100.0 * (
        p.score_edge_weight * values["edge_score"]
        + p.score_productive_frequency_weight * values["productive_frequency_score"]
        + p.score_holding_weight * values["holding_score"]
        + p.score_drawdown_weight * values["drawdown_risk"]
        + p.score_drawdown_duration_weight * values["drawdown_duration_risk"]
        + p.score_loss_streak_weight * values["loss_streak_risk"]
        + p.score_overtrading_weight * values["overtrading_risk"]
    )
    if not math.isfinite(score):
        return _invalid_v5(panel, "NONFINITE_SCORE", "score is not finite")
    return RawMultiPairScore(
        score_version=RAW_MULTIPAIR_SCORE_V5_VERSION,
        policy_hash=panel.policy_hash,
        panel_id=panel.panel_id,
        timeframe=panel.timeframe,
        status=RawMultiPairStatus.VALID,
        score=score,
        reason_code="SCORED",
        reason_detail=None,
        pair_components=tuple(components),
        aggregate_components=aggregate,
    )


def rescore_v5(
    timeframe: str,
    scenarios: Iterable[PairScenario],
    *,
    profiles: Iterable[str] = ("edge", "balanced", "productive"),
    data_manifest_hash: str | None = None,
) -> dict[str, RawMultiPairScore]:
    """Evaluate identical raw evidence under all requested V5 policies."""

    frozen = tuple(scenarios)
    return {
        profile: score_raw_multipair_v5(
            RawMultiPairPanelV5(
                timeframe=timeframe,
                policy=policy_from_profile(profile),
                data_manifest_hash=data_manifest_hash,
            ),
            frozen,
        )
        for profile in profiles
    }


def _invalid_v5(panel: RawMultiPairPanelV5, code: str, detail: str) -> RawMultiPairScore:
    return RawMultiPairScore(
        score_version=RAW_MULTIPAIR_SCORE_V5_VERSION,
        policy_hash=panel.policy_hash,
        panel_id=panel.panel_id,
        timeframe=panel.timeframe,
        status=RawMultiPairStatus.INVALID,
        score=None,
        reason_code=code,
        reason_detail=detail,
    )
