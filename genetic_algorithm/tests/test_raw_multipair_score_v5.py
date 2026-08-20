"""Regression tests for the policy-experiment V5 raw score."""

from __future__ import annotations

from dataclasses import replace

import pytest

from genetic_algorithm.evaluation.raw_multipair_score import (
    HARDCORE_DEVELOPMENT_PAIRS,
    HARDCORE_VALIDATION_PAIRS,
    PairScenario,
)
from genetic_algorithm.evaluation.raw_multipair_score_v5 import (
    RAW_MULTIPAIR_SCORE_V5_VERSION,
    RawMultiPairPanelV5,
    RawMultiPairStatus,
    V5_BALANCED_POLICY,
    V5_EDGE_POLICY,
    V5_PRODUCTIVE_POLICY,
    policy_from_profile,
    rescore_v5,
    score_raw_multipair_v5,
)


PAIRS = HARDCORE_DEVELOPMENT_PAIRS + HARDCORE_VALIDATION_PAIRS


def _scenario(pair: str, timeframe: str, **changes: object) -> PairScenario:
    panel = RawMultiPairPanelV5(timeframe=timeframe)
    target = panel.policy.activity_target(timeframe)
    holding_median, holding_p90 = panel.policy.holding_targets(timeframe)
    values: dict[str, object] = {
        "pair": pair,
        "timeframe": timeframe,
        "success": True,
        "trade_count": int(target * panel.calendar_months),
        "active_months": int(panel.calendar_months * 0.8),
        "net_return": 0.20 * panel.calendar_years,
        "net_expectancy": 0.005,
        "profit_factor": 1.5,
        "profit_factor_censored": False,
        "median_holding_hours": holding_median,
        "p90_holding_hours": holding_p90,
        "max_drawdown": 0.10,
        "max_drawdown_duration_days": 60.0,
        "max_consecutive_losses": 4,
    }
    values.update(changes)
    return PairScenario(**values)  # type: ignore[arg-type]


def _scenarios(timeframe: str = "1h") -> list[PairScenario]:
    return [_scenario(pair, timeframe) for pair in PAIRS]


def test_v5_profiles_are_hash_distinct_and_rescore_same_evidence():
    scenarios = [_scenario(pair, "1h", trade_count=80, active_months=14) for pair in PAIRS]
    scores = rescore_v5("1h", scenarios)

    assert set(scores) == {"edge", "balanced", "productive"}
    assert {item.score_version for item in scores.values()} == {RAW_MULTIPAIR_SCORE_V5_VERSION}
    assert len({item.policy_hash for item in scores.values()}) == 3
    assert all(item.is_valid for item in scores.values())
    assert scores["edge"].score > scores["balanced"].score > scores["productive"].score


def test_productive_policy_rewards_more_profitable_activity_but_not_loss_activity():
    sparse = [_scenario(pair, "1h", trade_count=80, active_months=16) for pair in PAIRS]
    profitable_active = [_scenario(pair, "1h", trade_count=350, active_months=30) for pair in PAIRS]
    losing_active = [
        _scenario(
            pair,
            "1h",
            trade_count=350,
            active_months=30,
            net_return=-0.20 * RawMultiPairPanelV5("1h").calendar_years,
            net_expectancy=-0.005,
            profit_factor=0.5,
        )
        for pair in PAIRS
    ]
    panel = RawMultiPairPanelV5("1h", policy=V5_PRODUCTIVE_POLICY)

    sparse_score = score_raw_multipair_v5(panel, sparse).score
    active_score = score_raw_multipair_v5(panel, profitable_active).score
    losing_score = score_raw_multipair_v5(panel, losing_active).score
    assert active_score is not None and sparse_score is not None and losing_score is not None
    assert active_score > sparse_score
    assert losing_score < sparse_score


def test_4h_panel_is_exact_and_uses_six_trades_per_month_reference():
    panel = RawMultiPairPanelV5("4h")
    assert panel.period_start.isoformat() == "2023-05-18T04:00:00+00:00"
    assert panel.period_end.isoformat() == "2026-03-26T20:00:00+00:00"
    assert panel.policy.activity_target("4h") == 6.0
    assert panel.policy.holding_targets("4h") == (24.0, 72.0)
    assert score_raw_multipair_v5(panel, _scenarios("4h")).is_valid


def test_unknown_profile_and_bad_pair_evidence_fail_closed():
    with pytest.raises(ValueError, match="unknown v5 policy"):
        policy_from_profile("fast-money")
    invalid = score_raw_multipair_v5(
        RawMultiPairPanelV5("15m", policy=V5_BALANCED_POLICY),
        _scenarios("15m")[:-1],
    )
    assert invalid.status is RawMultiPairStatus.INVALID
    assert invalid.score is None


@pytest.mark.parametrize("policy", [V5_EDGE_POLICY, V5_BALANCED_POLICY, V5_PRODUCTIVE_POLICY])
def test_policy_benefit_weights_are_conserved(policy):
    assert policy.score_edge_weight + policy.score_productive_frequency_weight + policy.score_holding_weight == pytest.approx(0.92)
