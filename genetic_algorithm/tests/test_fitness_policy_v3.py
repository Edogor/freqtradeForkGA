"""Counterexample tests for feasibility-first evolutionary ordering."""

from __future__ import annotations

from genetic_algorithm.config.schema import (
    config_contract_shape_v2,
    deep_merge,
    validate_config_or_raise,
)
from genetic_algorithm.evaluation.fitness_policy_v3 import (
    FeasibilityFirstPolicyV3,
    apply_feasibility_first_v3,
)


def _policy() -> FeasibilityFirstPolicyV3:
    return FeasibilityFirstPolicyV3(
        enabled=True,
        policy_version="fitness-v3-test",
    )


def _metrics(**overrides):
    metrics = {
        "num_trades": 80,
        "profit": 8.0,
        "profit_factor": 1.5,
        "per_pair_profit": {
            "BTC/USDT": 5.0,
            "SOL/USDT": 2.0,
            "XRP/USDT": -1.0,
        },
        "max_drawdown": 0.12,
        "max_drawdown_duration_days": 40.0,
    }
    metrics.update(overrides)
    return metrics


def test_negative_edge_cannot_outrank_a_robust_candidate():
    misleading = _metrics(profit=-2.0, profit_factor=0.8, win_rate=0.95)
    robust = _metrics()

    misleading_score = apply_feasibility_first_v3(100.0, misleading, _policy())
    robust_score = apply_feasibility_first_v3(0.01, robust, _policy())

    assert misleading["feasibility_stage_name"] == "EVIDENCE"
    assert robust["feasibility_stage_name"] == "RISK_FEASIBLE"
    assert misleading_score < robust_score


def test_large_single_pair_loss_is_non_compensable():
    concentrated = _metrics(
        profit=30.0,
        per_pair_profit={
            "BTC/USDT": 40.0,
            "SOL/USDT": 30.0,
            "XRP/USDT": -20.0,
        },
    )
    robust = _metrics()

    concentrated_score = apply_feasibility_first_v3(100.0, concentrated, _policy())
    robust_score = apply_feasibility_first_v3(0.01, robust, _policy())

    assert concentrated["feasibility_stage_name"] == "POSITIVE_EDGE"
    assert concentrated_score < robust_score


def test_no_trade_candidate_stays_in_bottom_band():
    metrics = _metrics(num_trades=0)
    score = apply_feasibility_first_v3(100.0, metrics, _policy())

    assert metrics["feasibility_stage_name"] == "NO_EVIDENCE"
    assert score < 0.2


def test_holding_targets_are_soft_and_do_not_change_feasibility_stage():
    short = _metrics(trades=[{"trade_duration": 12 * 60}] * 20)
    long = _metrics(trades=[{"trade_duration": 120 * 60}] * 20)

    short_score = apply_feasibility_first_v3(0.5, short, _policy())
    long_score = apply_feasibility_first_v3(0.5, long, _policy())

    assert short["feasibility_stage"] == long["feasibility_stage"] == 4
    assert short["feasibility_holding_target_score"] == 1.0
    assert 0 < long["feasibility_holding_target_score"] < 1.0
    assert long_score < short_score


def test_policy_is_accepted_by_strict_config_contract():
    config = deep_merge(
        config_contract_shape_v2(),
        {
            "config_schema_version": 2,
            "safety_profile": {"name": "schema-test", "enforce": False},
            "fitness_policy_v3": _policy().model_dump(mode="python"),
        },
    )

    validate_config_or_raise(config)
