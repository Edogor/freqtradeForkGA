"""Tests for T2.2 — Tail-risk metric primitives + fitness integration."""
from __future__ import annotations

import math

import pytest

from genetic_algorithm.evaluation import tail_risk as tr


# ── cvar ──────────────────────────────────────────────────────────────


def test_cvar_empty_returns_zero():
    assert tr.cvar([]) == 0.0


def test_cvar_picks_worst_5pct():
    returns = list(range(100))  # 0..99
    # 5% tail of 100 = 5 worst values: 0,1,2,3,4 → mean 2.0
    assert tr.cvar(returns, 0.95) == pytest.approx(2.0)


def test_cvar_negative_returns():
    returns = [-10.0, -5.0, -1.0, 0.0, 5.0]
    # 5% of 5 = 0.25 → rounded up to 1 → worst value -10
    assert tr.cvar(returns, 0.95) == pytest.approx(-10.0)


def test_cvar_invalid_confidence_returns_zero():
    assert tr.cvar([1.0, 2.0], confidence=0.0) == 0.0
    assert tr.cvar([1.0, 2.0], confidence=1.0) == 0.0


# ── recovery_factor ───────────────────────────────────────────────────


def test_recovery_factor_basic():
    assert tr.recovery_factor(20.0, 5.0) == pytest.approx(4.0)
    assert tr.recovery_factor(20.0, -5.0) == pytest.approx(4.0)  # sign-agnostic


def test_recovery_factor_zero_drawdown():
    # Positive profit with zero drawdown → "very high but finite"
    rf = tr.recovery_factor(10.0, 0.0)
    assert rf > 0 and math.isfinite(rf)
    # Zero/negative profit with zero drawdown → 0
    assert tr.recovery_factor(0.0, 0.0) == 0.0
    assert tr.recovery_factor(-5.0, 0.0) == 0.0


def test_recovery_factor_handles_nan():
    assert tr.recovery_factor(float("nan"), 5.0) == 0.0
    assert tr.recovery_factor(5.0, float("inf")) == 0.0


# ── ulcer_index ───────────────────────────────────────────────────────


def test_ulcer_index_flat_curve_is_zero():
    assert tr.ulcer_index([100.0, 100.0, 100.0]) == 0.0


def test_ulcer_index_monotone_rising_is_zero():
    assert tr.ulcer_index([100.0, 110.0, 120.0]) == 0.0


def test_ulcer_index_drawdown_produces_positive():
    # 100 → 80 → 100: 33% drop then recovery
    # Drawdowns (pct): 0, -20, 0
    # RMS: sqrt((0+400+0)/3) ≈ 11.55
    val = tr.ulcer_index([100.0, 80.0, 100.0])
    assert val == pytest.approx(math.sqrt(400 / 3), rel=1e-3)


def test_ulcer_index_short_curve_returns_zero():
    assert tr.ulcer_index([]) == 0.0
    assert tr.ulcer_index([100.0]) == 0.0


def test_ulcer_index_skips_nonpositive_balances():
    # Negative balance entries are ignored, not treated as zero-divide.
    val = tr.ulcer_index([100.0, -1.0, 80.0, 100.0])
    assert math.isfinite(val) and val >= 0


# ── max_adverse_excursion ─────────────────────────────────────────────


def test_mae_empty_returns_zero():
    assert tr.max_adverse_excursion([]) == 0.0


def test_mae_returns_absolute_median():
    # Five trades: -10, -5, -2, -1, 0 → median abs is 2
    assert tr.max_adverse_excursion([-10, -5, -2, -1, 0]) == pytest.approx(2.0)


def test_mae_even_count_averages_middle_two():
    # [-10, -5, -2, -1] → sorted abs [1,2,5,10] → median (2+5)/2 = 3.5
    assert tr.max_adverse_excursion([-10, -5, -2, -1]) == pytest.approx(3.5)


def test_mae_ignores_nan():
    val = tr.max_adverse_excursion([-5.0, float("nan"), -3.0])
    assert math.isfinite(val)


# ── compute_tail_risk_metrics ─────────────────────────────────────────


def test_bundle_returns_all_four_keys():
    out = tr.compute_tail_risk_metrics(
        trade_returns=[-0.05, 0.02, 0.03, -0.01],
        equity_curve=[100, 95, 97, 100],
        per_trade_mae=[-2.0, -3.0, -1.0],
        net_profit=5.0,
        max_drawdown=5.0,
    )
    assert set(out) == {"tail_cvar_95", "tail_recovery_factor",
                        "tail_ulcer_index", "tail_mae_median"}
    for v in out.values():
        assert math.isfinite(v)


def test_bundle_safe_on_empty_inputs():
    out = tr.compute_tail_risk_metrics()
    assert all(v == 0.0 for v in out.values())


def test_bundle_falls_back_to_sum_of_returns_for_net_profit():
    out = tr.compute_tail_risk_metrics(
        trade_returns=[1.0, 2.0, 3.0],
        max_drawdown=2.0,
        # No explicit net_profit → uses sum(returns) = 6.0
    )
    assert out["tail_recovery_factor"] == pytest.approx(3.0)


# ── normalised_tail_score ─────────────────────────────────────────────


def test_normalised_tail_score_in_unit_interval():
    # Wide range of inputs.
    for raw in [
        {"tail_cvar_95": -100, "tail_recovery_factor": 0,
         "tail_ulcer_index": 50, "tail_mae_median": 50},
        {"tail_cvar_95": 0, "tail_recovery_factor": 100,
         "tail_ulcer_index": 0, "tail_mae_median": 0},
        {},
    ]:
        norm = tr.normalised_tail_score(raw)
        for k, v in norm.items():
            assert 0.0 <= v <= 1.0, f"{k}={v}"


def test_normalised_score_directionality():
    # Worse CVaR (more negative) → lower normalised score
    a = tr.normalised_tail_score({"tail_cvar_95": -10})["tail_cvar_95_norm"]
    b = tr.normalised_tail_score({"tail_cvar_95": -40})["tail_cvar_95_norm"]
    assert a > b
    # Higher recovery → higher score
    a = tr.normalised_tail_score({"tail_recovery_factor": 1})["tail_recovery_factor_norm"]
    b = tr.normalised_tail_score({"tail_recovery_factor": 4})["tail_recovery_factor_norm"]
    assert b > a
    # Higher ulcer → lower score
    a = tr.normalised_tail_score({"tail_ulcer_index": 1})["tail_ulcer_index_norm"]
    b = tr.normalised_tail_score({"tail_ulcer_index": 10})["tail_ulcer_index_norm"]
    assert a > b


# ── Regression: fitness unchanged when tail weights = 0 ───────────────


def test_fitness_unchanged_without_tail_weights():
    """Default config has no tail weights → tail metrics must be inert."""
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator

    # Minimal config — won't actually run a backtest, only calculate_fitness.
    base_cfg = {}
    evaluator = FitnessEvaluator(base_cfg)
    metrics_no_tail = {
        "profit": 5.0, "sharpe_ratio": 1.5, "sortino_ratio": 2.0,
        "profit_factor": 1.8, "max_drawdown": 0.1, "win_rate": 0.55,
        "num_trades": 50,
    }
    metrics_with_tail = dict(metrics_no_tail)
    metrics_with_tail.update({
        "tail_cvar_95": -30.0,
        "tail_recovery_factor": 0.1,
        "tail_ulcer_index": 15.0,
        "tail_mae_median": 12.0,
    })
    f_no = evaluator.calculate_fitness(metrics_no_tail)
    f_with = evaluator.calculate_fitness(metrics_with_tail)
    # With zero weights, tail metrics may not change fitness at all.
    assert f_no == pytest.approx(f_with, abs=1e-6)


def test_fitness_decreases_when_tail_weight_punishes_bad_tail():
    """Enabling tail weight + bad CVaR should *lower* fitness."""
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator

    cfg_weighted = {
        "fitness_weights": {
            "profit": 0.3, "sharpe_ratio": 0.1, "drawdown": 0.1,
            "tail_cvar_95": 0.2,  # opt-in
        },
    }
    ev = FitnessEvaluator(cfg_weighted)
    metrics_good_tail = {
        "profit": 5.0, "sharpe_ratio": 1.5, "max_drawdown": 0.1,
        "num_trades": 50, "tail_cvar_95": -2.0,  # mild
    }
    metrics_bad_tail = dict(metrics_good_tail)
    metrics_bad_tail["tail_cvar_95"] = -40.0  # nasty tail
    f_good = ev.calculate_fitness(metrics_good_tail)
    f_bad = ev.calculate_fitness(metrics_bad_tail)
    assert f_good > f_bad
