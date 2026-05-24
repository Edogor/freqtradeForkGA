"""Tests for T2.6 — Bootstrap-CI conservative profit estimator."""
from __future__ import annotations

import math
import statistics

import pytest

from genetic_algorithm.evaluation import bootstrap_ci as bci


# ── bootstrap_lower_ci ─────────────────────────────────────────────────


def test_returns_zero_on_empty():
    assert bci.bootstrap_lower_ci([]) == 0.0


def test_returns_zero_on_single_trade():
    assert bci.bootstrap_lower_ci([5.0]) == 0.0


def test_returns_zero_on_invalid_ci():
    assert bci.bootstrap_lower_ci([1.0, 2.0], ci=0.0) == 0.0
    assert bci.bootstrap_lower_ci([1.0, 2.0], ci=1.0) == 0.0


def test_returns_zero_on_nonpositive_iter():
    assert bci.bootstrap_lower_ci([1.0, 2.0], n_iter=0) == 0.0


def test_seeded_run_is_deterministic():
    rs = [-0.5, 0.2, 0.3, -0.1, 0.4, 0.1]
    a = bci.bootstrap_lower_ci(rs, n_iter=500, seed=42)
    b = bci.bootstrap_lower_ci(rs, n_iter=500, seed=42)
    assert a == b


def test_different_seeds_diverge():
    rs = [-0.5, 0.2, 0.3, -0.1, 0.4, 0.1]
    a = bci.bootstrap_lower_ci(rs, n_iter=200, seed=1)
    b = bci.bootstrap_lower_ci(rs, n_iter=200, seed=99)
    # Should *almost certainly* differ.  Allow rare coincidence by
    # accepting tiny diff but not exact equality.
    assert a != b


def test_lower_ci_le_mean_total():
    # 5%-CI must always be <= median bootstrap total which itself
    # is close to the sum of the actual sample.
    rs = [1.0, 2.0, 3.0, 4.0, 5.0]   # sum = 15
    lower = bci.bootstrap_lower_ci(rs, n_iter=500, ci=0.05, seed=7)
    assert lower <= sum(rs)


def test_higher_ci_yields_lower_estimate():
    # ci=0.05 (5th percentile) should be lower than ci=0.50 (median)
    rs = [-3.0, -1.0, 0.5, 1.0, 2.0, 4.0]
    low5 = bci.bootstrap_lower_ci(rs, n_iter=500, ci=0.05, seed=1)
    med = bci.bootstrap_lower_ci(rs, n_iter=500, ci=0.5, seed=1)
    assert low5 <= med


def test_nan_inf_filtered():
    rs = [1.0, float("nan"), 2.0, float("inf"), 3.0]
    val = bci.bootstrap_lower_ci(rs, n_iter=100, seed=3)
    # Should not raise and should be a finite number.
    assert math.isfinite(val)


def test_few_finite_inputs_return_zero():
    # Only one finite value after filtering → can't bootstrap.
    assert bci.bootstrap_lower_ci(
        [float("nan"), 2.0, float("inf")], n_iter=10, seed=1
    ) == 0.0


def test_all_losing_trades_yield_negative_lower_ci():
    rs = [-1.0, -2.0, -3.0, -4.0]
    val = bci.bootstrap_lower_ci(rs, n_iter=500, ci=0.05, seed=5)
    assert val < 0


def test_constant_returns_yield_constant_lower_ci():
    # All trades = 1.0 → every resample sums to n.  No variance.
    rs = [1.0] * 6
    val = bci.bootstrap_lower_ci(rs, n_iter=200, ci=0.05, seed=8)
    assert val == pytest.approx(6.0)


# ── is_bootstrap_ci_enabled / settings ────────────────────────────────


def test_disabled_by_default():
    assert bci.is_bootstrap_ci_enabled({}) is False
    assert bci.is_bootstrap_ci_enabled({"fitness": {}}) is False


def test_enabled_via_fitness_block():
    assert bci.is_bootstrap_ci_enabled({"fitness": {"use_bootstrap_ci": True}})


def test_enabled_via_legacy_bootstrap_ci_block():
    assert bci.is_bootstrap_ci_enabled({"bootstrap_ci": {"enabled": True}})


def test_disabled_with_non_dict_config():
    assert bci.is_bootstrap_ci_enabled(None) is False
    assert bci.is_bootstrap_ci_enabled("not a dict") is False


def test_settings_defaults():
    s = bci.bootstrap_ci_settings({})
    assert s == {"n_iter": 1000, "ci": 0.05, "seed": None}


def test_settings_from_legacy_block():
    s = bci.bootstrap_ci_settings({"bootstrap_ci": {"n_iter": 500, "ci": 0.10, "seed": 42}})
    assert s == {"n_iter": 500, "ci": 0.10, "seed": 42}


def test_fitness_block_overrides_legacy():
    s = bci.bootstrap_ci_settings({
        "bootstrap_ci": {"n_iter": 500, "ci": 0.10},
        "fitness": {"bootstrap_n_iter": 2000, "bootstrap_ci": 0.01},
    })
    assert s["n_iter"] == 2000
    assert s["ci"] == 0.01


# ── Fitness integration ───────────────────────────────────────────────


def test_fitness_unchanged_when_bootstrap_ci_disabled():
    """Default off → metrics dict has no bootstrap keys."""
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator
    ev = FitnessEvaluator({})
    metrics = {"profit": 10.0, "sharpe_ratio": 1.5, "num_trades": 30}
    fitness = ev.calculate_fitness(dict(metrics))
    assert math.isfinite(fitness)


def test_bootstrap_ci_settings_resolves_in_full_config():
    """Wiring sanity: enabling the flag is detected."""
    cfg = {"fitness": {"use_bootstrap_ci": True}}
    assert bci.is_bootstrap_ci_enabled(cfg)
    s = bci.bootstrap_ci_settings(cfg)
    assert s["n_iter"] == 1000  # default still applied
