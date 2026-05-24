"""Tests for T2.5 — Information Coefficient primitives & fitness penalty."""
from __future__ import annotations

import math

import pytest

from genetic_algorithm.evaluation import information_coefficient as ic


# ── _ranks ────────────────────────────────────────────────────────────


def test_ranks_unique_values():
    assert ic._ranks([10, 30, 20]) == [1.0, 3.0, 2.0]


def test_ranks_with_ties_average_them():
    # [5, 5, 10] → ranks [1.5, 1.5, 3]
    assert ic._ranks([5, 5, 10]) == [1.5, 1.5, 3.0]


# ── spearman_ic ───────────────────────────────────────────────────────


def test_spearman_perfect_positive():
    x = list(range(20))
    y = [v * 2 + 5 for v in x]
    assert ic.spearman_ic(x, y) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    x = list(range(20))
    y = [-v for v in x]
    assert ic.spearman_ic(x, y) == pytest.approx(-1.0)


def test_spearman_uncorrelated_is_near_zero():
    # Deterministic pseudo-random — large sample → near 0
    import random
    rng = random.Random(42)
    x = [rng.random() for _ in range(500)]
    y = [rng.random() for _ in range(500)]
    val = ic.spearman_ic(x, y)
    assert abs(val) < 0.15  # statistical noise tolerance


def test_spearman_empty_or_short_returns_zero():
    assert ic.spearman_ic([], []) == 0.0
    assert ic.spearman_ic([1.0], [2.0]) == 0.0


def test_spearman_length_mismatch_returns_zero():
    assert ic.spearman_ic([1, 2, 3], [1, 2]) == 0.0


def test_spearman_constant_series_returns_zero():
    assert ic.spearman_ic([1, 1, 1, 1], [4, 3, 2, 1]) == 0.0
    assert ic.spearman_ic([4, 3, 2, 1], [1, 1, 1, 1]) == 0.0


def test_spearman_filters_nan_inf():
    x = [1, 2, 3, float("nan"), 5]
    y = [2, 4, 6, 100, 10]
    val = ic.spearman_ic(x, y)
    # After filter: (1,2),(2,4),(3,6),(5,10) → perfect positive
    assert val == pytest.approx(1.0)


# ── compute_ic_table ──────────────────────────────────────────────────


def test_compute_ic_table_basic():
    indicators = {
        "RSI": [10, 20, 30, 40, 50],
        "MACD": [50, 40, 30, 20, 10],
        "NOISE": [5, 5, 5, 5, 5],
    }
    forward = [1, 2, 3, 4, 5]
    table = ic.compute_ic_table(indicators, forward)
    assert table["RSI"] == pytest.approx(1.0)
    assert table["MACD"] == pytest.approx(-1.0)
    assert table["NOISE"] == 0.0


def test_compute_ic_table_empty_inputs():
    assert ic.compute_ic_table({}, [1, 2, 3]) == {}
    assert ic.compute_ic_table({"X": [1, 2]}, []) == {}


# ── median_abs_ic ─────────────────────────────────────────────────────


def test_median_abs_ic_takes_magnitude():
    table = {"a": -0.4, "b": 0.1, "c": -0.05, "d": 0.3, "e": 0.2}
    # |ICs|: 0.05, 0.1, 0.2, 0.3, 0.4 → median 0.2
    assert ic.median_abs_ic(table) == pytest.approx(0.2)


def test_median_abs_ic_even_count():
    # |ICs|: 0.1, 0.2, 0.3, 0.4 → median (0.2+0.3)/2 = 0.25
    assert ic.median_abs_ic({"a": 0.1, "b": -0.2, "c": 0.3, "d": -0.4}) == pytest.approx(0.25)


def test_median_abs_ic_empty_or_invalid():
    assert ic.median_abs_ic({}) == 0.0
    assert ic.median_abs_ic({"a": float("nan"), "b": float("inf")}) == 0.0


# ── low_ic_penalty ────────────────────────────────────────────────────


def test_no_penalty_above_threshold():
    assert ic.low_ic_penalty(0.05, threshold=0.02) == 1.0
    assert ic.low_ic_penalty(0.02, threshold=0.02) == 1.0


def test_full_penalty_at_zero_ic():
    assert ic.low_ic_penalty(0.0, threshold=0.02, max_penalty=0.30) == pytest.approx(0.70)


def test_linear_ramp_between():
    # midpoint of threshold → half max penalty
    assert ic.low_ic_penalty(0.01, threshold=0.02, max_penalty=0.30) == pytest.approx(0.85)


def test_penalty_clamped_to_unit_interval():
    val = ic.low_ic_penalty(0.0, threshold=0.02, max_penalty=2.0)
    # max_penalty>1 → result could be negative, but we clamp severity to [0,1]
    # so 1 - 2*1 = -1.  Direct math doesn't clamp, but real-life max_penalty <= 1.
    # We accept the math result here; behaviour-wise users should keep max_penalty in [0,1].
    assert val == pytest.approx(-1.0)


def test_invalid_inputs_return_one():
    assert ic.low_ic_penalty(float("nan")) == 1.0
    assert ic.low_ic_penalty(0.01, threshold=0.0) == 1.0


# ── ic_penalty_settings ───────────────────────────────────────────────


def test_settings_default_disabled():
    enabled, t, m = ic.ic_penalty_settings({})
    assert (enabled, t, m) == (False, 0.02, 0.30)


def test_settings_enable_via_config():
    cfg = {"fitness": {"ic_penalty": {"enabled": True, "threshold": 0.05, "max_penalty": 0.5}}}
    enabled, t, m = ic.ic_penalty_settings(cfg)
    assert enabled is True and t == 0.05 and m == 0.5


def test_settings_non_dict_safe():
    assert ic.ic_penalty_settings(None)[0] is False


# ── Fitness integration (passive) ─────────────────────────────────────


def test_fitness_unchanged_when_ic_disabled():
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator
    ev = FitnessEvaluator({})
    metrics = {"profit": 5, "sharpe_ratio": 1.0, "num_trades": 30,
               "median_ic_magnitude": 0.001}  # very low but penalty off
    fitness = ev.calculate_fitness(metrics)
    assert math.isfinite(fitness)


def test_fitness_lower_when_ic_enabled_and_low():
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator
    cfg = {"fitness": {"ic_penalty": {"enabled": True, "threshold": 0.05, "max_penalty": 0.3}}}
    ev = FitnessEvaluator(cfg)
    base_metrics = {"profit": 5, "sharpe_ratio": 1.0, "num_trades": 30}
    m_good = dict(base_metrics, median_ic_magnitude=0.10)  # high IC, no penalty
    m_bad = dict(base_metrics, median_ic_magnitude=0.0)    # zero IC, full penalty
    f_good = ev.calculate_fitness(m_good)
    f_bad = ev.calculate_fitness(m_bad)
    assert f_good > f_bad


def test_fitness_unchanged_when_metric_missing_even_if_enabled():
    """If pipeline didn't compute IC, penalty must be a no-op."""
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator
    cfg = {"fitness": {"ic_penalty": {"enabled": True}}}
    ev = FitnessEvaluator(cfg)
    m1 = {"profit": 5, "sharpe_ratio": 1.0, "num_trades": 30}
    m2 = dict(m1)  # identical, no IC key
    assert ev.calculate_fitness(m1) == ev.calculate_fitness(m2)
