"""Tests for surrogate v2 (T3.8) — Expected Improvement acquisition.

Covers predict_with_uncertainty, expected_improvement math, and
filter_candidates_ei end-to-end with a real (tiny) RandomForest.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from genetic_algorithm.evaluation.surrogate import SurrogateModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**over):
    base = {
        "surrogate": {
            "enabled": True,
            "min_training_samples": 10,
            "filter_percentile": 60,
            "retrain_interval": 1,
            "validation_fraction": 0.3,
            "model": "random_forest",
            "mutate_skipped": False,
        },
        "mutation_rate": 0.1,
    }
    base["surrogate"].update(over)
    return base


def _make_gene(rsi_threshold: float = 30.0):
    """Minimal gene shape consumed by extract_features (65 features)."""
    ind = SimpleNamespace(
        type="RSI",
        parameters={"period": 14},
        weight=1.0,
    )
    entry = SimpleNamespace(
        operator="<",
        indicator_id=1,
        secondary_indicator_id=None,
        threshold=rsi_threshold,
        logic="AND",
    )
    return SimpleNamespace(
        indicators=[ind],
        entry_conditions=[entry],
        exit_conditions=[],
        short_entry_conditions=[],
        short_exit_conditions=[],
        stoploss=-0.05,
        trailing_stop=False,
        trailing_stop_positive=None,
        trailing_stop_positive_offset=None,
        max_open_trades=3,
        minimal_roi={"0": 0.05, "30": 0.02},
        timeframe="1h",
        informative_timeframes=[],
        can_short=False,
    )


def _make_ind(fitness=None, gene=None):
    ind = MagicMock()
    ind.evaluated = fitness is not None
    ind.fitness = fitness
    ind.strategy_gene = gene or _make_gene()
    ind.surrogate_evaluated = False
    ind.metrics = {}
    return ind


def _trained_surrogate():
    """Build and train a surrogate on a handful of synthetic samples."""
    pytest.importorskip("sklearn")
    surr = SurrogateModel(_config())
    # Feed varied training data with a clear signal
    fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.5, 0.6]
    pop = []
    for f in fitnesses:
        ind = _make_ind(fitness=f, gene=_make_gene(rsi_threshold=f * 100))
        pop.append(ind)
    surr.add_training_data(pop)
    assert surr.maybe_retrain()
    assert surr.ready
    return surr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictWithUncertainty:
    def test_returns_none_when_not_ready(self):
        surr = SurrogateModel(_config())
        assert surr.predict_with_uncertainty(_make_gene()) is None

    def test_returns_mean_and_std(self):
        surr = _trained_surrogate()
        out = surr.predict_with_uncertainty(_make_gene())
        assert out is not None
        mean, std = out
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0.0
        # With 100 trees on noisy data we should see some disagreement
        # somewhere in the input space.
        any_variance = False
        for thr in (10.0, 50.0, 95.0):
            _, s = surr.predict_with_uncertainty(_make_gene(rsi_threshold=thr))
            if s > 0:
                any_variance = True
                break
        assert any_variance, "Expected at least one point with non-zero std"


class TestExpectedImprovementMath:
    def test_returns_none_when_not_ready(self):
        surr = SurrogateModel(_config())
        assert surr.expected_improvement(_make_gene(), best_so_far=0.5) is None

    def test_zero_when_mean_below_incumbent_and_no_uncertainty(self):
        surr = _trained_surrogate()
        # Monkey-patch to force zero std
        original = surr.predict_with_uncertainty
        surr.predict_with_uncertainty = lambda g: (0.1, 0.0)
        try:
            ei = surr.expected_improvement(_make_gene(), best_so_far=0.9)
            assert ei == 0.0
        finally:
            surr.predict_with_uncertainty = original

    def test_strictly_positive_with_uncertainty_even_below_incumbent(self):
        surr = _trained_surrogate()
        surr.predict_with_uncertainty = lambda g: (0.4, 0.5)
        ei = surr.expected_improvement(_make_gene(), best_so_far=0.5, xi=0.0)
        # std > 0 + mean close to incumbent → EI must be > 0
        assert ei > 0.0

    def test_grows_with_predicted_mean(self):
        surr = _trained_surrogate()
        surr.predict_with_uncertainty = lambda g: (0.6, 0.1)
        ei_low = surr.expected_improvement(_make_gene(), best_so_far=0.5)
        surr.predict_with_uncertainty = lambda g: (0.9, 0.1)
        ei_high = surr.expected_improvement(_make_gene(), best_so_far=0.5)
        assert ei_high > ei_low

    def test_grows_with_uncertainty_when_mean_near_incumbent(self):
        surr = _trained_surrogate()
        surr.predict_with_uncertainty = lambda g: (0.5, 0.01)
        ei_low_var = surr.expected_improvement(_make_gene(), best_so_far=0.5, xi=0.0)
        surr.predict_with_uncertainty = lambda g: (0.5, 0.5)
        ei_high_var = surr.expected_improvement(_make_gene(), best_so_far=0.5, xi=0.0)
        assert ei_high_var > ei_low_var


class TestFilterCandidatesEi:
    def test_returns_all_when_not_ready(self):
        surr = SurrogateModel(_config())
        inds = [_make_ind() for _ in range(5)]
        to_bt, to_skip = surr.filter_candidates_ei(inds, best_so_far=0.5)
        assert to_bt == inds
        assert to_skip == []

    def test_splits_population_according_to_percentile(self):
        surr = _trained_surrogate()
        inds = [_make_ind(gene=_make_gene(rsi_threshold=t)) for t in range(10, 100, 5)]
        to_bt, to_skip = surr.filter_candidates_ei(inds, best_so_far=0.6)
        assert len(to_bt) + len(to_skip) == len(inds)
        # filter_percentile=60 → ~60% of scored go to backtest
        assert len(to_bt) >= 1
        assert len(to_skip) >= 1

    def test_skipped_individuals_are_marked(self):
        surr = _trained_surrogate()
        inds = [_make_ind(gene=_make_gene(rsi_threshold=t)) for t in range(10, 100, 5)]
        to_bt, to_skip = surr.filter_candidates_ei(inds, best_so_far=0.6)
        for ind in to_skip:
            assert ind.evaluated is True
            assert ind.surrogate_evaluated is True
            assert ind.metrics.get("acquisition") == "expected_improvement"
            assert "surrogate_ei" in ind.metrics
