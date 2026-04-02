"""Tests for evaluation modules: deflated_sharpe, cpcv, and utils/dynamic_bounds."""

import math
import random
import pytest
import numpy as np

# ═══════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio
# ═══════════════════════════════════════════════════════════════
from genetic_algorithm.evaluation.deflated_sharpe import (
    expected_max_sharpe,
    calculate_dsr,
    compute_return_statistics,
    deflated_sharpe_penalty,
)


class TestExpectedMaxSharpe:
    def test_single_trial_returns_zero(self):
        assert expected_max_sharpe(1) == 0.0

    def test_increases_with_more_trials(self):
        e5 = expected_max_sharpe(5)
        e50 = expected_max_sharpe(50)
        e500 = expected_max_sharpe(500)
        assert e5 < e50 < e500

    def test_normal_returns_baseline(self):
        """With normal returns (skew=0, kurtosis=3), should match known values."""
        e = expected_max_sharpe(100, skewness=0.0, kurtosis=3.0)
        assert 2.0 < e < 3.5  # Approx E[max] for 100 standard normals

    def test_positive_result_for_many_trials(self):
        e = expected_max_sharpe(1000)
        assert e > 0


class TestCalculateDSR:
    def test_high_sharpe_many_returns_gives_high_dsr(self):
        dsr = calculate_dsr(
            observed_sharpe=3.0, n_trials=10,
            n_returns=500, skewness=0.0, kurtosis=3.0,
        )
        assert dsr > 0.5

    def test_low_sharpe_many_trials_gives_low_dsr(self):
        dsr = calculate_dsr(
            observed_sharpe=0.5, n_trials=1000,
            n_returns=100, skewness=0.0, kurtosis=3.0,
        )
        assert dsr < 0.5

    def test_too_few_returns(self):
        dsr = calculate_dsr(observed_sharpe=2.0, n_trials=10, n_returns=5)
        assert dsr == 0.0

    def test_single_trial_easier_to_pass(self):
        dsr_1 = calculate_dsr(observed_sharpe=1.5, n_trials=1, n_returns=200)
        dsr_100 = calculate_dsr(observed_sharpe=1.5, n_trials=100, n_returns=200)
        assert dsr_1 > dsr_100

    def test_returns_between_zero_and_one(self):
        dsr = calculate_dsr(
            observed_sharpe=1.0, n_trials=50,
            n_returns=200, skewness=0.0, kurtosis=3.0,
        )
        assert 0.0 <= dsr <= 1.0


class TestComputeReturnStatistics:
    def test_empty_list(self):
        stats = compute_return_statistics([])
        assert stats['n_returns'] == 0
        assert stats['sharpe_ratio'] == 0.0

    def test_dict_trades(self):
        trades = [{'profit_ratio': 0.02}, {'profit_ratio': -0.01}, {'profit_ratio': 0.03}]
        stats = compute_return_statistics(trades)
        assert stats['n_returns'] == 3
        assert stats['mean'] > 0

    def test_numeric_list(self):
        stats = compute_return_statistics([0.01, 0.02, -0.005, 0.015])
        assert stats['n_returns'] == 4
        assert stats['std'] > 0

    def test_single_trade(self):
        stats = compute_return_statistics([0.05])
        assert stats['n_returns'] == 1
        assert stats['sharpe_ratio'] == 0.0  # Can't compute with 1 obs

    def test_ignores_nan(self):
        stats = compute_return_statistics([0.01, float('nan'), 0.02])
        assert stats['n_returns'] == 2


class TestDeflatedSharpePenalty:
    def test_insufficient_data_returns_no_penalty(self):
        penalty, info = deflated_sharpe_penalty(
            observed_sharpe=1.0, n_trials=1, n_returns=10,
        )
        assert penalty == 1.0
        assert info['dsr_skipped'] is True

    def test_high_sharpe_low_penalty(self):
        penalty, info = deflated_sharpe_penalty(
            observed_sharpe=3.0, n_trials=10, n_returns=500,
            skewness=0.0, kurtosis=3.0, penalty_weight=0.15,
        )
        assert penalty > 0.90

    def test_penalty_bounded(self):
        penalty, info = deflated_sharpe_penalty(
            observed_sharpe=0.1, n_trials=500, n_returns=100,
            penalty_weight=0.15,
        )
        assert 1.0 - 0.15 <= penalty <= 1.0


# ═══════════════════════════════════════════════════════════════
# CPCV + PBO
# ═══════════════════════════════════════════════════════════════
from genetic_algorithm.evaluation.cpcv import (
    generate_cpcv_paths,
    create_time_blocks,
    get_train_test_indices,
    compute_pbo,
    cpcv_penalty,
)


class TestGenerateCPCVPaths:
    def test_c_6_2_gives_15_paths(self):
        paths = generate_cpcv_paths(n_groups=6, n_test_groups=2)
        assert len(paths) == 15  # C(6,2)

    def test_c_4_2_gives_6_paths(self):
        paths = generate_cpcv_paths(n_groups=4, n_test_groups=2)
        assert len(paths) == 6

    def test_max_paths_subsampling(self):
        paths = generate_cpcv_paths(n_groups=10, n_test_groups=5, max_paths=10)
        assert len(paths) == 10

    def test_train_test_partition(self):
        """Train and test should be disjoint and cover all groups."""
        paths = generate_cpcv_paths(n_groups=6, n_test_groups=2)
        for train, test in paths:
            all_groups = set(train) | set(test)
            assert all_groups == set(range(6))
            assert len(set(train) & set(test)) == 0

    def test_invalid_n_test_groups(self):
        assert generate_cpcv_paths(n_groups=4, n_test_groups=5) == []
        assert generate_cpcv_paths(n_groups=4, n_test_groups=0) == []

    def test_invalid_n_groups(self):
        assert generate_cpcv_paths(n_groups=1, n_test_groups=1) == []


class TestCreateTimeBlocks:
    def test_block_count_matches(self):
        info = create_time_blocks(n_samples=1000, n_groups=5)
        assert len(info['blocks']) == 5

    def test_blocks_cover_all_samples(self):
        info = create_time_blocks(n_samples=1000, n_groups=4)
        first_start = info['blocks'][0][0]
        last_end = info['blocks'][-1][1]
        assert first_start == 0
        assert last_end == 1000

    def test_purge_and_embargo_sizes(self):
        info = create_time_blocks(n_samples=1000, n_groups=5, purge_pct=0.02, embargo_pct=0.01)
        assert info['purge_size'] == 20
        assert info['embargo_size'] == 10


class TestGetTrainTestIndices:
    def test_no_overlap(self):
        info = create_time_blocks(n_samples=1000, n_groups=5)
        train, test = get_train_test_indices(info, [0, 1, 2], [3, 4])
        assert len(set(train) & set(test)) == 0

    def test_purge_removes_boundary_samples(self):
        info = create_time_blocks(n_samples=1000, n_groups=5, purge_pct=0.05)
        train_purged, _ = get_train_test_indices(info, [0, 1, 2], [3, 4])
        train_no_purge_info = create_time_blocks(n_samples=1000, n_groups=5, purge_pct=0.0, embargo_pct=0.0)
        train_no_purge, _ = get_train_test_indices(train_no_purge_info, [0, 1, 2], [3, 4])
        assert len(train_purged) < len(train_no_purge)


class TestComputePBO:
    def test_perfect_strategies_low_pbo(self):
        """IS winner also best OOS → PBO ≈ 0."""
        n_paths, n_strategies = 10, 5
        # Strategy 0 always best IS and OOS
        is_perfs = np.zeros((n_paths, n_strategies))
        oos_perfs = np.zeros((n_paths, n_strategies))
        for i in range(n_paths):
            is_perfs[i] = [5, 4, 3, 2, 1]
            oos_perfs[i] = [5, 4, 3, 2, 1]
        pbo, details = compute_pbo(is_perfs, oos_perfs)
        assert pbo == 0.0

    def test_inverted_strategies_high_pbo(self):
        """IS winner is worst OOS → PBO = 1."""
        n_paths, n_strategies = 10, 5
        is_perfs = np.zeros((n_paths, n_strategies))
        oos_perfs = np.zeros((n_paths, n_strategies))
        for i in range(n_paths):
            is_perfs[i] = [5, 4, 3, 2, 1]
            oos_perfs[i] = [1, 2, 3, 4, 5]  # Completely reversed
        pbo, details = compute_pbo(is_perfs, oos_perfs)
        assert pbo == 1.0

    def test_insufficient_data(self):
        pbo, details = compute_pbo(np.array([]).reshape(0, 0), np.array([]).reshape(0, 0))
        assert pbo == 0.0
        assert 'error' in details

    def test_pbo_between_zero_and_one(self):
        rng = np.random.RandomState(42)
        n_paths, n_strats = 20, 5
        is_perfs = rng.randn(n_paths, n_strats)
        oos_perfs = rng.randn(n_paths, n_strats)
        pbo, _ = compute_pbo(is_perfs, oos_perfs)
        assert 0.0 <= pbo <= 1.0


class TestCPCVPenalty:
    def test_low_pbo_no_penalty(self):
        assert cpcv_penalty(0.1) == 1.0

    def test_high_pbo_has_penalty(self):
        p = cpcv_penalty(0.9, pbo_threshold=0.5, penalty_weight=0.20)
        assert p < 1.0
        assert p >= 0.80

    def test_very_high_pbo_near_floor(self):
        p = cpcv_penalty(0.99, penalty_weight=0.20)
        assert abs(p - 0.80) < 0.05

    def test_penalty_monotonically_decreasing(self):
        values = [cpcv_penalty(pbo / 10.0) for pbo in range(3, 10)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]


# ═══════════════════════════════════════════════════════════════
# Dynamic Bounds
# ═══════════════════════════════════════════════════════════════
from genetic_algorithm.utils.dynamic_bounds import (
    initialise_bounds,
    mutate_bounds,
    sample_from_bounds,
    crossover_bounds,
)


class TestInitialiseBounds:
    def test_uses_config_range(self):
        config = {'RSI': {'period': [5, 30]}}
        bounds = initialise_bounds('RSI', {'period': 14}, config)
        assert bounds['period'] == (5, 30)

    def test_fallback_symmetric_window(self):
        bounds = initialise_bounds('CUSTOM', {'period': 20}, {})
        lo, hi = bounds['period']
        assert lo < 20 < hi

    def test_multiple_params(self):
        config = {'MACD': {'fast': [5, 20], 'slow': [15, 50]}}
        bounds = initialise_bounds('MACD', {'fast': 12, 'slow': 26}, config)
        assert 'fast' in bounds
        assert 'slow' in bounds


class TestMutateBounds:
    def test_output_keys_match_input(self):
        bounds = {'period': (5.0, 30.0), 'multiplier': (1.0, 5.0)}
        params = {'period': 14, 'multiplier': 2.5}
        result = mutate_bounds(bounds, params, mutation_strength=0.1, rng=random.Random(42))
        assert set(result.keys()) == set(bounds.keys())

    def test_bounds_remain_valid(self):
        """min <= max should always hold."""
        rng = random.Random(42)
        bounds = {'p': (10.0, 20.0)}
        params = {'p': 15}
        for _ in range(100):
            result = mutate_bounds(bounds, params, mutation_strength=0.3, rng=rng)
            lo, hi = result['p']
            assert lo <= hi

    def test_minimum_span_prevents_collapse(self):
        rng = random.Random(42)
        bounds = {'p': (10.0, 20.0)}
        params = {'p': 15}
        for _ in range(50):
            bounds = mutate_bounds(bounds, params, mutation_strength=0.9, rng=rng)
            lo, hi = bounds['p']
            assert (hi - lo) >= 1.0  # min_span = max(1, 10*0.1) = 1


class TestSampleFromBounds:
    def test_integer_sampling(self):
        rng = random.Random(42)
        bounds = {'period': (5.0, 30.0)}
        for _ in range(100):
            val = sample_from_bounds('period', bounds, (1, 50), is_int=True, rng=rng)
            assert isinstance(val, int)
            assert 5 <= val <= 30

    def test_float_sampling(self):
        rng = random.Random(42)
        bounds = {'factor': (0.5, 2.0)}
        for _ in range(100):
            val = sample_from_bounds('factor', bounds, (0.0, 5.0), is_int=False, rng=rng)
            assert 0.5 <= val <= 2.0

    def test_fallback_when_no_bounds(self):
        rng = random.Random(42)
        val = sample_from_bounds('unknown', None, (10, 20), is_int=True, rng=rng)
        assert 10 <= val <= 20


class TestCrossoverBounds:
    def test_child_has_all_parent_keys(self):
        b1 = {'a': (1.0, 5.0), 'b': (10.0, 20.0)}
        b2 = {'a': (2.0, 6.0), 'c': (0.1, 0.5)}
        child = crossover_bounds(b1, b2, rng=random.Random(42))
        assert 'a' in child
        assert 'b' in child
        assert 'c' in child

    def test_inherits_from_parents(self):
        rng = random.Random(42)
        b1 = {'x': (1.0, 5.0)}
        b2 = {'x': (2.0, 6.0)}
        child = crossover_bounds(b1, b2, rng=rng)
        assert child['x'] in [b1['x'], b2['x']]

    def test_none_parents(self):
        assert crossover_bounds(None, None) == {}

    def test_one_none_parent(self):
        b1 = {'a': (1.0, 5.0)}
        child = crossover_bounds(b1, None, rng=random.Random(42))
        assert child['a'] == b1['a']
