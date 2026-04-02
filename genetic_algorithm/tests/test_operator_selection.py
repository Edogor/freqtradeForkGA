"""Tests for genetic_algorithm.core.operator_selection — AdaptiveOperatorSelector."""

import pytest
import random

from genetic_algorithm.core.operator_selection import AdaptiveOperatorSelector


def _make_config(enabled=True, window_size=50, min_prob=0.10, credit_type='fitness_improvement'):
    return {
        'adaptive_operators': {
            'enabled': enabled,
            'window_size': window_size,
            'min_probability': min_prob,
            'credit_type': credit_type,
        }
    }


class TestDisabledAOS:
    def test_disabled_selects_randomly(self):
        aos = AdaptiveOperatorSelector(_make_config(enabled=False))
        # Should not crash and should return valid operators
        cx = aos.select_crossover()
        mut = aos.select_mutation()
        assert cx in AdaptiveOperatorSelector.CROSSOVER_OPS
        assert mut in AdaptiveOperatorSelector.MUTATION_OPS

    def test_disabled_update_is_noop(self):
        aos = AdaptiveOperatorSelector(_make_config(enabled=False))
        # Probabilities should stay uniform
        initial_cx = dict(aos.crossover_probs)
        aos.update_probabilities()
        assert aos.crossover_probs == initial_cx


class TestInitialState:
    def test_uniform_crossover_probabilities(self):
        aos = AdaptiveOperatorSelector(_make_config())
        n = len(AdaptiveOperatorSelector.CROSSOVER_OPS)
        for op, prob in aos.crossover_probs.items():
            assert abs(prob - 1.0 / n) < 1e-9

    def test_uniform_mutation_probabilities(self):
        aos = AdaptiveOperatorSelector(_make_config())
        n = len(AdaptiveOperatorSelector.MUTATION_OPS)
        for op, prob in aos.mutation_probs.items():
            assert abs(prob - 1.0 / n) < 1e-9


class TestSelectionWeighting:
    def test_select_crossover_respects_probabilities(self):
        """After heavy bias, the favored operator should appear most often."""
        aos = AdaptiveOperatorSelector(_make_config())
        # Bias 'uniform' crossover
        for _ in range(100):
            aos.record_outcome('crossover', 'uniform', 0.3, 0.9)  # +0.6 credit
        aos.update_probabilities()

        assert aos.crossover_probs['uniform'] > aos.crossover_probs['single_point']

    def test_select_mutation_respects_probabilities(self):
        aos = AdaptiveOperatorSelector(_make_config())
        for _ in range(100):
            aos.record_outcome('mutation', 'parameter', 0.2, 0.8)
        aos.update_probabilities()

        assert aos.mutation_probs['parameter'] > aos.mutation_probs['logic_toggle']


class TestCreditComputation:
    def test_fitness_improvement_positive(self):
        aos = AdaptiveOperatorSelector(_make_config(credit_type='fitness_improvement'))
        credit = aos._compute_credit(0.3, 0.7)
        assert credit == pytest.approx(0.4)

    def test_fitness_improvement_negative_clamped_to_zero(self):
        aos = AdaptiveOperatorSelector(_make_config(credit_type='fitness_improvement'))
        credit = aos._compute_credit(0.8, 0.5)
        assert credit == 0.0

    def test_rank_improvement_binary(self):
        aos = AdaptiveOperatorSelector(_make_config(credit_type='rank_improvement'))
        assert aos._compute_credit(0.3, 0.5) == 1.0
        assert aos._compute_credit(0.5, 0.3) == 0.0


class TestMinimumProbabilityFloor:
    def test_no_operator_below_floor(self):
        """Even with no credits, floor is enforced."""
        aos = AdaptiveOperatorSelector(_make_config(min_prob=0.10))
        # Give all credit to one operator
        for _ in range(200):
            aos.record_outcome('crossover', 'uniform', 0.2, 0.9)
        aos.update_probabilities()

        for op, prob in aos.crossover_probs.items():
            assert prob >= 0.099  # Allow tiny float rounding

    def test_probabilities_sum_to_one(self):
        aos = AdaptiveOperatorSelector(_make_config())
        for _ in range(100):
            aos.record_outcome('mutation', 'parameter', 0.1, 0.9)
            aos.record_outcome('mutation', 'indicator_add', 0.1, 0.5)
        aos.update_probabilities()

        total = sum(aos.mutation_probs.values())
        assert abs(total - 1.0) < 1e-6


class TestRecordOutcome:
    def test_unknown_operator_ignored(self):
        aos = AdaptiveOperatorSelector(_make_config())
        # Should not raise
        aos.record_outcome('crossover', 'nonexistent_op', 0.3, 0.7)
        assert len(aos._crossover_credits.get('nonexistent_op', [])) == 0

    def test_credits_accumulate(self):
        aos = AdaptiveOperatorSelector(_make_config(window_size=10))
        for _ in range(5):
            aos.record_outcome('crossover', 'single_point', 0.3, 0.5)
        assert len(aos._crossover_credits['single_point']) == 5

    def test_window_size_respected(self):
        aos = AdaptiveOperatorSelector(_make_config(window_size=5))
        for _ in range(20):
            aos.record_outcome('crossover', 'single_point', 0.3, 0.5)
        assert len(aos._crossover_credits['single_point']) == 5


class TestSerialization:
    def test_round_trip(self):
        aos = AdaptiveOperatorSelector(_make_config())
        for _ in range(10):
            aos.record_outcome('crossover', 'uniform', 0.2, 0.8)
            aos.record_outcome('mutation', 'parameter', 0.1, 0.9)
        aos.update_probabilities()

        data = aos.to_dict()
        aos2 = AdaptiveOperatorSelector(_make_config())
        aos2.load_from_dict(data)

        assert aos2.crossover_probs == pytest.approx(aos.crossover_probs, abs=1e-9)
        assert aos2.mutation_probs == pytest.approx(aos.mutation_probs, abs=1e-9)

    def test_get_report_structure(self):
        aos = AdaptiveOperatorSelector(_make_config())
        report = aos.get_report()
        assert 'crossover_probabilities' in report
        assert 'mutation_probabilities' in report
        assert 'crossover_credits_sizes' in report
        assert 'mutation_credits_sizes' in report


class TestEnforceFloor:
    def test_all_equal_probabilities(self):
        probs = {'a': 0.5, 'b': 0.5}
        result = AdaptiveOperatorSelector._enforce_floor(probs, 0.1)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_one_very_low(self):
        probs = {'a': 0.01, 'b': 0.99}
        result = AdaptiveOperatorSelector._enforce_floor(probs, 0.1)
        assert result['a'] >= 0.1
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_floor_above_uniform_capped(self):
        """Floor can't exceed 1/n (uniform)."""
        probs = {'a': 0.01, 'b': 0.99}
        result = AdaptiveOperatorSelector._enforce_floor(probs, 0.6)
        # With 2 ops, max floor = 0.5
        assert result['a'] >= 0.49  # ≈ 0.5
        assert abs(sum(result.values()) - 1.0) < 1e-6
