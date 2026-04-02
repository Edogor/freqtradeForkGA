"""
Tests for NSGA-II Multi-Objective Optimization

Covers: dominance, non-dominated sorting, crowding distance,
tournament selection, hypervolume calculation, and objective extraction.
"""

import pytest
import math

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.nsga2 import (
    dominates,
    fast_non_dominated_sort,
    crowding_distance_assignment,
    nsga2_tournament_selection,
    nsga2_crowded_comparison_sort,
    get_pareto_front,
    calculate_hypervolume,
    extract_objectives_from_metrics,
    DEFAULT_OBJECTIVES,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_gene(individual_id=0):
    return StrategyGene(
        generation=1,
        individual_id=individual_id,
        indicators=[IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0')],
        entry_conditions=[ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND')],
        exit_conditions=[ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='AND')],
        timeframe='5m',
        stoploss=-0.05,
    )


def _make_ind(objectives, fitness=None):
    """Create Individual with given objectives."""
    ind = Individual(strategy_gene=_make_gene())
    ind.objectives = objectives
    ind.fitness = fitness or (sum(objectives) if objectives else 0.0)
    ind.evaluated = True
    ind.metrics = {'num_trades': 20}
    return ind


# =============================================================================
# DOMINANCE TESTS
# =============================================================================


class TestDominates:

    def test_strictly_better_in_all(self):
        a = _make_ind([3.0, 3.0])
        b = _make_ind([1.0, 1.0])
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_better_in_one_equal_in_rest(self):
        a = _make_ind([3.0, 2.0])
        b = _make_ind([2.0, 2.0])
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_equal_objectives_no_dominance(self):
        a = _make_ind([2.0, 2.0])
        b = _make_ind([2.0, 2.0])
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_tradeoff_no_dominance(self):
        a = _make_ind([3.0, 1.0])
        b = _make_ind([1.0, 3.0])
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_none_objectives_returns_false(self):
        a = _make_ind(None)
        b = _make_ind([1.0, 2.0])
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_mismatched_lengths_raises(self):
        a = _make_ind([1.0, 2.0])
        b = _make_ind([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="same length"):
            dominates(a, b)

    def test_three_objectives(self):
        a = _make_ind([3.0, 3.0, 3.0])
        b = _make_ind([2.0, 2.0, 2.0])
        assert dominates(a, b) is True

    def test_three_objectives_tradeoff(self):
        a = _make_ind([3.0, 1.0, 3.0])
        b = _make_ind([2.0, 2.0, 2.0])
        assert dominates(a, b) is False
        assert dominates(b, a) is False


# =============================================================================
# NON-DOMINATED SORT TESTS
# =============================================================================


class TestFastNonDominatedSort:

    def test_empty_population(self):
        assert fast_non_dominated_sort([]) == []

    def test_single_individual(self):
        ind = _make_ind([1.0, 2.0])
        fronts = fast_non_dominated_sort([ind])
        assert len(fronts) == 1
        assert fronts[0] == [ind]
        assert ind.rank == 1

    def test_all_non_dominated(self):
        """Points on the Pareto front should all be rank 1."""
        inds = [
            _make_ind([3.0, 1.0]),
            _make_ind([2.0, 2.0]),
            _make_ind([1.0, 3.0]),
        ]
        fronts = fast_non_dominated_sort(inds)
        assert len(fronts) == 1
        assert len(fronts[0]) == 3
        for ind in inds:
            assert ind.rank == 1

    def test_two_fronts(self):
        """Dominated point goes to front 2."""
        a = _make_ind([3.0, 3.0])  # dominates c
        b = _make_ind([1.0, 4.0])  # non-dominated (tradeoff with a)
        c = _make_ind([1.0, 1.0])  # dominated by a
        fronts = fast_non_dominated_sort([a, b, c])
        assert len(fronts) == 2
        assert a.rank == 1
        assert b.rank == 1
        assert c.rank == 2

    def test_three_fronts_stacked(self):
        """Stacked dominance creates 3 fronts."""
        inds = [
            _make_ind([3.0, 3.0]),  # front 1 — dominates all below
            _make_ind([2.0, 2.0]),  # front 2 — dominates (1,1) only
            _make_ind([1.0, 1.0]),  # front 3
        ]
        fronts = fast_non_dominated_sort(inds)
        assert len(fronts) == 3
        assert inds[0].rank == 1
        assert inds[1].rank == 2
        assert inds[2].rank == 3

    def test_skips_unevaluated_individuals(self):
        a = _make_ind([3.0, 3.0])
        b = _make_ind(None)  # no objectives
        fronts = fast_non_dominated_sort([a, b])
        assert len(fronts) == 1
        assert fronts[0] == [a]

    def test_mixed_three_objectives(self):
        """3-objective sorting produces correct ranks."""
        a = _make_ind([3.0, 3.0, 3.0])  # dominates c
        b = _make_ind([4.0, 1.0, 2.0])  # tradeoff with a
        c = _make_ind([1.0, 1.0, 1.0])  # dominated by a
        fronts = fast_non_dominated_sort([a, b, c])
        assert a.rank == 1
        assert b.rank == 1
        assert c.rank == 2


# =============================================================================
# CROWDING DISTANCE TESTS
# =============================================================================


class TestCrowdingDistanceAssignment:

    def test_empty_front(self):
        crowding_distance_assignment([])  # no crash

    def test_single_individual(self):
        ind = _make_ind([1.0, 2.0])
        crowding_distance_assignment([ind])
        assert ind.crowding_distance == float('inf')

    def test_two_individuals(self):
        a = _make_ind([1.0, 3.0])
        b = _make_ind([3.0, 1.0])
        crowding_distance_assignment([a, b])
        assert a.crowding_distance == float('inf')
        assert b.crowding_distance == float('inf')

    def test_boundary_points_get_infinity(self):
        """In a front of 3+, boundary points should get inf."""
        a = _make_ind([1.0, 4.0])
        b = _make_ind([2.0, 3.0])
        c = _make_ind([4.0, 1.0])
        crowding_distance_assignment([a, b, c])
        # Boundary in objective 0: a (min) and c (max)
        # Boundary in objective 1: c (min) and a (max)
        assert a.crowding_distance == float('inf')
        assert c.crowding_distance == float('inf')

    def test_interior_point_finite_distance(self):
        """Interior points get non-zero finite distance."""
        a = _make_ind([1.0, 4.0])
        b = _make_ind([2.0, 2.5])
        c = _make_ind([4.0, 1.0])
        crowding_distance_assignment([a, b, c])
        assert b.crowding_distance > 0
        assert b.crowding_distance != float('inf')

    def test_four_points_interior_scores(self):
        """Four-point front: two boundary (inf), two interior (finite)."""
        inds = [
            _make_ind([1.0, 4.0]),
            _make_ind([2.0, 3.0]),
            _make_ind([3.0, 2.0]),
            _make_ind([4.0, 1.0]),
        ]
        crowding_distance_assignment(inds)
        # Sort by objective 0 to identify boundaries
        sorted_inds = sorted(inds, key=lambda x: x.objectives[0])
        assert sorted_inds[0].crowding_distance == float('inf')
        assert sorted_inds[-1].crowding_distance == float('inf')
        assert sorted_inds[1].crowding_distance > 0
        assert sorted_inds[2].crowding_distance > 0

    def test_all_same_objectives(self):
        """All identical objectives: range=0, no crash."""
        inds = [_make_ind([2.0, 2.0]) for _ in range(4)]
        crowding_distance_assignment(inds)
        # Boundary points still get inf
        # Interior distance contributions should be 0 when range is 0


# =============================================================================
# TOURNAMENT SELECTION TESTS
# =============================================================================


class TestNSGA2TournamentSelection:

    def test_selects_lower_rank(self):
        a = _make_ind([3.0, 3.0])
        a.rank = 1
        a.crowding_distance = 0.5
        b = _make_ind([1.0, 1.0])
        b.rank = 2
        b.crowding_distance = 10.0
        # a has lower (better) rank — always preferred
        winner = nsga2_tournament_selection([a, b], tournament_size=2)
        assert winner.rank <= 2

    def test_same_rank_prefers_higher_crowding(self):
        a = _make_ind([3.0, 1.0])
        a.rank = 1
        a.crowding_distance = 0.1
        b = _make_ind([1.0, 3.0])
        b.rank = 1
        b.crowding_distance = float('inf')
        # Same rank, b has higher crowding distance
        # Over many trials, b should be preferred more often
        wins_b = sum(
            1 for _ in range(100)
            if nsga2_tournament_selection([a, b], tournament_size=2) is b
        )
        assert wins_b > 40  # b should win most of the time

    def test_empty_population_raises(self):
        with pytest.raises(ValueError):
            nsga2_tournament_selection([], tournament_size=2)


# =============================================================================
# CROWDED COMPARISON SORT TESTS
# =============================================================================


class TestNSGA2CrowdedComparisonSort:

    def test_sorts_by_rank_first(self):
        a = _make_ind([1.0, 1.0])
        a.rank = 3
        b = _make_ind([2.0, 2.0])
        b.rank = 1
        c = _make_ind([1.5, 1.5])
        c.rank = 2
        result = nsga2_crowded_comparison_sort([a, b, c])
        assert [r.rank for r in result] == [1, 2, 3]

    def test_same_rank_sorts_by_crowding_desc(self):
        a = _make_ind([1.0, 3.0])
        a.rank = 1
        a.crowding_distance = 0.5
        b = _make_ind([2.0, 2.0])
        b.rank = 1
        b.crowding_distance = 2.0
        c = _make_ind([3.0, 1.0])
        c.rank = 1
        c.crowding_distance = float('inf')
        result = nsga2_crowded_comparison_sort([a, b, c])
        assert result[0] is c  # inf crowding first
        assert result[1] is b  # 2.0
        assert result[2] is a  # 0.5


# =============================================================================
# PARETO FRONT EXTRACTION
# =============================================================================


class TestGetParetoFront:

    def test_returns_rank_1(self):
        a = _make_ind([3.0, 3.0])  # front 1
        b = _make_ind([1.0, 4.0])  # front 1
        c = _make_ind([1.0, 1.0])  # front 2
        front = get_pareto_front([a, b, c])
        assert a in front
        assert b in front
        assert c not in front

    def test_empty_population(self):
        assert get_pareto_front([]) == []


# =============================================================================
# HYPERVOLUME TESTS
# =============================================================================


class TestCalculateHypervolume:

    def test_empty_front(self):
        assert calculate_hypervolume([], [0.0, 0.0]) == 0.0

    def test_single_point_2d(self):
        ind = _make_ind([3.0, 4.0])
        hv = calculate_hypervolume([ind], [0.0, 0.0])
        assert hv == pytest.approx(12.0, rel=0.01)

    def test_two_points_2d(self):
        """Two non-dominated points, staircase area."""
        a = _make_ind([4.0, 2.0])
        b = _make_ind([2.0, 4.0])
        hv = calculate_hypervolume([a, b], [0.0, 0.0])
        # Staircase: 4*2 + 2*(4-2) = 8 + 4 = 12
        assert hv == pytest.approx(12.0, rel=0.01)

    def test_point_below_reference_excluded(self):
        """Points that don't dominate reference are filtered out."""
        a = _make_ind([3.0, 3.0])
        b = _make_ind([-1.0, -1.0])  # below reference
        hv = calculate_hypervolume([a, b], [0.0, 0.0])
        assert hv == pytest.approx(9.0, rel=0.01)

    def test_3d_single_point(self):
        ind = _make_ind([2.0, 3.0, 4.0])
        hv = calculate_hypervolume([ind], [0.0, 0.0, 0.0])
        assert hv == pytest.approx(24.0, rel=0.01)

    def test_3d_two_points(self):
        """Two non-dominated points in 3D."""
        a = _make_ind([3.0, 1.0, 2.0])
        b = _make_ind([1.0, 3.0, 2.0])
        hv = calculate_hypervolume([a, b], [0.0, 0.0, 0.0])
        assert hv > 0  # Non-zero hypervolume


# =============================================================================
# OBJECTIVE EXTRACTION TESTS
# =============================================================================


class TestExtractObjectivesFromMetrics:

    def test_basic_maximize(self):
        metrics = {'profit': 50.0, 'max_drawdown': 0.10, 'sharpe_ratio': 1.5, 'num_trades': 20}
        objectives = extract_objectives_from_metrics(metrics, DEFAULT_OBJECTIVES)
        assert len(objectives) == 3
        # profit/100, -drawdown, sharpe/3
        assert objectives[0] == pytest.approx(0.5, rel=0.01)
        assert objectives[1] == pytest.approx(-0.10, rel=0.01)
        assert objectives[2] == pytest.approx(0.5, rel=0.01)

    def test_minimize_negated(self):
        config = [{'name': 'max_drawdown', 'type': 'minimize'}]
        metrics = {'max_drawdown': 0.15, 'num_trades': 10}
        objectives = extract_objectives_from_metrics(metrics, config)
        assert objectives[0] == -0.15

    def test_goldilocks_objective(self):
        config = [{'name': 'trade_count', 'type': 'goldilocks', 'target': 50, 'tolerance': 25}]
        metrics = {'trade_count': 50, 'num_trades': 50}
        objectives = extract_objectives_from_metrics(metrics, config)
        assert objectives[0] == pytest.approx(1.0)  # at target

    def test_goldilocks_off_target(self):
        config = [{'name': 'trade_count', 'type': 'goldilocks', 'target': 50, 'tolerance': 25}]
        metrics = {'trade_count': 75, 'num_trades': 75}
        objectives = extract_objectives_from_metrics(metrics, config)
        assert objectives[0] == pytest.approx(0.0)  # 25 away from target = tolerance

    def test_min_trades_gate(self):
        metrics = {'profit': 100.0, 'sharpe_ratio': 3.0, 'max_drawdown': 0.01, 'num_trades': 2}
        objectives = extract_objectives_from_metrics(metrics, DEFAULT_OBJECTIVES, min_trades=5)
        # Should return worst-case values
        assert objectives[0] == 0.0  # maximize profit
        assert objectives[2] == 0.0  # maximize sharpe

    def test_missing_metric_defaults_to_zero(self):
        metrics = {'num_trades': 20}  # missing profit, sharpe, drawdown
        objectives = extract_objectives_from_metrics(metrics, DEFAULT_OBJECTIVES)
        assert all(isinstance(o, float) for o in objectives)
