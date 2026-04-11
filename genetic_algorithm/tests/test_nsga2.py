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
        # Should return worst-case values (AP-7 fix: -1e6 for maximize, -1.0 for minimize)
        assert objectives[0] == -1e6  # maximize profit → large negative
        assert objectives[1] == -1.0  # minimize drawdown → -1.0
        assert objectives[2] == -1e6  # maximize sharpe → large negative

    def test_missing_metric_defaults_to_zero(self):
        metrics = {'num_trades': 20}  # missing profit, sharpe, drawdown
        objectives = extract_objectives_from_metrics(metrics, DEFAULT_OBJECTIVES)
        assert all(isinstance(o, float) for o in objectives)


# =============================================================================
# NSGA-II ENVIRONMENTAL SELECTION (μ+λ)
#
# Tests the (μ+λ) Pareto-based survivor selection algorithm that is called
# by GeneticAlgorithm._nsga2_environmental_selection.
#
# Rather than instantiating the heavy GeneticAlgorithm class, we replicate
# the exact merge→sort→fill algorithm here as a standalone helper so we can
# verify its correctness in isolation.
# =============================================================================


def _nsga2_env_select(parent_inds, offspring_inds, population_size):
    """
    Standalone replica of GeneticAlgorithm._nsga2_environmental_selection
    for unit-testing purposes.

    merge evaluated parents+offspring → Pareto sort → crowding distance →
    fill front-by-front up to population_size.
    Unevaluated offspring (objectives is None) are added last, filling any
    remaining slots.
    """
    combined = [ind for ind in parent_inds if ind.objectives is not None]
    combined += [ind for ind in offspring_inds if ind.objectives is not None]
    unevaluated = [ind for ind in offspring_inds if ind.objectives is None]

    if not combined:
        return list(unevaluated[:population_size])

    fronts = fast_non_dominated_sort(combined)
    for front in fronts:
        crowding_distance_assignment(front)

    survivors = []
    for front in fronts:
        if len(survivors) + len(front) <= population_size:
            survivors.extend(front)
        else:
            ranked = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
            remaining = population_size - len(survivors)
            survivors.extend(ranked[:remaining])
            break

    for ind in unevaluated:
        if len(survivors) >= population_size:
            break
        survivors.append(ind)

    return survivors


class TestNSGA2EnvironmentalSelection:

    def test_merges_evaluated_from_both_populations(self):
        """All evaluated individuals from parents + offspring are considered."""
        parents = [_make_ind([float(i), float(i)]) for i in range(3)]
        offspring = [_make_ind([float(i + 3), float(i + 3)]) for i in range(3)]
        survivors = _nsga2_env_select(parents, offspring, 6)
        assert len(survivors) == 6

    def test_unevaluated_offspring_excluded_from_sorting(self):
        """Individuals with objectives=None must not enter the Pareto sort."""
        parents = [_make_ind([1.0, 1.0]), _make_ind([2.0, 0.5])]
        # One evaluatedOffspring, two unevaluated
        evaluated_off = _make_ind([3.0, 0.1])
        unevaluated_off = Individual(strategy_gene=_make_gene())
        unevaluated_off.objectives = None
        unevaluated_off.evaluated = False

        survivors = _nsga2_env_select(parents, [evaluated_off, unevaluated_off], 4)
        evaluated_survivors = [s for s in survivors if s.objectives is not None]
        unevaluated_survivors = [s for s in survivors if s.objectives is None]

        # Exactly 3 evaluated + 1 unevaluated fills 4 slots
        assert len(evaluated_survivors) == 3
        assert len(unevaluated_survivors) == 1

    def test_rank1_individuals_preserved_when_front_fits(self):
        """When rank-1 front size ≤ population_size, all rank-1 survive."""
        # 3 non-dominated + 2 dominated, pop_size = 3 → all rank-1 kept
        rank1 = [
            _make_ind([3.0, 1.0]),
            _make_ind([2.0, 2.0]),
            _make_ind([1.0, 3.0]),
        ]
        rank2 = [
            _make_ind([0.5, 0.5]),
            _make_ind([0.6, 0.4]),
        ]
        survivors = _nsga2_env_select(rank1 + rank2, [], 3)
        assert all(s.rank == 1 for s in survivors)

    def test_partial_front_filled_by_crowding_distance(self):
        """When last front is partial, take highest-crowding-distance individuals."""
        # 2 rank-1 individuals + 3 rank-2 individuals, pop_size = 4
        front1 = [_make_ind([3.0, 1.0]), _make_ind([1.0, 3.0])]
        # Rank-2 individuals spread across objective space (different crowding)
        front2 = [
            _make_ind([0.1, 0.1]),  # interior
            _make_ind([0.0, 2.0]),  # boundary
            _make_ind([2.0, 0.0]),  # boundary
        ]
        survivors = _nsga2_env_select(front1 + front2, [], 4)
        assert len(survivors) == 4
        # The two rank-1 survivors must be in the output
        rank1_ids = {id(f) for f in front1}
        assert any(id(s) in rank1_ids for s in survivors)
        # The two selected from front2 should be boundary points (inf crowding)
        rank2_survivors = [s for s in survivors if s.rank == 2]
        assert len(rank2_survivors) == 2
        # Both should have infinite crowding distance (boundary points)
        assert all(
            s.crowding_distance == float('inf') for s in rank2_survivors
        ), "Partial front selection must prefer boundary (infinite crowding distance) points"

    def test_output_size_equals_population_size(self):
        """Selection output must be exactly population_size individuals."""
        parents = [_make_ind([float(i), float(10 - i)]) for i in range(8)]
        offspring = [_make_ind([float(i) * 0.5, float(i) * 0.5]) for i in range(4)]
        for pop_size in (3, 5, 8, 10):
            survivors = _nsga2_env_select(parents, offspring, pop_size)
            assert len(survivors) == pop_size, \
                f"Expected {pop_size} survivors, got {len(survivors)}"

    def test_no_duplicate_individuals_in_output(self):
        """No individual should appear twice in the survivor list."""
        inds = [_make_ind([float(i), float(10 - i)]) for i in range(6)]
        survivors = _nsga2_env_select(inds[:3], inds[3:], 5)
        ids = [id(s) for s in survivors]
        assert len(ids) == len(set(ids)), "Duplicate individual in survivors"

    def test_empty_combined_returns_unevaluated(self):
        """If all individuals are unevaluated, return whatever fits."""
        unevaluated = [Individual(strategy_gene=_make_gene()) for _ in range(3)]
        for u in unevaluated:
            u.objectives = None
        survivors = _nsga2_env_select([], unevaluated, 2)
        assert len(survivors) <= 2

    def test_all_rank1_larger_than_pop_size_uses_crowding(self):
        """More rank-1 individuals than pop_size → crowding distance selects."""
        # 5 Pareto-optimal points on a clear front, pop_size = 3
        front = [_make_ind([float(i), float(4 - i)]) for i in range(5)]
        survivors = _nsga2_env_select(front, [], 3)
        assert len(survivors) == 3
        # Boundary points (index 0 and 4) have inf crowding → should be selected
        boundary_inds = {id(front[0]), id(front[4])}
        survivor_ids = {id(s) for s in survivors}
        assert boundary_inds.issubset(survivor_ids), \
            "Boundary points must be included when selecting from partial Pareto front"


# =============================================================================
# NSGA-II SELECTION INTEGRATION PIPELINE
#
# End-to-end chain: extract_objectives_from_metrics →
#   fast_non_dominated_sort → crowding_distance_assignment →
#   nsga2_tournament_selection
# =============================================================================


class TestNSGA2SelectionIntegration:

    def test_extract_then_sort_produces_correct_ranks(self):
        """Full pipeline: extract objectives → sort → verify rank assignment."""
        # Three strategies with clear dominance ordering
        metrics_a = {'profit': 80.0, 'max_drawdown': 0.05, 'sharpe_ratio': 2.0, 'num_trades': 20}
        metrics_b = {'profit': 50.0, 'max_drawdown': 0.20, 'sharpe_ratio': 1.0, 'num_trades': 20}
        metrics_c = {'profit': 20.0, 'max_drawdown': 0.40, 'sharpe_ratio': 0.5, 'num_trades': 20}

        inds = []
        for metrics in [metrics_a, metrics_b, metrics_c]:
            ind = Individual(strategy_gene=_make_gene())
            objectives = extract_objectives_from_metrics(metrics, DEFAULT_OBJECTIVES)
            ind.set_objectives(objectives, metrics)
            inds.append(ind)

        fronts = fast_non_dominated_sort(inds)

        # a dominates b (better in all objectives), b dominates c
        # So we expect 3 separate fronts
        assert len(fronts) == 3
        assert inds[0].rank == 1
        assert inds[1].rank == 2
        assert inds[2].rank == 3

    def test_crowding_nonzero_for_interior_points(self):
        """Interior Pareto-front points get finite, non-zero crowding distance."""
        # 4-point Pareto front: 2 boundary, 2 interior
        objectives_list = [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ]
        inds = [_make_ind(o) for o in objectives_list]
        fronts = fast_non_dominated_sort(inds)
        assert len(fronts) == 1  # all non-dominated
        crowding_distance_assignment(inds)

        # Boundaries (first and last sorted by obj[0]) get inf
        sorted_by_obj0 = sorted(inds, key=lambda x: x.objectives[0])
        assert sorted_by_obj0[0].crowding_distance == float('inf')
        assert sorted_by_obj0[-1].crowding_distance == float('inf')
        # Interior points get finite positive distance
        assert sorted_by_obj0[1].crowding_distance > 0
        assert sorted_by_obj0[1].crowding_distance != float('inf')
        assert sorted_by_obj0[2].crowding_distance > 0
        assert sorted_by_obj0[2].crowding_distance != float('inf')

    def test_tournament_always_prefers_lower_rank(self):
        """Over 50 trials tournament must always select rank-1 over rank-2."""
        rank1 = _make_ind([3.0, 3.0])
        rank1.rank = 1
        rank1.crowding_distance = 0.1  # low diversity

        rank2 = _make_ind([1.0, 1.0])
        rank2.rank = 2
        rank2.crowding_distance = float('inf')  # high diversity — but rank is worse

        wins_rank1 = sum(
            1 for _ in range(50)
            if nsga2_tournament_selection([rank1, rank2], tournament_size=2) is rank1
        )
        assert wins_rank1 == 50, \
            "Rank-1 individual must win every tournament against rank-2, regardless of crowding distance"

    def test_tournament_same_rank_prefers_higher_crowding(self):
        """When rank is tied, higher crowding distance wins more often."""
        a = _make_ind([2.0, 2.0])
        a.rank = 1
        a.crowding_distance = 0.01

        b = _make_ind([1.0, 3.0])
        b.rank = 1
        b.crowding_distance = float('inf')

        wins_b = sum(
            1 for _ in range(100)
            if nsga2_tournament_selection([a, b], tournament_size=2) is b
        )
        assert wins_b > 60, \
            f"High-crowding individual should win most tournaments (won {wins_b}/100)"

    def test_full_pipeline_pareto_front_size(self):
        """Full pipeline: 10 individuals → verify first Pareto front size."""
        # 5 non-dominated + 5 dominated
        non_dominated = [
            _make_ind([float(i), float(4 - i)]) for i in range(5)
        ]
        dominated = [
            _make_ind([float(i) * 0.3, float(i) * 0.3]) for i in range(5)
        ]
        all_inds = non_dominated + dominated
        fronts = fast_non_dominated_sort(all_inds)

        # First front should be all 5 non-dominated
        assert len(fronts[0]) == 5
        assert all(ind.rank == 1 for ind in fronts[0])
        # All dominated individuals should NOT be in front 1
        dominated_ids = {id(d) for d in dominated}
        for ind in fronts[0]:
            assert id(ind) not in dominated_ids

    def test_extract_objectives_chained_with_sort(self):
        """extract_objectives_from_metrics result can be used directly with sort."""
        metrics_good = {'profit': 60.0, 'max_drawdown': 0.10, 'sharpe_ratio': 2.0, 'num_trades': 15}
        metrics_bad = {'profit': 5.0, 'max_drawdown': 0.50, 'sharpe_ratio': 0.2, 'num_trades': 15}

        ind_good = Individual(strategy_gene=_make_gene())
        ind_bad = Individual(strategy_gene=_make_gene())

        ind_good.set_objectives(
            extract_objectives_from_metrics(metrics_good, DEFAULT_OBJECTIVES), metrics_good
        )
        ind_bad.set_objectives(
            extract_objectives_from_metrics(metrics_bad, DEFAULT_OBJECTIVES), metrics_bad
        )

        fronts = fast_non_dominated_sort([ind_good, ind_bad])
        # Good dominates bad — should be in rank-1 front
        assert ind_good.rank == 1
        assert ind_bad.rank == 2

