"""Tests for genetic_algorithm.core.culling — StrategyCuller."""

import pytest
from unittest.mock import MagicMock

from genetic_algorithm.core.culling import StrategyCuller, CullReport
from genetic_algorithm.core.population import Population


def _make_individual(fitness=0.5, profit=10.0, num_trades=50,
                     max_drawdown=0.1, no_trades=False, val_profit=None):
    """Create a mock individual with given metrics."""
    ind = MagicMock()
    ind.id = f"ind_{id(ind)}"
    ind.raw_fitness = fitness
    ind.metrics = {
        'profit': profit,
        'num_trades': num_trades,
        'max_drawdown': max_drawdown,
        'no_trades': no_trades,
    }
    if val_profit is not None:
        ind.metrics['val_profit'] = val_profit
    return ind


def _make_population(individuals):
    """Create a mock population."""
    pop = MagicMock(spec=Population)
    pop.individuals = list(individuals)
    pop.remove_individual = lambda ind: pop.individuals.remove(ind)
    return pop


class TestCullReportProperties:
    def test_num_removed(self):
        r = CullReport(generation=1, total_before=10, total_after=7)
        assert r.num_removed == 3

    def test_num_removed_no_change(self):
        r = CullReport(generation=0, total_before=10, total_after=10)
        assert r.num_removed == 0


class TestCullerDisabled:
    def test_disabled_does_nothing(self):
        culler = StrategyCuller({'strategy_culling': {'enabled': False}})
        pop = _make_population([_make_individual() for _ in range(5)])
        report = culler.cull_population(pop, generation=1)
        assert report.total_before == report.total_after
        assert report.num_removed == 0

    def test_skip_first_generation(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': True,
                'rules': {'zero_trades': {'enabled': True}},
            }
        })
        ind = _make_individual(num_trades=0, no_trades=True)
        pop = _make_population([ind])
        report = culler.cull_population(pop, generation=0)
        assert report.num_removed == 0


class TestZeroTradesRule:
    def test_removes_zero_trade_individual(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': False,
                'min_survivors': 1,
                'rules': {'zero_trades': {'enabled': True}},
            }
        })
        good = _make_individual(num_trades=50)
        bad = _make_individual(num_trades=0, no_trades=True)
        pop = _make_population([good, bad])
        report = culler.cull_population(pop, generation=1)
        assert report.num_removed == 1
        assert report.reasons.get('zero_trades') == 1

    def test_no_trades_flag_triggers_cull(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': False,
                'min_survivors': 1,
                'rules': {'zero_trades': {'enabled': True}},
            }
        })
        ind = _make_individual(num_trades=0, no_trades=True)
        pop = _make_population([ind, _make_individual()])
        report = culler.cull_population(pop, generation=1)
        assert 'zero_trades' in report.reasons


class TestNegativeProfitHighFitnessRule:
    def test_culls_high_fitness_losing_strategy(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': False,
                'min_survivors': 1,
                'rules': {
                    'zero_trades': {'enabled': False},
                    'negative_profit_high_fitness': {
                        'enabled': True,
                        'profit_threshold': -1.0,
                        'min_fitness_for_cull': 0.35,
                    },
                },
            }
        })
        bad = _make_individual(fitness=0.5, profit=-5.0)
        good = _make_individual(fitness=0.5, profit=10.0)
        pop = _make_population([bad, good])
        report = culler.cull_population(pop, generation=1)
        assert report.reasons.get('negative_profit_high_fitness') == 1


class TestExtremeDrawdownRule:
    def test_culls_extreme_drawdown(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': False,
                'min_survivors': 1,
                'rules': {
                    'zero_trades': {'enabled': False},
                    'extreme_drawdown': {
                        'enabled': True,
                        'max_drawdown': 0.40,
                    },
                },
            }
        })
        bad = _make_individual(max_drawdown=0.60)
        ok = _make_individual(max_drawdown=0.20)
        pop = _make_population([bad, ok])
        report = culler.cull_population(pop, generation=1)
        assert report.reasons.get('extreme_drawdown') == 1


class TestMinTradesFloorRule:
    def test_culls_below_min_trades(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': False,
                'min_survivors': 1,
                'rules': {
                    'zero_trades': {'enabled': False},
                    'min_trades_floor': {
                        'enabled': True,
                        'min_trades': 50,
                    },
                },
            }
        })
        few = _make_individual(num_trades=10)
        enough = _make_individual(num_trades=100)
        pop = _make_population([few, enough])
        report = culler.cull_population(pop, generation=1)
        assert report.reasons.get('min_trades_floor') == 1


class TestValidationFailureRule:
    def test_culls_bad_validation(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': False,
                'min_survivors': 1,
                'rules': {
                    'zero_trades': {'enabled': False},
                    'validation_failure': {
                        'enabled': True,
                        'min_val_profit': -5.0,
                    },
                },
            }
        })
        bad = _make_individual(val_profit=-10.0)
        ok = _make_individual(val_profit=5.0)
        pop = _make_population([bad, ok])
        report = culler.cull_population(pop, generation=1)
        assert report.reasons.get('validation_failure') == 1


class TestEliteProtection:
    def test_elites_not_culled(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': False,
                'protect_elites': True,
                'min_survivors': 1,
                'rules': {'zero_trades': {'enabled': True}},
            }
        })
        elite = _make_individual(fitness=0.9, num_trades=0, no_trades=True)
        normal = _make_individual(fitness=0.1, num_trades=0, no_trades=True)
        filler = _make_individual(fitness=0.5, num_trades=100)
        pop = _make_population([elite, normal, filler])
        report = culler.cull_population(pop, generation=1, elite_size=1)
        # The elite (highest fitness) should be protected;
        # only the lower-fitness zero-trade individual should be removed
        assert report.num_removed == 1


class TestMinSurvivorsGuard:
    def test_respects_min_survivors(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': False,
                'min_survivors': 3,
                'rules': {'zero_trades': {'enabled': True}},
            }
        })
        inds = [_make_individual(num_trades=0, no_trades=True) for _ in range(5)]
        pop = _make_population(inds)
        report = culler.cull_population(pop, generation=1)
        assert report.total_after >= 3


class TestAggregateStats:
    def test_stats_accumulate(self):
        culler = StrategyCuller({
            'strategy_culling': {
                'enabled': True,
                'skip_first_generation': False,
                'min_survivors': 1,
                'rules': {'zero_trades': {'enabled': True}},
            }
        })

        for gen in range(3):
            inds = [
                _make_individual(num_trades=0, no_trades=True),
                _make_individual(num_trades=100),
            ]
            pop = _make_population(inds)
            culler.cull_population(pop, generation=gen + 1)

        stats = culler.get_aggregate_stats()
        assert stats['total_culled'] == 3
        assert stats['reasons']['zero_trades'] == 3
