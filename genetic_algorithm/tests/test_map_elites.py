"""
Tests for MAP-Elites Behavioral Archive

Covers: behavior computation, discretization, cell placement/replacement,
exploration bonus, diverse sampling, and serialization.
"""

import pytest
from unittest.mock import MagicMock

from genetic_algorithm.core.map_elites import MAPElitesArchive, _compute_behavior


# =============================================================================
# HELPERS
# =============================================================================


def _make_ind(fitness=0.5, trade_count=30, trading_days=90, avg_duration_hours=6.0):
    """Create a mock individual with behavior-relevant metrics."""
    ind = MagicMock()
    ind.fitness = fitness
    ind.evaluated = True
    ind.metrics = {
        'trade_count': trade_count,
        'trading_days': trading_days,
        'avg_duration_hours': avg_duration_hours,
    }
    ind.strategy_gene = MagicMock()
    ind.strategy_gene.to_dict.return_value = {'timeframe': '1h'}
    return ind


def _default_config(enabled=True):
    return {
        'map_elites': {
            'enabled': enabled,
            'frequency_bins': 5,
            'duration_bins': 5,
            'exploration_bonus': 0.05,
            'injection_count': 3,
        }
    }


# =============================================================================
# BEHAVIOR COMPUTATION TESTS
# =============================================================================


class TestComputeBehavior:

    def test_basic_metrics(self):
        ind = _make_ind(trade_count=30, trading_days=90, avg_duration_hours=12.0)
        result = _compute_behavior(ind)
        assert result is not None
        trades_per_month, avg_hours = result
        assert trades_per_month == pytest.approx(10.0)  # 30/90*30
        assert avg_hours == pytest.approx(12.0)

    def test_missing_duration_returns_none(self):
        ind = MagicMock()
        ind.metrics = {'trade_count': 20, 'trading_days': 60}
        result = _compute_behavior(ind)
        assert result is None

    def test_zero_trading_days(self):
        ind = _make_ind(trading_days=0, avg_duration_hours=5.0, trade_count=10)
        result = _compute_behavior(ind)
        assert result is not None
        # Falls back to float(total_trades) when days=0
        trades_per_month, _ = result
        assert trades_per_month == 10.0

    def test_large_duration_converted_from_minutes(self):
        ind = _make_ind(avg_duration_hours=3600)  # > 1000, treated as minutes
        result = _compute_behavior(ind)
        assert result is not None
        _, avg_hours = result
        assert avg_hours == pytest.approx(60.0)

    def test_win_lose_duration_fallback(self):
        ind = MagicMock()
        ind.metrics = {
            'trade_count': 20,
            'trading_days': 60,
            'winning_avg_duration': 8.0,
            'losing_avg_duration': 4.0,
            'win_rate': 0.6,
        }
        result = _compute_behavior(ind)
        assert result is not None
        _, avg_hours = result
        assert avg_hours == pytest.approx(8.0 * 0.6 + 4.0 * 0.4)

    def test_no_metrics_returns_none(self):
        ind = MagicMock()
        ind.metrics = None
        result = _compute_behavior(ind)
        assert result is None


# =============================================================================
# ARCHIVE INITIALIZATION TESTS
# =============================================================================


class TestMAPElitesArchiveInit:

    def test_default_config(self):
        archive = MAPElitesArchive(_default_config())
        assert archive.enabled is True
        assert archive.freq_bins == 5
        assert archive.dur_bins == 5
        assert archive.total_cells == 25
        assert archive.filled_cells == 0
        assert archive.occupancy == 0.0

    def test_disabled(self):
        archive = MAPElitesArchive(_default_config(enabled=False))
        assert archive.enabled is False

    def test_custom_bins(self):
        config = {'map_elites': {'enabled': True, 'frequency_bins': 3, 'duration_bins': 4}}
        archive = MAPElitesArchive(config)
        assert archive.total_cells == 12


# =============================================================================
# DISCRETIZATION TESTS
# =============================================================================


class TestDiscretize:

    def test_low_values(self):
        archive = MAPElitesArchive(_default_config())
        cell = archive._discretize(0.5, 0.3)
        assert cell == (0, 0)

    def test_high_values(self):
        archive = MAPElitesArchive(_default_config())
        cell = archive._discretize(100, 200)
        assert cell[0] == 4  # capped to freq_bins - 1
        assert cell[1] == 4  # capped to dur_bins - 1

    def test_mid_values(self):
        archive = MAPElitesArchive(_default_config())
        # edges: [0, 2, 5, 10, 25] for freq, [0, 1, 4, 12, 48] for dur
        cell = archive._discretize(7, 8)
        assert cell[0] == 2  # 5 <= 7 < 10 → bin 2
        assert cell[1] == 2  # 4 <= 8 < 12 → bin 2


# =============================================================================
# UPDATE (PLACEMENT) TESTS
# =============================================================================


class TestUpdate:

    def test_place_single_individual(self):
        archive = MAPElitesArchive(_default_config())
        ind = _make_ind(fitness=0.5, trade_count=30, trading_days=90, avg_duration_hours=6.0)
        placements = archive.update([ind])
        assert placements == 1
        assert archive.filled_cells == 1

    def test_disabled_archive_no_placements(self):
        archive = MAPElitesArchive(_default_config(enabled=False))
        ind = _make_ind()
        placements = archive.update([ind])
        assert placements == 0

    def test_replace_with_better_fitness(self):
        archive = MAPElitesArchive(_default_config())
        ind1 = _make_ind(fitness=0.3, trade_count=10, trading_days=90, avg_duration_hours=2.0)
        ind2 = _make_ind(fitness=0.8, trade_count=10, trading_days=90, avg_duration_hours=2.0)
        archive.update([ind1])
        assert archive.filled_cells == 1
        placements = archive.update([ind2])
        assert placements == 1
        assert archive.filled_cells == 1  # same cell, replaced
        assert archive.total_replacements == 1

    def test_no_replace_with_worse_fitness(self):
        archive = MAPElitesArchive(_default_config())
        ind1 = _make_ind(fitness=0.8, trade_count=10, trading_days=90, avg_duration_hours=2.0)
        ind2 = _make_ind(fitness=0.3, trade_count=10, trading_days=90, avg_duration_hours=2.0)
        archive.update([ind1])
        placements = archive.update([ind2])
        assert placements == 0  # worse fitness, not placed

    def test_skips_unevaluated(self):
        archive = MAPElitesArchive(_default_config())
        ind = _make_ind()
        ind.evaluated = False
        placements = archive.update([ind])
        assert placements == 0

    def test_skips_none_fitness(self):
        archive = MAPElitesArchive(_default_config())
        ind = _make_ind()
        ind.fitness = None
        placements = archive.update([ind])
        assert placements == 0

    def test_multiple_individuals_different_cells(self):
        archive = MAPElitesArchive(_default_config())
        ind1 = _make_ind(fitness=0.5, trade_count=3, trading_days=90, avg_duration_hours=0.5)
        ind2 = _make_ind(fitness=0.6, trade_count=60, trading_days=90, avg_duration_hours=100.0)
        placements = archive.update([ind1, ind2])
        assert placements == 2
        assert archive.filled_cells == 2

    def test_occupancy_calculation(self):
        archive = MAPElitesArchive(_default_config())
        ind = _make_ind()
        archive.update([ind])
        assert archive.occupancy == pytest.approx(1 / 25)


# =============================================================================
# EXPLORATION BONUS TESTS
# =============================================================================


class TestExplorationBonus:

    def test_bonus_for_empty_cell(self):
        archive = MAPElitesArchive(_default_config())
        ind = _make_ind(trade_count=30, trading_days=90, avg_duration_hours=6.0)
        bonus = archive.get_exploration_bonus(ind)
        assert bonus == 0.05

    def test_no_bonus_for_occupied_cell(self):
        archive = MAPElitesArchive(_default_config())
        ind = _make_ind(trade_count=30, trading_days=90, avg_duration_hours=6.0)
        archive.update([ind])
        bonus = archive.get_exploration_bonus(ind)
        assert bonus == 0.0

    def test_no_bonus_when_disabled(self):
        archive = MAPElitesArchive(_default_config(enabled=False))
        ind = _make_ind()
        bonus = archive.get_exploration_bonus(ind)
        assert bonus == 0.0


# =============================================================================
# DIVERSE SAMPLING TESTS
# =============================================================================


class TestSampleDiverse:

    def test_empty_archive(self):
        archive = MAPElitesArchive(_default_config())
        assert archive.sample_diverse(3) == []

    def test_fewer_cells_than_requested(self):
        archive = MAPElitesArchive(_default_config())
        ind = _make_ind()
        archive.update([ind])
        result = archive.sample_diverse(5)
        assert len(result) == 1

    def test_returns_requested_count(self):
        archive = MAPElitesArchive(_default_config())
        # Place individuals in different cells
        inds = [
            _make_ind(fitness=0.3, trade_count=1, trading_days=90, avg_duration_hours=0.5),
            _make_ind(fitness=0.4, trade_count=8, trading_days=90, avg_duration_hours=3.0),
            _make_ind(fitness=0.5, trade_count=30, trading_days=90, avg_duration_hours=20.0),
            _make_ind(fitness=0.6, trade_count=80, trading_days=90, avg_duration_hours=100.0),
        ]
        archive.update(inds)
        result = archive.sample_diverse(3)
        assert len(result) == 3


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:

    def test_round_trip(self):
        archive = MAPElitesArchive(_default_config())
        ind = _make_ind(fitness=0.5, trade_count=15, trading_days=90, avg_duration_hours=6.0)
        archive.update([ind])
        state = archive.to_dict()
        assert 'grid' in state
        assert state['total_placements'] == 1

    def test_get_report(self):
        archive = MAPElitesArchive(_default_config())
        ind = _make_ind()
        archive.update([ind])
        report = archive.get_report()
        assert 'occupancy' in report
        assert 'filled_cells' in report
        assert 'total_cells' in report
