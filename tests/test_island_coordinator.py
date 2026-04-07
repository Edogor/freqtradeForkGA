"""Tests for island migration utilities and IslandCoordinator facade."""

import copy
import random
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from genetic_algorithm.engine.migration import (
    AggregateStats,
    get_top_individuals,
    inject_migrants,
    migrate,
    migrate_fully_connected,
    migrate_hierarchical,
    migrate_ring,
    migrate_tournament,
    TOPOLOGIES,
)
from genetic_algorithm.engine.islands import IslandCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeGene:
    generation: int = 0
    individual_id: int = 0

    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self):
        return {"gen": self.generation, "id": self.individual_id}


@dataclass
class _FakeIndividual:
    strategy_gene: _FakeGene = field(default_factory=_FakeGene)
    raw_fitness: float = 0.5
    fitness: float = 0.5
    evaluated: bool = True
    metrics: dict = field(default_factory=dict)
    id: str = "ind"


@dataclass
class _FakePopulation:
    individuals: list = field(default_factory=list)


def _make_pop(n=5, base_fitness=0.5):
    """Create a population with n individuals at varying fitness levels."""
    inds = []
    for i in range(n):
        ind = _FakeIndividual(
            strategy_gene=_FakeGene(generation=0, individual_id=i),
            raw_fitness=base_fitness + i * 0.1,
            fitness=base_fitness + i * 0.1,
            id=f"ind-{i}",
        )
        inds.append(ind)
    return _FakePopulation(individuals=inds)


def _make_populations(island_names, pop_size=5, base_fitnesses=None):
    """Create a dict of populations for multiple islands."""
    pops = {}
    for i, name in enumerate(island_names):
        base = (base_fitnesses or [0.5])[i % len(base_fitnesses or [0.5])]
        pops[name] = _make_pop(pop_size, base)
    return pops


# =====================================================================
# AggregateStats
# =====================================================================


class TestAggregateStats:
    def test_defaults(self):
        s = AggregateStats()
        assert s.best_fitness == 0.0
        assert s.genetic_diversity is None
        assert s.holdout_avg_degradation is None

    def test_custom_values(self):
        s = AggregateStats(best_fitness=1.5, generation=10)
        assert s.best_fitness == 1.5
        assert s.generation == 10


# =====================================================================
# get_top_individuals
# =====================================================================


class TestGetTopIndividuals:
    def test_returns_top_n(self):
        pops = {"island_a": _make_pop(5, base_fitness=0.1)}
        top = get_top_individuals(pops, "island_a", 2)
        assert len(top) == 2
        assert top[0].raw_fitness >= top[1].raw_fitness

    def test_missing_island(self):
        assert get_top_individuals({}, "missing", 3) == []

    def test_filters_zero_fitness(self):
        pop = _FakePopulation(
            individuals=[
                _FakeIndividual(raw_fitness=0.0, id="zero"),
                _FakeIndividual(raw_fitness=0.5, id="good"),
            ]
        )
        top = get_top_individuals({"a": pop}, "a", 5)
        assert len(top) == 1
        assert top[0].id == "good"

    def test_filters_none_fitness(self):
        pop = _FakePopulation(
            individuals=[
                _FakeIndividual(raw_fitness=None, id="none"),
                _FakeIndividual(raw_fitness=0.3, id="ok"),
            ]
        )
        top = get_top_individuals({"a": pop}, "a", 5)
        assert len(top) == 1

    def test_count_exceeds_available(self):
        pops = {"a": _make_pop(3)}
        top = get_top_individuals(pops, "a", 10)
        assert len(top) == 3


# =====================================================================
# inject_migrants
# =====================================================================


class TestInjectMigrants:
    def test_replaces_worst(self):
        pops = {"target": _make_pop(5, base_fitness=0.1)}
        donors = [_FakeIndividual(raw_fitness=9.0, id="donor")]
        replaced = inject_migrants(pops, "target", donors, generation=5, source="src")
        assert replaced == 1
        # The worst individual (fitness 0.1) should be replaced
        fitnesses = [ind.raw_fitness for ind in pops["target"].individuals]
        assert None in fitnesses or any(
            ind.metrics.get("origin") == "migrant_from_src"
            for ind in pops["target"].individuals
        )

    def test_migrants_marked_unevaluated(self):
        pops = {"t": _make_pop(3)}
        donors = [_FakeIndividual(raw_fitness=5.0)]
        inject_migrants(pops, "t", donors, generation=1, source="s")
        migrant = [
            ind for ind in pops["t"].individuals
            if ind.metrics.get("origin") == "migrant_from_s"
        ]
        assert len(migrant) == 1
        assert migrant[0].evaluated is False

    def test_empty_migrants(self):
        pops = {"t": _make_pop(3)}
        assert inject_migrants(pops, "t", [], generation=0) == 0

    def test_missing_target(self):
        assert inject_migrants({}, "nope", [_FakeIndividual()], generation=0) == 0

    def test_source_in_origin(self):
        pops = {"t": _make_pop(3)}
        donors = [_FakeIndividual(raw_fitness=5.0)]
        inject_migrants(pops, "t", donors, generation=1, source="island_x")
        origins = [
            ind.metrics.get("origin", "")
            for ind in pops["t"].individuals
        ]
        assert "migrant_from_island_x" in origins


# =====================================================================
# Migration topologies
# =====================================================================


class TestMigrateRing:
    def test_ring_sends_to_next(self):
        names = ["a", "b", "c"]
        pops = _make_populations(names, 5, [0.1, 0.5, 0.9])
        events = migrate_ring(names, pops, count=1, generation=0)
        # Should have 3 events: a→b, b→c, c→a
        assert len(events) == 3
        froms = [e["from"] for e in events]
        assert set(froms) == {"a", "b", "c"}

    def test_ring_wraps_around(self):
        names = ["x", "y"]
        pops = _make_populations(names, 3, [0.5, 0.5])
        events = migrate_ring(names, pops, count=1, generation=0)
        destinations = {e["to"] for e in events}
        assert destinations == {"x", "y"}


class TestMigrateFullyConnected:
    def test_all_to_all(self):
        names = ["a", "b", "c"]
        pops = _make_populations(names, 5, [0.5, 0.5, 0.5])
        events = migrate_fully_connected(names, pops, count=1, generation=0)
        # 3 sources × 2 targets each = 6 events
        assert len(events) == 6


class TestMigrateTournament:
    def test_winner_sends_to_loser(self):
        random.seed(42)
        names = ["strong", "weak"]
        pops = _make_populations(names, 5, [0.9, 0.1])
        events = migrate_tournament(names, pops, count=1, generation=0)
        assert len(events) == 1
        # Strong island should win
        assert events[0]["from"] == "strong"
        assert events[0]["to"] == "weak"

    def test_odd_number_of_islands(self):
        names = ["a", "b", "c"]
        pops = _make_populations(names, 3, [0.5, 0.5, 0.5])
        events = migrate_tournament(names, pops, count=1, generation=0)
        # Only 1 pair from 3 islands
        assert len(events) == 1


class TestMigrateHierarchical:
    def test_bidirectional_exchange(self):
        names = ["a", "b"]
        pops = _make_populations(names, 5, [0.5, 0.5])
        events = migrate_hierarchical(names, pops, count=1, generation=0)
        # Both directions
        assert len(events) == 2
        dirs = {(e["from"], e["to"]) for e in events}
        assert ("a", "b") in dirs or ("b", "a") in dirs


class TestMigrateDispatch:
    def test_all_topologies_registered(self):
        assert set(TOPOLOGIES.keys()) == {
            "ring", "fully_connected", "tournament", "hierarchical",
        }

    def test_dispatch_ring(self):
        names = ["a", "b"]
        pops = _make_populations(names, 3, [0.5, 0.5])
        events = migrate("ring", names, pops, count=1, generation=0)
        assert len(events) >= 1

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown migration topology"):
            migrate("star", ["a"], {}, count=1, generation=0)


# =====================================================================
# IslandCoordinator
# =====================================================================


class TestIslandCoordinator:
    def test_evolve_delegates(self):
        backend = MagicMock()
        backend.evolve.return_value = ["result"]
        coord = IslandCoordinator(backend)
        result = coord.evolve()
        backend.evolve.assert_called_once()
        assert result == ["result"]

    def test_backend_property(self):
        backend = MagicMock()
        coord = IslandCoordinator(backend)
        assert coord.backend is backend

    def test_island_names(self):
        backend = MagicMock()
        ic_a = MagicMock()
        ic_a.name = "a"
        ic_b = MagicMock()
        ic_b.name = "b"
        backend.island_configs = [ic_a, ic_b]
        coord = IslandCoordinator(backend)
        assert coord.island_names == ["a", "b"]

    def test_island_names_empty(self):
        backend = MagicMock(spec=[])
        coord = IslandCoordinator(backend)
        assert coord.island_names == []

    def test_from_config_no_island_model_raises(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("genetic_algorithm:\n  generations: 5\n")
        with pytest.raises(ValueError, match="No island model enabled"):
            IslandCoordinator.from_config(str(cfg))

    def test_from_config_both_enabled_raises(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "island_model:\n  enabled: true\n"
            "generic_island_model:\n  enabled: true\n"
        )
        with pytest.raises(ValueError, match="Both"):
            IslandCoordinator.from_config(str(cfg))

    def test_from_config_regime(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("island_model:\n  enabled: true\n")
        with patch(
            "genetic_algorithm.core.island_model.IslandModelEvolution"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            coord = IslandCoordinator.from_config(str(cfg))
            mock_cls.assert_called_once()
            assert coord.backend is mock_cls.return_value

    def test_from_config_generic(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("generic_island_model:\n  enabled: true\n")
        with patch(
            "genetic_algorithm.core.generic_island_model.GenericIslandModelEvolution"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            coord = IslandCoordinator.from_config(str(cfg))
            mock_cls.assert_called_once()
            assert coord.backend is mock_cls.return_value
