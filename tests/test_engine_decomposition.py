"""Tests for CheckpointManager and AdaptiveController (Phase 1 extracted modules)."""

import json
import random
from pathlib import Path

import pytest

from genetic_algorithm.engine.adaptive import AdaptiveController
from genetic_algorithm.engine.checkpoint import CheckpointManager
from genetic_algorithm.engine.checkpoint_contract import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    CheckpointProvenanceV3,
)
from genetic_algorithm.engine.population import Population, PopulationStats
from genetic_algorithm.genome.gene import ConditionGene, IndicatorGene, StrategyGene
from genetic_algorithm.genome.individual import Individual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_individual(gen=0, ind_id=0, fitness=0.5):
    """Create a minimal valid Individual for testing."""
    ind_gene = IndicatorGene(type='RSI', parameters={'period': 14})
    cond = ConditionGene(indicator='RSI', operator='>', threshold=70)
    gene = StrategyGene(
        generation=gen, individual_id=ind_id,
        indicators=[ind_gene],
        entry_conditions=[cond],
        exit_conditions=[cond],
    )
    ind = Individual(strategy_gene=gene)
    ind.fitness = fitness
    ind.raw_fitness = fitness
    ind.evaluated = True
    ind.metrics = {'profit': fitness * 100}
    return ind


def _make_population(size=5, gen=0):
    """Create a population with `size` individuals."""
    pop = Population(size=size, generation=gen)
    for i in range(size):
        pop.add_individual(_make_individual(gen=gen, ind_id=i, fitness=0.1 * (i + 1)))
    return pop


def _make_stats(gen=0):
    """Create a minimal PopulationStats."""
    return PopulationStats(
        generation=gen, size=5,
        best_fitness=0.8, avg_fitness=0.5, worst_fitness=0.1,
    )


def _provenance(**overrides):
    values = {
        "engine_kind": "STANDARD",
        "config_hash": "1" * 64,
        "code_manifest_hash": "2" * 64,
        "data_manifest_hash": "3" * 64,
    }
    values.update(overrides)
    return CheckpointProvenanceV3(**values)


# =========================================================================
# CheckpointManager tests
# =========================================================================

class TestCheckpointSave:
    """Test save and atomic write mechanics."""

    def test_save_creates_file(self, tmp_path):
        mgr = CheckpointManager(tmp_path / 'ckpt', checkpoint_interval=5)
        pop = _make_population()
        ga_state = {
            'best_individual': _make_individual(fitness=0.9),
            'best_fitness_ever': 0.9,
            'no_improvement_count': 2,
            'current_mutation_rate': 0.15,
            'base_mutation_rate': 0.1,
            'catastrophic_restart_needed': False,
            'total_generations': 10,
        }
        stats = [_make_stats(i) for i in range(3)]
        config_snap = {'genetic_algorithm': {'population_size': 5}}

        path = mgr.save(pop, 2, ga_state, config_snap, stats, provenance=_provenance())
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data['generation'] == 2
        assert data['version'] == 3
        assert data['resume_eligible'] is True
        assert 'checksum' in data

    def test_save_no_tmp_left(self, tmp_path):
        """Atomic write should not leave .tmp files."""
        mgr = CheckpointManager(tmp_path / 'ckpt')
        pop = _make_population(size=2)
        ga_state = {'best_individual': None, 'best_fitness_ever': 0, 'total_generations': 5}
        mgr.save(pop, 0, ga_state, {}, [])
        assert list((tmp_path / 'ckpt').glob('*.tmp')) == []

    def test_save_explicit_filepath(self, tmp_path):
        mgr = CheckpointManager(tmp_path / 'ckpt')
        pop = _make_population(size=2)
        ga_state = {'best_individual': None, 'total_generations': 5}
        explicit = str(tmp_path / 'custom.json')
        path = mgr.save(pop, 1, ga_state, {}, [], filepath=explicit)
        assert path == explicit
        assert Path(explicit).exists()


class TestCheckpointLoad:
    """Test load and state restoration."""

    def _save_and_load(self, tmp_path, ga_state=None, pop_size=5, gen=3):
        mgr = CheckpointManager(tmp_path / 'ckpt')
        pop = _make_population(size=pop_size, gen=gen)
        if ga_state is None:
            ga_state = {
                'best_individual': _make_individual(fitness=0.9),
                'best_fitness_ever': 0.9,
                'no_improvement_count': 3,
                'current_mutation_rate': 0.2,
                'base_mutation_rate': 0.1,
                'catastrophic_restart_needed': True,
                'total_generations': 20,
            }
        stats = [_make_stats(i) for i in range(gen + 1)]
        conf = {'genetic_algorithm': {'population_size': pop_size}}
        provenance = _provenance()
        path = mgr.save(pop, gen, ga_state, conf, stats, provenance=provenance)
        return mgr.load(path, pop_size, expected_provenance=provenance)

    def test_load_returns_tuple(self, tmp_path):
        pop, start_gen, state = self._save_and_load(tmp_path)
        assert isinstance(pop, Population)
        assert start_gen == 4  # gen 3 → resume at 4
        assert 'ga_state' in state

    def test_load_population_size(self, tmp_path):
        pop, _, _ = self._save_and_load(tmp_path, pop_size=5, gen=2)
        assert len(pop.individuals) == 5

    def test_load_ga_state(self, tmp_path):
        _, _, state = self._save_and_load(tmp_path)
        ga = state['ga_state']
        assert ga['best_fitness_ever'] == 0.9
        assert ga['no_improvement_count'] == 3
        assert ga['catastrophic_restart_needed'] is True
        # best_individual should be restored as Individual
        assert hasattr(ga['best_individual'], 'fitness')

    def test_load_generation_stats(self, tmp_path):
        _, _, state = self._save_and_load(tmp_path, gen=3)
        stats = state['generation_stats']
        assert len(stats) == 4
        assert stats[0].generation == 0
        assert stats[0].best_fitness == 0.8

    def test_checksum_mismatch_blocks_resume(self, tmp_path):
        mgr = CheckpointManager(tmp_path / 'ckpt')
        pop = _make_population(size=2)
        ga_state = {'best_individual': None, 'total_generations': 5}
        provenance = _provenance()
        path = mgr.save(pop, 0, ga_state, {}, [], provenance=provenance)
        # Corrupt the file
        data = json.loads(Path(path).read_text())
        data['generation'] = 999
        Path(path).write_text(json.dumps(data, indent=2, default=str))
        with pytest.raises(CheckpointIntegrityError, match='checksum mismatch'):
            mgr.load(path, 2, expected_provenance=provenance)

    def test_provenance_mismatch_blocks_resume(self, tmp_path):
        mgr = CheckpointManager(tmp_path / 'ckpt')
        pop = _make_population(size=2)
        expected = _provenance()
        path = mgr.save(
            pop,
            0,
            {'best_individual': None, 'total_generations': 5},
            {},
            [],
            provenance=expected,
        )
        changed = _provenance(data_manifest_hash="4" * 64)
        with pytest.raises(CheckpointCompatibilityError, match='data_manifest_hash'):
            mgr.load(path, 2, expected_provenance=changed)

    def test_diagnostic_checkpoint_cannot_resume(self, tmp_path):
        mgr = CheckpointManager(tmp_path / 'ckpt')
        path = mgr.save(
            _make_population(size=2),
            0,
            {'best_individual': None, 'total_generations': 5},
            {},
            [],
        )
        with pytest.raises(CheckpointCompatibilityError, match='diagnostic-only'):
            mgr.load(path, 2, expected_provenance=_provenance())


class TestCheckpointLegacy:
    """Test legacy checkpoint format."""

    def test_save_legacy(self, tmp_path):
        mgr = CheckpointManager(tmp_path / 'ckpt')
        pop = _make_population(size=3)
        ga_state = {
            'best_individual': _make_individual(fitness=0.7),
            'best_fitness_ever': 0.7,
            'current_mutation_rate': 0.1,
            'random_seed': 42,
        }
        stats = [_make_stats(0)]
        mgr.save_legacy(pop, 1, ga_state, stats)
        legacy_path = tmp_path / 'ckpt' / 'latest_checkpoint.json'
        assert legacy_path.exists()
        data = json.loads(legacy_path.read_text())
        assert data['generation'] == 1
        assert len(data['population']) == 3

    def test_load_legacy_missing(self, tmp_path):
        mgr = CheckpointManager(tmp_path / 'ckpt')
        assert mgr.load_legacy() is None

    def test_load_legacy_roundtrip(self, tmp_path):
        mgr = CheckpointManager(tmp_path / 'ckpt')
        pop = _make_population(size=3, gen=2)
        ga_state = {
            'best_individual': _make_individual(fitness=0.6),
            'best_fitness_ever': 0.6,
            'current_mutation_rate': 0.12,
            'no_improvement_count': 1,
            'random_seed': None,
        }
        stats = [_make_stats(0), _make_stats(1)]
        mgr.save_legacy(pop, 2, ga_state, stats)
        loaded = mgr.load_legacy()
        assert loaded is not None
        restored_pop, restored_state = mgr.restore_from_legacy(loaded, 3)
        assert len(restored_pop.individuals) == 3
        assert restored_state['ga_state']['best_fitness_ever'] == 0.6


class TestCheckpointShouldSave:
    def test_interval_check(self):
        mgr = CheckpointManager('/tmp/unused', checkpoint_interval=5)
        assert not mgr.should_save(0)
        assert not mgr.should_save(3)
        assert mgr.should_save(4)  # (4+1) % 5 == 0
        assert not mgr.should_save(5)
        assert mgr.should_save(9)

    def test_interval_zero_never_saves(self):
        mgr = CheckpointManager('/tmp/unused', checkpoint_interval=0)
        assert not mgr.should_save(0)
        assert not mgr.should_save(100)


class TestCheckpointRandomState:
    """Test random state capture and restore."""

    def test_python_random_state_roundtrip(self, tmp_path):
        mgr = CheckpointManager(tmp_path / 'ckpt')
        pop = _make_population(size=2)
        ga_state = {'best_individual': None, 'total_generations': 5}
        # Set known seed, save
        random.seed(42)
        provenance = _provenance()
        path = mgr.save(pop, 0, ga_state, {}, [], provenance=provenance)
        # Generate some numbers
        vals_after_save = [random.random() for _ in range(5)]
        # Load (restores seed state as of save time)
        mgr.load(path, 2, expected_provenance=provenance)
        vals_after_load = [random.random() for _ in range(5)]
        # Should match: same state → same sequence
        assert vals_after_save == vals_after_load


# =========================================================================
# AdaptiveController tests
# =========================================================================

def _make_config(mutation_rate=0.1, adaptive=True, patience=10):
    return {
        'genetic_algorithm': {
            'mutation_rate': mutation_rate,
            'adaptive_mutation': adaptive,
            'max_adaptation_factor': 2.0,
            'adaptation_step': 0.1,
            'max_mutation_rate': 0.65,
            'mutation_cooldown_factor': 0.5,
            'convergence_patience': patience,
        }
    }


class TestAdaptiveConvergence:
    """Test convergence detection."""

    def test_no_convergence_when_improving(self):
        ctrl = AdaptiveController(_make_config(patience=5))
        ctrl.record_new_best()
        assert ctrl.check_convergence() is False
        assert ctrl.no_improvement_count == 0

    def test_convergence_at_patience(self):
        ctrl = AdaptiveController(_make_config(patience=3))
        for _ in range(3):
            result = ctrl.check_convergence()
        assert result is True
        assert ctrl.no_improvement_count == 3

    def test_improvement_resets_counter(self):
        ctrl = AdaptiveController(_make_config(patience=10))
        ctrl.check_convergence()  # +1
        ctrl.check_convergence()  # +2
        assert ctrl.no_improvement_count == 2
        ctrl.record_new_best()
        ctrl.check_convergence()
        assert ctrl.no_improvement_count == 0


class TestAdaptiveMutationRate:
    """Test adaptive mutation rate logic."""

    def test_rate_increases_when_stuck(self):
        ctrl = AdaptiveController(_make_config(mutation_rate=0.1, patience=10))
        base = ctrl.mutation_rate
        ctrl.check_convergence()  # no_improvement = 1
        assert ctrl.mutation_rate > base

    def test_rate_capped_at_max(self):
        ctrl = AdaptiveController(_make_config(mutation_rate=0.1, patience=20))
        for _ in range(20):
            ctrl.check_convergence()
        assert ctrl.mutation_rate <= 0.65

    def test_rate_cools_down_on_improvement(self):
        ctrl = AdaptiveController(_make_config(mutation_rate=0.1, patience=10))
        # Escalate
        for _ in range(5):
            ctrl.check_convergence()
        escalated = ctrl.mutation_rate
        assert escalated > 0.1
        # Improve
        ctrl.record_new_best()
        ctrl.check_convergence()
        assert ctrl.mutation_rate < escalated
        assert ctrl.mutation_rate >= 0.1

    def test_non_adaptive_stays_at_base(self):
        ctrl = AdaptiveController(_make_config(mutation_rate=0.2, adaptive=False))
        ctrl.check_convergence()
        ctrl.check_convergence()
        assert ctrl.mutation_rate == 0.2


class TestCatastrophicRestart:
    """Test catastrophic restart flagging."""

    def test_flag_at_half_patience(self):
        ctrl = AdaptiveController(_make_config(patience=10))
        for _ in range(5):
            ctrl.check_convergence()
        assert ctrl.catastrophic_restart_needed is True

    def test_clear_resets_state(self):
        ctrl = AdaptiveController(_make_config(patience=10))
        for _ in range(5):
            ctrl.check_convergence()
        assert ctrl.catastrophic_restart_needed is True
        ctrl.clear_catastrophic_restart()
        assert ctrl.catastrophic_restart_needed is False
        assert ctrl.no_improvement_count == 0
        assert ctrl.mutation_rate == ctrl.base_mutation_rate

    def test_not_flagged_with_patience_1(self):
        """Patience=1 → half_patience=0 → should never flag."""
        ctrl = AdaptiveController(_make_config(patience=1))
        ctrl.check_convergence()
        assert ctrl.catastrophic_restart_needed is False


class TestAdaptiveStatePersistence:
    """Test get_state / load_state roundtrip."""

    def test_state_roundtrip(self):
        ctrl = AdaptiveController(_make_config(patience=10))
        for _ in range(3):
            ctrl.check_convergence()
        state = ctrl.get_state()
        assert state['no_improvement_count'] == 3
        assert state['mutation_rate'] > 0.1

        ctrl2 = AdaptiveController(_make_config(patience=10))
        ctrl2.load_state(state)
        assert ctrl2.no_improvement_count == 3
        assert ctrl2.mutation_rate == state['mutation_rate']
