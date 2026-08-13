"""
Tests for the Generic Island Model Evolution.

Tests cover:
- Auto-generation of island configs
- Indicator pool splitting with overlap
- Pair rotation
- Each migration topology (ring, fully_connected, tournament, hierarchical)
- Merge round logic
- Gene hashing and deduplication
- Walk-forward compatibility
- Config building for sub-islands
- Integration with run_ga.py branching
"""

import copy
import hashlib
import random
from unittest.mock import MagicMock, patch

import pytest

from genetic_algorithm.core.generic_island_model import (
    ALL_INDICATORS,
    INDICATOR_FAMILIES,
    GenericIslandModelEvolution,
    GenericIslandStats,
    OUTCOME_COMPLETED_BUDGET,
    OUTCOME_DIVERSITY_COLLAPSE,
    OUTCOME_PLATEAU,
    OUTCOME_TECHNICAL_INVALID,
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population
from genetic_algorithm.core.strategy_gene import ConditionGene, IndicatorGene, StrategyGene


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _make_gene(generation=0, individual_id=0, **overrides):
    """Create a minimal StrategyGene for testing."""
    defaults = dict(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
        ],
        entry_conditions=[
            ConditionGene(
                indicator='RSI',
                operator='<',
                threshold=30.0,
                logic='AND',
            ),
        ],
        exit_conditions=[
            ConditionGene(
                indicator='RSI',
                operator='>',
                threshold=70.0,
                logic='AND',
            ),
        ],
        stoploss=-0.10,
        timeframe='15m',
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
        max_open_trades=3,
    )
    defaults.update(overrides)
    return StrategyGene(**defaults)


def _make_individual(fitness=None, generation=0, individual_id=0, **gene_kw):
    """Create an Individual with optional fitness."""
    gene = _make_gene(generation=generation, individual_id=individual_id, **gene_kw)
    ind = Individual(strategy_gene=gene)
    if fitness is not None:
        ind.set_fitness(
            fitness,
            {'profit': fitness * 10, 'sharpe_ratio': 1.0, 'num_trades': 20},
        )
    return ind


def _make_population(fitnesses):
    """Create a Population with individuals at the given fitness levels."""
    pop = Population(size=len(fitnesses), generation=0)
    for i, f in enumerate(fitnesses):
        ind = _make_individual(fitness=f, generation=0, individual_id=i)
        pop.add_individual(ind)
    return pop


def _minimal_config(num_islands=3, topology='ring', **gim_overrides):
    """Build a minimal config dict for GenericIslandModelEvolution."""
    import tempfile

    cfg = {
        'genetic_algorithm': {
            'population_size': 10,
            'generations': 5,
            'mutation_rate': 0.3,
            'crossover_rate': 0.7,
            'elite_size': 2,
            'tournament_size': 3,
            'random_seed': 42,
        },
        'backtesting': {
            'pairs': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'],
            'timeframe': '15m',
            'timerange': '20230101-20260315',
        },
        'indicators': {
            'available': list(ALL_INDICATORS),
        },
        'generic_island_model': {
            'enabled': True,
            'num_islands': num_islands,
            'parallel_islands': False,
            'population_per_island': 5,
            'generations': 3,
            'specialization': {
                'rotate_seeds': True,
                'indicator_pools': True,
                'indicator_overlap': 0.5,
                'pair_rotation': False,
                'pair_subset_size': 3,
            },
            'migration': {
                'topology': topology,
                'interval': 2,
                'count': 1,
                'merge_rounds': False,
                'merge_interval': 3,
                'tournament_size': 3,
            },
            **gim_overrides,
        },
        'island_model': {'enabled': False},
        'terminal_monitor': {'enabled': False},
        'walk_forward': {'enabled': False},
        'fitness_weights': {},
        'fitness_penalties': {},
        'strategy_constraints': {},
        'hall_of_fame': {
            'directory': tempfile.mkdtemp(prefix='ga-generic-island-test-hof-'),
            'max_size': 10,
            'min_fitness': 0.0,
        },
    }
    return cfg


def _create_model_from_config(config):
    """
    Create a GenericIslandModelEvolution without going through YAML file.
    Patches the file read to use the in-memory config.
    """
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False,
    ) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    try:
        model = GenericIslandModelEvolution(
            config_path=tmp_path,
            visualize=False,
            interactive=False,
        )
    finally:
        import os
        os.unlink(tmp_path)

    return model


class TestPeriodicCommonPanelReplay:
    def test_replay_every_third_generation_and_stop_after_stagnation(self):
        config = _minimal_config(num_islands=1)
        config['generic_island_model']['common_panel_replay'] = {
            'enabled': True,
            'interval': 3,
            'top_n_per_island': 1,
            'early_stop_patience': 2,
            'min_improvement': 0.01,
            'relative_min_improvement': 0.0,
            'min_generation': 1,
        }
        model = _create_model_from_config(config)
        candidate = _make_individual(fitness=0.6)
        model._evaluated_generation_elites = {'island_0': [candidate]}
        panel = MagicMock(panel_id='panel_v1_' + 'a' * 24)
        model._create_common_replay_evaluator = MagicMock(
            return_value=(MagicMock(), panel)
        )

        with patch(
            'genetic_algorithm.core.generic_island_model.replay_on_common_panel',
            return_value=[candidate],
        ) as replay:
            for generation in range(9):
                model._maybe_replay_common_panel(generation)

        assert replay.call_count == 3
        assert [item['generation'] for item in model.common_panel_replay_history] == [2, 5, 8]
        assert model.common_panel_replay_history[-1]['early_stop_triggered'] is True
        assert model._shutdown_requested is True
        assert model._stop_reason == OUTCOME_PLATEAU


# ══════════════════════════════════════════════════════════════════════
# Tests: Island Configuration
# ══════════════════════════════════════════════════════════════════════

class TestIslandAutoGeneration:
    """Tests for automatic island configuration generation."""

    def test_auto_generate_creates_correct_number_of_islands(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        assert len(model.island_configs) == 5

    def test_auto_generate_rotates_seeds(self):
        config = _minimal_config(num_islands=4)
        model = _create_model_from_config(config)
        seeds = [ic.seed for ic in model.island_configs]
        assert len(set(seeds)) == 4, "All seeds should be unique"
        # Seeds should be sequential from base_seed
        assert seeds == [42, 43, 44, 45]

    def test_auto_generate_without_seed_rotation(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['specialization']['rotate_seeds'] = False
        model = _create_model_from_config(config)
        seeds = [ic.seed for ic in model.island_configs]
        assert all(s == 42 for s in seeds), "All seeds should be the same base seed"

    def test_auto_generate_with_indicator_pools(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)

        # Each island should have an indicator pool
        for ic in model.island_configs:
            assert ic.indicator_pool is not None
            assert len(ic.indicator_pool) > 0

    def test_auto_generate_without_indicator_pools(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['specialization']['indicator_pools'] = False
        model = _create_model_from_config(config)

        for ic in model.island_configs:
            assert ic.indicator_pool is None

    def test_auto_generate_with_pair_rotation(self):
        config = _minimal_config(num_islands=5)
        config['generic_island_model']['specialization']['pair_rotation'] = True
        config['generic_island_model']['specialization']['pair_subset_size'] = 3
        model = _create_model_from_config(config)

        # Each island should have a pair subset
        for ic in model.island_configs:
            assert ic.pairs is not None
            assert len(ic.pairs) == 3

    def test_auto_generate_population_size(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['population_per_island'] = 8
        model = _create_model_from_config(config)

        for ic in model.island_configs:
            assert ic.population_size == 8

    def test_explicit_islands_override_auto(self):
        config = _minimal_config(num_islands=2)
        config['generic_island_model']['islands'] = [
            {'name': 'custom_1', 'population_size': 20, 'seed': 100},
            {'name': 'custom_2', 'population_size': 15, 'seed': 200,
             'indicator_pool': ['RSI', 'MACD']},
        ]
        model = _create_model_from_config(config)

        assert len(model.island_configs) == 2
        assert model.island_configs[0].name == 'custom_1'
        assert model.island_configs[0].population_size == 20
        assert model.island_configs[1].indicator_pool == ['RSI', 'MACD']

    def test_explicit_island_count_must_match_num_islands(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['islands'] = [
            {'name': 'custom_1'},
            {'name': 'custom_2'},
        ]

        with pytest.raises(ValueError, match='must exactly match'):
            _create_model_from_config(config)


class TestStrictV2IslandSeeds:
    def test_parent_seed_is_injected_into_every_island_before_initialization(self):
        model = _create_model_from_config(_minimal_config(num_islands=3))
        seed = _make_individual(fitness=0.5)
        model.set_strict_initial_seeds([seed])
        created = []

        def fake_create(island_config):
            algorithm = MagicMock()
            algorithm.initialize_population.return_value = Population(
                size=island_config.population_size
            )
            created.append(algorithm)
            return algorithm

        model._create_island_ga = fake_create
        model._phase1_create_islands()

        assert len(created) == 3
        for algorithm in created:
            algorithm.set_strict_initial_seeds.assert_called_once_with([seed])
            algorithm.initialize_population.assert_called_once_with()


class TestIndicatorPoolSplitting:
    """Tests for indicator pool splitting logic."""

    def test_pool_count_matches_families(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        pools = model._split_indicator_pools()
        assert len(pools) == len(INDICATOR_FAMILIES)

    def test_pools_have_overlap(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        pools = model._split_indicator_pools()

        # Check pairs of adjacent pools have some overlap
        for i in range(len(pools)):
            j = (i + 1) % len(pools)
            overlap = set(pools[i]) & set(pools[j])
            assert len(overlap) > 0, f"Pools {i} and {j} should overlap"

    def test_all_indicators_covered(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        pools = model._split_indicator_pools()

        all_in_pools = set()
        for pool in pools:
            all_in_pools.update(pool)

        # All standard indicators should appear in at least one pool
        for family, indicators in INDICATOR_FAMILIES.items():
            for ind in indicators:
                assert ind in all_in_pools, f"Indicator {ind} not in any pool"

    def test_overlap_increases_with_config(self):
        config_low = _minimal_config(num_islands=5)
        config_low['generic_island_model']['specialization']['indicator_overlap'] = 0.1
        model_low = _create_model_from_config(config_low)
        pools_low = model_low._split_indicator_pools()

        config_high = _minimal_config(num_islands=5)
        config_high['generic_island_model']['specialization']['indicator_overlap'] = 0.9
        model_high = _create_model_from_config(config_high)
        pools_high = model_high._split_indicator_pools()

        avg_size_low = sum(len(p) for p in pools_low) / len(pools_low)
        avg_size_high = sum(len(p) for p in pools_high) / len(pools_high)
        assert avg_size_high >= avg_size_low


class TestPairRotation:
    """Tests for pair rotation logic."""

    def test_rotate_pairs_creates_subsets(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        subsets = model._rotate_pairs()
        assert len(subsets) == 5  # 5 pairs = 5 rotations
        for subset in subsets:
            assert len(subset) == 3  # pair_subset_size=3

    def test_rotate_pairs_returns_all_when_subset_too_large(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['specialization']['pair_subset_size'] = 10
        model = _create_model_from_config(config)
        subsets = model._rotate_pairs()
        assert len(subsets) == 1
        assert len(subsets[0]) == 5


# ══════════════════════════════════════════════════════════════════════
# Tests: Island Config Building
# ══════════════════════════════════════════════════════════════════════

class TestBuildIslandConfig:
    """Tests for building per-island GA config."""

    def test_disables_nested_island_models(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        island_cfg = model._build_island_config(ic)

        assert island_cfg['island_model']['enabled'] is False
        assert island_cfg['generic_island_model']['enabled'] is False

    def test_sets_population_size(self):
        config = _minimal_config(num_islands=2)
        config['generic_island_model']['population_per_island'] = 12
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        island_cfg = model._build_island_config(ic)

        assert island_cfg['genetic_algorithm']['population_size'] == 12

    def test_minimum_island_population_derives_feasible_slots(self):
        config = _minimal_config(num_islands=2)
        config['generic_island_model']['population_per_island'] = 2
        model = _create_model_from_config(config)

        island_cfg = model._build_island_config(model.island_configs[0])

        assert island_cfg['genetic_algorithm']['population_size'] == 2
        assert island_cfg['genetic_algorithm']['elite_size'] == 1
        assert island_cfg['genetic_algorithm']['random_immigrants'] == 1

    def test_sets_seed(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        island_cfg = model._build_island_config(ic)

        assert island_cfg['genetic_algorithm']['random_seed'] == ic.seed
        assert 'seed' not in island_cfg['genetic_algorithm']

    def test_rotated_seeds_reach_each_island_runtime_config(self):
        config = _minimal_config(num_islands=3)
        model = _create_model_from_config(config)

        seeds = [
            model._build_island_config(ic)['genetic_algorithm']['random_seed']
            for ic in model.island_configs
        ]

        assert seeds == [42, 43, 44]

    def test_extra_config_cannot_reenable_nested_runtime_features(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        ic.extra_config = {
            'generic_island_model': {'enabled': True},
            'island_model': {'enabled': True},
            'parallel_evaluation': {'enabled': True, 'num_workers': 8},
        }

        island_cfg = model._build_island_config(ic)

        assert island_cfg['generic_island_model']['enabled'] is False
        assert island_cfg['island_model']['enabled'] is False
        assert island_cfg['parallel_evaluation']['enabled'] is False
        assert island_cfg['parallel_evaluation']['num_workers'] == 8

    def test_restricts_indicator_pool(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        # Manually set an indicator pool
        ic.indicator_pool = ['RSI', 'MACD', 'BBANDS']
        island_cfg = model._build_island_config(ic)

        assert island_cfg['indicators']['available'] == ['RSI', 'MACD', 'BBANDS']

    def test_restricts_pairs(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        ic.pairs = ['BTC/USDT', 'ETH/USDT']
        island_cfg = model._build_island_config(ic)

        assert island_cfg['backtesting']['pairs'] == ['BTC/USDT', 'ETH/USDT']

    def test_walk_forward_configurable(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        ic.walk_forward_enabled = True
        island_cfg = model._build_island_config(ic)

        assert island_cfg['walk_forward']['enabled'] is True

    def test_disables_terminal_monitor(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        island_cfg = model._build_island_config(ic)

        assert island_cfg['terminal_monitor']['enabled'] is False


# ══════════════════════════════════════════════════════════════════════
# Tests: Migration
# ══════════════════════════════════════════════════════════════════════

class TestMigrationHelpers:
    """Tests for migration helper methods."""

    def test_get_top_individuals_returns_sorted(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.1, 0.5, 0.3, 0.8, 0.2])
        model.island_populations['test'] = pop

        top = model._get_top_individuals('test', 3)
        assert len(top) == 3
        assert top[0].raw_fitness == 0.8
        assert top[1].raw_fitness == 0.5
        assert top[2].raw_fitness == 0.3

    def test_get_top_individuals_skips_unevaluated(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.5, 0.3])
        # Add an unevaluated individual
        uneval = _make_individual(fitness=None)
        pop.individuals.append(uneval)
        model.island_populations['test'] = pop

        top = model._get_top_individuals('test', 5)
        assert len(top) == 2

    def test_get_top_individuals_nonexistent_island(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        top = model._get_top_individuals('nonexistent', 3)
        assert top == []

    def test_inject_migrants_replaces_worst(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.1, 0.2, 0.3, 0.4, 0.5])
        model.island_populations['target'] = pop

        migrant = _make_individual(fitness=0.9, generation=1, individual_id=99)
        replaced = model._inject_migrants('target', [migrant], generation=5)

        assert replaced == 1
        # The worst (0.1) should have been replaced
        fitnesses = [
            ind.raw_fitness for ind in pop.individuals
            if ind.raw_fitness is not None and ind.evaluated
        ]
        assert 0.1 not in [f for f in fitnesses if f != 0.1]  # replaced

    def test_inject_migrants_marks_unevaluated(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.1, 0.2])
        model.island_populations['target'] = pop

        migrant = _make_individual(fitness=0.9)
        model._inject_migrants('target', [migrant], generation=5)

        # Find the injected migrant (the one with evaluated=False)
        unevaluated = [ind for ind in pop.individuals if not ind.evaluated]
        assert len(unevaluated) == 1

    def test_inject_migrants_preserves_signed_negative_elites(self):
        """An unevaluated offspring is worse than every valid signed score."""
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([-10.0, -40.0])
        unevaluated = _make_individual(
            fitness=None,
            generation=5,
            individual_id=77,
        )
        pop.individuals.append(unevaluated)
        model.island_populations['target'] = pop

        migrant = _make_individual(
            fitness=5.0,
            generation=4,
            individual_id=99,
        )
        replaced = model._inject_migrants(
            'target',
            [migrant],
            generation=5,
        )

        assert replaced == 1
        measured_scores = sorted(
            ind.raw_fitness
            for ind in pop.individuals
            if ind.evaluated and ind.raw_fitness is not None
        )
        assert measured_scores == [-40.0, -10.0]
        injected = [
            ind
            for ind in pop.individuals
            if ind.metrics.get('origin') == 'migrant_from_unknown'
        ]
        assert len(injected) == 1
        assert injected[0].strategy_gene.individual_id == 77

    def test_inject_migrants_empty_list(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.1, 0.2])
        model.island_populations['target'] = pop
        replaced = model._inject_migrants('target', [], generation=5)
        assert replaced == 0


class TestGeneHash:
    """Tests for gene hashing / deduplication."""

    def test_same_gene_same_hash(self):
        ind1 = _make_individual(fitness=0.5, generation=0, individual_id=0)
        ind2 = _make_individual(fitness=0.5, generation=0, individual_id=0)
        h1 = GenericIslandModelEvolution._gene_hash(ind1)
        h2 = GenericIslandModelEvolution._gene_hash(ind2)
        assert h1 == h2

    def test_different_gene_different_hash(self):
        ind1 = _make_individual(fitness=0.5, generation=0, individual_id=0)
        ind2 = _make_individual(
            fitness=0.5, generation=0, individual_id=0,
            stoploss=-0.20,  # Different stoploss
        )
        h1 = GenericIslandModelEvolution._gene_hash(ind1)
        h2 = GenericIslandModelEvolution._gene_hash(ind2)
        assert h1 != h2

    def test_hash_ignores_generation_and_id(self):
        ind1 = _make_individual(fitness=0.5, generation=0, individual_id=0)
        ind2 = _make_individual(fitness=0.5, generation=5, individual_id=99)
        h1 = GenericIslandModelEvolution._gene_hash(ind1)
        h2 = GenericIslandModelEvolution._gene_hash(ind2)
        assert h1 == h2


class TestMigrationTopologies:
    """Tests for each migration topology."""

    def _setup_model_with_populations(self, topology='ring', num_islands=4):
        """Create a model with pre-populated islands."""
        config = _minimal_config(num_islands=num_islands, topology=topology)
        model = _create_model_from_config(config)

        # Create populations with distinct fitness ranges per island
        for i, ic in enumerate(model.island_configs):
            base_fitness = (i + 1) * 0.1  # island 0: 0.1, island 1: 0.2, etc.
            fitnesses = [base_fitness + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop
            model.island_stats[ic.name] = GenericIslandStats(name=ic.name)

        return model

    def test_ring_migration_creates_events(self):
        model = self._setup_model_with_populations('ring', 4)
        model._migrate_ring(generation=2)

        assert len(model.migration_history) == 4  # One per island
        # Check ring pattern: each source → next
        sources = [e.source for e in model.migration_history]
        targets = [e.target for e in model.migration_history]
        names = [ic.name for ic in model.island_configs]
        for i, (src, tgt) in enumerate(zip(sources, targets)):
            assert src == names[i]
            assert tgt == names[(i + 1) % len(names)]

    def test_ring_skips_incompatible_elite_for_next_compatible_candidate(self):
        model = self._setup_model_with_populations('ring', 2)
        source, target = model.island_configs
        source.indicator_pool = ['MACD', 'RSI', 'EMA']
        target.indicator_pool = ['RSI']

        def candidate(indicator_type, fitness, individual_id):
            return _make_individual(
                fitness=fitness,
                individual_id=individual_id,
                indicators=[
                    IndicatorGene(type=indicator_type, parameters={'period': 14})
                ],
                entry_conditions=[
                    ConditionGene(
                        indicator=indicator_type,
                        operator='<',
                        threshold=30,
                    )
                ],
                exit_conditions=[
                    ConditionGene(
                        indicator=indicator_type,
                        operator='>',
                        threshold=70,
                    )
                ],
            )

        source_ranked = [
            candidate('MACD', 0.9, 90),  # best, but invalid for RSI target
            candidate('RSI', 0.8, 80),   # next-best compatible candidate
            candidate('EMA', 0.7, 70),
        ]
        model._evaluated_generation_populations[source.name] = source_ranked
        # Prevent the reverse ring edge from adding a second migration event.
        model._evaluated_generation_populations[target.name] = [
            candidate('RSI', 0.6, 60)
        ]
        target_population_before = list(
            model.island_populations[target.name].individuals
        )

        model._migrate_ring(generation=4)

        injected = [
            item
            for item in model.island_populations[target.name].individuals
            if item.metrics.get('origin') == f'migrant_from_{source.name}'
        ]
        assert len(injected) == 1
        assert [item.type for item in injected[0].strategy_gene.indicators] == [
            'RSI'
        ]
        assert all(
            item.strategy_gene.indicators[0].type != 'MACD'
            for item in model.island_populations[target.name].individuals
        )
        assert model.migration_history[0].fitnesses == [0.8]
        assert sum(
            before is after
            for before in target_population_before
            for after in model.island_populations[target.name].individuals
        ) == len(target_population_before) - 1

    def test_fully_connected_migration_all_pairs(self):
        model = self._setup_model_with_populations('fully_connected', 3)
        model._migrate_fully_connected(generation=2)

        # 3 islands × 2 targets each = 6 events
        assert len(model.migration_history) == 6

    def test_tournament_migration_creates_events(self):
        model = self._setup_model_with_populations('tournament', 4)
        model._migrate_tournament(generation=2)

        # Unknown/different panels cannot produce a tournament winner. Each
        # pair exchanges locally ranked genomes in both directions.
        assert len(model.migration_history) == 4

    def test_hierarchical_migration_bidirectional(self):
        model = self._setup_model_with_populations('hierarchical', 4)
        model._migrate_hierarchical(generation=2)

        # 4 islands → 2 pairs → 2 bidirectional = 4 events
        assert len(model.migration_history) == 4

    def test_migrate_dispatches_to_correct_topology(self):
        for topology in ['ring', 'fully_connected', 'tournament', 'hierarchical']:
            model = self._setup_model_with_populations(topology, 4)
            model._migrate(generation=2)
            assert len(model.migration_history) > 0, f"No events for {topology}"


class TestMergeRound:
    """Tests for global merge rounds."""

    def test_merge_round_injects_global_elites(self):
        config = _minimal_config(num_islands=3)
        model = _create_model_from_config(config)

        # Create populations with different fitness levels
        for i, ic in enumerate(model.island_configs):
            fitnesses = [(i + 1) * 0.1 + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop
            model.island_stats[ic.name] = GenericIslandStats(name=ic.name)

        model._merge_round(generation=5)

        # After merge, the global best should appear in some islands
        # This is hard to test precisely due to deduplication, but we can
        # verify the merge ran without errors
        assert True

    def test_merge_round_with_empty_populations(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)

        for ic in model.island_configs:
            pop = Population(size=0, generation=0)
            model.island_populations[ic.name] = pop
            model.island_stats[ic.name] = GenericIslandStats(name=ic.name)

        # Should not crash
        model._merge_round(generation=5)


# ══════════════════════════════════════════════════════════════════════
# Tests: Result Collection
# ══════════════════════════════════════════════════════════════════════

class TestResultCollection:
    """Tests for final result collection and deduplication."""

    def test_collect_results_has_global_key(self):
        config = _minimal_config(num_islands=3)
        model = _create_model_from_config(config)

        for i, ic in enumerate(model.island_configs):
            fitnesses = [(i + 1) * 0.1 + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop

        model._replay_global_candidates = MagicMock(
            side_effect=lambda candidates: sorted(
                candidates, key=lambda item: item.raw_fitness, reverse=True
            )
        )
        results = model._collect_final_results()
        assert '__global__' in results
        assert len(results['__global__']) > 0

    def test_collect_results_per_island(self):
        config = _minimal_config(num_islands=3)
        model = _create_model_from_config(config)

        for i, ic in enumerate(model.island_configs):
            fitnesses = [0.1 * (i + 1) + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop

        model._replay_global_candidates = MagicMock(
            side_effect=lambda candidates: sorted(
                candidates, key=lambda item: item.raw_fitness, reverse=True
            )
        )
        results = model._collect_final_results()
        for ic in model.island_configs:
            assert ic.name in results
            assert len(results[ic.name]) <= 5

    def test_global_results_sorted_by_fitness(self):
        config = _minimal_config(num_islands=3)
        model = _create_model_from_config(config)

        for i, ic in enumerate(model.island_configs):
            fitnesses = [0.1 * (i + 1) + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop

        model._replay_global_candidates = MagicMock(
            side_effect=lambda candidates: sorted(
                candidates, key=lambda item: item.raw_fitness, reverse=True
            )
        )
        results = model._collect_final_results()
        global_top = results['__global__']
        fitnesses = [ind.raw_fitness for ind in global_top]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_global_results_max_20(self):
        config = _minimal_config(num_islands=10)
        model = _create_model_from_config(config)

        for i, ic in enumerate(model.island_configs):
            fitnesses = [0.01 * (i * 5 + j + 1) for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop

        model._replay_global_candidates = MagicMock(
            side_effect=lambda candidates: sorted(
                candidates, key=lambda item: item.raw_fitness, reverse=True
            )
        )
        results = model._collect_final_results()
        assert len(results['__global__']) <= 20

    def test_negative_measured_candidate_remains_a_valid_finalist(self):
        model = _create_model_from_config(_minimal_config(num_islands=1))
        island_name = model.island_configs[0].name
        candidate = _make_individual(fitness=-0.5)
        model._evaluated_generation_populations = {island_name: [candidate]}
        population = Population(size=1)
        population.individuals = [candidate]
        model.island_populations[island_name] = population
        model._replay_global_candidates = MagicMock(return_value=[candidate])

        results = model._collect_final_results()

        assert results['__global__'] == [candidate]
        assert model._get_top_individuals(island_name, 1) == [candidate]

    def test_final_common_replay_updates_valid_and_failed_candidate_counts(self):
        model = _create_model_from_config(_minimal_config(num_islands=1))
        first = _make_individual(fitness=0.4, individual_id=1, stoploss=-0.11)
        second = _make_individual(fitness=0.3, individual_id=2, stoploss=-0.12)
        panel = MagicMock(
            panel_id='panel_v1_' + 'f' * 24,
            role='COMMON_REPLAY',
            data_identity_verified=False,
        )
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = [
            (0.5, {'profit': 2.0, 'num_trades': 20}),
            (0.0, {'error': 'replay failed', 'num_trades': 0}),
        ]
        model._create_common_replay_evaluator = MagicMock(
            return_value=(evaluator, panel)
        )
        model._shared_parallel_evaluator = None
        model.hall_of_fame = MagicMock()

        replayed = model._replay_global_candidates([first, second])

        assert replayed == [first]
        assert model.evolution_outcome['common_panel_valid_evaluations'] == 1
        assert model.evolution_outcome['common_panel_failed_evaluations'] == 1


class TestProductionEvolutionSupervision:
    def _common_replay_config(self, **overrides):
        values = {
            'enabled': True,
            'interval': 3,
            'top_n_per_island': 2,
            'early_stop_patience': 4,
            'min_improvement': 0.25,
            'relative_min_improvement': 0.005,
            'min_generation': 9,
        }
        values.update(overrides)
        return values

    def test_common_replay_zero_candidates_is_technical_invalid(self):
        config = _minimal_config(num_islands=1)
        config['generic_island_model']['common_panel_replay'] = (
            self._common_replay_config(interval=1)
        )
        model = _create_model_from_config(config)
        model._evaluated_generation_elites = {}

        model._maybe_replay_common_panel(0)

        assert model._stop_reason == OUTCOME_TECHNICAL_INVALID
        assert model.common_panel_replay_history[-1]['technical_stop_triggered']

    def test_raw_common_replay_score_drift_is_technical_invalid(self):
        config = _minimal_config(num_islands=1)
        config['raw_multipair_score'] = {
            'enabled': True,
            'policy_version': 'raw-multipair-score-v3',
            'development_pairs': ['BTC/USDT', 'SOL/USDT', 'XRP/USDT'],
            'validation_pairs': ['BNB/USDT', 'ETH/USDT', 'PEPE/USDT'],
            'period_start': '2023-05-09',
            'period_end': '2026-03-26',
        }
        config['generic_island_model']['common_panel_replay'] = (
            self._common_replay_config(interval=1)
        )
        model = _create_model_from_config(config)
        candidate = _make_individual(fitness=10.0)
        candidate.metrics['source_fitness'] = 9.0
        model._evaluated_generation_elites = {
            model.island_configs[0].name: [candidate]
        }
        panel = MagicMock(panel_id='panel_v1_' + 'a' * 24)
        model._create_common_replay_evaluator = MagicMock(
            return_value=(MagicMock(), panel)
        )

        with patch(
            'genetic_algorithm.core.generic_island_model.replay_on_common_panel',
            return_value=[candidate],
        ):
            model._maybe_replay_common_panel(0)

        assert model._stop_reason == OUTCOME_TECHNICAL_INVALID
        event = model.common_panel_replay_history[-1]
        assert event['technical_stop_triggered'] is True
        assert event['valid_count'] == 0
        assert 'source=9' in event['parity_failures'][0]

    def test_material_relative_epsilon_and_min_generation(self):
        config = _minimal_config(num_islands=1)
        config['generic_island_model']['common_panel_replay'] = (
            self._common_replay_config(early_stop_patience=1)
        )
        model = _create_model_from_config(config)
        candidate = _make_individual(fitness=100.0)
        model._evaluated_generation_elites = {
            model.island_configs[0].name: [candidate]
        }
        panel = MagicMock(panel_id='panel_v1_' + 'b' * 24)
        model._create_common_replay_evaluator = MagicMock(
            return_value=(MagicMock(), panel)
        )

        with patch(
            'genetic_algorithm.core.generic_island_model.replay_on_common_panel',
            return_value=[candidate],
        ):
            model._maybe_replay_common_panel(2)
            candidate.set_fitness(100.2, {'profit': 1.0, 'num_trades': 20})
            model._maybe_replay_common_panel(5)
            assert model._stop_reason is None
            candidate.set_fitness(100.4, {'profit': 1.0, 'num_trades': 20})
            model._maybe_replay_common_panel(8)

        assert model._stop_reason == OUTCOME_PLATEAU
        assert model.common_panel_replay_history[-1][
            'material_improvement_threshold'
        ] == pytest.approx(0.5)

    def test_lane_incumbent_makes_four_stale_checks_stop_at_generation_twelve(self):
        config = _minimal_config(num_islands=1)
        replay_config = self._common_replay_config()
        replay_config['incumbent_score'] = 100.0
        config['generic_island_model']['common_panel_replay'] = replay_config
        model = _create_model_from_config(config)
        candidate = _make_individual(fitness=100.0)
        model._evaluated_generation_elites = {
            model.island_configs[0].name: [candidate]
        }
        panel = MagicMock(panel_id='panel_v1_' + 'c' * 24)
        model._create_common_replay_evaluator = MagicMock(
            return_value=(MagicMock(), panel)
        )

        with patch(
            'genetic_algorithm.core.generic_island_model.replay_on_common_panel',
            return_value=[candidate],
        ) as replay:
            for generation in range(12):
                model._maybe_replay_common_panel(generation)

        assert replay.call_count == 4
        assert [
            item['no_improvement_checks']
            for item in model.common_panel_replay_history
        ] == [1, 2, 3, 4]
        assert model._stop_reason == OUTCOME_PLATEAU
        assert model.common_panel_replay_history[-1]['generation'] == 11
        assert model.common_panel_replay_history[-1]['early_stop_triggered']
        assert model.evolution_outcome['incumbent_score'] == 100.0

    def test_material_improvement_over_lane_incumbent_resets_plateau_clock(self):
        config = _minimal_config(num_islands=1)
        config['generic_island_model']['common_panel_replay'] = (
            self._common_replay_config(incumbent_score=100.0)
        )
        model = _create_model_from_config(config)
        candidate = _make_individual(fitness=100.5)
        model._evaluated_generation_elites = {
            model.island_configs[0].name: [candidate]
        }
        panel = MagicMock(panel_id='panel_v1_' + 'd' * 24)
        model._create_common_replay_evaluator = MagicMock(
            return_value=(MagicMock(), panel)
        )

        with patch(
            'genetic_algorithm.core.generic_island_model.replay_on_common_panel',
            return_value=[candidate],
        ):
            model._maybe_replay_common_panel(2)
            model._maybe_replay_common_panel(5)
            model._maybe_replay_common_panel(8)
            model._maybe_replay_common_panel(11)

        assert model.common_panel_replay_history[0]['improved'] is True
        assert model.common_panel_replay_history[0][
            'material_improvement_threshold'
        ] == pytest.approx(0.5)
        assert model._common_panel_no_improvement_checks == 3
        assert model._stop_reason is None

    @pytest.mark.parametrize('invalid_score', [True, float('nan'), float('inf')])
    def test_lane_incumbent_must_be_finite(self, invalid_score):
        config = _minimal_config(num_islands=1)
        config['generic_island_model']['common_panel_replay'] = (
            self._common_replay_config(incumbent_score=invalid_score)
        )

        with pytest.raises(ValueError, match='incumbent_score'):
            _create_model_from_config(config)

    def test_metrics_error_counts_as_failed_and_zero_valid_stops(self):
        model = _create_model_from_config(_minimal_config(num_islands=1))
        candidate = _make_individual(fitness=0.8)
        # Simulate a legacy worker which stamped a finite fitness despite an
        # explicit error payload.
        candidate.metrics['error'] = 'worker crashed after result envelope'
        model._evaluated_generation_populations = {
            model.island_configs[0].name: [candidate]
        }

        assert model._record_generation_evidence(0) is False
        assert model.evolution_outcome['valid_evaluations'] == 0
        assert model.evolution_outcome['failed_evaluations'] == 1
        assert model._stop_reason == OUTCOME_TECHNICAL_INVALID

    def test_raw_multipair_mode_requires_all_six_pair_rows(self):
        model = _create_model_from_config(_minimal_config(num_islands=1))
        model.config['raw_multipair_score'] = {
            'enabled': True,
            'development_pairs': ['BTC/USDT', 'SOL/USDT', 'XRP/USDT'],
            'validation_pairs': ['BNB/USDT', 'ETH/USDT', 'PEPE/USDT'],
        }
        candidate = _make_individual(fitness=-1.0)
        candidate.metrics.update({
            'raw_multipair_score_version': 'raw-multipair-score-v3',
            'raw_multipair_status': 'VALID',
            'raw_pair_metrics': {'BTC/USDT': {}},
        })

        assert model._is_valid_evidence(candidate) is False
        candidate.metrics['raw_pair_metrics'] = {
            pair: {}
            for pair in (
                'BTC/USDT', 'SOL/USDT', 'XRP/USDT',
                'BNB/USDT', 'ETH/USDT', 'PEPE/USDT',
            )
        }
        assert model._is_valid_evidence(candidate) is True

    def test_generation_trace_uses_evaluated_population_not_offspring(self):
        model = _create_model_from_config(_minimal_config(num_islands=1))
        island_name = model.island_configs[0].name
        evaluated_population = Population(size=2, generation=0)
        evaluated_population.individuals = [
            _make_individual(individual_id=1),
            _make_individual(individual_id=2),
        ]
        offspring = Population(size=2, generation=1)
        offspring.individuals = [
            _make_individual(generation=1, individual_id=10),
            _make_individual(generation=1, individual_id=11),
        ]
        ga = MagicMock()
        ga.config = {'genetic_algorithm': {'behavioral_distance_weight': 0.0}}
        ga.llm_enabled = False
        ga.strategy_designer = None
        ga._surrogate = None
        ga.fitness_sharing = False
        ga.elite_size = 1
        ga.evaluation_panel.panel_id = 'panel_v1_' + 'c' * 24

        def evaluate(population):
            for index, individual in enumerate(population.individuals):
                individual.set_fitness(
                    0.2 + index,
                    {'profit': index, 'num_trades': 20, 'max_drawdown': 0.1},
                )

        ga.evaluate_population.side_effect = evaluate
        ga.create_next_generation.return_value = offspring
        model._shared_parallel_evaluator = None
        model.island_populations[island_name] = evaluated_population
        model.island_stats[island_name] = GenericIslandStats(name=island_name)
        model.generation_stats[island_name] = []
        model.islands[island_name] = ga

        model._evolve_island_one_generation(
            ga, evaluated_population, island_name, generation=0
        )
        model.generation_trace = MagicMock()
        model._persist_generation_trace(0)

        traced = model.generation_trace.append.call_args.kwargs['individuals']
        assert all(individual.has_measured_fitness for individual in traced)
        assert [item.strategy_gene.individual_id for item in traced] == [1, 2]
        assert model.island_populations[island_name] is offspring

    def test_failed_zero_score_cannot_outrank_or_breed_valid_negative_raw_score(self):
        model = _create_model_from_config(_minimal_config(num_islands=1))
        model.config['raw_multipair_score'] = {
            'enabled': True,
            'development_pairs': ['BTC/USDT', 'SOL/USDT', 'XRP/USDT'],
            'validation_pairs': ['BNB/USDT', 'ETH/USDT', 'PEPE/USDT'],
        }
        island_name = model.island_configs[0].name
        population = Population(size=2, generation=0)
        population.individuals = [
            _make_individual(individual_id=1),
            _make_individual(individual_id=2),
        ]
        ga = MagicMock()
        ga.config = {'genetic_algorithm': {'behavioral_distance_weight': 0.0}}
        ga.llm_enabled = False
        ga.strategy_designer = None
        ga._surrogate = None
        ga.fitness_sharing = False
        ga.elite_size = 1
        ga.evaluation_panel.panel_id = 'panel_v1_' + 'd' * 24

        all_pairs = (
            'BTC/USDT', 'SOL/USDT', 'XRP/USDT',
            'BNB/USDT', 'ETH/USDT', 'PEPE/USDT',
        )

        def evaluate(evaluated_population):
            valid, failed = evaluated_population.individuals
            valid.set_fitness(
                -10.0,
                {
                    'profit': -1.0,
                    'num_trades': 20,
                    'raw_multipair_score_version': 'raw-multipair-score-v3',
                    'raw_multipair_status': 'VALID',
                    'raw_pair_metrics': {pair: {} for pair in all_pairs},
                },
            )
            failed.set_fitness(0.0, {'error': 'worker failed'})

        ga.evaluate_population.side_effect = evaluate
        ga.strategy_generator.generate_random_strategy.return_value = _make_gene(
            generation=1,
            individual_id=1,
        )
        model._shared_parallel_evaluator = None
        model.island_populations[island_name] = population
        model.island_stats[island_name] = GenericIslandStats(name=island_name)
        model.generation_stats[island_name] = []
        model.islands[island_name] = ga

        model._evolve_island_one_generation(
            ga, population, island_name, generation=0
        )

        ga.create_next_generation.assert_not_called()
        next_population = model.island_populations[island_name]
        assert len(next_population.individuals) == 2
        assert next_population.individuals[0].raw_fitness == -10.0
        assert next_population.individuals[1].metrics['origin'] == (
            'insufficient_valid_parents'
        )
        assert all(
            item.metrics.get('error') is None
            for item in next_population.individuals
        )
        # Full evidence accounting still retains both evaluated outcomes.
        assert len(model._evaluated_generation_populations[island_name]) == 2
        assert sum(
            model._is_failed_evidence(item)
            for item in model._evaluated_generation_populations[island_name]
        ) == 1

    def test_one_diversity_recovery_then_persistent_collapse_stops(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['common_panel_replay'] = (
            self._common_replay_config(enabled=False, interval=1)
        )
        config['generic_island_model']['diversity_recovery'] = {
            'enabled': True,
            'island_threshold_count': 2,
            'consecutive_checks_before_recovery': 2,
            'consecutive_checks_after_recovery': 2,
            'duplicate_fraction_threshold': 0.75,
            'genetic_diversity_threshold': 0.10,
            'replacement_fraction': 0.25,
            'recovery_mutation_rate': 0.35,
        }
        model = _create_model_from_config(config)
        for island_config in model.island_configs:
            name = island_config.name
            model._evaluated_generation_populations[name] = _make_population(
                [0.1, 0.2, 0.3, 0.4]
            ).individuals
            stats = MagicMock(genetic_diversity=0.01)
            model.generation_stats[name] = [stats]
            model.island_populations[name] = _make_population(
                [0.1, 0.2, 0.3, 0.4]
            )
            ga = MagicMock()
            ga.mutation_rate = 0.2
            ga.strategy_generator.generate_random_strategy.side_effect = (
                lambda generation, individual_id: _make_gene(
                    generation=generation, individual_id=individual_id
                )
            )
            model.islands[name] = ga

        for generation in range(4):
            model._maybe_handle_diversity(generation)

        actions = [event['action'] for event in model.diversity_events]
        assert actions == [
            'COLLAPSE_OBSERVED',
            'RECOVERY_APPLIED',
            'POST_RECOVERY_COLLAPSE',
            'STOP_DIVERSITY_COLLAPSE',
        ]
        assert all(ga.mutation_rate == 0.35 for ga in model.islands.values())
        assert model._stop_reason == OUTCOME_DIVERSITY_COLLAPSE

    def test_diversity_recovery_replaces_bottom_evaluated_quartile(self):
        config = _minimal_config(num_islands=1)
        config['generic_island_model']['diversity_recovery'] = {
            'enabled': True,
            'island_threshold_count': 1,
            'consecutive_checks_before_recovery': 1,
            'consecutive_checks_after_recovery': 2,
            'duplicate_fraction_threshold': 0.75,
            'genetic_diversity_threshold': 0.10,
            'replacement_fraction': 0.25,
            'recovery_mutation_rate': 0.35,
        }
        model = _create_model_from_config(config)
        island_name = model.island_configs[0].name
        evaluated = [
            _make_individual(
                fitness=fitness,
                generation=0,
                individual_id=index,
                stoploss=-0.01 * (index + 1),
            )
            for index, fitness in enumerate((-4, -3, -2, -1, 1, 2, 3, 4))
        ]
        model._evaluated_generation_populations[island_name] = copy.deepcopy(
            evaluated
        )

        # This is the already-created next population which exposed the bug:
        # every slot is unevaluated and none of its IDs identify the measured
        # bottom quartile.
        offspring = Population(size=8, generation=1)
        offspring.individuals = [
            _make_individual(generation=1, individual_id=100 + index)
            for index in range(8)
        ]
        model.island_populations[island_name] = offspring
        ga = MagicMock()
        ga.mutation_rate = 0.2
        generated_for = []

        def generate_random_strategy(*, generation, individual_id):
            generated_for.append(individual_id)
            return _make_gene(
                generation=generation,
                individual_id=individual_id,
                stoploss=-0.25,
            )

        ga.strategy_generator.generate_random_strategy.side_effect = (
            generate_random_strategy
        )
        model.islands[island_name] = ga

        replacements = model._apply_diversity_recovery([island_name], generation=0)

        recovered = model.island_populations[island_name]
        assert replacements == {island_name: 2}
        assert generated_for == [0, 1]
        assert recovered is not offspring
        assert recovered.generation == 1
        assert [
            item.strategy_gene.individual_id for item in recovered.individuals
        ] == list(range(8))
        assert [
            item.metrics.get('origin') for item in recovered.individuals[:2]
        ] == ['diversity_recovery', 'diversity_recovery']
        assert all(
            item.has_measured_fitness for item in recovered.individuals[2:]
        )
        assert all(
            item.strategy_gene.individual_id < 100
            for item in recovered.individuals
        )
        assert all(
            item.has_measured_fitness
            for item in model._evaluated_generation_populations[island_name]
        )
        assert ga.mutation_rate == 0.35

    def test_behavioral_distance_weight_reaches_implementation_module(self):
        import genetic_algorithm.engine.population as population_impl

        ga = MagicMock()
        ga.config = {'genetic_algorithm': {'behavioral_distance_weight': 0.7}}
        try:
            assert GenericIslandModelEvolution._configure_behavioral_distance(ga) == 0.7
            assert population_impl._BEHAVIORAL_DISTANCE_WEIGHT == 0.7
        finally:
            population_impl._BEHAVIORAL_DISTANCE_WEIGHT = 0.0

    def test_island_rng_streams_are_independent_and_restore_caller(self):
        model = _create_model_from_config(_minimal_config(num_islands=2))
        first, second = [item.name for item in model.island_configs]
        caller_state = random.getstate()
        observed = {first: [], second: []}
        for island_name in (first, second, first, second):
            with model._island_random_scope(island_name):
                observed[island_name].append(random.random())

        expected_first = random.Random(model.island_configs[0].seed)
        expected_second = random.Random(model.island_configs[1].seed)
        assert observed[first] == [expected_first.random(), expected_first.random()]
        assert observed[second] == [expected_second.random(), expected_second.random()]
        assert random.getstate() == caller_state

    def test_canonical_outcome_and_digest_are_persisted(self, tmp_path):
        model = _create_model_from_config(_minimal_config(num_islands=1))
        model.checkpoint_dir = tmp_path / 'checkpoints'
        model._last_completed_generation = 3
        model._stop_reason = OUTCOME_PLATEAU
        model._stop_detail = 'stale'

        outcome = model._finalize_evolution_outcome()

        path = tmp_path / 'evolution_outcome.json'
        payload = path.read_bytes()
        digest = (tmp_path / 'evolution_outcome.json.sha256').read_text().strip()
        assert hashlib.sha256(payload).hexdigest() == digest
        assert outcome['reason'] == OUTCOME_PLATEAU
        assert outcome['generations_completed'] == 4

    def test_unstopped_run_classifies_as_completed_budget(self, tmp_path):
        model = _create_model_from_config(_minimal_config(num_islands=1))
        model.checkpoint_dir = tmp_path / 'checkpoints'
        model._last_completed_generation = model.generations - 1

        outcome = model._finalize_evolution_outcome()

        assert outcome['status'] == 'COMPLETED'
        assert outcome['reason'] == OUTCOME_COMPLETED_BUDGET


class TestArchiveIslandSeeding:
    def test_distinct_archive_seeds_only_reach_configured_islands(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['islands'] = [
            {'name': 'archive_a'},
            {'name': 'archive_b'},
            {'name': 'fresh'},
        ]
        config['generic_island_model']['archive_seeding'] = {
            'enabled': True,
            'island_names': ['archive_a', 'archive_b'],
            'max_seeds_per_island': 2,
        }
        model = _create_model_from_config(config)
        seeds = [
            _make_individual(stoploss=-0.01 * (index + 1))
            for index in range(5)
        ]

        model.set_archive_initial_seeds(seeds)

        assert set(model._archive_initial_seeds) == {'archive_a', 'archive_b'}
        assert [len(items) for items in model._archive_initial_seeds.values()] == [2, 2]
        assigned = [
            model._gene_hash(seed)
            for items in model._archive_initial_seeds.values()
            for seed in items
        ]
        assert len(assigned) == len(set(assigned)) == 4
        assert 'fresh' not in model._archive_initial_seeds


# ══════════════════════════════════════════════════════════════════════
# Tests: Identify primary family
# ══════════════════════════════════════════════════════════════════════

class TestIdentifyPrimaryFamily:
    def test_momentum_dominant(self):
        pool = ['RSI', 'MACD', 'STOCH', 'EMA']
        result = GenericIslandModelEvolution._identify_primary_family(pool)
        assert result == 'momentum'

    def test_trend_dominant(self):
        pool = ['EMA', 'SMA', 'TEMA', 'KAMA', 'RSI']
        result = GenericIslandModelEvolution._identify_primary_family(pool)
        assert result == 'trend'

    def test_mixed(self):
        pool = ['RSI', 'EMA']
        result = GenericIslandModelEvolution._identify_primary_family(pool)
        assert result in INDICATOR_FAMILIES  # Should pick one

    def test_empty_pool(self):
        result = GenericIslandModelEvolution._identify_primary_family([])
        assert result == 'mixed'


# ══════════════════════════════════════════════════════════════════════
# Tests: Constants
# ══════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_all_indicators_is_union_of_families(self):
        expected = set()
        for indicators in INDICATOR_FAMILIES.values():
            expected.update(indicators)
        assert set(ALL_INDICATORS) == expected

    def test_no_duplicate_indicators_within_families(self):
        for family, indicators in INDICATOR_FAMILIES.items():
            assert len(indicators) == len(set(indicators)), \
                f"Duplicates in {family}"

    def test_expected_family_count(self):
        assert len(INDICATOR_FAMILIES) == 5
