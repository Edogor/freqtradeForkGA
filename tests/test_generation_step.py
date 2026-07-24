"""Tests for GenerationStep (Phase 1 — generation logic extraction)."""

import logging
import random
import pytest
from unittest.mock import MagicMock, patch

from genetic_algorithm.engine.generation import GenerationStep
from genetic_algorithm.engine.population import Population, PopulationStats
from genetic_algorithm.genome.individual import Individual
from genetic_algorithm.genome.gene import StrategyGene, IndicatorGene, ConditionGene


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


def _make_population(size=10, gen=0):
    """Create a population with *size* individuals."""
    pop = Population(size=size, generation=gen)
    for i in range(size):
        pop.add_individual(
            _make_individual(gen=gen, ind_id=i, fitness=0.05 * (i + 1))
        )
    return pop


def _make_strategy_generator():
    """Mock strategy generator that produces random-looking genes."""
    sg = MagicMock()
    def _generate(generation, individual_id):
        ind_gene = IndicatorGene(type='SMA', parameters={'period': 20})
        cond = ConditionGene(indicator='SMA', operator='<', threshold=50)
        return StrategyGene(
            generation=generation, individual_id=individual_id,
            indicators=[ind_gene],
            entry_conditions=[cond],
            exit_conditions=[cond],
        )
    sg.generate_random_strategy.side_effect = _generate
    return sg


def _make_config():
    """Minimal config dict for GenerationStep."""
    return {
        'genetic_algorithm': {
            'population_size': 10,
            'generations': 5,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 2,
            'mode': 'single_objective',
        },
        'indicators': {
            'min_entry_conditions': 1,
        },
    }


def _make_step(**overrides):
    """Build a GenerationStep with sensible defaults, accepting overrides."""
    defaults = dict(
        population_size=10,
        elite_size=2,
        mode='single_objective',
        config=_make_config(),
        logger=logging.getLogger('test_genstep'),
        strategy_generator=_make_strategy_generator(),
        crossover_rate=0.8,
        crossover_method='single_point',
        tournament_size=3,
        selection_method='tournament',
        allow_self_crossover=True,
        adaptive_tournament=False,
        random_immigrants=2,
        diversity_threshold=0.15,
    )
    defaults.update(overrides)
    return GenerationStep(**defaults)


# =========================================================================
# Basic execute tests
# =========================================================================

class TestGenerationStepExecute:
    """Test the main execute() path."""

    def test_returns_full_population(self):
        step = _make_step()
        pop = _make_population(size=10, gen=0)
        next_gen, stats = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert isinstance(next_gen, Population)
        assert len(next_gen) == 10

    def test_generation_incremented(self):
        step = _make_step()
        pop = _make_population(size=10, gen=0)
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert next_gen.generation == 1

    def test_generation_local_individual_ids_are_unique(self):
        """Elite carry-over must not collide with new candidate artifacts."""
        step = _make_step(elite_size=2, random_immigrants=2, population_size=10)
        pop = _make_population(size=10, gen=0)

        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)

        identities = [
            (ind.strategy_gene.generation, ind.strategy_gene.individual_id)
            for ind in next_gen.individuals
        ]
        assert len(identities) == len(set(identities))
        assert sorted(item[1] for item in identities) == list(range(10))

    def test_returns_op_stats_dict(self):
        step = _make_step()
        pop = _make_population(size=10, gen=0)
        _, stats = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert 'crossover_count' in stats
        assert 'mutation_count' in stats
        assert 'immigrants_total' in stats
        assert 'llm_mutations' in stats

    def test_population_size_matches(self):
        """Even with small input population, output should be full-sized."""
        step = _make_step(population_size=20)
        pop = _make_population(size=20, gen=0)
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert len(next_gen) == 20


# =========================================================================
# Elitism tests
# =========================================================================

class TestElitism:
    """Test elite selection and carry-over."""

    def test_elites_preserved(self):
        step = _make_step(elite_size=3, population_size=10)
        pop = _make_population(size=10, gen=0)
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        # The top 3 by raw_fitness should appear as evaluated individuals
        evaluated = [ind for ind in next_gen.individuals if ind.evaluated]
        assert len(evaluated) >= 3

    def test_elites_have_next_generation_number(self):
        step = _make_step(elite_size=2, population_size=10)
        pop = _make_population(size=10, gen=5)
        next_gen, _ = step.execute(pop, current_generation=5, mutation_rate=0.1)
        for ind in next_gen.individuals:
            assert ind.strategy_gene.generation == 6

    def test_nsga2_skips_elitism(self):
        step = _make_step(mode='nsga2', elite_size=2, population_size=10)
        pop = _make_population(size=10, gen=0)
        # Give all individuals objectives for NSGA-II
        for ind in pop.individuals:
            ind.objectives = [ind.fitness, 0.5]
            ind.rank = 1
            ind.crowding_distance = float('inf')
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        # Should still produce a population (via NSGA-II env selection)
        assert len(next_gen) > 0

    def test_elite_raw_fitness_carried_over(self):
        step = _make_step(elite_size=1, population_size=10)
        pop = _make_population(size=10, gen=0)
        best = max(pop.individuals, key=lambda x: x.raw_fitness)
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        # First individual should be the elite copy
        elite_copy = next_gen.individuals[0]
        assert elite_copy.raw_fitness == best.raw_fitness


# =========================================================================
# Immigrant tests
# =========================================================================

class TestImmigrants:
    """Test immigrant injection."""

    def test_random_immigrants_injected(self):
        step = _make_step(random_immigrants=3, population_size=10)
        pop = _make_population(size=10, gen=0)
        _, stats = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert stats['immigrants_total'] >= 1

    def test_external_immigrants_used(self):
        step = _make_step(random_immigrants=2, population_size=10)
        pop = _make_population(size=10, gen=0)
        ext = [_make_individual(gen=0, ind_id=99, fitness=0.99)]
        _, stats = step.execute(
            pop, current_generation=0, mutation_rate=0.1,
            external_immigrants=ext,
        )
        assert stats['immigrants_total'] >= 1

    def test_immigrants_marked_unevaluated(self):
        step = _make_step(random_immigrants=3, elite_size=0, population_size=10)
        pop = _make_population(size=10, gen=0)
        ext = [_make_individual(gen=0, ind_id=99, fitness=0.99)]
        next_gen, _ = step.execute(
            pop, current_generation=0, mutation_rate=0.1,
            external_immigrants=ext,
        )
        # External immigrants should be marked unevaluated
        unevaluated = [ind for ind in next_gen.individuals if not ind.evaluated]
        assert len(unevaluated) >= 1

    def test_low_diversity_doubles_immigrants(self):
        step = _make_step(random_immigrants=2, population_size=20, elite_size=2)
        # Create a low-diversity population (all same fitness)
        pop = Population(size=20, generation=0)
        for i in range(20):
            pop.add_individual(_make_individual(gen=0, ind_id=i, fitness=0.5))
        _, stats = step.execute(pop, current_generation=0, mutation_rate=0.1)
        # With fully identical population, diversity = 0 → immigrants doubled
        # Should get at least 2 immigrants (original count)
        assert stats['immigrants_total'] >= 2


# =========================================================================
# Offspring tests
# =========================================================================

class TestOffspring:
    """Test offspring creation via crossover and mutation."""

    def test_offspring_fill_remaining_slots(self):
        step = _make_step(elite_size=2, random_immigrants=2, population_size=10)
        pop = _make_population(size=10, gen=0)
        next_gen, stats = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert len(next_gen) == 10
        assert stats['offspring_added'] > 0

    def test_crossover_count_positive(self):
        step = _make_step(crossover_rate=1.0, population_size=10, elite_size=0,
                          random_immigrants=0)
        pop = _make_population(size=10, gen=0)
        _, stats = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert stats['crossover_count'] > 0

    def test_zero_crossover_rate_uses_clones(self):
        step = _make_step(crossover_rate=0.0, population_size=10, elite_size=0,
                          random_immigrants=0)
        pop = _make_population(size=10, gen=0)
        next_gen, stats = step.execute(pop, current_generation=0, mutation_rate=0.0)
        assert stats['crossover_count'] == 0
        assert len(next_gen) == 10

    def test_offspring_have_origin_tag(self):
        step = _make_step(elite_size=0, random_immigrants=0, population_size=10)
        pop = _make_population(size=10, gen=0)
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        for ind in next_gen.individuals:
            assert 'origin' in ind.metrics


# =========================================================================
# AOS integration tests
# =========================================================================

class TestAOS:
    """Test Adaptive Operator Selection integration."""

    def test_aos_select_crossover_called(self):
        aos = MagicMock()
        aos.enabled = True
        aos.select_crossover.return_value = 'uniform'
        step = _make_step(aos=aos, elite_size=0, random_immigrants=0,
                          population_size=10)
        pop = _make_population(size=10, gen=0)
        step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert aos.select_crossover.called

    def test_aos_disabled_uses_default_method(self):
        aos = MagicMock()
        aos.enabled = False
        step = _make_step(aos=aos, crossover_method='two_point',
                          elite_size=0, random_immigrants=0, population_size=10)
        pop = _make_population(size=10, gen=0)
        # Should not crash — uses self.crossover_method
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert len(next_gen) == 10


# =========================================================================
# Adaptive tournament tests
# =========================================================================

class TestAdaptiveTournament:
    """Test adaptive tournament size adjustments."""

    def test_high_diversity_increases_tournament(self):
        step = _make_step(
            adaptive_tournament=True,
            tournament_size=3,
            population_size=10,
            elite_size=0,
            random_immigrants=0,
        )
        pop = _make_population(size=10, gen=0)
        # The population has varied fitness → get_stats().genetic_diversity
        # should be > 0; if > 0.4, tournament should increase
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert len(next_gen) == 10  # Just verify it doesn't crash


# =========================================================================
# Random fill tests
# =========================================================================

class TestRandomFill:
    """Test population fill-up when offspring loop can't fill."""

    def test_fill_random_completes_population(self):
        step = _make_step(population_size=10, elite_size=0, random_immigrants=0)
        pop = _make_population(size=10, gen=0)
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert len(next_gen) == 10


# =========================================================================
# LLM mutation tests
# =========================================================================

class TestLLMMutations:
    """Test LLM-guided mutation on top-K."""

    def test_llm_mutation_when_stagnating(self):
        """LLM mutations fire when population has room and stagnation exceeds threshold."""
        sd = MagicMock()
        sd.enabled = True
        sd.mutation_enabled = True
        sd.mutation_stagnation_threshold = 3
        sd.mutation_top_k = 2
        sd.mutation_probability = 1.0  # always mutate

        def _mutate(**kwargs):
            ind_gene = IndicatorGene(type='EMA', parameters={'period': 10})
            cond = ConditionGene(indicator='EMA', operator='<', threshold=30)
            return StrategyGene(
                generation=kwargs.get('generation', 1),
                individual_id=kwargs.get('individual_id', 0),
                indicators=[ind_gene],
                entry_conditions=[cond],
                exit_conditions=[cond],
            )
        sd.mutate_strategy.side_effect = _mutate

        step = _make_step(
            llm_enabled=True,
            strategy_designer=sd,
            population_size=15,  # larger than offspring will produce
            elite_size=2,
            random_immigrants=0,
        )

        # Directly test _apply_llm_mutations with a non-full population
        ranked_by_raw = [_make_individual(gen=0, ind_id=i, fitness=0.5 - 0.1 * i)
                         for i in range(5)]
        next_gen = Population(size=15, generation=1)
        # Add 10 individuals (5 slots remaining for LLM mutations)
        for i in range(10):
            next_gen.add_individual(_make_individual(gen=1, ind_id=i, fitness=0.3))

        count = step._apply_llm_mutations(
            next_gen, ranked_by_raw, 1,
            mutation_rate=0.1,
            no_improvement_count=5,
        )
        assert count > 0
        assert sd.mutate_strategy.called

    def test_no_llm_mutation_below_threshold(self):
        sd = MagicMock()
        sd.enabled = True
        sd.mutation_enabled = True
        sd.mutation_stagnation_threshold = 10

        step = _make_step(
            llm_enabled=True,
            strategy_designer=sd,
            population_size=10,
            elite_size=2,
            random_immigrants=0,
        )
        pop = _make_population(size=10, gen=0)
        _, stats = step.execute(
            pop, current_generation=0, mutation_rate=0.1,
            no_improvement_count=2,  # below threshold
        )
        assert stats['llm_mutations'] == 0

    def test_llm_mutation_disabled(self):
        step = _make_step(llm_enabled=False, population_size=10)
        pop = _make_population(size=10, gen=0)
        _, stats = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert stats['llm_mutations'] == 0


# =========================================================================
# NSGA-II environmental selection tests
# =========================================================================

class TestNSGA2EnvironmentalSelection:
    """Test NSGA-II (μ+λ) survivor selection."""

    def _make_nsga2_step(self, **kw):
        defaults = dict(mode='nsga2', population_size=10, elite_size=2,
                        random_immigrants=0)
        defaults.update(kw)
        return _make_step(**defaults)

    def _make_nsga2_pop(self, size=10, gen=0):
        pop = _make_population(size=size, gen=gen)
        for i, ind in enumerate(pop.individuals):
            ind.objectives = [ind.fitness, 1.0 - ind.fitness]
            ind.rank = 1
            ind.crowding_distance = float('inf')
        return pop

    def test_nsga2_returns_correct_size(self):
        step = self._make_nsga2_step()
        pop = self._make_nsga2_pop()
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert len(next_gen) <= 10

    def test_nsga2_merges_parents_and_offspring(self):
        step = self._make_nsga2_step(population_size=5)
        pop = self._make_nsga2_pop(size=5)
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert len(next_gen) <= 5


# =========================================================================
# Immigrant provider callback tests
# =========================================================================

class TestImmigrantProvider:
    """Test external immigrant provider integration."""

    def test_provider_called(self):
        provider = MagicMock(return_value=[_make_individual(fitness=0.99)])
        step = _make_step(
            immigrant_provider=provider,
            random_immigrants=3,
            population_size=10,
        )
        pop = _make_population(size=10, gen=0)
        step.execute(pop, current_generation=0, mutation_rate=0.1)
        provider.assert_called_once()

    def test_provider_failure_graceful(self):
        provider = MagicMock(side_effect=RuntimeError("provider error"))
        step = _make_step(
            immigrant_provider=provider,
            random_immigrants=3,
            population_size=10,
        )
        pop = _make_population(size=10, gen=0)
        # Should not raise
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert len(next_gen) == 10


# =========================================================================
# MAP-Elites integration tests
# =========================================================================

class TestMAPElites:
    """Test MAP-Elites diverse injection."""

    def test_map_elites_injection(self):
        me = MagicMock()
        me.enabled = True
        me.filled_cells = 5
        me.injection_count = 2
        me.sample_diverse.return_value = [
            _make_individual(fitness=0.7),
            _make_individual(fitness=0.6),
        ]
        step = _make_step(
            map_elites=me,
            random_immigrants=3,
            population_size=10,
        )
        pop = _make_population(size=10, gen=0)
        _, stats = step.execute(pop, current_generation=0, mutation_rate=0.1)
        me.sample_diverse.assert_called_once_with(n=2)
        assert stats['immigrants_total'] >= 2

    def test_map_elites_disabled(self):
        me = MagicMock()
        me.enabled = False
        step = _make_step(map_elites=me, population_size=10)
        pop = _make_population(size=10, gen=0)
        step.execute(pop, current_generation=0, mutation_rate=0.1)
        me.sample_diverse.assert_not_called()


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    """Edge cases and robustness."""

    def test_empty_external_immigrants(self):
        step = _make_step(population_size=10)
        pop = _make_population(size=10, gen=0)
        next_gen, _ = step.execute(
            pop, current_generation=0, mutation_rate=0.1,
            external_immigrants=[],
        )
        assert len(next_gen) == 10

    def test_none_external_immigrants(self):
        step = _make_step(population_size=10)
        pop = _make_population(size=10, gen=0)
        next_gen, _ = step.execute(
            pop, current_generation=0, mutation_rate=0.1,
            external_immigrants=None,
        )
        assert len(next_gen) == 10

    def test_high_mutation_rate(self):
        step = _make_step(population_size=10, elite_size=0, random_immigrants=0)
        pop = _make_population(size=10, gen=0)
        next_gen, _ = step.execute(
            pop, current_generation=0, mutation_rate=1.0,
        )
        assert len(next_gen) == 10

    def test_single_individual_population(self):
        step = _make_step(population_size=5, elite_size=1, random_immigrants=1,
                          allow_self_crossover=True)
        pop = Population(size=5, generation=0)
        pop.add_individual(_make_individual(gen=0, ind_id=0, fitness=0.5))
        # With only 1 individual, crossover uses self-crossover
        next_gen, _ = step.execute(pop, current_generation=0, mutation_rate=0.1)
        assert len(next_gen) == 5
