"""Focused tests for signed fitness sharing."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from genetic_algorithm.engine.population import apply_fitness_sharing


@dataclass
class _FitnessOnlyIndividual:
    raw_fitness: float
    fitness: float
    shared_updates: int = 0

    def set_shared_fitness(self, shared_fitness: float) -> None:
        self.fitness = shared_fitness
        self.shared_updates += 1


def _population(*raw_fitnesses: float):
    return SimpleNamespace(
        individuals=[
            _FitnessOnlyIndividual(raw_fitness=value, fitness=value)
            for value in raw_fitnesses
        ]
    )


def test_positive_fitness_sharing_keeps_legacy_division_semantics():
    population = _population(6.0, 3.0)

    apply_fitness_sharing(
        population,
        sigma_share=1.0,
        distance_matrix=[[0.0, 0.0], [0.0, 0.0]],
    )

    assert [individual.fitness for individual in population.individuals] == [3.0, 1.5]


def test_negative_and_zero_fitness_are_shared_without_reversing_sign_order():
    population = _population(0.0, -1.0, -2.0)

    apply_fitness_sharing(
        population,
        sigma_share=1.0,
        distance_matrix=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )

    shared = [individual.fitness for individual in population.individuals]
    assert shared == pytest.approx([0.0, -3.0, -6.0])
    assert shared[0] > shared[1] > shared[2]
    assert [individual.shared_updates for individual in population.individuals] == [1, 1, 1]
