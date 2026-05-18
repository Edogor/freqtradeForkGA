"""
Shared migration and island-evolution utilities.

Extracted from ``island_model.py`` and ``generic_island_model.py`` to
eliminate code duplication.  Both island implementations can import
these helpers instead of maintaining their own copies.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from genetic_algorithm.engine.population import Population
    from genetic_algorithm.genome.individual import Individual

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared dataclass
# ------------------------------------------------------------------

@dataclass
class AggregateStats:
    """Aggregate statistics across all islands for monitor display.

    Previously duplicated as ``_AggregateStats`` in both island modules.
    """

    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    genetic_diversity: Optional[float] = None
    generation: int = 0
    # Fields the monitor may access
    best_raw_fitness: Optional[float] = None
    median_fitness: Optional[float] = None
    diversity_score: Optional[float] = None
    holdout_avg_degradation: Optional[float] = None
    holdout_best_degradation: Optional[float] = None
    holdout_num_evaluated: Optional[int] = None
    holdout_num_profitable: Optional[int] = None


# ------------------------------------------------------------------
# Migration helpers
# ------------------------------------------------------------------

def get_top_individuals(
    populations: Dict[str, "Population"],
    island_name: str,
    count: int,
) -> list:
    """Return top-*count* individuals from *island_name* by raw_fitness.

    Filters out individuals with ``raw_fitness`` that is ``None`` or ≤ 0.
    """
    pop = populations.get(island_name)
    if pop is None:
        return []

    ranked = sorted(
        [
            ind for ind in pop.individuals
            if ind.raw_fitness is not None and ind.raw_fitness > 0
        ],
        key=lambda x: x.raw_fitness,
        reverse=True,
    )
    return ranked[:count]


def inject_migrants(
    populations: Dict[str, "Population"],
    target_island: str,
    migrants: list,
    generation: int,
    source: str = "unknown",
) -> int:
    """Replace worst individuals in *target_island* with deep-copied *migrants*.

    Returns the number of individuals replaced.
    """
    pop = populations.get(target_island)
    if pop is None or not migrants:
        return 0

    from genetic_algorithm.genome.individual import Individual

    sorted_inds = sorted(
        pop.individuals,
        key=lambda x: x.raw_fitness if x.raw_fitness is not None else -1,
    )

    replaced = 0
    for migrant in migrants:
        if replaced >= len(sorted_inds):
            break

        gene_copy = migrant.strategy_gene.copy()
        gene_copy.generation = generation
        gene_copy.individual_id = sorted_inds[replaced].strategy_gene.individual_id

        new_ind = Individual(strategy_gene=gene_copy)
        new_ind.evaluated = False
        new_ind.metrics = {"origin": f"migrant_from_{source}"}

        idx = pop.individuals.index(sorted_inds[replaced])
        pop.individuals[idx] = new_ind
        replaced += 1

    return replaced


# ------------------------------------------------------------------
# Migration topologies
# ------------------------------------------------------------------

def migrate_ring(
    island_names: List[str],
    populations: Dict[str, "Population"],
    count: int,
    generation: int,
) -> List[dict]:
    """Ring topology: island *i* sends top-*count* to island *(i+1) % N*."""
    events = []
    n = len(island_names)
    for i, src in enumerate(island_names):
        dst = island_names[(i + 1) % n]
        top = get_top_individuals(populations, src, count)
        if top:
            replaced = inject_migrants(populations, dst, top, generation, source=src)
            events.append({"from": src, "to": dst, "count": replaced})
    return events


def migrate_fully_connected(
    island_names: List[str],
    populations: Dict[str, "Population"],
    count: int,
    generation: int,
) -> List[dict]:
    """Fully connected: every island sends top individuals to every other.

    For diversity, each target receives a *different random sample* of the
    source's top-2× individuals (same approach as generic island model).
    """
    events = []
    for src in island_names:
        top = get_top_individuals(populations, src, count * 2)
        if not top:
            continue
        for dst in island_names:
            if dst == src:
                continue
            sample = random.sample(top, min(count, len(top)))
            replaced = inject_migrants(populations, dst, sample, generation, source=src)
            events.append({"from": src, "to": dst, "count": replaced})
    return events


def migrate_tournament(
    island_names: List[str],
    populations: Dict[str, "Population"],
    count: int,
    generation: int,
) -> List[dict]:
    """Tournament: random pairs, winner (higher best fitness) sends to loser."""
    events = []
    names = list(island_names)
    random.shuffle(names)
    for i in range(0, len(names) - 1, 2):
        a, b = names[i], names[i + 1]
        top_a = get_top_individuals(populations, a, 1)
        top_b = get_top_individuals(populations, b, 1)
        fit_a = top_a[0].raw_fitness if top_a else 0
        fit_b = top_b[0].raw_fitness if top_b else 0
        if fit_a >= fit_b:
            winner, loser = a, b
        else:
            winner, loser = b, a
        top = get_top_individuals(populations, winner, count)
        if top:
            replaced = inject_migrants(populations, loser, top, generation, source=winner)
            events.append({"from": winner, "to": loser, "count": replaced})
    return events


def migrate_hierarchical(
    island_names: List[str],
    populations: Dict[str, "Population"],
    count: int,
    generation: int,
) -> List[dict]:
    """Hierarchical: random pairs, bidirectional exchange."""
    events = []
    names = list(island_names)
    random.shuffle(names)
    for i in range(0, len(names) - 1, 2):
        a, b = names[i], names[i + 1]
        top_a = get_top_individuals(populations, a, count)
        top_b = get_top_individuals(populations, b, count)
        if top_a:
            r = inject_migrants(populations, b, top_a, generation, source=a)
            events.append({"from": a, "to": b, "count": r})
        if top_b:
            r = inject_migrants(populations, a, top_b, generation, source=b)
            events.append({"from": b, "to": a, "count": r})
    return events


# Topology dispatch table
TOPOLOGIES = {
    "ring": migrate_ring,
    "fully_connected": migrate_fully_connected,
    "tournament": migrate_tournament,
    "hierarchical": migrate_hierarchical,
}


def migrate(
    topology: str,
    island_names: List[str],
    populations: Dict[str, "Population"],
    count: int,
    generation: int,
) -> List[dict]:
    """Dispatch to the named migration topology.

    Raises ``ValueError`` if *topology* is unknown.
    """
    fn = TOPOLOGIES.get(topology)
    if fn is None:
        raise ValueError(
            f"Unknown migration topology {topology!r}. "
            f"Choose from: {', '.join(TOPOLOGIES)}"
        )
    return fn(island_names, populations, count, generation)
