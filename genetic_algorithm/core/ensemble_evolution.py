"""
Ensemble Co-Evolution

Two-level GA: the inner level evolves individual strategies (existing engine),
the outer level evolves *portfolios* of strategies with capital-weight genes.

Usage:
    from genetic_algorithm.core.ensemble_evolution import EnsembleEvolver

    evolver = EnsembleEvolver(config, candidate_pool)
    best_portfolio = evolver.run()

Designed to run as a finishing phase after the main strategy evolution.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from genetic_algorithm.evaluation.portfolio_backtester import (
    PortfolioBacktester,
    PortfolioResult,
    StrategySlot,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Portfolio Gene — represents a portfolio of strategies + capital weights
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioGene:
    """Genetic representation of a portfolio."""

    strategy_ids: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    fitness: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def normalise_weights(self):
        """Normalise weights to sum to 1."""
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]

    def to_dict(self) -> dict:
        return {
            'strategy_ids': list(self.strategy_ids),
            'weights': list(self.weights),
            'fitness': self.fitness,
            'metrics': dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PortfolioGene':
        pg = cls(
            strategy_ids=data.get('strategy_ids', []),
            weights=data.get('weights', []),
            fitness=data.get('fitness', 0.0),
            metrics=data.get('metrics', {}),
        )
        return pg


# ──────────────────────────────────────────────────────────────────────
# Ensemble Evolver
# ──────────────────────────────────────────────────────────────────────

class EnsembleEvolver:
    """
    Outer-level GA that evolves portfolio compositions from a fixed
    candidate pool of already-evolved strategies.

    Parameters (from config['ensemble']):
        enabled: bool (default False)
        population_size: int (default 30)
        generations: int (default 20)
        portfolio_size: int (default 3-5 strategies per portfolio)
        mutation_rate: float (default 0.3)
        crossover_rate: float (default 0.7)
        fitness_metric: str — one of 'sharpe', 'sortino', 'profit', 'combined'
        elite_size: int (default 2)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        candidate_pool: list,
    ):
        """
        Args:
            config: full GA config dict (reads config['ensemble'])
            candidate_pool: list of evaluated Individual objects with metrics.
        """
        self.config = config
        ens_cfg = config.get('ensemble', {})
        self.pop_size = ens_cfg.get('population_size', 30)
        self.generations = ens_cfg.get('generations', 20)
        self.portfolio_min = ens_cfg.get('portfolio_size_min', 3)
        self.portfolio_max = ens_cfg.get('portfolio_size_max', 5)
        self.mutation_rate = ens_cfg.get('mutation_rate', 0.30)
        self.crossover_rate = ens_cfg.get('crossover_rate', 0.70)
        self.fitness_metric = ens_cfg.get('fitness_metric', 'combined')
        self.elite_size = ens_cfg.get('elite_size', 2)

        # Build candidate lookup {id: Individual}
        self._candidates: Dict[str, Any] = {}
        for ind in candidate_pool:
            mp = (ind.metrics or {}).get('monthly_profits')
            if mp and len(mp) >= 2:
                self._candidates[ind.id] = ind
        self._candidate_ids = list(self._candidates.keys())

        if len(self._candidate_ids) < self.portfolio_min:
            logger.warning(
                f"[ENSEMBLE] Only {len(self._candidate_ids)} viable candidates "
                f"(need >= {self.portfolio_min}). Ensemble evolution skipped."
            )

    # ─── Public API ────────────────────────────────────────────────

    def run(self) -> Optional[PortfolioGene]:
        """
        Run the outer-level portfolio GA.

        Returns the best PortfolioGene or None if not enough candidates.
        """
        if len(self._candidate_ids) < self.portfolio_min:
            return None

        population = self._init_population()
        best: Optional[PortfolioGene] = None

        for gen in range(self.generations):
            # Evaluate
            for pg in population:
                self._evaluate(pg)

            # Sort by fitness (descending)
            population.sort(key=lambda p: p.fitness, reverse=True)
            if best is None or population[0].fitness > best.fitness:
                best = population[0]

            logger.info(
                f"[ENSEMBLE] Gen {gen+1}/{self.generations}: "
                f"best={population[0].fitness:.4f}, "
                f"avg={sum(p.fitness for p in population)/len(population):.4f}"
            )

            if gen == self.generations - 1:
                break

            # Selection + reproduction
            next_pop: List[PortfolioGene] = []
            # Elitism
            for i in range(min(self.elite_size, len(population))):
                next_pop.append(population[i])

            while len(next_pop) < self.pop_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                if random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = PortfolioGene(
                        strategy_ids=list(p1.strategy_ids),
                        weights=list(p1.weights),
                    )
                if random.random() < self.mutation_rate:
                    self._mutate(child)
                child.normalise_weights()
                next_pop.append(child)

            population = next_pop

        return best

    # ─── Internal helpers ──────────────────────────────────────────

    def _init_population(self) -> List[PortfolioGene]:
        """Create random portfolio population."""
        pop: List[PortfolioGene] = []
        for _ in range(self.pop_size):
            size = random.randint(self.portfolio_min, self.portfolio_max)
            ids = random.sample(self._candidate_ids, min(size, len(self._candidate_ids)))
            weights = [random.random() for _ in ids]
            pg = PortfolioGene(strategy_ids=ids, weights=weights)
            pg.normalise_weights()
            pop.append(pg)
        return pop

    def _evaluate(self, pg: PortfolioGene):
        """Evaluate a portfolio gene using PortfolioBacktester."""
        slots: List[StrategySlot] = []
        for sid, w in zip(pg.strategy_ids, pg.weights):
            ind = self._candidates.get(sid)
            if ind is None:
                continue
            m = ind.metrics or {}
            slots.append(StrategySlot(
                strategy_id=sid,
                weight=w,
                monthly_profits=list(m.get('monthly_profits', [])),
                trade_profit_ratios=list(m.get('trade_profit_ratios', [])),
                total_profit_pct=m.get('profit', 0.0),
                total_trades=m.get('num_trades', 0),
            ))
        if not slots:
            pg.fitness = 0.0
            return

        result = PortfolioBacktester.evaluate_portfolio(slots)
        pg.fitness = self._score(result)
        pg.metrics = {
            'total_profit_pct': result.total_profit_pct,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'total_trades': result.total_trades,
            'strategy_count': result.strategy_count,
        }

    def _score(self, result: PortfolioResult) -> float:
        """Convert portfolio result to a single fitness score."""
        if self.fitness_metric == 'sharpe':
            return max(0.0, result.sharpe_ratio)
        elif self.fitness_metric == 'sortino':
            return max(0.0, result.sortino_ratio)
        elif self.fitness_metric == 'profit':
            return result.total_profit_pct
        else:
            # Combined: weighted blend
            sharpe_norm = max(0.0, result.sharpe_ratio) / 5.0
            dd_score = 1.0 - min(result.max_drawdown, 1.0)
            profit_norm = max(0.0, result.total_profit_pct + 50.0) / 250.0
            return 0.4 * sharpe_norm + 0.3 * dd_score + 0.3 * profit_norm

    def _tournament_select(self, pop: List[PortfolioGene], k: int = 3) -> PortfolioGene:
        """Tournament selection."""
        contestants = random.sample(pop, min(k, len(pop)))
        return max(contestants, key=lambda p: p.fitness)

    def _crossover(self, p1: PortfolioGene, p2: PortfolioGene) -> PortfolioGene:
        """Uniform crossover: merge strategy sets, average overlapping weights."""
        merged: Dict[str, float] = {}
        for sid, w in zip(p1.strategy_ids, p1.weights):
            merged[sid] = w
        for sid, w in zip(p2.strategy_ids, p2.weights):
            if sid in merged:
                merged[sid] = (merged[sid] + w) / 2.0
            elif random.random() < 0.5:
                merged[sid] = w
        # Trim to max size
        if len(merged) > self.portfolio_max:
            sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
            merged = dict(sorted_items[:self.portfolio_max])
        # Ensure min size
        while len(merged) < self.portfolio_min and self._candidate_ids:
            extra = random.choice(self._candidate_ids)
            if extra not in merged:
                merged[extra] = random.random()
        ids = list(merged.keys())
        weights = [merged[sid] for sid in ids]
        return PortfolioGene(strategy_ids=ids, weights=weights)

    def _mutate(self, pg: PortfolioGene):
        """Mutate portfolio: swap a strategy or perturb weights."""
        if not pg.strategy_ids:
            return
        r = random.random()
        if r < 0.4 and len(self._candidate_ids) > len(pg.strategy_ids):
            # Swap one strategy for a new one
            idx = random.randrange(len(pg.strategy_ids))
            current_set = set(pg.strategy_ids)
            available = [c for c in self._candidate_ids if c not in current_set]
            if available:
                pg.strategy_ids[idx] = random.choice(available)
                pg.weights[idx] = random.random()
        elif r < 0.7:
            # Perturb a weight
            idx = random.randrange(len(pg.weights))
            pg.weights[idx] = max(0.01, pg.weights[idx] + random.gauss(0, 0.15))
        else:
            # Add or remove a strategy
            if len(pg.strategy_ids) > self.portfolio_min and random.random() < 0.5:
                idx = random.randrange(len(pg.strategy_ids))
                pg.strategy_ids.pop(idx)
                pg.weights.pop(idx)
            elif len(pg.strategy_ids) < self.portfolio_max:
                current_set = set(pg.strategy_ids)
                available = [c for c in self._candidate_ids if c not in current_set]
                if available:
                    pg.strategy_ids.append(random.choice(available))
                    pg.weights.append(random.random())
