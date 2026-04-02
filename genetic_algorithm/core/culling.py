"""
Strategy Culling Module

Removes low-quality individuals from the population during evolution
based on configurable rules. Prevents bad strategies from consuming
evaluation budget and polluting the gene pool.

Usage:
    culler = StrategyCuller(config, logger)
    report = culler.cull_population(population, generation=3, elite_size=2)
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population

logger = logging.getLogger(__name__)


@dataclass
class CullReport:
    """Summary of a culling operation on a population."""
    generation: int
    total_before: int
    total_after: int = 0
    removed: List[Dict] = field(default_factory=list)  # [{id, reason, fitness, profit, trades}]
    reasons: Dict[str, int] = field(default_factory=dict)  # {reason: count}

    @property
    def num_removed(self) -> int:
        return self.total_before - self.total_after


class StrategyCuller:
    """
    Configurable strategy culling engine.

    Reads rules from the ``strategy_culling`` config section and removes
    individuals that match any enabled rule, respecting elite protection
    and minimum survivor guarantees.

    Config example::

        strategy_culling:
          enabled: true
          protect_elites: true
          min_survivors: 4
          skip_first_generation: true
          rules:
            zero_trades:
              enabled: true
            negative_profit_high_fitness:
              enabled: true
              profit_threshold: -1.0
              min_fitness_for_cull: 0.35
            extreme_drawdown:
              enabled: false
              max_drawdown: 0.40
            min_trades_floor:
              enabled: true
              min_trades: 50
            validation_failure:
              enabled: false
              min_val_profit: -5.0
    """

    def __init__(self, config: dict, log: Optional[logging.Logger] = None):
        cfg = config.get('strategy_culling', {})
        self.enabled: bool = cfg.get('enabled', False)
        self.protect_elites: bool = cfg.get('protect_elites', True)
        self.min_survivors: int = cfg.get('min_survivors', 4)
        self.skip_first_generation: bool = cfg.get('skip_first_generation', True)
        self.rules_cfg: dict = cfg.get('rules', {})
        self.log = log or logger

        # Aggregate stats across all generations (thread-safe)
        self._stats_lock = threading.Lock()
        self.total_culled: int = 0
        self.total_reason_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cull_population(
        self,
        population: Population,
        generation: int,
        elite_size: int = 0,
    ) -> CullReport:
        """
        Apply culling rules to a population in-place.

        Args:
            population: The population to cull (modified in-place).
            generation: Current generation number (0-indexed).
            elite_size: Number of top individuals protected from culling.

        Returns:
            CullReport with details of removed individuals.
        """
        report = CullReport(
            generation=generation,
            total_before=len(population.individuals),
        )

        if not self.enabled:
            report.total_after = report.total_before
            return report

        if self.skip_first_generation and generation == 0:
            self.log.debug("[CULL] Skipping culling for generation 0")
            report.total_after = report.total_before
            return report

        # Determine protected elite set
        protected_ids = set()
        if self.protect_elites and elite_size > 0:
            ranked = sorted(
                population.individuals,
                key=lambda x: x.raw_fitness if x.raw_fitness is not None else float('-inf'),
                reverse=True,
            )
            for ind in ranked[:elite_size]:
                protected_ids.add(id(ind))

        # Evaluate all rules, collect removal candidates
        to_remove: List[tuple] = []  # [(individual, reason)]
        for ind in population.individuals:
            if id(ind) in protected_ids:
                continue
            reason = self._evaluate_rules(ind)
            if reason:
                to_remove.append((ind, reason))

        # Enforce min_survivors guard
        max_removable = max(0, len(population.individuals) - self.min_survivors)
        if len(to_remove) > max_removable:
            self.log.warning(
                "[CULL] Gen %d: Would remove %d but min_survivors=%d limits to %d",
                generation, len(to_remove), self.min_survivors, max_removable,
            )
            to_remove = to_remove[:max_removable]

        # Remove and log
        for ind, reason in to_remove:
            profit = ind.metrics.get('profit', 0)
            trades = ind.metrics.get('num_trades', 0)
            fitness_val = ind.raw_fitness if ind.raw_fitness is not None else 0
            self.log.info(
                "[CULL] Removed %s: %s (fitness=%.4f, profit=%.2f%%, trades=%d)",
                ind.id, reason, fitness_val, profit, trades,
            )
            population.remove_individual(ind)
            report.removed.append({
                'id': ind.id,
                'reason': reason,
                'fitness': fitness_val,
                'profit': profit,
                'trades': trades,
            })
            report.reasons[reason] = report.reasons.get(reason, 0) + 1

        report.total_after = len(population.individuals)

        # Update aggregate stats (thread-safe for island-model parallelism)
        with self._stats_lock:
            self.total_culled += report.num_removed
            for reason, count in report.reasons.items():
                self.total_reason_counts[reason] = self.total_reason_counts.get(reason, 0) + count

        # Summary log
        if report.num_removed > 0:
            reason_parts = [f"{r}={c}" for r, c in sorted(report.reasons.items())]
            self.log.info(
                "[CULL SUMMARY] Gen %d: Removed %d/%d individuals. Reasons: %s",
                generation, report.num_removed, report.total_before,
                ', '.join(reason_parts),
            )

        return report

    def get_aggregate_stats(self) -> dict:
        """Return aggregate culling stats for final experiment summary."""
        return {
            'total_culled': self.total_culled,
            'reasons': dict(self.total_reason_counts),
        }

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rules(self, ind: Individual) -> Optional[str]:
        """
        Check all enabled rules against an individual.

        Returns the name of the first matching rule, or None.
        """
        metrics = ind.metrics
        fitness = ind.raw_fitness if ind.raw_fitness is not None else 0

        # Rule: zero_trades
        r = self.rules_cfg.get('zero_trades', {})
        if r.get('enabled', True):  # On by default
            if metrics.get('num_trades', 0) == 0 or metrics.get('no_trades', False):
                return 'zero_trades'

        # Rule: negative_profit_high_fitness
        r = self.rules_cfg.get('negative_profit_high_fitness', {})
        if r.get('enabled', False):
            profit_threshold = r.get('profit_threshold', -1.0)
            min_fitness = r.get('min_fitness_for_cull', 0.35)
            profit = metrics.get('profit', 0)
            if profit < profit_threshold and fitness > min_fitness:
                return 'negative_profit_high_fitness'

        # Rule: extreme_drawdown
        r = self.rules_cfg.get('extreme_drawdown', {})
        if r.get('enabled', False):
            max_dd = r.get('max_drawdown', 0.40)
            dd = metrics.get('max_drawdown', 0)
            if dd > max_dd:
                return 'extreme_drawdown'

        # Rule: min_trades_floor
        r = self.rules_cfg.get('min_trades_floor', {})
        if r.get('enabled', False):
            min_trades = r.get('min_trades', 50)
            trades = metrics.get('num_trades', 0)
            if 0 < trades < min_trades:
                return 'min_trades_floor'

        # Rule: validation_failure
        r = self.rules_cfg.get('validation_failure', {})
        if r.get('enabled', False):
            min_val_profit = r.get('min_val_profit', -5.0)
            val_profit = metrics.get('val_profit')
            if val_profit is not None and val_profit < min_val_profit:
                return 'validation_failure'

        return None
