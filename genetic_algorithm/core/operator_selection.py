"""
Adaptive Operator Selection (AOS)

Tracks which genetic operators (crossover types, mutation types) produce
the fittest offspring and adapts selection probabilities accordingly.

Each operator accumulates credit from offspring fitness improvements.
Probabilities are updated per-generation using a sliding-window average.

Config:
    adaptive_operators:
        enabled: true
        window_size: 50         # Offspring results to keep per operator
        min_probability: 0.10   # Floor — no operator can drop below 10%
        credit_type: "fitness_improvement"  # or "rank_improvement"

Usage:
    aos = AdaptiveOperatorSelector(config)
    # During mating:
    cx_type = aos.select_crossover()
    mut_type = aos.select_mutation()
    # After offspring evaluation:
    aos.record_outcome('crossover', cx_type, parent_fitness, offspring_fitness)
    aos.record_outcome('mutation', mut_type, parent_fitness, offspring_fitness)
    # At generation end:
    aos.update_probabilities()
"""

import logging
import random
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AdaptiveOperatorSelector:
    """Selects genetic operators with probabilities adapted by performance."""

    # Operator registries — names must match what crossover.py / mutation.py use
    CROSSOVER_OPS = ['single_point', 'uniform', 'component']
    MUTATION_OPS = [
        'parameter',          # Mutate indicator parameters / ROI / stoploss
        'indicator_add',      # Add a new indicator
        'indicator_remove',   # Remove an indicator
        'indicator_replace',  # Swap one indicator for another
        'condition_operator', # Change condition operator (>, <, cross_above…)
        'condition_threshold',# Perturb condition threshold
        'logic_toggle',       # Flip AND ↔ OR
    ]

    def __init__(self, config: Dict[str, Any]):
        aos_config = config.get('adaptive_operators', {})
        self.enabled = aos_config.get('enabled', False)
        self.window_size = aos_config.get('window_size', 50)
        self.min_prob = aos_config.get('min_probability', 0.10)
        self.credit_type = aos_config.get('credit_type', 'fitness_improvement')

        # Per-operator sliding windows of (credit_value,)
        self._crossover_credits: Dict[str, deque] = {
            op: deque(maxlen=self.window_size) for op in self.CROSSOVER_OPS
        }
        self._mutation_credits: Dict[str, deque] = {
            op: deque(maxlen=self.window_size) for op in self.MUTATION_OPS
        }

        # Current probabilities (uniform initially)
        self.crossover_probs: Dict[str, float] = {
            op: 1.0 / len(self.CROSSOVER_OPS) for op in self.CROSSOVER_OPS
        }
        self.mutation_probs: Dict[str, float] = {
            op: 1.0 / len(self.MUTATION_OPS) for op in self.MUTATION_OPS
        }

        # Statistics for logging
        self._generation_outcomes: int = 0

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_crossover(self) -> str:
        """Pick a crossover operator weighted by current probabilities."""
        if not self.enabled:
            return random.choice(self.CROSSOVER_OPS)
        ops = list(self.crossover_probs.keys())
        weights = [self.crossover_probs[op] for op in ops]
        return random.choices(ops, weights=weights, k=1)[0]

    def select_mutation(self) -> str:
        """Pick a mutation operator weighted by current probabilities."""
        if not self.enabled:
            return random.choice(self.MUTATION_OPS)
        ops = list(self.mutation_probs.keys())
        weights = [self.mutation_probs[op] for op in ops]
        return random.choices(ops, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Credit recording
    # ------------------------------------------------------------------

    def record_outcome(self, operator_class: str, operator_name: str,
                       parent_fitness: float, offspring_fitness: float) -> None:
        """Record the result of applying an operator.

        Args:
            operator_class: 'crossover' or 'mutation'
            operator_name: Name of the specific operator used
            parent_fitness: Best parent fitness (for crossover: max of two parents)
            offspring_fitness: Offspring fitness after evaluation
        """
        credit = self._compute_credit(parent_fitness, offspring_fitness)

        if operator_class == 'crossover':
            if operator_name in self._crossover_credits:
                self._crossover_credits[operator_name].append(credit)
        elif operator_class == 'mutation':
            if operator_name in self._mutation_credits:
                self._mutation_credits[operator_name].append(credit)

        self._generation_outcomes += 1

    def _compute_credit(self, parent_fitness: float, offspring_fitness: float) -> float:
        """Compute credit value for an operator application."""
        if self.credit_type == 'fitness_improvement':
            # Positive credit for improvement, zero floor (no negative credits)
            return max(0.0, offspring_fitness - parent_fitness)
        elif self.credit_type == 'rank_improvement':
            # Binary: 1.0 if offspring is better, 0.0 otherwise
            return 1.0 if offspring_fitness > parent_fitness else 0.0
        else:
            return max(0.0, offspring_fitness - parent_fitness)

    # ------------------------------------------------------------------
    # Probability update
    # ------------------------------------------------------------------

    def update_probabilities(self) -> None:
        """Recompute operator probabilities from accumulated credits.

        Called once per generation after all offspring are evaluated.
        Uses the adaptive pursuit method with a minimum probability floor.
        """
        if not self.enabled:
            return

        self.crossover_probs = self._recompute(self._crossover_credits, self.CROSSOVER_OPS)
        self.mutation_probs = self._recompute(self._mutation_credits, self.MUTATION_OPS)

        logger.info(
            f"[AOS] Updated probabilities (from {self._generation_outcomes} outcomes) — "
            f"CX: {self._format_probs(self.crossover_probs)} | "
            f"MUT top-3: {self._format_probs(self.mutation_probs, top_n=3)}"
        )
        self._generation_outcomes = 0

    def _recompute(self, credit_windows: Dict[str, deque],
                   all_ops: List[str]) -> Dict[str, float]:
        """Recompute probabilities from credit windows."""
        epsilon = 0.01  # Prevents division by zero

        avg_credits = {}
        for op in all_ops:
            window = credit_windows.get(op, deque())
            if len(window) > 0:
                avg_credits[op] = sum(window) / len(window)
            else:
                avg_credits[op] = epsilon

        total = sum(avg_credits.values()) + epsilon * len(all_ops)
        raw_probs = {op: (avg_credits[op] + epsilon) / total for op in all_ops}

        # Enforce minimum probability floor
        probs = self._enforce_floor(raw_probs, self.min_prob)
        return probs

    @staticmethod
    def _enforce_floor(probs: Dict[str, float], floor: float) -> Dict[str, float]:
        """Enforce a minimum probability for each operator, redistributing excess."""
        n = len(probs)
        max_floor = 1.0 / n  # Can't exceed uniform
        floor = min(floor, max_floor)

        adjusted = {}
        deficit = 0.0
        above_floor = []

        for op, p in probs.items():
            if p < floor:
                deficit += floor - p
                adjusted[op] = floor
            else:
                above_floor.append(op)
                adjusted[op] = p

        # Redistribute deficit from operators above floor
        if above_floor and deficit > 0:
            total_above = sum(adjusted[op] for op in above_floor)
            for op in above_floor:
                adjusted[op] -= deficit * (adjusted[op] / total_above) if total_above > 0 else 0
                adjusted[op] = max(adjusted[op], floor)

        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {op: p / total for op, p in adjusted.items()}

        return adjusted

    @staticmethod
    def _format_probs(probs: Dict[str, float], top_n: int = 0) -> str:
        """Format probabilities for logging."""
        sorted_ops = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        if top_n > 0:
            sorted_ops = sorted_ops[:top_n]
        return ", ".join(f"{op}={p:.2f}" for op, p in sorted_ops)

    # ------------------------------------------------------------------
    # Serialization (for checkpoint persistence)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize AOS state for checkpointing."""
        return {
            'enabled': self.enabled,
            'crossover_probs': dict(self.crossover_probs),
            'mutation_probs': dict(self.mutation_probs),
            'crossover_credits': {k: list(v) for k, v in self._crossover_credits.items()},
            'mutation_credits': {k: list(v) for k, v in self._mutation_credits.items()},
        }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore AOS state from checkpoint."""
        self.crossover_probs = data.get('crossover_probs', self.crossover_probs)
        self.mutation_probs = data.get('mutation_probs', self.mutation_probs)
        for op, credits in data.get('crossover_credits', {}).items():
            if op in self._crossover_credits:
                self._crossover_credits[op] = deque(credits, maxlen=self.window_size)
        for op, credits in data.get('mutation_credits', {}).items():
            if op in self._mutation_credits:
                self._mutation_credits[op] = deque(credits, maxlen=self.window_size)

    def get_report(self) -> Dict[str, Any]:
        """Return a summary report for logging/analysis."""
        return {
            'crossover_probabilities': dict(self.crossover_probs),
            'mutation_probabilities': dict(self.mutation_probs),
            'crossover_credits_sizes': {k: len(v) for k, v in self._crossover_credits.items()},
            'mutation_credits_sizes': {k: len(v) for k, v in self._mutation_credits.items()},
        }
