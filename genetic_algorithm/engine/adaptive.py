"""
Adaptive Controller — Manages adaptive mutation rates and convergence detection.

Extracted from GeneticAlgorithm.check_convergence() to enable independent
testing and reuse across different engine configurations.

Responsibilities:
  - Track improvement/stagnation across generations
  - Adaptive mutation rate: escalate when stuck, cool down on improvement
  - Convergence detection with configurable patience
  - Catastrophic restart flagging (40% pop replacement when deeply stuck)
"""

import logging
from typing import Optional


class AdaptiveController:
    """Controls adaptive parameters and convergence for the evolution engine."""

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        ga_cfg = config.get('genetic_algorithm', {})

        # Mutation rate tracking
        self.base_mutation_rate: float = ga_cfg.get('mutation_rate', 0.1)
        self.mutation_rate: float = self.base_mutation_rate
        self.adaptive_mutation: bool = ga_cfg.get('adaptive_mutation', True)
        self.max_adaptation_factor: float = ga_cfg.get('max_adaptation_factor', 2.0)
        self.adaptation_step: float = ga_cfg.get('adaptation_step', 0.1)
        self.max_mutation_rate: float = ga_cfg.get('max_mutation_rate', 0.65)
        self.mutation_cooldown_factor: float = ga_cfg.get('mutation_cooldown_factor', 0.5)

        # Convergence
        self.convergence_patience: int = ga_cfg.get('convergence_patience', 10)
        self.no_improvement_count: int = 0

        # Per-generation flags
        self._new_best_this_gen: bool = False
        self._catastrophic_restart_needed: bool = False

        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_new_best(self):
        """Call when a new best individual is found this generation."""
        self.no_improvement_count = 0
        self._new_best_this_gen = True

    def check_convergence(self) -> bool:
        """
        Update stagnation counter and adaptive rates. Call once per generation
        AFTER best-individual detection.

        Returns:
            True if evolution should stop (converged).
        """
        if not self._new_best_this_gen:
            self.no_improvement_count += 1
        self._new_best_this_gen = False

        self._update_mutation_rate()
        self._check_catastrophic_restart()

        if self.no_improvement_count >= self.convergence_patience:
            self.logger.info(
                f"Converged: No improvement for {self.convergence_patience} generations"
            )
            return True

        return False

    @property
    def catastrophic_restart_needed(self) -> bool:
        return self._catastrophic_restart_needed

    def clear_catastrophic_restart(self):
        """Reset the flag after the restart has been performed."""
        self._catastrophic_restart_needed = False
        self.no_improvement_count = 0
        self.mutation_rate = self.base_mutation_rate

    def get_state(self) -> dict:
        """Return serialisable state for checkpointing."""
        return {
            'mutation_rate': self.mutation_rate,
            'base_mutation_rate': self.base_mutation_rate,
            'no_improvement_count': self.no_improvement_count,
            'catastrophic_restart_needed': self._catastrophic_restart_needed,
        }

    def load_state(self, state: dict):
        """Restore from a checkpoint dict."""
        self.mutation_rate = state.get('mutation_rate', self.mutation_rate)
        self.base_mutation_rate = state.get('base_mutation_rate', self.base_mutation_rate)
        self.no_improvement_count = state.get('no_improvement_count', 0)
        self._catastrophic_restart_needed = state.get('catastrophic_restart_needed', False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_mutation_rate(self):
        """Adjust mutation rate based on stagnation."""
        if not self.adaptive_mutation:
            self.mutation_rate = self.base_mutation_rate
            return

        if self.no_improvement_count > 0:
            # Escalate: factor grows linearly with stuck count
            adaptation_factor = min(
                self.max_adaptation_factor,
                1.0 + (self.no_improvement_count * self.adaptation_step),
            )
            self.mutation_rate = min(
                self.max_mutation_rate,
                self.base_mutation_rate * adaptation_factor,
            )
            self.logger.info(
                f"Adaptive mutation: rate increased to {self.mutation_rate:.3f} "
                f"(factor={adaptation_factor:.2f}, no improvement for "
                f"{self.no_improvement_count} gens)"
            )
        else:
            # Cool down: exponential decay toward base rate
            excess = self.mutation_rate - self.base_mutation_rate
            if excess > 1e-6:
                self.mutation_rate = self.base_mutation_rate + excess * self.mutation_cooldown_factor
                self.logger.info(
                    f"Adaptive mutation: cooling down to {self.mutation_rate:.3f} "
                    f"(cooldown factor={self.mutation_cooldown_factor})"
                )
            else:
                self.mutation_rate = self.base_mutation_rate

    def _check_catastrophic_restart(self):
        """Flag for catastrophic restart at half patience."""
        half_patience = self.convergence_patience // 2
        if half_patience > 0 and self.no_improvement_count == half_patience:
            self._catastrophic_restart_needed = True
            self.logger.warning(
                f"[CATASTROPHIC RESTART] Stagnation for {half_patience} gens "
                f"— flagged for 40% population replacement"
            )
