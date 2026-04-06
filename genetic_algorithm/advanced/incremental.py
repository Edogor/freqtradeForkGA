"""
Phase 4.1 — Incremental Evolution Pipeline

Resume evolution when new market data arrives. Only re-evaluate on the
new time window.  Keeps strategies fresh without a full restart.

Workflow:
  1.  Load the latest checkpoint (population + GA state)
  2.  Detect new data by comparing the saved timerange with the current config
  3.  Re-evaluate the existing population on the *new* window only
  4.  Run N additional generations of evolution on the updated data
  5.  Save a fresh checkpoint with the extended timerange

Config:
    incremental_evolution:
        enabled: true
        source_checkpoint: "auto"       # "auto" = latest, or explicit path
        new_generations: 5              # Extra gens to evolve after re-eval
        re_eval_top_n: 0                # 0 = re-eval ALL; N = only top-N
        keep_unevaluated: false         # Keep un-re-evaluated individuals?
        merge_hof: true                 # Merge old HoF into seed population

Usage:
    from genetic_algorithm.core.incremental_evolution import IncrementalEvolver
    evolver = IncrementalEvolver(config)
    population = evolver.resume(ga)
"""

import logging
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class IncrementalEvolver:
    """Orchestrates incremental (data-append) evolution runs."""

    def __init__(self, config: Dict[str, Any]):
        inc_cfg = config.get('incremental_evolution', {})
        self.enabled = inc_cfg.get('enabled', False)
        self.source_checkpoint = inc_cfg.get('source_checkpoint', 'auto')
        self.new_generations = inc_cfg.get('new_generations', 5)
        self.re_eval_top_n = inc_cfg.get('re_eval_top_n', 0)
        self.keep_unevaluated = inc_cfg.get('keep_unevaluated', False)
        self.merge_hof = inc_cfg.get('merge_hof', True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resume(self, ga) -> 'Population':
        """Resume evolution from a prior checkpoint with (potentially) new data.

        Args:
            ga: A fully-initialised GeneticAlgorithm instance whose config
                may carry an updated ``backtesting.timerange``.

        Returns:
            The evolved Population after *new_generations* additional gens.
        """
        if not self.enabled:
            raise RuntimeError("IncrementalEvolver called but not enabled")

        checkpoint_path = self._resolve_checkpoint(ga)
        if checkpoint_path is None:
            logger.warning("[INCREMENTAL] No checkpoint found — falling back to fresh start")
            return None

        logger.info(f"[INCREMENTAL] Resuming from {checkpoint_path}")
        population, saved_gen = ga.load_checkpoint(checkpoint_path)

        # Compare timeranges
        old_range = self._extract_timerange_from_checkpoint(checkpoint_path)
        new_range = ga.config.get('backtesting', {}).get('timerange', '')
        if old_range and new_range and old_range != new_range:
            logger.info(f"[INCREMENTAL] Timerange changed: {old_range} → {new_range}")
        else:
            logger.info("[INCREMENTAL] Timerange unchanged — re-evaluating with current config")

        # Mark individuals for re-evaluation
        self._invalidate_for_re_eval(population)

        # Optionally merge HoF strategies as immigrants
        if self.merge_hof:
            self._inject_hof(ga, population)

        # Override generation counter so the engine runs *new_generations* more
        original_total = ga.generations
        ga.generations = saved_gen + self.new_generations
        ga.current_generation = saved_gen

        logger.info(f"[INCREMENTAL] Will run {self.new_generations} additional generations "
                     f"(gen {saved_gen + 1} → {ga.generations})")

        return population

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint(self, ga) -> Optional[str]:
        """Find the checkpoint file to resume from."""
        if self.source_checkpoint and self.source_checkpoint != 'auto':
            p = Path(self.source_checkpoint)
            return str(p) if p.exists() else None

        # Auto-detect: newest checkpoint in the checkpoint directory
        ckpt_dir = getattr(ga, 'checkpoint_dir', None)
        if ckpt_dir is None:
            return None

        candidates = sorted(
            glob.glob(str(Path(ckpt_dir) / 'checkpoint_gen*.json')),
            key=lambda f: Path(f).stat().st_mtime,
        )
        return candidates[-1] if candidates else None

    def _extract_timerange_from_checkpoint(self, path: str) -> Optional[str]:
        """Read the config_snapshot.backtesting.timerange from a checkpoint."""
        import json
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data.get('config_snapshot', {}).get('backtesting', {}).get('timerange')
        except Exception:
            return None

    def _invalidate_for_re_eval(self, population):
        """Mark individuals as unevaluated so the engine re-backtests them."""
        individuals = list(population.individuals)
        if self.re_eval_top_n > 0:
            # Only re-evaluate top N; optionally drop the rest
            individuals.sort(key=lambda i: i.fitness if i.fitness is not None else -1e9, reverse=True)
            to_keep = individuals[:self.re_eval_top_n]
            to_drop = individuals[self.re_eval_top_n:]

            for ind in to_keep:
                ind.evaluated = False
                ind.fitness = None
                ind.raw_fitness = None

            if not self.keep_unevaluated:
                for ind in to_drop:
                    population.individuals.remove(ind)
                logger.info(f"[INCREMENTAL] Dropped {len(to_drop)} low-rank individuals")
            else:
                for ind in to_drop:
                    ind.evaluated = False
                    ind.fitness = None
                    ind.raw_fitness = None

            logger.info(f"[INCREMENTAL] Marked {len(to_keep)} individuals for re-evaluation")
        else:
            for ind in individuals:
                ind.evaluated = False
                ind.fitness = None
                ind.raw_fitness = None
            logger.info(f"[INCREMENTAL] Marked ALL {len(individuals)} individuals for re-evaluation")

    def _inject_hof(self, ga, population):
        """Merge Hall-of-Fame strategies into the population as immigrants."""
        try:
            hof = getattr(ga, 'hall_of_fame', None)
            if hof is None:
                return
            hof_individuals = hof.get_individuals(5)
            injected = 0
            for ind in hof_individuals:
                clone = type(ind)(strategy_gene=ind.strategy_gene.copy())
                clone.evaluated = False
                clone.fitness = None
                clone.raw_fitness = None
                clone.metrics = {'origin': 'incremental_hof_seed'}
                population.add_individual(clone)
                injected += 1
            if injected:
                logger.info(f"[INCREMENTAL] Injected {injected} HoF strategies into population")
        except Exception as e:
            logger.warning(f"[INCREMENTAL] Failed to inject HoF: {e}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'source_checkpoint': self.source_checkpoint,
            'new_generations': self.new_generations,
            're_eval_top_n': self.re_eval_top_n,
        }
