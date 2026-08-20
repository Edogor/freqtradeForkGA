"""
Checkpoint Manager — Save/restore evolution state.

Extracted from evolution.py to enable independent testing and reuse.
The manager handles:
  - Full checkpoint save/load (version 3, fail-closed provenance contract)
  - Legacy checkpoint for web dashboard compatibility
  - Atomic writes (temp file + rename) to prevent corruption
  - Random state serialization for reproducible resume
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from genetic_algorithm.engine.checkpoint_contract import (
    CHECKPOINT_VERSION,
    CheckpointCompatibilityError,
    CheckpointProvenanceV3,
    parse_checkpoint_provenance,
    seal_checkpoint,
    verify_resume_checkpoint,
)
from genetic_algorithm.engine.population import Population, PopulationStats
from genetic_algorithm.genome.individual import Individual


class CheckpointManager:
    """Manages saving and loading evolution checkpoints."""

    def __init__(self, checkpoint_dir: str | Path,
                 checkpoint_interval: int = 5,
                 logger: Optional[logging.Logger] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.logger = logger or logging.getLogger(__name__)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_temps()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_save(self, generation: int) -> bool:
        """Return True if a checkpoint should be saved at this generation."""
        return self.checkpoint_interval > 0 and (generation + 1) % self.checkpoint_interval == 0

    def save(self, population: Population, generation: int,
             ga_state: Dict[str, Any],
             config_snapshot: Dict[str, Any],
             generation_stats: List[PopulationStats],
             extras: Optional[Dict[str, Any]] = None,
             filepath: Optional[str] = None,
             monitor=None,
             provenance: CheckpointProvenanceV3 | Dict[str, Any] | None = None) -> str:
        """
        Save full evolution state to a checkpoint file.

        Args:
            population: Current population.
            generation: Current generation number.
            ga_state: Dict with keys like best_individual, best_fitness_ever,
                      no_improvement_count, current_mutation_rate, base_mutation_rate,
                      catastrophic_restart_needed.
            config_snapshot: Subset of config to store for reference.
            generation_stats: List of PopulationStats from all generations.
            extras: Optional dict with feature_tracker, holdout_state,
                    aos_state, surrogate_state, map_elites_state data.
            filepath: If None, auto-generated in checkpoint_dir.
            monitor: Optional monitor to notify on save.
            provenance: Immutable run identity.  Without it, the checkpoint is
                        saved for diagnostics but cannot be resumed.

        Returns:
            Path to the saved checkpoint file.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(self.checkpoint_dir / f"checkpoint_gen{generation}_{timestamp}.json")

        extras = extras or {}
        parsed_provenance = (
            parse_checkpoint_provenance(provenance) if provenance is not None else None
        )

        checkpoint = {
            'version': CHECKPOINT_VERSION,
            'resume_eligible': parsed_provenance is not None,
            'provenance': (
                parsed_provenance.model_dump(mode='json')
                if parsed_provenance is not None
                else None
            ),
            'timestamp': datetime.now().isoformat(),
            'generation': generation,
            'total_generations': ga_state.get('total_generations', 0),
            'population_size': population.size,

            # Full population state
            'population': {
                'size': population.size,
                'generation': population.generation,
                'individuals': [ind.to_dict() for ind in population.individuals],
            },

            # GA engine state
            'ga_state': {
                'best_individual': (ga_state['best_individual'].to_dict()
                                    if ga_state.get('best_individual') else None),
                'best_fitness_ever': ga_state.get('best_fitness_ever', 0.0),
                'no_improvement_count': ga_state.get('no_improvement_count', 0),
                'current_mutation_rate': ga_state.get('current_mutation_rate',
                                                       ga_state.get('mutation_rate', 0.1)),
                'base_mutation_rate': ga_state.get('base_mutation_rate', 0.1),
                'catastrophic_restart_needed': ga_state.get('catastrophic_restart_needed', False),
            },

            # Optional sub-system state
            'feature_tracker': extras.get('feature_tracker'),
            'holdout_state': extras.get('holdout_state'),
            'aos_state': extras.get('aos_state'),
            'surrogate_state': extras.get('surrogate_state'),
            'map_elites_state': extras.get('map_elites_state'),

            # Generation history
            'generation_stats': [
                {
                    'generation': s.generation,
                    'best_fitness': s.best_fitness,
                    'avg_fitness': s.avg_fitness,
                    'worst_fitness': s.worst_fitness,
                    'genetic_diversity': s.genetic_diversity,
                    'best_raw_fitness': s.best_raw_fitness,
                    'avg_raw_fitness': s.avg_raw_fitness,
                }
                for s in generation_stats
            ],

            # Config snapshot for reference
            'config_snapshot': config_snapshot,

            # Random state for reproducible resume
            'random_state': self._capture_random_state(),
        }

        # Atomic write
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = seal_checkpoint(checkpoint)

        tmp_path = filepath + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        os.replace(tmp_path, filepath)

        self.logger.info(f"[CHECKPOINT] Saved generation {generation} to {filepath}")

        # Legacy format + monitor notification
        try:
            self.save_legacy(population, generation, ga_state, generation_stats)
            if monitor is not None:
                monitor.on_checkpoint_saved(generation, filepath)
        except Exception as e:
            self.logger.debug(f"[CHECKPOINT] Monitor/legacy notification failed: {e}")

        return filepath

    def load(
        self,
        filepath: str,
        population_size: int,
        *,
        expected_provenance: CheckpointProvenanceV3 | Dict[str, Any] | None = None,
    ) -> Tuple[Population, int, Dict[str, Any]]:
        """
        Load evolution state from a checkpoint file.

        Args:
            filepath: Path to checkpoint JSON file.
            population_size: Current configured population size.
            expected_provenance: Exact immutable identity of the current run.

        Returns:
            Tuple of (population, start_generation, state_dict).
            state_dict contains: ga_state, feature_tracker, holdout_state,
            aos_state, surrogate_state, map_elites_state, generation_stats.
        """
        self.logger.info(f"[CHECKPOINT] Loading from {filepath}")

        with open(filepath, 'r') as f:
            checkpoint = json.load(f)

        if expected_provenance is None:
            raise CheckpointCompatibilityError(
                "checkpoint load requires the current immutable provenance"
            )
        checkpoint = verify_resume_checkpoint(checkpoint, expected_provenance)

        saved_gen = checkpoint['generation']
        ckpt_version = checkpoint['version']

        # Resume only accepts the exact v3 population format.  Legacy formats
        # remain readable through load_legacy() for dashboard/migration tools.
        pop_raw = checkpoint['population']
        if not isinstance(pop_raw, dict):
            raise CheckpointCompatibilityError("checkpoint population is not v3")
        individuals_data = pop_raw['individuals']
        pop_gen = pop_raw.get('generation', saved_gen)
        pop_size = pop_raw.get('size', population_size)
        if pop_size != population_size:
            raise CheckpointCompatibilityError(
                f"checkpoint population size differs: {pop_size} != {population_size}"
            )
        if len(individuals_data) != pop_size:
            raise CheckpointCompatibilityError(
                "checkpoint population count differs from its declared size"
            )

        population = Population(size=pop_size, generation=pop_gen)
        for ind_data in individuals_data:
            population.add_individual(Individual.from_dict(ind_data))

        self.logger.info(
            f"[CHECKPOINT] Restored population: {len(population.individuals)} "
            f"individuals from generation {saved_gen} (v{ckpt_version} format)"
        )

        ga_state = checkpoint.get('ga_state', {})
        if ga_state.get('best_individual'):
            ga_state['best_individual'] = Individual.from_dict(ga_state['best_individual'])

        # Restore generation stats
        stats_data = checkpoint.get('generation_stats', [])
        restored_stats: List[PopulationStats] = []
        for s in stats_data:
            stat = PopulationStats(
                generation=s.get('generation', 0),
                size=population_size,
                best_fitness=s.get('best_fitness', 0),
                avg_fitness=s.get('avg_fitness', 0),
                worst_fitness=s.get('worst_fitness', 0),
                best_raw_fitness=s.get('best_raw_fitness'),
                avg_raw_fitness=s.get('avg_raw_fitness'),
            )
            stat.genetic_diversity = s.get('genetic_diversity')
            restored_stats.append(stat)

        # Restore random state
        self._restore_random_state(checkpoint.get('random_state', {}))

        start_generation = saved_gen + 1
        self.logger.info(f"[CHECKPOINT] Will resume from generation {start_generation + 1}")

        state = {
            'ga_state': ga_state,
            'feature_tracker': checkpoint.get('feature_tracker'),
            'holdout_state': checkpoint.get('holdout_state', {}),
            'aos_state': checkpoint.get('aos_state'),
            'surrogate_state': checkpoint.get('surrogate_state'),
            'map_elites_state': checkpoint.get('map_elites_state'),
            'generation_stats': restored_stats,
        }

        return population, start_generation, state

    # ------------------------------------------------------------------
    # Legacy checkpoint (web dashboard compatibility)
    # ------------------------------------------------------------------

    def save_legacy(self, population: Population, generation: int,
                    ga_state: Dict[str, Any],
                    generation_stats: List[PopulationStats]):
        """Save checkpoint in legacy format for web dashboard."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_ind = ga_state.get('best_individual')
        checkpoint = {
            'generation': generation,
            'population': [ind.to_dict() for ind in population.individuals],
            'population_size': population.size,
            'best_individual': best_ind.to_dict() if best_ind else None,
            'best_fitness_ever': ga_state.get('best_fitness_ever', 0.0),
            'no_improvement_count': ga_state.get('no_improvement_count', 0),
            'mutation_rate': ga_state.get('current_mutation_rate',
                                          ga_state.get('mutation_rate', 0.1)),
            'generation_stats': [
                {
                    'generation': s.generation,
                    'size': s.size,
                    'best_fitness': s.best_fitness,
                    'avg_fitness': s.avg_fitness,
                    'worst_fitness': s.worst_fitness,
                    'best_raw_fitness': s.best_raw_fitness,
                    'avg_raw_fitness': s.avg_raw_fitness,
                    'genetic_diversity': s.genetic_diversity,
                    'holdout_avg_degradation': s.holdout_avg_degradation,
                    'holdout_best_degradation': s.holdout_best_degradation,
                    'holdout_num_evaluated': s.holdout_num_evaluated,
                    'holdout_num_profitable': s.holdout_num_profitable,
                }
                for s in generation_stats
            ],
            'random_seed': ga_state.get('random_seed'),
            'timestamp': time.time(),
        }

        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.json'
        temp_path = self.checkpoint_dir / 'latest_checkpoint.tmp'

        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            temp_path.rename(checkpoint_path)
        except Exception as e:
            self.logger.debug(f"[CHECKPOINT] Legacy checkpoint write failed: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def load_legacy(self) -> Optional[Dict[str, Any]]:
        """Load latest legacy checkpoint if available."""
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.json'
        if not checkpoint_path.exists():
            return None
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            self.logger.info(
                f"[CHECKPOINT] Found legacy checkpoint at generation {checkpoint['generation']} "
                f"(saved at {checkpoint.get('timestamp', 'unknown')})"
            )
            return checkpoint
        except Exception as e:
            self.logger.error(f"[CHECKPOINT] Failed to load legacy checkpoint: {e}")
            return None

    def restore_from_legacy(self, checkpoint: Dict[str, Any],
                            population_size: int) -> Tuple[Population, Dict[str, Any]]:
        """
        Restore from a legacy checkpoint.

        Returns:
            Tuple of (population, state_dict) where state_dict has keys
            matching ga_state + generation_stats.
        """
        ga_state = {
            'best_fitness_ever': checkpoint.get('best_fitness_ever', 0.0),
            'no_improvement_count': checkpoint.get('no_improvement_count', 0),
            'current_mutation_rate': checkpoint.get('mutation_rate', 0.1),
        }

        best_ind_data = checkpoint.get('best_individual')
        if best_ind_data:
            ga_state['best_individual'] = Individual.from_dict(best_ind_data)

        population = Population(
            size=checkpoint.get('population_size', population_size),
            generation=checkpoint['generation'],
        )
        for ind_dict in checkpoint['population']:
            population.add_individual(Individual.from_dict(ind_dict))

        restored_stats: List[PopulationStats] = []
        for s in checkpoint.get('generation_stats', []):
            stats = PopulationStats(
                generation=s.get('generation', 0),
                size=s.get('size', population_size),
                best_fitness=s.get('best_fitness', 0),
                avg_fitness=s.get('avg_fitness', 0),
                worst_fitness=s.get('worst_fitness', 0),
                best_raw_fitness=s.get('best_raw_fitness'),
                avg_raw_fitness=s.get('avg_raw_fitness'),
                genetic_diversity=s.get('genetic_diversity'),
                holdout_avg_degradation=s.get('holdout_avg_degradation'),
                holdout_best_degradation=s.get('holdout_best_degradation'),
                holdout_num_evaluated=s.get('holdout_num_evaluated'),
                holdout_num_profitable=s.get('holdout_num_profitable'),
            )
            restored_stats.append(stats)

        self.logger.info(
            f"[CHECKPOINT] Restored: generation={checkpoint['generation']}, "
            f"population={len(population.individuals)}, "
            f"best_fitness={ga_state['best_fitness_ever']:.4f}"
        )

        state = {
            'ga_state': ga_state,
            'generation_stats': restored_stats,
            'current_generation': checkpoint['generation'],
        }
        return population, state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cleanup_stale_temps(self):
        """Remove stale .tmp files from previous crashes."""
        for tmp_file in self.checkpoint_dir.glob('*.tmp'):
            try:
                tmp_file.unlink()
                self.logger.debug(f"Cleaned up stale temp file: {tmp_file}")
            except Exception as e:
                self.logger.warning(f"Failed to clean temp file {tmp_file}: {e}")

    @staticmethod
    def _capture_random_state() -> Dict[str, Any]:
        """Capture Python + NumPy random state for reproducible resume."""
        state: Dict[str, Any] = {'python': random.getstate()}
        try:
            import numpy as np
            np_state = np.random.get_state()
            state['numpy'] = (
                np_state[0],
                np_state[1].tolist(),
                int(np_state[2]),
                int(np_state[3]),
                float(np_state[4]),
            )
        except ImportError:
            pass
        return state

    def _restore_random_state(self, random_state: Dict[str, Any]):
        """Restore Python + NumPy random state."""
        if 'python' in random_state:
            try:
                py_state = random_state['python']
                random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))
                self.logger.debug("[CHECKPOINT] Restored Python random state")
            except Exception as e:
                self.logger.warning(f"[CHECKPOINT] Failed to restore Python random state: {e}")
        if 'numpy' in random_state:
            try:
                import numpy as np
                np_state = random_state['numpy']
                np.random.set_state((
                    np_state[0],
                    np.array(np_state[1], dtype=np.uint32),
                    int(np_state[2]),
                    int(np_state[3]),
                    float(np_state[4]),
                ))
                self.logger.debug("[CHECKPOINT] Restored NumPy random state")
            except Exception as e:
                self.logger.warning(f"[CHECKPOINT] Failed to restore NumPy random state: {e}")
