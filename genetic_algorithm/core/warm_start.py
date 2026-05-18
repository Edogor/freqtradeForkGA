"""
Warm-Start Module

Loads populations from previous experiment checkpoints or Hall of Fame archives
to seed a new evolution run, enabling experiments to build on each other.

Config:
    warm_start:
        enabled: true
        source_experiment: "E170"          # Experiment ID or checkpoint path
        source_type: "population"          # "population", "hof", or "both"
        top_n: 15                          # Max individuals to import
        source_checkpoint: null            # Explicit path (overrides source_experiment)

Usage:
    loader = WarmStartLoader(config, logger)
    individuals = loader.load()
    # Inject into initialize_population()
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene

logger = logging.getLogger(__name__)


class WarmStartLoader:
    """Loads seed individuals from previous experiment artifacts."""

    def __init__(self, config: Dict[str, Any], ga_logger: Optional[logging.Logger] = None):
        self.config = config
        self.ws_config = config.get('warm_start', {})
        self.logger = ga_logger or logger
        self.enabled = self.ws_config.get('enabled', False)
        self.source_type = self.ws_config.get('source_type', 'population')
        self.top_n = self.ws_config.get('top_n', 15)
        self.source_experiment = self.ws_config.get('source_experiment', '')
        self.source_checkpoint = self.ws_config.get('source_checkpoint')
        # Base directories for discovery
        self._checkpoint_dir = Path(config.get('genetic_algorithm', {}).get(
            'checkpoint_dir', 'genetic_algorithm/data/checkpoints'))
        self._hof_dir = Path(config.get('genetic_algorithm', {}).get(
            'hall_of_fame_dir', 'genetic_algorithm/data/hall_of_fame'))
        self._output_dir = Path('genetic_algorithm/output')

    def load(self) -> List[Individual]:
        """Load individuals according to configuration.

        Returns list of Individual objects with reset generation/id/fitness
        ready for evaluation in a new run.
        """
        if not self.enabled:
            return []

        individuals: List[Individual] = []
        try:
            if self.source_type in ('population', 'both'):
                individuals.extend(self._load_from_checkpoint())

            if self.source_type in ('hof', 'both'):
                individuals.extend(self._load_from_hof())

        except Exception as e:
            self.logger.warning(f"[WARM-START] Failed to load: {e}")
            return []

        # Deduplicate by gene dict hash
        seen = set()
        unique: List[Individual] = []
        for ind in individuals:
            key = json.dumps(ind.strategy_gene.to_dict(), sort_keys=True, default=str)
            h = hash(key)
            if h not in seen:
                seen.add(h)
                unique.append(ind)

        # Take top_n best (sorted by fitness descending, unknowns last)
        unique.sort(key=lambda i: i.fitness if i.fitness is not None else -999, reverse=True)
        result = unique[:self.top_n]

        # Reset identifiers so they integrate cleanly into the new population
        for idx, ind in enumerate(result):
            ind.strategy_gene.generation = 0
            ind.strategy_gene.individual_id = 9000 + idx  # High IDs to avoid collisions
            ind.fitness = None
            ind.raw_fitness = None
            ind.evaluated = False
            if hasattr(ind, 'metrics'):
                ind.metrics = {}
            if hasattr(ind, 'metadata') and isinstance(ind.metadata, dict):
                ind.metadata['origin'] = f'warm_start_{self.source_type}'
                ind.metadata['source_experiment'] = self.source_experiment or 'unknown'

        self.logger.info(f"[WARM-START] Loaded {len(result)} individuals "
                         f"(source_type={self.source_type}, experiment={self.source_experiment})")
        return result

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_from_checkpoint(self) -> List[Individual]:
        """Load individuals from a checkpoint file."""
        filepath = self._resolve_checkpoint_path()
        if filepath is None:
            self.logger.warning("[WARM-START] No checkpoint found for source experiment")
            return []

        self.logger.info(f"[WARM-START] Loading population from checkpoint: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)

        pop_data = data.get('population', {})
        individuals_data = pop_data.get('individuals', [])
        if not individuals_data:
            self.logger.warning("[WARM-START] Checkpoint has no individuals")
            return []

        result = []
        for ind_data in individuals_data:
            try:
                ind = Individual.from_dict(ind_data)
                result.append(ind)
            except Exception as e:
                self.logger.debug(f"[WARM-START] Skipping individual: {e}")
        return result

    def _load_from_hof(self) -> List[Individual]:
        """Load individuals from Hall of Fame JSON files."""
        # Try experiment-specific HoF first, then global
        hof_files = []
        if self.source_experiment:
            pattern = f"*{self.source_experiment}*"
            hof_files = sorted(self._hof_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not hof_files:
            hof_files = sorted(self._hof_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        result = []
        for hof_file in hof_files[:3]:  # Check up to 3 most recent
            try:
                with open(hof_file, 'r') as f:
                    data = json.load(f)
                entries = data.get('entries', [])
                for entry in entries:
                    gene_dict = entry.get('gene_dict') or entry.get('strategy_gene')
                    if gene_dict:
                        gene = StrategyGene.from_dict(gene_dict)
                        gene.assign_instance_ids()
                        ind = Individual(strategy_gene=gene)
                        ind.fitness = entry.get('fitness')
                        result.append(ind)
            except Exception as e:
                self.logger.debug(f"[WARM-START] Skipping HoF file {hof_file}: {e}")

        return result

    def _resolve_checkpoint_path(self) -> Optional[Path]:
        """Find the checkpoint file for the configured source experiment."""
        # Explicit path takes priority
        if self.source_checkpoint:
            p = Path(self.source_checkpoint)
            if p.exists():
                return p
            self.logger.warning(f"[WARM-START] Explicit checkpoint path not found: {p}")

        if not self.source_experiment:
            # No experiment specified — find the most recent checkpoint globally
            return self._find_latest_checkpoint(self._checkpoint_dir)

        # Search for experiment-specific checkpoints
        # Convention: checkpoint files contain experiment ID in path or filename
        exp_id = self.source_experiment
        candidates = []

        # Check experiment-specific subdirectory
        exp_dir = self._checkpoint_dir / exp_id
        if exp_dir.is_dir():
            candidates = sorted(exp_dir.glob("checkpoint_gen*.json"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Check output directory for experiment results
        if not candidates:
            for output_file in self._output_dir.glob(f"*{exp_id}*checkpoint*.json"):
                candidates.append(output_file)

        # Fall back to any checkpoint containing the experiment ID in filename
        if not candidates:
            for f in self._checkpoint_dir.rglob("checkpoint_gen*.json"):
                if exp_id.lower() in f.stem.lower() or exp_id.lower() in str(f.parent).lower():
                    candidates.append(f)

        if candidates:
            # Most recent first
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]

        return None

    @staticmethod
    def _find_latest_checkpoint(directory: Path) -> Optional[Path]:
        """Find the most recently modified checkpoint in a directory tree."""
        checkpoints = sorted(
            directory.rglob("checkpoint_gen*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return checkpoints[0] if checkpoints else None
