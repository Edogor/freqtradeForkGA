"""
Generic Island Model Evolution

A configurable N-island system where independent GA populations evolve
strategies with different specializations (indicator pools, pair subsets,
seeds) and periodically exchange individuals through multiple migration
topologies.

Unlike the regime-locked ``IslandModelEvolution``, every island receives
the **full** dataset (or a configured pair subset) and can optionally
enable walk-forward validation.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  Island 0          Island 1         ...  Island N-1  │
    │  (momentum)        (trend)               (mixed)     │
    │  ┌──────────┐     ┌──────────┐     ┌──────────┐     │
    │  │ pop=10   │◄───►│ pop=10   │◄───►│ pop=10   │     │
    │  │ seed=42  │     │ seed=43  │     │ seed=56  │     │
    │  └──────────┘     └──────────┘     └──────────┘     │
    │       ↕ ring / fully_connected / tournament / merge  │
    └──────────────────────────────────────────────────────┘

Usage:
    from genetic_algorithm.core.generic_island_model import (
        GenericIslandModelEvolution,
    )
    evo = GenericIslandModelEvolution("config/ga_config_generic_island.yaml")
    results = evo.evolve()
"""

import copy
import glob
import hashlib
import json
import logging
import math
import os
import random
import signal
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from genetic_algorithm.config.invariants import derive_island_population_slots
from genetic_algorithm.core.culling import StrategyCuller
from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.core.hall_of_fame import HallOfFame
from genetic_algorithm.core.individual import (
    FITNESS_EVIDENCE_BACKTEST_FAILED,
    Individual,
)
from genetic_algorithm.core.population import (
    Population,
    apply_fitness_sharing,
    calculate_pairwise_distances,
)
from genetic_algorithm.engine.checkpoint_contract import (
    CHECKPOINT_VERSION,
    CheckpointCompatibilityError,
    checkpoint_generation_from_path,
    checkpoint_provenance_from_config,
    latest_checkpoint_by_generation,
    seal_checkpoint,
    verify_resume_checkpoint,
)
from genetic_algorithm.engine.island_results import extract_island_finalists
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.evaluation.panel_contract import (
    PANEL_ROLE_COMMON_REPLAY,
    build_evaluation_panel,
    replay_on_common_panel,
)
from genetic_algorithm.orchestration.generation_trace_v2 import GenerationTraceWriterV2


logger = logging.getLogger(__name__)


OUTCOME_PLATEAU = "PLATEAU"
OUTCOME_TECHNICAL_INVALID = "TECHNICAL_INVALID"
OUTCOME_DIVERSITY_COLLAPSE = "DIVERSITY_COLLAPSE"
OUTCOME_COMPLETED_BUDGET = "COMPLETED_BUDGET"


# ══════════════════════════════════════════════════════════════════════
# Indicator families for auto-generation
# ══════════════════════════════════════════════════════════════════════

INDICATOR_FAMILIES: Dict[str, List[str]] = {
    'momentum': ['RSI', 'MACD', 'STOCH', 'CCI', 'MFI', 'ROC', 'WILLR'],
    'trend': ['EMA', 'SMA', 'TEMA', 'KAMA', 'SUPERTREND', 'AROON', 'ICHIMOKU', 'PSAR'],
    'volatility': ['BBANDS', 'ATR', 'DONCHIAN'],
    'volume': ['OBV', 'CMF', 'VROC', 'VWAP'],
    'candlestick': [
        'CDL_ENGULFING', 'CDL_HAMMER', 'CDL_DOJI', 'CDL_MORNINGSTAR',
        'CDL_EVENINGSTAR', 'CDL_SHOOTINGSTAR', 'CDL_HARAMI', 'CDL_PIERCING',
        'CDL_DARKCLOUD', 'CDL_3WHITESOLDIERS', 'CDL_3BLACKCROWS',
    ],
}

ALL_INDICATORS = [ind for family in INDICATOR_FAMILIES.values() for ind in family]


# ══════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GenericIslandConfig:
    """Configuration for a single generic island."""
    name: str
    population_size: int = 10
    generations: Optional[int] = None  # None = inherit from top-level
    seed: int = 42
    indicator_pool: Optional[List[str]] = None  # None = all indicators
    pairs: Optional[List[str]] = None  # None = inherit from base config
    walk_forward_enabled: Optional[bool] = None  # None = inherit
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericMigrationConfig:
    """Configuration for migration between generic islands."""
    topology: str = 'ring'  # ring, fully_connected, tournament, hierarchical
    interval: int = 3  # every N generations
    count: int = 2  # top-N individuals to migrate
    merge_rounds: bool = False  # periodic global pool-and-redistribute
    merge_interval: int = 5  # if merge_rounds=True, every N gens
    tournament_size: int = 3  # if topology=tournament


@dataclass
class GenericIslandStats:
    """Statistics for one generic island across evolution."""
    name: str
    best_fitness: float = 0.0
    best_profit: float = 0.0
    avg_fitness: float = 0.0
    generations_completed: int = 0
    migrants_sent: int = 0
    migrants_received: int = 0


@dataclass
class GenericMigrationEvent:
    """Record of a migration event."""
    generation: int
    source: str
    target: str
    count: int
    fitnesses: List[float]


@dataclass
class _AggregateStats:
    """Lightweight aggregate stats for the terminal monitor."""
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    genetic_diversity: Optional[float] = None
    generation: int = 0
    best_raw_fitness: Optional[float] = None
    median_fitness: Optional[float] = None
    diversity_score: Optional[float] = None
    holdout_avg_degradation: Optional[float] = None
    holdout_best_degradation: Optional[float] = None
    holdout_num_evaluated: Optional[int] = None
    holdout_num_profitable: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════
# Generic Island Model Evolution
# ══════════════════════════════════════════════════════════════════════

class GenericIslandModelEvolution:
    """
    Orchestrates N independent GeneticAlgorithm instances as peer islands
    with configurable migration topologies.

    Unlike the regime-locked ``IslandModelEvolution``, islands are NOT
    tied to regime segments.  Each island gets the full dataset (or an
    optional pair subset) and evolves independently with its own seed,
    optional indicator pool restriction, and optional walk-forward.
    """

    def __init__(
        self,
        config_path: str,
        visualize: bool = False,
        interactive: bool = True,
    ):
        from genetic_algorithm.config.schema import load_config

        self.config = load_config(config_path)

        self.config_path = config_path
        self.visualize = visualize
        self.interactive = interactive
        self.logger = logging.getLogger(f"{__name__}.GenericIslandModel")

        # Parse generic_island_model config
        gim_cfg = self.config.get('generic_island_model', {})
        self.num_islands: int = gim_cfg.get('num_islands', 15)
        self.parallel_islands: bool = gim_cfg.get('parallel_islands', False)
        self.population_per_island: int = gim_cfg.get('population_per_island', 10)
        self.generations: int = gim_cfg.get(
            'generations',
            self.config.get('genetic_algorithm', {}).get('generations', 20),
        )

        # Specialization config
        spec_cfg = gim_cfg.get('specialization', {})
        self.rotate_seeds: bool = spec_cfg.get('rotate_seeds', True)
        self.use_indicator_pools: bool = spec_cfg.get('indicator_pools', True)
        self.indicator_overlap: float = spec_cfg.get('indicator_overlap', 0.5)
        self.pair_rotation: bool = spec_cfg.get('pair_rotation', False)
        self.pair_subset_size: int = spec_cfg.get('pair_subset_size', 3)

        # Migration config
        mig_cfg = gim_cfg.get('migration', {})
        self.migration = GenericMigrationConfig(
            topology=mig_cfg.get('topology', 'ring'),
            interval=mig_cfg.get('interval', 3),
            count=mig_cfg.get('count', 2),
            merge_rounds=mig_cfg.get('merge_rounds', False),
            merge_interval=mig_cfg.get('merge_interval', 5),
            tournament_size=mig_cfg.get('tournament_size', 3),
        )

        common_cfg = gim_cfg.get('common_panel_replay', {})
        self.common_panel_replay_enabled = bool(common_cfg.get('enabled', False))
        self.common_panel_replay_interval = int(common_cfg.get('interval', 3))
        self.common_panel_top_n = int(common_cfg.get('top_n_per_island', 3))
        self.common_panel_early_stop_patience = int(
            common_cfg.get('early_stop_patience', 4)
        )
        self.common_panel_min_improvement = float(
            common_cfg.get('min_improvement', 0.002)
        )
        # ``min_generation`` is one-based for operators (generation 9 means
        # the ninth fully evaluated population).  Defaults preserve the
        # historical behavior of allowing a stop at the first stale check.
        self.common_panel_min_generation = int(common_cfg.get('min_generation', 1))
        self.common_panel_relative_min_improvement = float(
            common_cfg.get('relative_min_improvement', 0.0)
        )
        incumbent_score = common_cfg.get('incumbent_score')
        if incumbent_score is not None:
            if (
                isinstance(incumbent_score, bool)
                or not isinstance(incumbent_score, (int, float))
                or not math.isfinite(float(incumbent_score))
            ):
                raise ValueError(
                    "generic_island_model.common_panel_replay.incumbent_score "
                    "must be null or a finite number"
                )
            incumbent_score = float(incumbent_score)
        self.common_panel_incumbent_score: Optional[float] = incumbent_score

        diversity_cfg = gim_cfg.get('diversity_recovery', {})
        self.diversity_recovery_enabled = bool(diversity_cfg.get('enabled', False))
        self.diversity_island_threshold_count = int(
            diversity_cfg.get('island_threshold_count', 9)
        )
        self.diversity_checks_before_recovery = int(
            diversity_cfg.get('consecutive_checks_before_recovery', 2)
        )
        self.diversity_checks_after_recovery = int(
            diversity_cfg.get('consecutive_checks_after_recovery', 2)
        )
        self.diversity_duplicate_fraction_threshold = float(
            diversity_cfg.get('duplicate_fraction_threshold', 0.75)
        )
        self.diversity_genetic_threshold = float(
            diversity_cfg.get('genetic_diversity_threshold', 0.10)
        )
        self.diversity_replacement_fraction = float(
            diversity_cfg.get('replacement_fraction', 0.25)
        )
        self.diversity_recovery_mutation_rate = float(
            diversity_cfg.get('recovery_mutation_rate', 0.35)
        )

        archive_cfg = gim_cfg.get('archive_seeding', {})
        self.archive_seeding_enabled = bool(archive_cfg.get('enabled', False))
        self.archive_seeding_island_names = list(archive_cfg.get('island_names', []))
        self.archive_seeding_max_per_island = int(
            archive_cfg.get('max_seeds_per_island', 4)
        )

        # Walk-forward config (per-island override)
        wf_cfg = gim_cfg.get('walk_forward', {})
        self.island_walk_forward: Optional[bool] = (
            wf_cfg.get('enabled') if 'enabled' in wf_cfg else None
        )

        # Base seed (from GA config or default)
        configured_seed = self.config.get('genetic_algorithm', {}).get('random_seed')
        self.base_seed: int = 42 if configured_seed is None else configured_seed

        # Build island configs
        explicit_islands = gim_cfg.get('islands', [])
        if explicit_islands:
            if len(explicit_islands) != self.num_islands:
                raise ValueError(
                    "generic_island_model.num_islands must exactly match the "
                    "number of explicit islands"
                )
            self.island_configs = self._parse_explicit_islands(explicit_islands)
        else:
            self.island_configs = self._auto_generate_islands()
        island_names = [island.name for island in self.island_configs]
        if len(set(island_names)) != len(island_names):
            raise ValueError("generic island names must be unique")
        unknown_archive_islands = sorted(
            set(self.archive_seeding_island_names) - set(island_names)
        )
        if unknown_archive_islands:
            raise ValueError(
                "archive_seeding references unknown islands: "
                + ", ".join(unknown_archive_islands)
            )
        if len(set(self.archive_seeding_island_names)) != len(
            self.archive_seeding_island_names
        ):
            raise ValueError("archive_seeding.island_names must be unique")

        # Runtime state
        self.islands: Dict[str, GeneticAlgorithm] = {}
        self.island_populations: Dict[str, Population] = {}
        self.island_stats: Dict[str, GenericIslandStats] = {}
        self.generation_stats: Dict[str, list] = {}
        self.migration_history: List[GenericMigrationEvent] = []
        hof_cfg = self.config.get('hall_of_fame', {})
        self.hall_of_fame = HallOfFame(
            directory=hof_cfg.get('directory', 'genetic_algorithm/data/hall_of_fame'),
            max_size=hof_cfg.get('max_size', 50),
            min_fitness=hof_cfg.get('min_fitness', 0.0),
        )

        # External migration (cross-machine strategy exchange)
        ext_cfg = gim_cfg.get('external_migration', {})
        self.external_migration_enabled: bool = ext_cfg.get('enabled', False)
        self.external_migration_dir: Path = Path(
            ext_cfg.get('directory', 'genetic_algorithm/data/incoming_migrants')
        )
        self.external_export_dir: Path = Path(
            ext_cfg.get('export_directory', 'genetic_algorithm/data/outgoing_migrants')
        )
        self.external_migration_interval: int = ext_cfg.get(
            'interval', self.migration.interval
        )
        self.external_migration_count: int = ext_cfg.get('count', 3)

        # Strategy culling
        self.culler = StrategyCuller(self.config, self.logger)
        if self.culler.enabled:
            self.logger.info("[CULL] Strategy culling enabled")

        # Checkpoint settings
        storage_config = self.config.get('storage', {})
        self.checkpoint_dir = Path(
            storage_config.get('checkpoint_dir', 'genetic_algorithm/data/checkpoints')
        )
        self.checkpoint_interval: int = storage_config.get('checkpoint_interval', 5)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.generation_trace = GenerationTraceWriterV2(
            self.checkpoint_dir.parent / "generation_trace_v2.jsonl"
        )
        self._checkpoint_requested = False  # for SIGUSR1 manual trigger

        # Thread safety
        self._hof_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._migration_lock = threading.Lock()
        self._shutdown_requested = False
        self._strict_initial_seeds: List[Individual] = []
        self._archive_initial_seeds: Dict[str, List[Individual]] = {}
        self._evaluated_generation_elites: Dict[str, List[Individual]] = {}
        self._evaluated_generation_populations: Dict[str, List[Individual]] = {}
        # A lane-global incumbent makes a new evolution a continuation of the
        # search rather than an isolated run.  The first replay is therefore
        # a genuine improvement check (and can be the first of four stale
        # checks) instead of silently resetting the plateau clock.
        self._common_panel_best: Optional[float] = self.common_panel_incumbent_score
        self._common_panel_no_improvement_checks = 0
        self.common_panel_replay_history: List[Dict[str, Any]] = []
        self._rng_lock = threading.RLock()
        self._island_random_states: Dict[str, Dict[str, Any]] = {}
        self._diversity_bad_checks = 0
        self._diversity_post_recovery_bad_checks = 0
        self._diversity_recovery_applied = False
        self.diversity_events: List[Dict[str, Any]] = []
        self._stop_reason: Optional[str] = None
        self._stop_detail: Optional[str] = None
        self._last_completed_generation = -1
        self.evolution_outcome: Dict[str, Any] = {
            'status': 'PENDING',
            'reason': None,
            'generations_completed': 0,
            'last_generation': None,
            'common_panel_curve': [],
            'valid_evaluations': 0,
            'failed_evaluations': 0,
            'common_panel_valid_evaluations': 0,
            'common_panel_failed_evaluations': 0,
            'diversity_events': [],
            'incumbent_score': self.common_panel_incumbent_score,
            'best_score': self.common_panel_incumbent_score,
            'evaluation_history': [],
            'detail': None,
        }

    def set_common_panel_incumbent(self, score: Optional[float]) -> None:
        """Set the prior lane-global score before the first replay.

        The config contract is
        ``generic_island_model.common_panel_replay.incumbent_score``.  This
        setter is intentionally also exposed for orchestration bridges which
        build an engine before materialising the final derived config.
        """

        if self.common_panel_replay_history:
            raise RuntimeError(
                "common-panel incumbent cannot change after replay has started"
            )
        if score is not None:
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise ValueError("common-panel incumbent must be null or finite")
            score = float(score)
        self.common_panel_incumbent_score = score
        self._common_panel_best = score
        self._common_panel_no_improvement_checks = 0
        self.evolution_outcome['incumbent_score'] = score
        self.evolution_outcome['best_score'] = score

    def set_strict_initial_seeds(self, seeds: List[Individual]) -> None:
        """Inject the same immutable parent set into every island.

        The per-island ``GeneticAlgorithm`` performs the canonical deep-copy
        and genome validation.  Keeping this hook at the island coordinator
        prevents V2 exploit/replication attempts from silently falling back to
        a scratch population.
        """

        if len(seeds) > min(
            island.population_size for island in self.island_configs
        ):
            raise ValueError("strict initial seeds exceed the smallest island population")
        if any(not isinstance(seed, Individual) for seed in seeds):
            raise TypeError("strict initial seeds must contain only Individual values")
        for island in self.island_configs:
            archive_count = len(self._archive_initial_seeds.get(island.name, []))
            if len(seeds) + archive_count > island.population_size:
                raise ValueError(
                    f'strict and archive seeds exceed population size for {island.name}'
                )
        self._strict_initial_seeds = list(seeds)

    def set_archive_initial_seeds(self, seeds: List[Individual]) -> None:
        """Distribute distinct archive champions only to configured islands."""

        if not self.archive_seeding_enabled:
            if seeds:
                raise ValueError("archive seeding is not enabled")
            self._archive_initial_seeds = {}
            return
        if not self.archive_seeding_island_names:
            raise ValueError("archive seeding requires at least one island name")
        if any(not isinstance(seed, Individual) for seed in seeds):
            raise TypeError("archive seeds must contain only Individual values")

        unique: List[Individual] = []
        seen: set[str] = set()
        for seed in seeds:
            fingerprint = self._gene_hash(seed)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(seed)

        assignments = {
            island_name: [] for island_name in self.archive_seeding_island_names
        }
        capacity = (
            len(self.archive_seeding_island_names)
            * self.archive_seeding_max_per_island
        )
        for index, seed in enumerate(unique[:capacity]):
            island_name = self.archive_seeding_island_names[
                index % len(self.archive_seeding_island_names)
            ]
            assignments[island_name].append(seed)

        island_sizes = {
            island.name: island.population_size for island in self.island_configs
        }
        for island_name, assigned in assignments.items():
            if len(assigned) + len(self._strict_initial_seeds) > island_sizes[island_name]:
                raise ValueError(
                    f'archive seeds exceed population size for {island_name}'
                )
        self._archive_initial_seeds = assignments

    @staticmethod
    def _capture_random_state() -> Dict[str, Any]:
        """Capture process RNGs used by legacy genome operators."""

        state: Dict[str, Any] = {'python': random.getstate()}
        try:
            import numpy as np

            state['numpy'] = np.random.get_state()
        except ImportError:
            pass
        return state

    @staticmethod
    def _restore_random_state(state: Dict[str, Any]) -> None:
        random.setstate(state['python'])
        if 'numpy' in state:
            try:
                import numpy as np

                np.random.set_state(state['numpy'])
            except ImportError:
                pass

    @staticmethod
    def _random_state_to_json(state: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {'python': state['python']}
        if 'numpy' in state:
            numpy_state = state['numpy']
            payload['numpy'] = (
                numpy_state[0],
                numpy_state[1].tolist(),
                int(numpy_state[2]),
                int(numpy_state[3]),
                float(numpy_state[4]),
            )
        return payload

    @staticmethod
    def _random_state_from_json(state: Dict[str, Any]) -> Dict[str, Any]:
        python_state = state['python']
        restored: Dict[str, Any] = {
            'python': (
                python_state[0], tuple(python_state[1]), python_state[2]
            )
        }
        if 'numpy' in state:
            try:
                import numpy as np

                numpy_state = state['numpy']
                restored['numpy'] = (
                    numpy_state[0],
                    np.array(numpy_state[1], dtype=np.uint32),
                    int(numpy_state[2]),
                    int(numpy_state[3]),
                    float(numpy_state[4]),
                )
            except ImportError:
                pass
        return restored

    def _initial_random_state_for_island(self, island_name: str) -> Dict[str, Any]:
        island_config = next(
            item for item in self.island_configs if item.name == island_name
        )
        state: Dict[str, Any] = {
            'python': random.Random(island_config.seed).getstate(),
        }
        try:
            import numpy as np

            state['numpy'] = np.random.RandomState(island_config.seed).get_state()
        except ImportError:
            pass
        return state

    @contextmanager
    def _island_random_scope(self, island_name: str):
        """Give an island an independent deterministic legacy RNG stream.

        A number of old operators still use module-level ``random`` and
        ``numpy.random``.  Swapping their state under one lock is the only
        safe way to isolate islands without rewriting every operator.  This
        intentionally serializes ``parallel_islands`` around GA work; the
        shared process evaluation pool still provides the expensive
        backtesting parallelism.
        """

        with self._rng_lock:
            caller_state = self._capture_random_state()
            island_state = self._island_random_states.get(island_name)
            if island_state is None:
                island_state = self._initial_random_state_for_island(island_name)
            self._restore_random_state(island_state)
            try:
                yield
            finally:
                self._island_random_states[island_name] = self._capture_random_state()
                self._restore_random_state(caller_state)

    @staticmethod
    def _configure_behavioral_distance(ga: GeneticAlgorithm) -> float:
        """Wire the configured behavioral blend into the implementation module.

        ``core.population`` is a compatibility shim.  Assigning its module
        global (as the standard GA historically did) does not update the
        global referenced by functions defined in ``engine.population``.
        Generic-island evaluation therefore binds both explicitly.
        """

        weight = float(
            ga.config.get('genetic_algorithm', {}).get(
                'behavioral_distance_weight', 0.0
            )
        )
        import genetic_algorithm.core.population as population_shim
        import genetic_algorithm.engine.population as population_impl

        population_shim._BEHAVIORAL_DISTANCE_WEIGHT = weight
        population_impl._BEHAVIORAL_DISTANCE_WEIGHT = weight
        return weight

    def _request_stop(self, reason: str, detail: str) -> None:
        """Set one terminal reason; technical invalidity outranks economics."""

        if self._stop_reason is None or (
            reason == OUTCOME_TECHNICAL_INVALID
            and self._stop_reason != OUTCOME_TECHNICAL_INVALID
        ):
            self._stop_reason = reason
            self._stop_detail = detail
        self._shutdown_requested = True

    @staticmethod
    def _is_failed_evidence(individual: Individual) -> bool:
        return bool(individual.metrics.get('error')) or (
            individual.fitness_evidence == FITNESS_EVIDENCE_BACKTEST_FAILED
        )

    def _is_valid_evidence(self, individual: Individual) -> bool:
        if (
            not individual.has_measured_fitness
            or self._is_failed_evidence(individual)
        ):
            return False
        raw_config = self.config.get('raw_multipair_score', {})
        if not raw_config.get('enabled', False):
            return True
        expected_pairs = {
            *raw_config.get('development_pairs', []),
            *raw_config.get('validation_pairs', []),
        }
        raw_pair_metrics = individual.metrics.get('raw_pair_metrics')
        return (
            individual.metrics.get('raw_multipair_score_version')
            == 'raw-multipair-score-v1'
            and individual.metrics.get('raw_multipair_status') == 'VALID'
            and isinstance(raw_pair_metrics, dict)
            and set(raw_pair_metrics) == expected_pairs
            and len(expected_pairs) == 6
        )

    def _record_generation_evidence(self, generation: int) -> bool:
        """Record evaluated evidence and fail fast when a generation has none."""

        individuals = [
            individual
            for island_name in sorted(self._evaluated_generation_populations)
            for individual in self._evaluated_generation_populations[island_name]
        ]
        valid = sum(self._is_valid_evidence(item) for item in individuals)
        failed = sum(self._is_failed_evidence(item) for item in individuals)
        record = {
            'generation': generation,
            'valid_evaluations': valid,
            'failed_evaluations': failed,
            'population_size': len(individuals),
        }
        self.evolution_outcome['evaluation_history'].append(record)
        self.evolution_outcome['valid_evaluations'] += valid
        self.evolution_outcome['failed_evaluations'] += failed
        if valid == 0:
            self._request_stop(
                OUTCOME_TECHNICAL_INVALID,
                f'generation {generation + 1} produced zero valid backtest evidence',
            )
            return False
        return True

    def _finalize_evolution_outcome(self) -> Dict[str, Any]:
        reason = self._stop_reason or OUTCOME_COMPLETED_BUDGET
        status = (
            'FAILED'
            if reason == OUTCOME_TECHNICAL_INVALID
            else 'EARLY_STOPPED'
            if reason in {OUTCOME_PLATEAU, OUTCOME_DIVERSITY_COLLAPSE}
            else 'COMPLETED'
        )
        self.evolution_outcome.update({
            'status': status,
            'reason': reason,
            'generations_completed': max(0, self._last_completed_generation + 1),
            'last_generation': (
                self._last_completed_generation
                if self._last_completed_generation >= 0
                else None
            ),
            'common_panel_curve': [
                {
                    key: event.get(key)
                    for key in (
                        'generation', 'best_fitness', 'median_fitness',
                        'improved', 'no_improvement_checks',
                    )
                }
                for event in self.common_panel_replay_history
            ],
            'diversity_events': copy.deepcopy(self.diversity_events),
            'incumbent_score': self.common_panel_incumbent_score,
            'best_score': self._common_panel_best,
            'detail': self._stop_detail,
        })
        self._persist_evolution_outcome()
        return copy.deepcopy(self.evolution_outcome)

    def get_evolution_outcome(self) -> Dict[str, Any]:
        """Return a JSON-safe snapshot for orchestration code."""

        return copy.deepcopy(self.evolution_outcome)

    def _persist_evolution_outcome(self) -> Path:
        """Atomically persist the canonical outcome and its SHA-256 digest."""

        path = self.checkpoint_dir.parent / 'evolution_outcome.json'
        payload = json.dumps(
            self.evolution_outcome,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False,
        ).encode('utf-8')
        digest = hashlib.sha256(payload).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        outcome_tmp = path.with_name(path.name + '.tmp')
        digest_path = path.with_name(path.name + '.sha256')
        digest_tmp = digest_path.with_name(digest_path.name + '.tmp')
        with outcome_tmp.open('wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(outcome_tmp, path)
        with digest_tmp.open('w', encoding='ascii') as handle:
            handle.write(digest + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(digest_tmp, digest_path)
        return path

    # ------------------------------------------------------------------
    # Island configuration building
    # ------------------------------------------------------------------

    def _parse_explicit_islands(
        self, island_defs: List[Dict[str, Any]],
    ) -> List[GenericIslandConfig]:
        """Parse explicitly defined island configs from YAML."""
        configs = []
        for i, idef in enumerate(island_defs):
            configs.append(GenericIslandConfig(
                name=idef.get('name', f'island_{i}'),
                population_size=idef.get(
                    'population_size', self.population_per_island,
                ),
                generations=idef.get('generations'),
                seed=idef.get('seed', self.base_seed + i),
                indicator_pool=idef.get('indicator_pool'),
                pairs=idef.get('pairs'),
                walk_forward_enabled=idef.get('walk_forward_enabled'),
                extra_config=idef.get('extra_config', {}),
            ))
        return configs

    def _auto_generate_islands(self) -> List[GenericIslandConfig]:
        """
        Auto-generate island configs with rotated seeds, optionally
        split indicator pools and rotated pair subsets.
        """
        indicator_pools = self._split_indicator_pools() if self.use_indicator_pools else None
        pair_subsets = self._rotate_pairs() if self.pair_rotation else None

        configs = []
        for i in range(self.num_islands):
            seed = (self.base_seed + i) if self.rotate_seeds else self.base_seed

            pool = None
            if indicator_pools:
                pool = indicator_pools[i % len(indicator_pools)] if i < len(indicator_pools) else indicator_pools[i % len(indicator_pools)]

            pairs = None
            if pair_subsets:
                pairs = pair_subsets[i % len(pair_subsets)]

            name_suffix = ''
            if pool:
                # Name after the primary family
                primary_family = self._identify_primary_family(pool)
                name_suffix = f'_{primary_family}'

            configs.append(GenericIslandConfig(
                name=f'island_{i}{name_suffix}',
                population_size=self.population_per_island,
                seed=seed,
                indicator_pool=pool,
                pairs=pairs,
            ))

        self.logger.info(
            "Auto-generated %d island configs (seeds=%s, indicator_pools=%s, pair_rotation=%s)",
            len(configs), self.rotate_seeds, self.use_indicator_pools, self.pair_rotation,
        )
        return configs

    def _split_indicator_pools(self) -> List[List[str]]:
        """
        Split indicators into overlapping pools by family.

        Each pool gets 2-3 families as its core, plus a random selection
        from other families for overlap.
        """
        families = list(INDICATOR_FAMILIES.keys())
        pools = []

        # Create pools — at least num_islands pools to avoid identical specialization
        n_pools = max(len(families), getattr(self, 'num_islands', len(families)))
        for i in range(n_pools):
            # Core: 2 adjacent families (wrap around with modulo)
            core_families = [families[i % len(families)], families[(i + 1) % len(families)]]
            core_indicators = []
            for fam in core_families:
                core_indicators.extend(INDICATOR_FAMILIES[fam])

            # Overlap: random picks from other families
            other_indicators = [
                ind for fam, inds in INDICATOR_FAMILIES.items()
                if fam not in core_families
                for ind in inds
            ]
            overlap_count = max(1, int(len(other_indicators) * self.indicator_overlap))
            rng = random.Random(self.base_seed + i)
            overlap_picks = rng.sample(
                other_indicators, min(overlap_count, len(other_indicators)),
            )

            pool = sorted(set(core_indicators + overlap_picks))
            pools.append(pool)

        return pools

    def _rotate_pairs(self) -> List[List[str]]:
        """
        Create rotated pair subsets from the base config's pair list.
        """
        base_pairs = self.config.get('backtesting', {}).get('pairs', ['BTC/USDT'])
        if len(base_pairs) <= self.pair_subset_size:
            return [base_pairs]

        subsets = []
        for i in range(len(base_pairs)):
            subset = []
            for j in range(self.pair_subset_size):
                subset.append(base_pairs[(i + j) % len(base_pairs)])
            subsets.append(subset)
        return subsets

    @staticmethod
    def _identify_primary_family(indicators: List[str]) -> str:
        """Determine which indicator family dominates a pool."""
        best_family = 'mixed'
        best_count = 0
        for family, members in INDICATOR_FAMILIES.items():
            count = sum(1 for ind in indicators if ind in members)
            if count > best_count:
                best_count = count
                best_family = family
        return best_family

    # ------------------------------------------------------------------
    # Build sub-GA for an island
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursively merge *override* into *base* in-place.

        Dicts are merged recursively so that a partial override like
        ``{'genetic_algorithm': {'selection_method': 'tournament'}}`` only
        touches the ``selection_method`` key rather than replacing the whole
        ``genetic_algorithm`` section.  All other types are replaced directly.
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                GenericIslandModelEvolution._deep_merge(base[key], value)
            else:
                base[key] = value

    def _build_island_config(self, ic: GenericIslandConfig) -> Dict[str, Any]:
        """
        Build a complete config dict for one island's GA, derived
        from the base config with island-specific overrides.
        """
        cfg = copy.deepcopy(self.config)

        # Override population size and related params
        elite_size, random_immigrants = derive_island_population_slots(
            ic.population_size
        )
        cfg['genetic_algorithm']['population_size'] = ic.population_size
        cfg['genetic_algorithm']['elite_size'] = elite_size
        cfg['genetic_algorithm']['random_immigrants'] = random_immigrants

        # GeneticAlgorithm consumes ``random_seed``.  Writing the historical
        # ``seed`` alias here made every island inherit the same base seed and
        # rendered seed rotation ineffective.
        cfg['genetic_algorithm']['random_seed'] = ic.seed
        cfg['genetic_algorithm'].pop('seed', None)

        # Set generations (per-island override or global)
        island_gens = ic.generations if ic.generations is not None else self.generations
        cfg['genetic_algorithm']['generations'] = island_gens

        # Walk-forward: configurable per island (unlike old model)
        wf_enabled = ic.walk_forward_enabled
        if wf_enabled is None:
            wf_enabled = self.island_walk_forward
        if wf_enabled is not None:
            cfg.setdefault('walk_forward', {})['enabled'] = wf_enabled

        # Indicator pool restriction (if specified)
        if ic.indicator_pool is not None:
            cfg.setdefault('indicators', {})['available'] = list(
                ic.indicator_pool
            )

        # Pair subset (if specified)
        if ic.pairs is not None:
            cfg.setdefault('backtesting', {})['pairs'] = list(ic.pairs)

        # Apply extra config overrides using deep merge so that partial
        # overrides (e.g. only genetic_algorithm.selection_method) don't
        # wipe sibling keys in the same section.
        self._deep_merge(cfg, ic.extra_config)

        # These invariants must be applied *after* extra_config so an island
        # override cannot accidentally recurse into another island model or
        # spawn its own monitor/process pool.  Preserve all sibling settings
        # in the resolved contract.
        cfg.setdefault('island_model', {})['enabled'] = False
        cfg.setdefault('generic_island_model', {})['enabled'] = False
        cfg.setdefault('terminal_monitor', {})['enabled'] = False
        if (
            cfg.get('safety_profile', {}).get('name')
            == 'automation_island_v2'
        ):
            # The coordinator owns the outer automation contract.  A child is
            # a standard GA with pair validation and must not satisfy the
            # top-level requirement that the island coordinator is enabled.
            cfg['safety_profile']['name'] = 'automation_island_child_v2'
        elif (
            cfg.get('safety_profile', {}).get('name')
            == 'hardcore_multipair_v1'
        ):
            # The immutable parent contract has already validated the twelve
            # explicit specialists and the complete six-pair panel.  A child
            # is a standard GA evaluated by the coordinator's shared pool, so
            # it must keep every raw-multipair safety invariant while
            # explicitly disabling recursive island orchestration.
            cfg['safety_profile']['name'] = 'hardcore_multipair_child_v1'

        # Disable per-island parallel evaluation — the island model
        # creates ONE shared ParallelEvaluator to avoid spawning
        # N_islands × N_workers processes (OOM on 16 GB systems).
        cfg.setdefault('parallel_evaluation', {})['enabled'] = False

        return cfg

    def _create_island_ga(self, ic: GenericIslandConfig) -> GeneticAlgorithm:
        """
        Create a GeneticAlgorithm instance for one island.

        Writes a temporary YAML config and instantiates a GA from it.
        Unlike the regime model, no regime segments are injected —
        each island gets the full dataset.
        """
        island_cfg = self._build_island_config(ic)

        # Write temp config
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False,
            prefix=f'generic_island_{ic.name}_',
        ) as tmp:
            yaml.dump(island_cfg, tmp)
            tmp_path = tmp.name

        try:
            ga = GeneticAlgorithm(
                config_path=tmp_path,
                visualize=False,
                interactive=False,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Share the hall of fame
        ga.hall_of_fame = self.hall_of_fame
        ga._evaluation_island = ic.name

        return ga

    # ------------------------------------------------------------------
    # Main evolution interface
    # ------------------------------------------------------------------

    def evolve(self) -> Dict[str, List[Individual]]:
        """
        Run the full generic island model evolution.

        Returns:
            Dict mapping island name -> list of top individuals.
        """
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        _sigusr1_available = hasattr(signal, 'SIGUSR1')
        original_sigusr1 = signal.getsignal(signal.SIGUSR1) if _sigusr1_available else None

        def _shutdown(signum, frame):
            if self._shutdown_requested:
                signal.signal(signal.SIGINT, original_sigint)
                raise KeyboardInterrupt
            self._shutdown_requested = True
            self.logger.warning("[SHUTDOWN] Graceful shutdown requested")

        def _checkpoint_now(signum, frame):
            self._checkpoint_requested = True
            self.logger.info("[CHECKPOINT] Manual checkpoint requested via SIGUSR1")

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        if _sigusr1_available:
            signal.signal(signal.SIGUSR1, _checkpoint_now)

        try:
            return self._evolve_inner()
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            if _sigusr1_available:
                signal.signal(signal.SIGUSR1, original_sigusr1)

    def _evolve_inner(self) -> Dict[str, List[Individual]]:
        start_time = time.time()

        # Terminal monitor
        from genetic_algorithm.monitor import create_monitor
        monitor_cfg = copy.deepcopy(self.config)
        monitor_cfg.setdefault('terminal_monitor', {})['enabled'] = (
            self.config.get('terminal_monitor', {}).get('enabled', True)
        )
        self.monitor = create_monitor(monitor_cfg)
        self.monitor.start(monitor_cfg)

        self.logger.info("=" * 70)
        self.logger.info("GENERIC ISLAND MODEL EVOLUTION STARTING")
        self.logger.info("=" * 70)
        self.logger.info("  Islands: %d", len(self.island_configs))
        self.logger.info("  Pop/island: %d", self.population_per_island)
        self.logger.info("  Generations: %d", self.generations)
        self.logger.info("  Migration: topology=%s interval=%d count=%d",
                         self.migration.topology, self.migration.interval,
                         self.migration.count)
        if self.migration.merge_rounds:
            self.logger.info("  Merge rounds: every %d gens", self.migration.merge_interval)
        self.logger.info("  Parallel: %s", self.parallel_islands)
        self.logger.info("=" * 70)

        # Create a SINGLE shared parallel evaluator for all islands.
        # Previously each island created its own ProcessPoolExecutor,
        # leading to N_islands × N_workers processes (~80 on 20 islands)
        # and OOM crashes on 16 GB systems.
        self._shared_parallel_evaluator = None
        parallel_config = self.config.get('parallel_evaluation', {})
        if parallel_config.get('enabled', False):
            from genetic_algorithm.evaluation.parallel import (
                ParallelEvaluator,
                is_parallel_available,
            )
            if is_parallel_available():
                self._shared_parallel_evaluator = ParallelEvaluator(
                    self.config,
                    num_workers=parallel_config.get('num_workers'),
                )
                self.logger.info(
                    "[ISLAND] Shared parallel evaluator: %d workers "
                    "(single pool for all %d islands)",
                    self._shared_parallel_evaluator.num_workers,
                    len(self.island_configs),
                )

        try:
            # ═══════════════════════════════════════════════════════════
            # PHASE 1: CREATE ISLANDS
            # ═══════════════════════════════════════════════════════════
            self._phase1_create_islands()

            self._enforce_search_panel_contract()

            # ── Try resuming from checkpoint ──
            resume_gen = self.load_island_checkpoint()
            if resume_gen is not None:
                self.logger.info(
                    "[ISLAND] Resuming island model from generation %d",
                    resume_gen,
                )
            else:
                resume_gen = 0

            # ═══════════════════════════════════════════════════════════
            # PHASE 2: EVOLUTION
            # ═══════════════════════════════════════════════════════════
            results = self._phase2_evolve(start_generation=resume_gen)

            # ═══════════════════════════════════════════════════════════
            # PHASE 3: REPORTING
            # ═══════════════════════════════════════════════════════════
            total_elapsed = time.time() - start_time
            self._phase3_report(results, total_elapsed)
            extract_island_finalists(
                results,
                expected_islands=[
                    island_config.name for island_config in self.island_configs
                ],
                require_finalists=True,
            )

            self.monitor.on_evolution_complete({
                'total_time': total_elapsed,
                'generations': self.evolution_outcome['generations_completed'],
                'islands': len(self.islands),
                'migrations': len(self.migration_history),
                'stop_reason': self.evolution_outcome['reason'],
            })

            return results
        except Exception as exc:
            self._request_stop(
                OUTCOME_TECHNICAL_INVALID,
                f'unhandled generic-island error: {type(exc).__name__}: {exc}',
            )
            try:
                self._finalize_evolution_outcome()
            except Exception:
                self.logger.exception("[OUTCOME] Failed to persist technical outcome")
            raise
        finally:
            # Always clean up the shared pool (crash, interrupt, or normal exit)
            if self._shared_parallel_evaluator is not None:
                self.logger.info("[ISLAND] Shutting down shared parallel evaluator")
                self._shared_parallel_evaluator.shutdown()
                self._shared_parallel_evaluator = None

    # ------------------------------------------------------------------
    # Phase 1: Create Islands
    # ------------------------------------------------------------------

    def _phase1_create_islands(self):
        """Create GA instances and initialize populations for all islands."""
        phase_start = time.time()
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  PHASE 1: CREATING %d ISLANDS", len(self.island_configs))
        self.logger.info("═" * 70)

        for ic in self.island_configs:
            ga = self._create_island_ga(ic)
            island_seeds = [
                *self._strict_initial_seeds,
                *self._archive_initial_seeds.get(ic.name, []),
            ]
            ga.set_strict_initial_seeds(island_seeds)
            self.islands[ic.name] = ga
            self.island_stats[ic.name] = GenericIslandStats(name=ic.name)
            self.generation_stats[ic.name] = []

            pop = ga.initialize_population()
            self.island_populations[ic.name] = pop
            # GeneticAlgorithm seeds the legacy global RNGs from ``ic.seed``;
            # capture the post-initialization position before another island
            # changes them.
            self._island_random_states[ic.name] = self._capture_random_state()

            pool_desc = (
                f"pool={len(ic.indicator_pool)} indicators"
                if ic.indicator_pool else "all indicators"
            )
            pairs_desc = (
                f"pairs={ic.pairs}" if ic.pairs else "all pairs"
            )
            self.logger.info(
                "  Island %-25s: pop=%d, seed=%d, %s, %s",
                ic.name, len(pop.individuals), ic.seed, pool_desc, pairs_desc,
            )

        phase_elapsed = time.time() - phase_start
        self.logger.info(
            "  Phase 1 complete: %.1f seconds. %d islands created.",
            phase_elapsed, len(self.islands),
        )

    def _enforce_search_panel_contract(self) -> bool:
        """Disable shared execution when island evaluator semantics differ."""
        search_panel_ids = {
            ga.evaluation_panel.panel_id for ga in self.islands.values()
        }
        self._search_panels_comparable = len(search_panel_ids) == 1
        if not self._search_panels_comparable and self._shared_parallel_evaluator:
            # One evaluator cannot truthfully execute several pair/cost/
            # fitness configs. Fall back to each island's own sequential
            # evaluator rather than stamp base-config results as local.
            self.logger.warning(
                "[ISLAND] Distinct evaluation panels detected; shared "
                "parallel evaluator disabled for provenance correctness"
            )
            self._shared_parallel_evaluator.shutdown()
            self._shared_parallel_evaluator = None
        return self._search_panels_comparable

    # ------------------------------------------------------------------
    # Phase 2: Evolution
    # ------------------------------------------------------------------

    def _phase2_evolve(self, start_generation: int = 0) -> Dict[str, List[Individual]]:
        """
        Run the generation loop with periodic migration across all islands.
        """
        phase_start = time.time()
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  PHASE 2: EVOLVING %d ISLANDS × %d GENERATIONS%s%s",
                         len(self.islands), self.generations,
                         " (PARALLEL)" if self.parallel_islands else "",
                         f" (resuming from gen {start_generation})" if start_generation > 0 else "")
        max_runtime_minutes = self.config.get('genetic_algorithm', {}).get('max_runtime_minutes', None)
        if max_runtime_minutes:
            self.logger.info("  Max runtime: %.0f minutes", max_runtime_minutes)
        self.logger.info("═" * 70)

        overall_best_individual = None
        self.evolution_outcome['status'] = 'RUNNING'
        self._last_completed_generation = start_generation - 1

        for gen in range(start_generation, self.generations):
            if self._shutdown_requested:
                self.logger.info("[SHUTDOWN] Stopping at generation %d", gen)
                break

            gen_start = time.time()
            self.logger.info("")
            self.logger.info("─" * 70)
            self.logger.info("GENERATION %d/%d", gen + 1, self.generations)
            self.logger.info("─" * 70)

            self.monitor.on_generation_start(gen, self.generations)

            # Evolve each island for one generation
            if self.parallel_islands and len(self.island_configs) > 1:
                self._evolve_all_islands_parallel(gen)
            else:
                for ic in self.island_configs:
                    self._evolve_island_one_generation(
                        self.islands[ic.name],
                        self.island_populations[ic.name],
                        ic.name,
                        gen,
                    )

            self._last_completed_generation = gen
            generation_has_evidence = self._record_generation_evidence(gen)
            if generation_has_evidence:
                self._maybe_replay_common_panel(gen)

            self._persist_generation_trace(gen)

            if generation_has_evidence and self._stop_reason is None:
                self._maybe_handle_diversity(gen)

            # Migration (skip generation 0 — populations not yet evaluated)
            if (
                not self._shutdown_requested
                and gen > 0
                and self.migration.interval > 0
                and (gen + 1) % self.migration.interval == 0
            ):
                self._migrate(gen)

            # External migration (cross-machine strategy exchange)
            if (
                not self._shutdown_requested
                and self.external_migration_enabled
                and gen > 0
                and self.external_migration_interval > 0
                and (gen + 1) % self.external_migration_interval == 0
            ):
                self._load_external_migrants(gen)
                self._export_for_external_migration(gen)

            # Merge rounds (global pool-and-redistribute)
            if (
                not self._shutdown_requested
                and gen > 0
                and self.migration.merge_rounds
                and self.migration.merge_interval > 0
                and (gen + 1) % self.migration.merge_interval == 0
            ):
                self._merge_round(gen)

            # ── Checkpoint save ──
            should_checkpoint = (
                self.checkpoint_interval > 0
                and (gen + 1) % self.checkpoint_interval == 0
            ) or self._checkpoint_requested
            if should_checkpoint:
                try:
                    self.save_island_checkpoint(gen)
                    self.export_best_strategies(gen)
                    self._checkpoint_requested = False
                except Exception as ckpt_err:
                    self.logger.error(
                        "[CHECKPOINT] Failed to save at gen %d: %s", gen, ckpt_err,
                    )

            # Log generation summary
            gen_elapsed = time.time() - gen_start
            self._log_generation_summary(gen, gen_elapsed)

            # A global best/average exists only when every island uses the
            # same exact evaluation panel.
            if self._search_panels_comparable:
                for ic in self.island_configs:
                    measured = [
                        individual
                        for individual in self._evaluated_generation_populations.get(
                            ic.name, []
                        )
                        if self._is_valid_evidence(individual)
                    ]
                    if measured:
                        cand = max(measured, key=lambda item: float(item.raw_fitness))
                        if (
                            overall_best_individual is None
                            or float(cand.raw_fitness)
                            > float(overall_best_individual.raw_fitness)
                        ):
                            overall_best_individual = cand
                            self.monitor.on_new_best(cand)
                agg_best = max(
                    (ist.best_fitness for ist in self.island_stats.values()),
                    default=0,
                )
                agg_avg = (
                    sum(ist.avg_fitness for ist in self.island_stats.values())
                    / max(len(self.island_stats), 1)
                )
            else:
                overall_best_individual = None
                agg_best = 0.0
                agg_avg = 0.0
            _agg_stats = _AggregateStats(
                best_fitness=agg_best,
                avg_fitness=agg_avg,
                worst_fitness=0,
                generation=gen,
            )
            self.monitor.on_generation_end(
                gen=gen,
                stats=_agg_stats,
                timing=None,
                best_individual=overall_best_individual,
                extras={
                    'island_count': len(self.islands),
                    'migrations': len(self.migration_history),
                    'global_fitness_comparable': self._search_panels_comparable,
                },
            )

            # Hard runtime cap: stop cleanly after max_runtime_minutes
            if max_runtime_minutes is not None:
                total_elapsed_min = (time.time() - phase_start) / 60.0
                if total_elapsed_min >= max_runtime_minutes:
                    self.logger.info(
                        "[RUNTIME] %.1f min elapsed ≥ limit %.0f min"
                        " — stopping after gen %d/%d",
                        total_elapsed_min, max_runtime_minutes,
                        gen + 1, self.generations,
                    )
                    break

            if self._shutdown_requested:
                break

        # Collect results: pool top-5 from every island, deduplicate
        results = self._collect_final_results()

        self._finalize_evolution_outcome()

        # Final checkpoint at end of evolution
        if self._last_completed_generation >= 0:
            try:
                self.save_island_checkpoint(self._last_completed_generation)
                self.export_best_strategies(self._last_completed_generation)
            except Exception as e:
                self.logger.error("[CHECKPOINT] Final save failed: %s", e)

        phase_elapsed = time.time() - phase_start
        self.logger.info("")
        self.logger.info(
            "  Phase 2 complete: %.1f seconds (%.1f minutes). "
            "%d migrations performed.",
            phase_elapsed, phase_elapsed / 60, len(self.migration_history),
        )

        return results

    def _persist_generation_trace(self, generation: int) -> None:
        """Persist comparable diagnostics before migration mutates populations."""

        for island_config in self.island_configs:
            island_name = island_config.name
            evaluated_population = self._evaluated_generation_populations.get(
                island_name
            )
            stats_history = self.generation_stats.get(island_name, [])
            if not stats_history or evaluated_population is None:
                raise RuntimeError(
                    f"missing generation statistics for {island_name} generation {generation}"
                )
            panel_id = self.islands[island_name].evaluation_panel.panel_id
            self.generation_trace.append(
                generation=generation,
                island_name=island_name,
                panel_id=panel_id,
                individuals=evaluated_population,
                stats=stats_history[-1],
            )

    # ------------------------------------------------------------------
    # Single-island evolution (one generation)
    # ------------------------------------------------------------------

    def _evolve_island_one_generation(
        self,
        ga: GeneticAlgorithm,
        population: Population,
        island_name: str,
        generation: int,
    ):
        with self._island_random_scope(island_name):
            self._configure_behavioral_distance(ga)
            return self._evolve_island_one_generation_unscoped(
                ga, population, island_name, generation
            )

    def _evolve_island_one_generation_unscoped(
        self,
        ga: GeneticAlgorithm,
        population: Population,
        island_name: str,
        generation: int,
    ):
        """
        Run one generation of evolution on a single island.

        Manually executes the core GA steps (evaluate -> select ->
        crossover/mutate -> next gen) without calling ga.evolve().
        """
        ga.current_generation = generation

        # Reset LLM generation budget if applicable
        if ga.llm_enabled and ga.strategy_designer:
            ga.strategy_designer.reset_generation_budget()

        # Inject shared parallel evaluator so all islands share ONE
        # worker pool instead of each spawning its own.
        if self._shared_parallel_evaluator is not None:
            ga.parallel_evaluator = self._shared_parallel_evaluator
            ga.parallel_enabled = True

        # Step 1: Evaluate fitness
        ga.evaluate_population(population)

        # Some legacy/third-party workers reported ``success=True`` while
        # returning ``metrics.error``.  The error marker is authoritative for
        # production evidence and must not leak into measured elites.
        for individual in population.individuals:
            if individual.metrics.get('error'):
                individual.fitness_evidence = FITNESS_EVIDENCE_BACKTEST_FAILED
                individual.metrics['fitness_evidence'] = (
                    FITNESS_EVIDENCE_BACKTEST_FAILED
                )

        # Keep the complete evaluated generation for trace/evidence accounting,
        # but never let technical failures compete with signed raw scores.  A
        # failed worker historically returned fitness=0 which can outrank a
        # perfectly valid (but negative) raw-multipair candidate and would then
        # leak into elitism and parent selection.
        evaluated_individuals = list(population.individuals)
        raw_multipair_enabled = bool(
            self.config.get('raw_multipair_score', {}).get('enabled', False)
        )
        reproduction_population = population
        if raw_multipair_enabled:
            reproduction_population = Population(
                size=population.size,
                generation=population.generation,
            )
            reproduction_population.individuals = [
                individual
                for individual in evaluated_individuals
                if self._is_valid_evidence(individual)
            ]

        # Step 1b: Feed evaluated population to surrogate model so it can
        # learn and (from the next generation onward) pre-filter candidates.
        # evaluate_population() is called directly here instead of via
        # ga.evolve(), so we must drive the surrogate manually.
        if hasattr(ga, '_surrogate') and ga._surrogate is not None and ga._surrogate.enabled:
            try:
                ga._surrogate.add_training_data(reproduction_population)
                ga._surrogate.maybe_retrain()
            except Exception as _surr_exc:
                self.logger.debug(
                    "[SURROGATE] Training update failed for island %s gen %d: %s",
                    island_name, generation, _surr_exc,
                )

        # Step 2: Fitness sharing
        if ga.fitness_sharing and len(reproduction_population.individuals) >= 2:
            distance_matrix = calculate_pairwise_distances(
                list(reproduction_population.individuals),
            )
            apply_fitness_sharing(
                reproduction_population, sigma_share=ga.sharing_radius,
                distance_matrix=distance_matrix,
            )

        # Step 2b: Strategy culling (after evaluation & sharing, before HoF)
        if self.culler.enabled and reproduction_population.individuals:
            self.culler.cull_population(
                reproduction_population,
                generation=generation,
                elite_size=ga.elite_size,
            )

        # Step 3: Get stats
        stats = reproduction_population.get_stats()
        # The trace still represents the complete evaluated generation; only
        # the fitness aggregates are restricted to valid evidence.
        stats.size = len(evaluated_individuals)

        # Update best
        measured_by_raw_score = sorted(
            [
                individual
                for individual in reproduction_population.individuals
                if self._is_valid_evidence(individual)
                and individual.raw_fitness is not None
            ],
            key=lambda individual: float(individual.raw_fitness),
            reverse=True,
        )
        best = measured_by_raw_score[:1]
        if best:
            best_ind = best[0]
            with self._stats_lock:
                ist = self.island_stats[island_name]
                if (
                    best_ind.raw_fitness is not None
                    and (
                        ist.generations_completed == 0
                        or best_ind.raw_fitness > ist.best_fitness
                    )
                ):
                    ist.best_fitness = best_ind.raw_fitness
                    profit = best_ind.metrics.get('profit', 0)
                    ist.best_profit = profit
                    trades = best_ind.metrics.get('num_trades', 0)
                    win_rate = best_ind.metrics.get('win_rate', 0.0)
                    self.logger.info(
                        "  [%s] NEW BEST: fitness=%.4f profit=%.2f%% trades=%d win_rate=%.2f",
                        island_name, ist.best_fitness, profit, trades, win_rate,
                    )
                ist.avg_fitness = stats.avg_fitness
                ist.generations_completed = generation + 1
        else:
            with self._stats_lock:
                ist = self.island_stats[island_name]
                ist.avg_fitness = stats.avg_fitness or 0.0
                ist.generations_completed = generation + 1

        # Local island scores are search signals only.  They may come from
        # different pair subsets or scoring overrides and therefore cannot
        # enter the shared Hall of Fame.  HOF update happens once after the
        # common-panel replay in _collect_final_results().

        # Step 5: Log island stats
        self.logger.info(
            "  [%-25s] best=%.4f avg=%.4f diversity=%.4f",
            island_name,
            stats.best_fitness or 0.0,
            stats.avg_fitness or 0.0,
            stats.genetic_diversity or 0,
        )

        # Step 6: Record generation stats
        stats.generation = generation
        with self._stats_lock:
            self.generation_stats[island_name].append(stats)
            self._evaluated_generation_populations[island_name] = copy.deepcopy(
                evaluated_individuals
            )
            self._evaluated_generation_elites[island_name] = copy.deepcopy(
                measured_by_raw_score[:self.common_panel_top_n]
            )

        # Step 6b: Record LLM strategy performance
        if ga.llm_enabled and ga.strategy_designer and ga.strategy_designer.enabled:
            try:
                ga.strategy_designer.record_llm_performance(generation, population)
            except Exception as e:
                self.logger.warning(
                    "LLM performance recording failed for %s gen %d: %s",
                    island_name, generation, e,
                )

        # Step 7: Create next generation
        if generation < self.generations - 1:
            if raw_multipair_enabled and len(reproduction_population.individuals) < 2:
                next_pop = self._fresh_generation_without_self_crossover(
                    ga,
                    reproduction_population,
                    generation=generation,
                    population_size=population.size,
                )
            else:
                next_pop = ga.create_next_generation(reproduction_population)
            self.island_populations[island_name] = next_pop

    @staticmethod
    def _fresh_generation_without_self_crossover(
        ga: GeneticAlgorithm,
        valid_population: Population,
        *,
        generation: int,
        population_size: int,
    ) -> Population:
        """Build a fresh generation when fewer than two valid parents exist."""

        next_generation = generation + 1
        next_pop = Population(size=population_size, generation=next_generation)
        if valid_population.individuals and ga.elite_size > 0:
            elite = copy.deepcopy(valid_population.individuals[0])
            elite.strategy_gene.generation = next_generation
            elite.strategy_gene.individual_id = 0
            next_pop.add_individual(elite)

        while len(next_pop) < population_size:
            individual_id = len(next_pop)
            gene = ga.strategy_generator.generate_random_strategy(
                generation=next_generation,
                individual_id=individual_id,
            )
            immigrant = Individual(strategy_gene=gene)
            immigrant.metrics['origin'] = 'insufficient_valid_parents'
            next_pop.add_individual(immigrant)
        return next_pop

    def _evolve_all_islands_parallel(self, generation: int):
        """Evolve all islands in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(len(self.island_configs), os.cpu_count() or 1)

        first_error = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for ic in self.island_configs:
                future = executor.submit(
                    self._evolve_island_one_generation,
                    self.islands[ic.name],
                    self.island_populations[ic.name],
                    ic.name,
                    generation,
                )
                futures[future] = ic.name

            for future in as_completed(futures):
                island_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(
                        "Island %s failed at generation %d: %s",
                        island_name, generation, e,
                    )
                    if first_error is None:
                        first_error = e
        if first_error is not None:
            raise RuntimeError(
                f'island worker failed in generation {generation + 1}'
            ) from first_error

    # ------------------------------------------------------------------
    # Checkpoint save / load / export
    # ------------------------------------------------------------------

    def save_island_checkpoint(self, generation: int) -> str:
        """
        Save the full island-model state to a single JSON checkpoint.

        Serialises every island population, hall of fame, island stats,
        migration history, and config snapshot so the run can be resumed
        from exactly this point.
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = str(
            self.checkpoint_dir / f"island_checkpoint_gen{generation}_{timestamp}.json"
        )

        # Collect all island populations
        island_data = {}
        for ic in self.island_configs:
            pop = self.island_populations.get(ic.name)
            if pop:
                island_data[ic.name] = {
                    'individuals': [ind.to_dict() for ind in pop.individuals],
                    'generation': pop.generation,
                    'size': pop.size,
                }

        # Collect island stats
        stats_data = {}
        for name, ist in self.island_stats.items():
            stats_data[name] = {
                'best_fitness': ist.best_fitness,
                'best_profit': ist.best_profit,
                'avg_fitness': ist.avg_fitness,
                'generations_completed': ist.generations_completed,
                'migrants_sent': ist.migrants_sent,
                'migrants_received': ist.migrants_received,
            }

        # Hall of fame
        hof_data = []
        try:
            for entry in self.hall_of_fame.entries:
                hof_data.append({
                    'fitness': entry.fitness,
                    'generation_found': getattr(entry, 'generation_found', 0),
                    'strategy_gene_dict': entry.strategy_gene_dict if hasattr(entry, 'strategy_gene_dict') else {},
                    'metrics': getattr(entry, 'metrics', {}),
                    'individual_id': getattr(entry, 'individual_id', 0),
                })
        except Exception as e:
            self.logger.warning("[CHECKPOINT] Could not serialise hall of fame: %s", e)

        provenance = checkpoint_provenance_from_config(
            self.config,
            engine_kind="GENERIC_ISLAND",
            island_names=(ic.name for ic in self.island_configs),
            required=False,
        )
        checkpoint = {
            'version': CHECKPOINT_VERSION,
            'resume_eligible': provenance is not None,
            'provenance': (
                provenance.model_dump(mode='json') if provenance is not None else None
            ),
            'type': 'island_model',
            'timestamp': datetime.now().isoformat(),
            'generation': generation,
            'total_generations': self.generations,
            'num_islands': len(self.island_configs),
            'population_per_island': self.population_per_island,
            'island_populations': island_data,
            'island_stats': stats_data,
            'hall_of_fame': hof_data,
            'migration_history_count': len(self.migration_history),
            'common_panel_replay': {
                'incumbent_score': self.common_panel_incumbent_score,
                'best_fitness': self._common_panel_best,
                'no_improvement_checks': self._common_panel_no_improvement_checks,
                'history': self.common_panel_replay_history,
            },
            'diversity_recovery': {
                'bad_checks': self._diversity_bad_checks,
                'post_recovery_bad_checks': self._diversity_post_recovery_bad_checks,
                'recovery_applied': self._diversity_recovery_applied,
                'events': self.diversity_events,
            },
            'evolution_outcome': self.evolution_outcome,
            'island_random_states': {
                name: self._random_state_to_json(state)
                for name, state in self._island_random_states.items()
            },
            'config_snapshot': {
                'genetic_algorithm': self.config.get('genetic_algorithm', {}),
                'backtesting': self.config.get('backtesting', {}),
                'generic_island_model': self.config.get('generic_island_model', {}),
                'fitness_weights': self.config.get('fitness_weights', {}),
            },
        }

        # Save numpy / python random state
        checkpoint['random_state'] = {'python': random.getstate()}
        try:
            import numpy as np
            np_state = np.random.get_state()
            checkpoint['random_state']['numpy'] = (
                np_state[0],
                np_state[1].tolist(),
                int(np_state[2]),
                int(np_state[3]),
                float(np_state[4]),
            )
        except ImportError:
            pass

        # Atomic write
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = seal_checkpoint(checkpoint)

        tmp_path = filepath + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        os.replace(tmp_path, filepath)

        self.logger.info(
            "[CHECKPOINT] Saved island model state at generation %d to %s",
            generation, filepath,
        )
        return filepath

    def load_island_checkpoint(self) -> Optional[int]:
        """
        Find and load the latest island checkpoint from *checkpoint_dir*.

        Restores island populations, stats, and hall of fame so evolution
        can resume from the next generation.

        Returns:
            The generation to resume FROM (i.e. next gen to run), or None
            if no checkpoint was found.
        """
        expected_provenance = checkpoint_provenance_from_config(
            self.config,
            engine_kind="GENERIC_ISLAND",
            island_names=(ic.name for ic in self.island_configs),
            required=False,
        )
        if expected_provenance is None:
            self.logger.info(
                "[CHECKPOINT] Resume disabled: immutable checkpoint provenance "
                "was not supplied"
            )
            return None

        filepath = latest_checkpoint_by_generation(
            self.checkpoint_dir.glob('island_checkpoint_gen*.json')
        )
        if filepath is None:
            return None
        self.logger.info("[CHECKPOINT] Loading island checkpoint: %s", filepath)

        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        checkpoint = verify_resume_checkpoint(checkpoint, expected_provenance)
        if checkpoint.get('type') != 'island_model':
            raise CheckpointCompatibilityError(
                "checkpoint type is not generic island_model"
            )

        saved_gen = checkpoint['generation']
        filename_gen = checkpoint_generation_from_path(filepath)
        if saved_gen != filename_gen:
            raise CheckpointCompatibilityError(
                "checkpoint generation differs from its filename"
            )

        # Restore island populations
        island_pop_data = checkpoint.get('island_populations', {})
        current_names = [ic.name for ic in self.island_configs]
        if list(island_pop_data) != current_names:
            raise CheckpointCompatibilityError(
                "checkpoint island populations differ in names/order"
            )

        restored_populations = {}
        for island_config in self.island_configs:
            pop_data = island_pop_data[island_config.name]
            expected_size = island_config.population_size
            saved_size = pop_data.get('size')
            individuals_data = pop_data.get('individuals')
            if saved_size != expected_size or not isinstance(individuals_data, list):
                raise CheckpointCompatibilityError(
                    f"checkpoint population shape differs for {island_config.name}"
                )
            pop = Population(
                size=saved_size,
                generation=pop_data.get('generation', saved_gen),
            )
            pop.individuals = []
            for ind_data in individuals_data:
                pop.add_individual(Individual.from_dict(ind_data))
            if len(pop.individuals) != expected_size:
                raise CheckpointCompatibilityError(
                    f"checkpoint individual count differs for {island_config.name}"
                )
            restored_populations[island_config.name] = pop

        stats_data = checkpoint.get('island_stats', {})
        if set(stats_data) != set(current_names):
            raise CheckpointCompatibilityError(
                "checkpoint island statistics differ from current islands"
            )

        self.island_populations.update(restored_populations)
        self.logger.info(
            "[CHECKPOINT] Restored %d/%d island populations from generation %d",
            len(restored_populations), len(self.island_configs), saved_gen,
        )

        # Restore island stats
        for name, sd in stats_data.items():
            if name in self.island_stats:
                ist = self.island_stats[name]
                ist.best_fitness = sd.get('best_fitness', 0.0)
                ist.best_profit = sd.get('best_profit', 0.0)
                ist.avg_fitness = sd.get('avg_fitness', 0.0)
                ist.generations_completed = sd.get('generations_completed', 0)
                ist.migrants_sent = sd.get('migrants_sent', 0)
                ist.migrants_received = sd.get('migrants_received', 0)

        common_state = checkpoint.get('common_panel_replay', {})
        if isinstance(common_state, dict):
            best = common_state.get('best_fitness')
            self._common_panel_best = float(best) if best is not None else None
            self._common_panel_no_improvement_checks = int(
                common_state.get('no_improvement_checks', 0)
            )
            history = common_state.get('history', [])
            self.common_panel_replay_history = (
                list(history) if isinstance(history, list) else []
            )

        diversity_state = checkpoint.get('diversity_recovery', {})
        if isinstance(diversity_state, dict):
            self._diversity_bad_checks = int(diversity_state.get('bad_checks', 0))
            self._diversity_post_recovery_bad_checks = int(
                diversity_state.get('post_recovery_bad_checks', 0)
            )
            self._diversity_recovery_applied = bool(
                diversity_state.get('recovery_applied', False)
            )
            events = diversity_state.get('events', [])
            self.diversity_events = list(events) if isinstance(events, list) else []

        stored_outcome = checkpoint.get('evolution_outcome')
        if isinstance(stored_outcome, dict):
            for key in self.evolution_outcome:
                if key in stored_outcome:
                    self.evolution_outcome[key] = stored_outcome[key]

        stored_island_states = checkpoint.get('island_random_states', {})
        if isinstance(stored_island_states, dict):
            restored_states = {}
            for name, state in stored_island_states.items():
                if name not in current_names or not isinstance(state, dict):
                    continue
                try:
                    restored_states[name] = self._random_state_from_json(state)
                except Exception as exc:
                    raise CheckpointCompatibilityError(
                        f'invalid RNG state for island {name}'
                    ) from exc
            if set(restored_states) == set(current_names):
                self._island_random_states = restored_states

        # Restore random state
        random_state = checkpoint.get('random_state', {})
        if 'python' in random_state:
            try:
                py_state = random_state['python']
                random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))
            except Exception as e:
                self.logger.warning("[CHECKPOINT] Failed to restore Python random state: %s", e)
        if 'numpy' in random_state:
            try:
                import numpy as np
                ns = random_state['numpy']
                np.random.set_state((
                    ns[0], np.array(ns[1], dtype=np.uint32),
                    int(ns[2]), int(ns[3]), float(ns[4]),
                ))
            except Exception as e:
                self.logger.warning("[CHECKPOINT] Failed to restore NumPy random state: %s", e)

        resume_gen = saved_gen + 1
        self.logger.info(
            "[CHECKPOINT] Will resume from generation %d/%d",
            resume_gen + 1, self.generations,
        )
        return resume_gen

    def export_best_strategies(self, generation: int, top_n: int = 10):
        """
        Export the top-N hall-of-fame strategies as standalone .py files
        to a backup directory alongside the checkpoint.
        """
        export_dir = self.checkpoint_dir / f"best_strategies_gen{generation}"
        export_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        try:
            from genetic_algorithm.core.strategy_gene import StrategyGene
            for i, entry in enumerate(self.hall_of_fame.entries[:top_n]):
                gene_dict = getattr(entry, 'strategy_gene_dict', None)
                if not gene_dict:
                    continue
                # Reconstruct StrategyGene from the stored dict
                try:
                    strategy_gene = StrategyGene.from_dict(gene_dict)
                except Exception as e:
                    self.logger.debug("[EXPORT] Could not reconstruct gene for HoF entry %d: %s", i, e)
                    continue
                # Regenerate strategy code
                ga = next(iter(self.islands.values()), None)
                if ga is None:
                    break
                try:
                    code = ga.strategy_generator.generate_strategy_code(strategy_gene)
                    fname = export_dir / f"hof_rank{i+1}_fitness{entry.fitness:.4f}.py"
                    with open(fname, 'w') as f:
                        f.write(code)
                    exported += 1
                except Exception as e:
                    self.logger.debug("[EXPORT] Failed to export HoF entry %d: %s", i, e)
        except Exception as e:
            self.logger.warning("[EXPORT] Strategy export failed: %s", e)

        if exported:
            self.logger.info(
                "[CHECKPOINT] Exported %d best strategies to %s",
                exported, export_dir,
            )

    # ------------------------------------------------------------------
    # Migration dispatch
    # ------------------------------------------------------------------

    def _migrate(self, generation: int):
        """Dispatch migration based on configured topology."""
        topology = self.migration.topology
        self.logger.info(
            "[MIGRATION] Gen %d — topology=%s, count=%d",
            generation + 1, topology, self.migration.count,
        )

        with self._migration_lock:
            if topology == 'ring':
                self._migrate_ring(generation)
            elif topology == 'fully_connected':
                self._migrate_fully_connected(generation)
            elif topology == 'tournament':
                self._migrate_tournament(generation)
            elif topology == 'hierarchical':
                self._migrate_hierarchical(generation)
            else:
                self.logger.warning(
                    "Unknown migration topology '%s', falling back to ring",
                    topology,
                )
                self._migrate_ring(generation)

    # ------------------------------------------------------------------
    # Migration topologies
    # ------------------------------------------------------------------

    def _migrate_ring(self, generation: int):
        """
        Ring topology: island[i] sends top-N to island[(i+1) % N].
        """
        names = [ic.name for ic in self.island_configs]
        n = len(names)

        for i in range(n):
            source = names[i]
            target = names[(i + 1) % n]

            top = self._get_top_individuals(
                source,
                self.migration.count,
                target_island=target,
            )
            if not top:
                continue

            replaced = self._inject_migrants(target, top, generation, source=source)
            fitnesses = [
                ind.raw_fitness for ind in top if ind.raw_fitness is not None
            ]

            self.migration_history.append(GenericMigrationEvent(
                generation=generation,
                source=source,
                target=target,
                count=replaced,
                fitnesses=fitnesses,
            ))

            with self._stats_lock:
                self.island_stats[source].migrants_sent += len(top)
                self.island_stats[target].migrants_received += replaced

            self.logger.info(
                "  %s → %s: %d migrants (fitnesses: %s)",
                source, target, replaced,
                [f"{f:.4f}" for f in fitnesses],
            )

    def _migrate_fully_connected(self, generation: int):
        """
        Fully connected: every island sends migrants to every other island.
        
        To preserve diversity, each source picks from a pool of top 3×N
        individuals and sends a different random sample of N to each target.
        """
        names = [ic.name for ic in self.island_configs]
        rng = random.Random(self.base_seed + generation)

        for source in names:
            # Fetch a larger pool so each target gets a different sample
            pool_size = self.migration.count * 3
            pool = self._get_top_individuals(source, pool_size)
            if not pool:
                continue

            for target in names:
                if target == source:
                    continue

                # Sample N from pool (with fallback to full pool if smaller)
                n = min(self.migration.count, len(pool))
                migrants = rng.sample(pool, n)

                replaced = self._inject_migrants(target, migrants, generation, source=source)
                fitnesses = [
                    ind.raw_fitness for ind in migrants if ind.raw_fitness is not None
                ]

                self.migration_history.append(GenericMigrationEvent(
                    generation=generation,
                    source=source,
                    target=target,
                    count=replaced,
                    fitnesses=fitnesses,
                ))

                with self._stats_lock:
                    self.island_stats[source].migrants_sent += len(migrants)
                    self.island_stats[target].migrants_received += replaced

            self.logger.info(
                "  %s → all: pool=%d, per-target=%d (best=%.4f)",
                source, len(pool), min(self.migration.count, len(pool)),
                max((ind.raw_fitness for ind in pool if ind.raw_fitness), default=0),
            )

    def _migrate_tournament(self, generation: int):
        """
        Tournament topology: pick random pairs, compare best individuals,
        inject winners into losers.
        """
        names = [ic.name for ic in self.island_configs]
        rng = random.Random(self.base_seed + generation)

        # Shuffle and pair up
        shuffled = list(names)
        rng.shuffle(shuffled)

        # Process pairs (drop last island if odd count)
        for i in range(0, len(shuffled) - 1, 2):
            island_a = shuffled[i]
            island_b = shuffled[i + 1]

            top_a = self._get_top_individuals(island_a, self.migration.count)
            top_b = self._get_top_individuals(island_b, self.migration.count)

            best_a = max(
                (ind.raw_fitness for ind in top_a if ind.raw_fitness),
                default=0,
            )
            best_b = max(
                (ind.raw_fitness for ind in top_b if ind.raw_fitness),
                default=0,
            )

            panel_ids = {
                ind.fitness_panel_id
                for ind in top_a + top_b
                if ind.fitness_panel_id is not None
            }
            panels_comparable = (
                bool(top_a and top_b)
                and len(panel_ids) == 1
                and all(ind.fitness_panel_id is not None for ind in top_a + top_b)
            )

            if not panels_comparable:
                # A tournament winner cannot be inferred from scores produced
                # on different pair/policy panels. Exchange locally ranked
                # genomes in both directions and force target re-evaluation.
                replaced_b = self._inject_migrants(
                    island_b, top_a, generation, source=island_a
                ) if top_a else 0
                replaced_a = self._inject_migrants(
                    island_a, top_b, generation, source=island_b
                ) if top_b else 0
                for source, target, migrants, replaced in (
                    (island_a, island_b, top_a, replaced_b),
                    (island_b, island_a, top_b, replaced_a),
                ):
                    if not migrants:
                        continue
                    self.migration_history.append(GenericMigrationEvent(
                        generation=generation,
                        source=source,
                        target=target,
                        count=replaced,
                        fitnesses=[
                            ind.raw_fitness for ind in migrants
                            if ind.raw_fitness is not None
                        ],
                    ))
                    with self._stats_lock:
                        self.island_stats[source].migrants_sent += len(migrants)
                        self.island_stats[target].migrants_received += replaced
                self.logger.info(
                    "  TOURNAMENT: %s ↔ %s (incomparable panels; "
                    "bidirectional re-evaluation)",
                    island_a, island_b,
                )
                continue

            if best_a >= best_b and top_a:
                winner, loser = island_a, island_b
                migrants = top_a
            elif top_b:
                winner, loser = island_b, island_a
                migrants = top_b
            else:
                continue

            replaced = self._inject_migrants(
                loser, migrants, generation, source=winner
            )
            fitnesses = [
                ind.raw_fitness for ind in migrants if ind.raw_fitness is not None
            ]

            self.migration_history.append(GenericMigrationEvent(
                generation=generation,
                source=winner,
                target=loser,
                count=replaced,
                fitnesses=fitnesses,
            ))

            with self._stats_lock:
                self.island_stats[winner].migrants_sent += len(migrants)
                self.island_stats[loser].migrants_received += replaced

            self.logger.info(
                "  TOURNAMENT: %s (%.4f) beats %s (%.4f) → %d migrants",
                winner, best_a if winner == island_a else best_b,
                loser, best_b if loser == island_b else best_a,
                replaced,
            )

    def _migrate_hierarchical(self, generation: int):
        """
        Hierarchical merge: pair islands, merge populations of each pair
        (top individuals from both replace worst in both).
        """
        names = [ic.name for ic in self.island_configs]
        rng = random.Random(self.base_seed + generation)

        shuffled = list(names)
        rng.shuffle(shuffled)

        for i in range(0, len(shuffled) - 1, 2):
            island_a = shuffled[i]
            island_b = shuffled[i + 1]

            top_a = self._get_top_individuals(island_a, self.migration.count)
            top_b = self._get_top_individuals(island_b, self.migration.count)

            # Bidirectional: A's best → B, B's best → A
            if top_a:
                replaced_b = self._inject_migrants(island_b, top_a, generation, source=island_a)
                fitnesses_a = [
                    ind.raw_fitness for ind in top_a if ind.raw_fitness is not None
                ]
                self.migration_history.append(GenericMigrationEvent(
                    generation=generation,
                    source=island_a,
                    target=island_b,
                    count=replaced_b,
                    fitnesses=fitnesses_a,
                ))
                with self._stats_lock:
                    self.island_stats[island_a].migrants_sent += len(top_a)
                    self.island_stats[island_b].migrants_received += replaced_b

            if top_b:
                replaced_a = self._inject_migrants(island_a, top_b, generation, source=island_b)
                fitnesses_b = [
                    ind.raw_fitness for ind in top_b if ind.raw_fitness is not None
                ]
                self.migration_history.append(GenericMigrationEvent(
                    generation=generation,
                    source=island_b,
                    target=island_a,
                    count=replaced_a,
                    fitnesses=fitnesses_b,
                ))
                with self._stats_lock:
                    self.island_stats[island_b].migrants_sent += len(top_b)
                    self.island_stats[island_a].migrants_received += replaced_a

            self.logger.info(
                "  HIERARCHICAL: %s ↔ %s (bidirectional exchange)",
                island_a, island_b,
            )

    # ------------------------------------------------------------------
    # Merge round (global pool-and-redistribute)
    # ------------------------------------------------------------------

    def _merge_round(self, generation: int):
        """
        Pool top-K from ALL islands, rank globally, redistribute
        top individuals back to all islands (replacing worst).
        """
        self.logger.info(
            "[MERGE] Global merge round at generation %d", generation + 1,
        )

        # Pool top individuals from every island
        pools: List[List[Individual]] = []
        for ic in self.island_configs:
            top = self._get_top_individuals(ic.name, self.migration.count * 2)
            pools.append(top)

        global_pool = [individual for pool in pools for individual in pool]

        if not global_pool:
            self.logger.info("  MERGE: No eligible individuals to merge")
            return

        top_global, panels_comparable, unique_count = self._select_cross_panel_pool(
            pools, self.migration.count
        )

        self.logger.info(
            "  MERGE: Pooled %d unique individuals, redistributing top %d",
            unique_count, len(top_global),
        )
        if not panels_comparable:
            self.logger.info(
                "  MERGE: panel scores differ; selected round-robin by local rank"
            )

        # Inject into every island
        for ic in self.island_configs:
            replaced = self._inject_migrants(ic.name, top_global, generation, source='global_merge')
            if replaced > 0:
                self.logger.info(
                    "    → %s: injected %d global elites", ic.name, replaced,
                )

    # ------------------------------------------------------------------
    # Migration helpers
    # ------------------------------------------------------------------

    def _get_top_individuals(
        self,
        island_name: str,
        count: int,
        target_island: Optional[str] = None,
    ) -> List[Individual]:
        """Get top-N compatible individuals ranked by raw fitness.

        When a target is supplied, incompatible elites are skipped before
        slicing so ring migration deterministically falls through to the next
        locally ranked compatible candidate.
        """
        pop = self.island_populations.get(island_name)
        if pop is None:
            return []
        source_individuals = self._evaluated_generation_populations.get(
            island_name, pop.individuals
        )

        ranked = sorted(
            [
                ind for ind in source_individuals
                if self._is_valid_evidence(ind)
                and ind.raw_fitness is not None
                and (
                    target_island is None
                    or self._is_indicator_pool_compatible(target_island, ind)
                )
            ],
            key=lambda x: x.raw_fitness,
            reverse=True,
        )
        return ranked[:count]

    def _is_indicator_pool_compatible(
        self,
        target_island: str,
        individual: Individual,
    ) -> bool:
        """Return whether every indicator gene is allowed on the target."""

        target_config = next(
            (
                island
                for island in self.island_configs
                if island.name == target_island
            ),
            None,
        )
        # Unknown targets preserve the legacy helper behavior used by
        # external callers.  A missing/empty pool means unrestricted.
        if target_config is None or not target_config.indicator_pool:
            return True
        allowed = set(target_config.indicator_pool)
        used = {
            indicator.type
            for indicator in individual.strategy_gene.indicators
        }
        return bool(used) and used.issubset(allowed)

    def _select_cross_panel_pool(
        self,
        pools: List[List[Individual]],
        count: int,
    ) -> tuple[List[Individual], bool, int]:
        """Select without comparing scores unless every panel is identical."""
        flattened = [individual for pool in pools for individual in pool]
        panel_ids = {individual.fitness_panel_id for individual in flattened}
        comparable = bool(flattened) and None not in panel_ids and len(panel_ids) == 1

        if comparable:
            ordered = sorted(
                flattened,
                key=lambda item: float(item.raw_fitness),
                reverse=True,
            )
        else:
            # Each pool is already locally ranked. Interleave ranks so no
            # island wins merely because its score scale is larger.
            ordered = []
            max_depth = max((len(pool) for pool in pools), default=0)
            for rank in range(max_depth):
                for pool in pools:
                    if rank < len(pool):
                        ordered.append(pool[rank])

        unique: List[Individual] = []
        seen: set[str] = set()
        for individual in ordered:
            fingerprint = self._gene_hash(individual)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(individual)
        return unique[:count], comparable, len(unique)

    def _inject_migrants(
        self,
        target_island: str,
        migrants: List[Individual],
        generation: int,
        source: str = 'unknown',
    ) -> int:
        """
        Inject migrant individuals into a target island's population,
        replacing the worst individuals.

        Returns the number of individuals replaced.
        """
        pop = self.island_populations.get(target_island)
        if pop is None or not migrants:
            return 0

        compatible_migrants = [
            migrant
            for migrant in migrants
            if self._is_indicator_pool_compatible(target_island, migrant)
        ]
        if not compatible_migrants:
            return 0

        # Sort population worst-first
        sorted_inds = sorted(
            pop.individuals,
            # Migration is injected into the already-created next generation,
            # which normally contains measured elites and unevaluated
            # offspring.  Raw multipair fitness is signed, so using ``-1`` for
            # missing fitness can rank a valid negative elite below an
            # unevaluated child.  Unmeasured slots are the replacement target
            # and must sort before every finite score.
            key=lambda x: (
                x.raw_fitness
                if x.raw_fitness is not None
                else float('-inf')
            ),
        )

        replaced = 0
        for migrant in compatible_migrants:
            if replaced >= len(sorted_inds):
                break

            # Deep-copy the migrant's gene so it's independent
            gene_copy = migrant.strategy_gene.copy()
            gene_copy.generation = generation
            gene_copy.individual_id = sorted_inds[replaced].strategy_gene.individual_id

            new_ind = Individual(strategy_gene=gene_copy)
            new_ind.evaluated = False  # Force re-evaluation
            new_ind.metrics = {'origin': f'migrant_from_{source}'}

            # Replace worst individual in-place
            idx = pop.individuals.index(sorted_inds[replaced])
            pop.individuals[idx] = new_ind
            replaced += 1

        return replaced

    # ------------------------------------------------------------------
    # External migration (cross-machine)
    # ------------------------------------------------------------------

    def _load_external_migrants(self, generation: int) -> int:
        """
        Load strategy individuals from the incoming_migrants directory.

        Other machines (or scripts) drop JSON files here containing
        serialized individuals. Each file is loaded, injected into a
        randomly chosen island, and then deleted to prevent re-processing.

        Returns the total number of migrants injected.
        """
        if not self.external_migration_dir.exists():
            return 0

        pattern = str(self.external_migration_dir / '*.json')
        files = sorted(glob.glob(pattern))
        if not files:
            return 0

        self.logger.info(
            "[EXT-MIGRATION] Gen %d — found %d incoming migrant files",
            generation + 1, len(files),
        )

        total_injected = 0
        island_names = [ic.name for ic in self.island_configs]

        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)

                individuals_data = data if isinstance(data, list) else data.get('individuals', [data])

                migrants = []
                for ind_data in individuals_data:
                    try:
                        ind = Individual.from_dict(ind_data)
                        ind.evaluated = False  # Force re-evaluation
                        migrants.append(ind)
                    except Exception as e:
                        self.logger.warning(
                            "[EXT-MIGRATION] Failed to parse individual from %s: %s",
                            fpath, e,
                        )

                if migrants:
                    # Distribute migrants across random islands
                    for migrant in migrants:
                        target = random.choice(island_names)
                        replaced = self._inject_migrants(target, [migrant], generation, source='external')
                        if replaced > 0:
                            total_injected += replaced
                            self.logger.info(
                                "[EXT-MIGRATION]   → injected into %s (fitness=%.4f)",
                                target, migrant.raw_fitness or 0,
                            )

                # Remove processed file
                os.remove(fpath)
                self.logger.debug("[EXT-MIGRATION] Processed and removed %s", fpath)

            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(
                    "[EXT-MIGRATION] Failed to read %s: %s", fpath, e,
                )

        if total_injected > 0:
            self.logger.info(
                "[EXT-MIGRATION] Gen %d — injected %d external migrants total",
                generation + 1, total_injected,
            )

        return total_injected

    def _export_for_external_migration(self, generation: int) -> int:
        """
        Export top individuals to the outgoing_migrants directory.

        A separate script (distribute_migrate.sh) picks these up and
        SCPs them to other machines' incoming_migrants directories.

        Returns the number of individuals exported.
        """
        self.external_export_dir.mkdir(parents=True, exist_ok=True)

        # Collect locally ranked candidates without comparing panel scores.
        pools: List[List[Individual]] = []

        for ic in self.island_configs:
            pools.append(
                self._get_top_individuals(ic.name, self.external_migration_count)
            )

        top_individuals, panels_comparable, _ = self._select_cross_panel_pool(
            pools, self.external_migration_count
        )
        if not panels_comparable and top_individuals:
            self.logger.info(
                "[EXT-MIGRATION] Incomparable panels; exporting round-robin local ranks"
            )

        if not top_individuals:
            return 0

        # Write as a single JSON file with timestamp
        export_data = {
            'source': os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown')),
            'generation': generation,
            'timestamp': time.time(),
            'individuals': [ind.to_dict() for ind in top_individuals],
        }

        filename = f"migrants_gen{generation:04d}_{int(time.time())}.json"
        export_path = self.external_export_dir / filename

        try:
            tmp_path = export_path.with_suffix('.tmp')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            os.replace(tmp_path, export_path)

            self.logger.info(
                "[EXT-MIGRATION] Gen %d — exported %d individuals to %s",
                generation + 1, len(top_individuals), export_path,
            )
        except IOError as e:
            self.logger.error("[EXT-MIGRATION] Failed to export: %s", e)
            return 0

        return len(top_individuals)

    @staticmethod
    def _gene_hash(ind: Individual) -> str:
        """Compute a hash of an individual's strategy gene for deduplication."""
        try:
            gene_dict = ind.strategy_gene.to_dict()
            # Remove volatile fields
            gene_dict.pop('generation', None)
            gene_dict.pop('individual_id', None)
            serialized = json.dumps(gene_dict, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()
        except Exception:
            logger.warning(f"[ISLAND] Gene hashing failed for individual {getattr(ind, 'id', '?')}", exc_info=True)
            return str(id(ind))

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def _collect_final_results(self) -> Dict[str, List[Individual]]:
        """
        Pool the configured local top candidates, replay unique phenotypes on one common
        panel, and return the comparable ranking under ``__global__``.
        """
        results: Dict[str, List[Individual]] = {
            island_config.name: [] for island_config in self.island_configs
        }

        # The production contract replays the top two per island.  Reusing the
        # same bound for final collection avoids an unbounded serial tail.
        global_pool: List[Individual] = []
        for ic in self.island_configs:
            evaluated = self._evaluated_generation_populations.get(ic.name)
            if evaluated is None:
                pop = self.island_populations.get(ic.name)
                evaluated = list(pop.individuals) if pop else []
            if evaluated:
                local_top = sorted(
                    [ind for ind in evaluated if self._is_valid_evidence(ind)],
                    key=lambda x: x.raw_fitness,
                    reverse=True,
                )[:self.common_panel_top_n]
                for individual in local_top:
                    individual.metrics['evaluation_island'] = ic.name
                results[ic.name] = local_top
                global_pool.extend(local_top)

        unique_global = self._replay_global_candidates(global_pool)
        results['__global__'] = unique_global[:20]

        self.logger.info(
            "Final results: %d unique strategies across %d islands (top-20 global)",
            len(unique_global), len(self.island_configs),
        )

        return results

    def _create_common_replay_evaluator(self):
        """Build the base-config evaluator used only for global comparison."""
        comparison_config = copy.deepcopy(self.config)
        comparison_config.setdefault('generic_island_model', {})['enabled'] = False
        comparison_config.setdefault('island_model', {})['enabled'] = False
        if comparison_config.get('regime_aware', {}).get('enabled', False):
            from genetic_algorithm.evaluation.regime_aware import (
                create_regime_aware_evaluator,
            )
            evaluator = create_regime_aware_evaluator(
                comparison_config, auto_detect=True
            )
        else:
            evaluator = FitnessEvaluator(comparison_config)
        panel = build_evaluation_panel(
            comparison_config,
            evaluator=evaluator,
            role=PANEL_ROLE_COMMON_REPLAY,
        )
        return evaluator, panel

    def _replay_global_candidates(
        self, candidates: List[Individual]
    ) -> List[Individual]:
        """Replay local finalists on one panel, then update the shared HOF."""
        if not candidates:
            return []
        unique_candidate_count = len({self._gene_hash(item) for item in candidates})
        evaluator, panel = self._create_common_replay_evaluator()
        replayed = replay_on_common_panel(
            candidates,
            evaluator=evaluator,
            panel=panel,
            logger=self.logger,
            parallel_evaluator=getattr(
                self, '_shared_parallel_evaluator', None
            ),
        )
        parity_failures = self._raw_replay_parity_failures(replayed)
        if parity_failures:
            self._request_stop(
                OUTCOME_TECHNICAL_INVALID,
                'raw common-panel replay changed deterministic source scores: '
                + '; '.join(parity_failures[:3]),
            )
            # One drift proves that the replay batch is not trustworthy.  Do
            # not salvage apparently matching siblings from the same batch or
            # allow any of them to reach HOF/strict replay.
            self.evolution_outcome['common_panel_failed_evaluations'] += (
                unique_candidate_count
            )
            return []
        replayed = [item for item in replayed if self._is_valid_evidence(item)]
        failed_count = max(0, unique_candidate_count - len(replayed))
        self.evolution_outcome['common_panel_valid_evaluations'] += len(replayed)
        self.evolution_outcome['common_panel_failed_evaluations'] += failed_count
        self.hall_of_fame.bind_panel(
            panel.panel_id,
            panel.role,
            data_identity_verified=panel.data_identity_verified,
        )
        if replayed:
            replay_population = Population(size=len(replayed))
            replay_population.individuals = replayed
            self.hall_of_fame.update(
                replay_population,
                generation=max(0, self.generations - 1),
            )
        return replayed

    def _raw_replay_parity_failures(
        self,
        candidates: List[Individual],
    ) -> List[str]:
        """Return deterministic score drifts for the immutable raw panel.

        Every raw-multipair search candidate has already been evaluated on the
        exact same six-pair panel.  Replaying an unchanged phenotype may
        refresh provenance, but it must not change its raw score.  Treat a
        drift as technical corruption (for example a generated-strategy file
        collision) instead of allowing it to steer plateau or archive state.
        """

        if not self.config.get('raw_multipair_score', {}).get('enabled', False):
            return []
        failures: List[str] = []
        for candidate in candidates:
            source = candidate.metrics.get('source_fitness')
            replay = candidate.raw_fitness
            if source is None or replay is None:
                continue
            try:
                source_value = float(source)
                replay_value = float(replay)
            except (TypeError, ValueError, OverflowError):
                failures.append(f'{candidate.id}:non-numeric score')
                continue
            if (
                not math.isfinite(source_value)
                or not math.isfinite(replay_value)
                or not math.isclose(
                    source_value,
                    replay_value,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                failures.append(
                    f'{candidate.id}:source={source_value:.12g},'
                    f'replay={replay_value:.12g}'
                )
        return failures

    def _maybe_handle_diversity(self, generation: int) -> None:
        """Recover one broad island collapse, then stop if it persists."""

        if (
            not self.diversity_recovery_enabled
            or self.common_panel_replay_interval <= 0
            or (generation + 1) % self.common_panel_replay_interval != 0
        ):
            return

        island_observations: Dict[str, Dict[str, float]] = {}
        collapsed_islands: List[str] = []
        for island_config in self.island_configs:
            island_name = island_config.name
            individuals = self._evaluated_generation_populations.get(island_name, [])
            unique_genomes = len({self._gene_hash(item) for item in individuals})
            duplicate_fraction = (
                0.0
                if not individuals
                else 1.0 - unique_genomes / len(individuals)
            )
            stats_history = self.generation_stats.get(island_name, [])
            genetic_diversity = (
                float(stats_history[-1].genetic_diversity or 0.0)
                if stats_history
                else 0.0
            )
            island_observations[island_name] = {
                'duplicate_fraction': duplicate_fraction,
                'genetic_diversity': genetic_diversity,
            }
            if (
                duplicate_fraction >= self.diversity_duplicate_fraction_threshold
                or genetic_diversity <= self.diversity_genetic_threshold
            ):
                collapsed_islands.append(island_name)

        broadly_collapsed = (
            len(collapsed_islands) >= self.diversity_island_threshold_count
        )
        action = 'HEALTHY'
        replacements: Dict[str, int] = {}
        if not broadly_collapsed:
            self._diversity_bad_checks = 0
            self._diversity_post_recovery_bad_checks = 0
        elif not self._diversity_recovery_applied:
            self._diversity_bad_checks += 1
            action = 'COLLAPSE_OBSERVED'
            if self._diversity_bad_checks >= self.diversity_checks_before_recovery:
                replacements = self._apply_diversity_recovery(
                    collapsed_islands, generation
                )
                self._diversity_recovery_applied = True
                self._diversity_bad_checks = 0
                action = 'RECOVERY_APPLIED'
        else:
            self._diversity_post_recovery_bad_checks += 1
            action = 'POST_RECOVERY_COLLAPSE'
            if (
                self._diversity_post_recovery_bad_checks
                >= self.diversity_checks_after_recovery
            ):
                action = 'STOP_DIVERSITY_COLLAPSE'
                self._request_stop(
                    OUTCOME_DIVERSITY_COLLAPSE,
                    'broad diversity collapse persisted after one recovery',
                )

        event_type = {
            'COLLAPSE_OBSERVED': 'COLLAPSE_OBSERVED',
            'POST_RECOVERY_COLLAPSE': 'COLLAPSE_OBSERVED',
            'RECOVERY_APPLIED': 'RECOVERY_APPLIED',
            'STOP_DIVERSITY_COLLAPSE': 'RECOVERY_FAILED',
        }.get(action)
        affected_observations = [
            island_observations[name] for name in collapsed_islands
        ]

        event = {
            'generation': generation + 1,
            'event_type': event_type,
            'affected_islands': collapsed_islands,
            'duplicate_fraction': max(
                (
                    item['duplicate_fraction']
                    for item in affected_observations
                ),
                default=0.0,
            ),
            'genetic_diversity': min(
                (
                    item['genetic_diversity']
                    for item in affected_observations
                ),
                default=1.0,
            ),
            'mutation_rate_after': (
                self.diversity_recovery_mutation_rate
                if action == 'RECOVERY_APPLIED'
                else None
            ),
            'collapsed_island_count': len(collapsed_islands),
            'collapsed_islands': collapsed_islands,
            'islands': island_observations,
            'action': action,
            'replacements': replacements,
            'bad_checks_before_recovery': self._diversity_bad_checks,
            'bad_checks_after_recovery': self._diversity_post_recovery_bad_checks,
        }
        if action != 'HEALTHY':
            self.diversity_events.append(event)
        self.logger.info(
            "[DIVERSITY] gen=%d collapsed=%d/%d action=%s",
            generation + 1,
            len(collapsed_islands),
            len(self.island_configs),
            action,
        )

    def _apply_diversity_recovery(
        self, island_names: List[str], generation: int
    ) -> Dict[str, int]:
        if generation >= self.generations - 1:
            return {island_name: 0 for island_name in island_names}
        replacements: Dict[str, int] = {}
        for island_name in island_names:
            ga = self.islands[island_name]
            ga.mutation_rate = max(
                float(ga.mutation_rate), self.diversity_recovery_mutation_rate
            )
            evaluated = self._evaluated_generation_populations.get(island_name)
            if not evaluated:
                replacements[island_name] = 0
                continue

            # ``island_populations`` already contains offspring for the next
            # generation, whose fitness is normally None.  Ranking that
            # population therefore replaced arbitrary list positions rather
            # than the weakest strategies which triggered recovery.  Build
            # the recovery generation from the immutable evaluated snapshot:
            # retain its best 75% and replace its measured bottom quartile
            # with fresh immigrants.  The snapshot itself remains untouched
            # for traces, replay evidence, and audit output.
            population = Population(size=len(evaluated), generation=generation + 1)
            population.individuals = copy.deepcopy(evaluated)
            nominal_replacement_count = max(
                1,
                int(len(population.individuals) * self.diversity_replacement_fraction),
            )
            failed_indices = [
                index
                for index, individual in enumerate(population.individuals)
                if not self._is_valid_evidence(individual)
            ]
            valid_indices = sorted(
                (
                    index
                    for index, individual in enumerate(population.individuals)
                    if self._is_valid_evidence(individual)
                ),
                key=lambda index: (
                    float(population.individuals[index].raw_fitness)
                    if population.individuals[index].raw_fitness is not None
                    else float('-inf'),
                    index,
                ),
            )
            replacement_count = max(
                nominal_replacement_count,
                len(failed_indices),
            )
            target_indices = failed_indices + valid_indices[
                :max(0, replacement_count - len(failed_indices))
            ]
            with self._island_random_scope(island_name):
                for target_index in target_indices:
                    target = population.individuals[target_index]
                    individual_id = target.strategy_gene.individual_id
                    gene = ga.strategy_generator.generate_random_strategy(
                        generation=generation + 1,
                        individual_id=individual_id,
                    )
                    replacement = Individual(strategy_gene=gene)
                    replacement.metrics['origin'] = 'diversity_recovery'
                    population.individuals[target_index] = replacement
            self.island_populations[island_name] = population
            replacements[island_name] = replacement_count
        return replacements

    def _maybe_replay_common_panel(self, generation: int) -> None:
        """Periodically compare local elites on one immutable common panel."""
        if (
            not self.common_panel_replay_enabled
            or self.common_panel_replay_interval <= 0
            or (generation + 1) % self.common_panel_replay_interval != 0
        ):
            return
        candidates = [
            individual
            for island_name in sorted(self._evaluated_generation_elites)
            for individual in self._evaluated_generation_elites[island_name]
        ]
        if not candidates:
            event = {
                'generation': generation,
                'panel_id': None,
                'candidate_count': 0,
                'valid_count': 0,
                'failed_count': 0,
                'best_fitness': None,
                'median_fitness': None,
                'improved': False,
                'material_improvement_threshold': None,
                'no_improvement_checks': self._common_panel_no_improvement_checks,
                'early_stop_triggered': False,
                'technical_stop_triggered': True,
            }
            self.common_panel_replay_history.append(event)
            self._request_stop(
                OUTCOME_TECHNICAL_INVALID,
                f'common-panel replay at generation {generation + 1} had no candidates',
            )
            return
        unique_candidate_count = len({self._gene_hash(item) for item in candidates})
        evaluator, panel = self._create_common_replay_evaluator()
        replayed = replay_on_common_panel(
            candidates,
            evaluator=evaluator,
            panel=panel,
            logger=self.logger,
            parallel_evaluator=getattr(
                self, '_shared_parallel_evaluator', None
            ),
        )
        parity_failures = self._raw_replay_parity_failures(replayed)
        if parity_failures:
            event = {
                'generation': generation,
                'panel_id': panel.panel_id,
                'candidate_count': unique_candidate_count,
                'valid_count': 0,
                'failed_count': len(parity_failures),
                'best_fitness': None,
                'median_fitness': None,
                'improved': False,
                'material_improvement_threshold': None,
                'no_improvement_checks': self._common_panel_no_improvement_checks,
                'early_stop_triggered': False,
                'technical_stop_triggered': True,
                'parity_failures': parity_failures,
            }
            self.common_panel_replay_history.append(event)
            self.evolution_outcome['common_panel_failed_evaluations'] += len(
                parity_failures
            )
            self._request_stop(
                OUTCOME_TECHNICAL_INVALID,
                'raw common-panel replay changed deterministic source scores: '
                + '; '.join(parity_failures[:3]),
            )
            return
        measured = [item for item in replayed if self._is_valid_evidence(item)]
        failed_count = max(0, unique_candidate_count - len(measured))
        self.evolution_outcome['common_panel_valid_evaluations'] += len(measured)
        self.evolution_outcome['common_panel_failed_evaluations'] += failed_count
        scores = sorted(
            (float(item.raw_fitness) for item in measured), reverse=True
        )
        if not scores:
            event = {
                'generation': generation,
                'panel_id': panel.panel_id,
                'candidate_count': unique_candidate_count,
                'valid_count': 0,
                'failed_count': failed_count,
                'best_fitness': None,
                'median_fitness': None,
                'improved': False,
                'material_improvement_threshold': None,
                'no_improvement_checks': self._common_panel_no_improvement_checks,
                'early_stop_triggered': False,
                'technical_stop_triggered': True,
            }
            self.common_panel_replay_history.append(event)
            self._request_stop(
                OUTCOME_TECHNICAL_INVALID,
                f'common-panel replay at generation {generation + 1} produced zero valid evidence',
            )
            return
        best = scores[0]
        threshold = (
            None
            if self._common_panel_best is None
            else max(
                self.common_panel_min_improvement,
                self.common_panel_relative_min_improvement
                * max(1.0, abs(self._common_panel_best)),
            )
        )
        improved = (
            self._common_panel_best is None
            or best - self._common_panel_best >= threshold
        )
        if improved:
            self._common_panel_best = best
            self._common_panel_no_improvement_checks = 0
        else:
            self._common_panel_no_improvement_checks += 1
        stopped = (
            self.common_panel_early_stop_patience > 0
            and self._common_panel_no_improvement_checks
            >= self.common_panel_early_stop_patience
            and generation + 1 >= self.common_panel_min_generation
        )
        event = {
            'generation': generation,
            'panel_id': panel.panel_id,
            'candidate_count': unique_candidate_count,
            'valid_count': len(measured),
            'failed_count': failed_count,
            'best_fitness': best,
            'median_fitness': scores[len(scores) // 2],
            'improved': improved,
            'material_improvement_threshold': threshold,
            'no_improvement_checks': self._common_panel_no_improvement_checks,
            'early_stop_triggered': stopped,
            'technical_stop_triggered': False,
        }
        self.common_panel_replay_history.append(event)
        self.logger.info(
            "[COMMON-PANEL] gen=%d candidates=%d best=%.4f improved=%s stale=%d/%d",
            generation + 1,
            len(measured),
            best,
            improved,
            self._common_panel_no_improvement_checks,
            self.common_panel_early_stop_patience,
        )
        if stopped:
            self._request_stop(
                OUTCOME_PLATEAU,
                f'{self._common_panel_no_improvement_checks} common-panel checks '
                'without material improvement',
            )
            self.logger.info(
                "[COMMON-PANEL] Early stop requested after %d checks without material improvement",
                self._common_panel_no_improvement_checks,
            )

    # ------------------------------------------------------------------
    # Phase 3: Reporting
    # ------------------------------------------------------------------

    def _phase3_report(
        self,
        results: Dict[str, List[Individual]],
        total_elapsed: float,
    ):
        """Generate final summary report."""
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  RESULTS SUMMARY")
        self.logger.info("═" * 70)
        self.logger.info("  Total time: %.1f seconds (%.1f minutes)",
                         total_elapsed, total_elapsed / 60)
        self.logger.info("  Islands: %d", len(self.islands))
        self.logger.info(
            "  Generations: %d/%d",
            self.evolution_outcome.get('generations_completed', self.generations),
            self.generations,
        )
        self.logger.info(
            "  Stop reason: %s",
            self.evolution_outcome.get('reason', OUTCOME_COMPLETED_BUDGET),
        )
        self.logger.info("  Migrations: %d events", len(self.migration_history))
        self.logger.info("")

        # Per-island summary
        for ic in self.island_configs:
            ist = self.island_stats[ic.name]
            self.logger.info(
                "── Island: %s ──", ic.name,
            )
            self.logger.info("  Best fitness:  %.4f", ist.best_fitness)
            self.logger.info("  Best profit:   %.2f%%", ist.best_profit)
            self.logger.info("  Avg fitness:   %.4f", ist.avg_fitness)
            self.logger.info("  Migrants sent: %d  received: %d",
                             ist.migrants_sent, ist.migrants_received)

            top5 = results.get(ic.name, [])
            for rank, ind in enumerate(top5[:3], 1):
                profit = ind.metrics.get('profit', 0)
                sharpe = ind.metrics.get('sharpe_ratio', 0)
                trades = ind.metrics.get('num_trades', 0)
                self.logger.info(
                    "    #%d: fitness=%.4f profit=%.2f%% sharpe=%.2f trades=%d",
                    rank, ind.raw_fitness or 0, profit, sharpe, trades,
                )
            self.logger.info("")

        # Global top-10
        global_top = results.get('__global__', [])
        if global_top:
            self.logger.info("── Global Top-10 (deduplicated) ──")
            for rank, ind in enumerate(global_top[:10], 1):
                profit = ind.metrics.get('profit', 0)
                sharpe = ind.metrics.get('sharpe_ratio', 0)
                trades = ind.metrics.get('num_trades', 0)
                self.logger.info(
                    "  #%d: fitness=%.4f profit=%.2f%% sharpe=%.2f trades=%d",
                    rank, ind.raw_fitness or 0, profit, sharpe, trades,
                )

        # Hall of fame
        if self.hall_of_fame.entries:
            self.logger.info("")
            self.logger.info("── Hall of Fame: %d entries ──",
                             len(self.hall_of_fame.entries))
            for i, entry in enumerate(self.hall_of_fame.entries[:5]):
                self.logger.info(
                    "  #%d: fitness=%.4f (gen %d)",
                    i + 1, entry.fitness, entry.generation_found,
                )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_generation_summary(self, gen: int, elapsed: float):
        """Log a compact summary of all islands for this generation."""
        parts = []
        for ic in self.island_configs:
            ist = self.island_stats[ic.name]
            parts.append(f"{ic.name}={ist.best_fitness:.4f}")

        # Truncate if too many islands
        if len(parts) > 8:
            summary = ', '.join(parts[:4]) + f' ... ({len(parts)} islands)'
        else:
            summary = ', '.join(parts)

        self.logger.info(
            "[SUMMARY] Gen %d/%d (%.1fs): %s | migrations=%d",
            gen + 1, self.generations, elapsed,
            summary, len(self.migration_history),
        )
