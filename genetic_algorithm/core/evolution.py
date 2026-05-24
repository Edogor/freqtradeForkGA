"""
Main Evolution Engine

Coordinates the genetic algorithm evolution process.
Supports both single-objective and multi-objective (NSGA-II) optimization.
"""

import os
import random
import json
import hashlib
import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from genetic_algorithm.core.population import (
    Population, PopulationStats, apply_fitness_sharing, calculate_pairwise_distances
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.selection import select_parents
from genetic_algorithm.core.crossover import crossover, _enforce_min_entry_conditions, _fix_invalid_operators
from genetic_algorithm.core.mutation import mutate
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.evaluation.regime_aware import create_regime_aware_evaluator
from genetic_algorithm.core.nsga2 import (
    fast_non_dominated_sort,
    crowding_distance_assignment,
    extract_objectives_from_metrics,
    get_pareto_front,
    nsga2_crowded_comparison_sort,
    DEFAULT_OBJECTIVES
)
from genetic_algorithm.evaluation.parallel import ParallelEvaluator, is_parallel_available
from genetic_algorithm.core.feature_importance import FeatureImportanceTracker
from genetic_algorithm.core.hall_of_fame import HallOfFame
from genetic_algorithm.core.culling import StrategyCuller
from genetic_algorithm.utils.run_diagnostics import RunDiagnostics
from genetic_algorithm.llm.designer import StrategyDesigner
from genetic_algorithm.monitor import create_monitor
from genetic_algorithm.engine.checkpoint import CheckpointManager
from genetic_algorithm.engine.adaptive import AdaptiveController
from genetic_algorithm.engine.generation import GenerationStep


class GeneticAlgorithm:
    """
    Main genetic algorithm engine for evolving trading strategies.
    
    Coordinates the entire evolution process:
    1. Initialize population
    2. Evaluate fitness
    3. Select parents
    4. Apply crossover and mutation
    5. Create next generation
    6. Repeat
    """
    
    def __init__(self, config_path: str = "genetic_algorithm/config/ga_config.yaml", 
                 visualize: bool = False, interactive: bool = True):
        """
        Initialize the genetic algorithm.
        
        Args:
            config_path: Path to configuration file
            visualize: Whether to enable live visualization
            interactive: Whether to use interactive plotting (only applies if visualize=True)
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Validate config before proceeding
        from genetic_algorithm.utils.config_validator import validate_and_log
        if not validate_and_log(self.config):
            raise ValueError("GA configuration has errors. Check logs above.")
        
        # Extract configuration
        ga_config = self.config['genetic_algorithm']
        
        # Set random seed for reproducibility if specified
        self.random_seed = ga_config.get('random_seed')
        if self.random_seed is not None:
            random.seed(self.random_seed)
            try:
                import numpy as np
                np.random.seed(self.random_seed)
            except ImportError:
                pass  # NumPy not available, skip
            self.logger.info(f"Random seed set to {self.random_seed} for reproducibility")
        
        # --- Core GA parameters ---
        self.population_size = ga_config['population_size']
        self.generations = ga_config['generations']
        self.mutation_rate = ga_config['mutation_rate']
        self.crossover_rate = ga_config['crossover_rate']
        self.elite_size = ga_config['elite_size']
        self.tournament_size = ga_config.get('tournament_size', 3)
        self.adaptive_tournament = ga_config.get('adaptive_tournament', False)
        self.selection_method = ga_config.get('selection_method', 'tournament')
        self.crossover_method = ga_config.get('crossover_method', 'single_point')
        self.convergence_patience = ga_config.get('convergence_patience', 10)
        self.max_runtime_minutes = ga_config.get('max_runtime_minutes', None)
        
        # NSGA-II multi-objective settings
        self.mode = ga_config.get('mode', 'single_objective')
        self.nsga2_config = self.config.get('nsga2', {})
        self.objectives_config = self.nsga2_config.get('objectives', DEFAULT_OBJECTIVES)
        self.pareto_front_size = self.nsga2_config.get('pareto_front_size', 20)
        
        # Diversity preservation settings
        self.fitness_sharing = ga_config.get('fitness_sharing', True)
        self.sharing_radius = ga_config.get('sharing_radius', 0.3)
        self.diversity_threshold = ga_config.get('diversity_threshold', 0.15)
        self.allow_self_crossover = ga_config.get('allow_self_crossover', True)
        # Scale immigrants to population size (at least 3, up to 20% of pop)
        # so that diversity injection is meaningful regardless of pop size.
        _default_immigrants = max(3, self.population_size // 5)
        self.random_immigrants = ga_config.get('random_immigrants', _default_immigrants)
        
        # Behavioral distance blending (0 = structural only, 1 = behavioral only)
        behavioral_weight = ga_config.get('behavioral_distance_weight', 0.0)
        import genetic_algorithm.core.population as _pop_mod
        _pop_mod._BEHAVIORAL_DISTANCE_WEIGHT = float(behavioral_weight)
        
        # --- Adaptive parameters ---
        self.base_mutation_rate = self.mutation_rate
        self.adaptive_mutation = ga_config.get('adaptive_mutation', True)
        self.max_adaptation_factor = ga_config.get('max_adaptation_factor', 2.0)
        self.adaptation_step = ga_config.get('adaptation_step', 0.1)
        self.max_mutation_rate = ga_config.get('max_mutation_rate', 0.65)
        self.mutation_cooldown_factor = ga_config.get('mutation_cooldown_factor', 0.5)
        
        # --- Adaptive controller (delegates adaptive mutation + convergence) ---
        self._adaptive = AdaptiveController(self.config, self.logger)
        
        # --- Evolution tracking state ---
        self.current_generation = 0
        self.best_individual: Optional[Individual] = None
        self.generation_stats: List[PopulationStats] = []
        self.no_improvement_count = 0
        self.best_fitness_ever = 0.0
        self._new_best_this_gen = False
        self._catastrophic_restart_needed = False

        # External control hooks (set by web RunManager when run via dashboard)
        self._web_stop_event = None
        self._web_pause_event = None
        self._web_injection_queue = None
        self._web_run_id: Optional[str] = None
        self._web_monitor = None

        # External strategy injection
        self._external_immigrants: List[Individual] = []
        self._immigrant_provider: Optional[Callable[['GeneticAlgorithm', int], List[Individual]]] = None
        self._sis_integrator = None  # Set by _setup_sis() if sis.enabled=true

        # Graceful shutdown flag
        self._shutdown_requested = False
        
        # --- Delegate subsystem initialisation to focused setup methods ---
        self.strategy_generator = StrategyGenerator(self.config)
        self._setup_evaluator()
        self._setup_parallel()
        self._setup_visualization(visualize, interactive)
        self._setup_hall_of_fame()
        self._setup_holdout()
        self._setup_llm()
        self._setup_sis()
        self._setup_diagnostics()

        # --- Adaptive Operator Selection (AOS) ---
        from genetic_algorithm.core.operator_selection import AdaptiveOperatorSelector
        self._aos = AdaptiveOperatorSelector(self.config)
        if self._aos.enabled:
            self.logger.info("[AOS] Adaptive Operator Selection enabled")

        # --- Surrogate pre-filtering ---
        from genetic_algorithm.evaluation.surrogate import SurrogateModel
        self._surrogate = SurrogateModel(self.config)
        if self._surrogate.enabled:
            self.logger.info("[SURROGATE] Surrogate pre-filtering enabled")

        # --- MAP-Elites quality-diversity archive ---
        from genetic_algorithm.core.map_elites import MAPElitesArchive
        self._map_elites = MAPElitesArchive(self.config)
        if self._map_elites.enabled:
            self.logger.info("[MAP-ELITES] Quality-diversity archive enabled")

        # --- Phase 4: Incremental Evolution ---
        from genetic_algorithm.core.incremental_evolution import IncrementalEvolver
        self._incremental = IncrementalEvolver(self.config)
        if self._incremental.enabled:
            self.logger.info("[INCREMENTAL] Incremental evolution enabled")

        # --- Phase 4: Strategy Lifecycle Manager ---
        from genetic_algorithm.core.lifecycle_manager import LifecycleManager
        self._lifecycle = LifecycleManager(self.config)
        if self._lifecycle.enabled:
            self._lifecycle.load_state()
            self.logger.info(f"[LIFECYCLE] Strategy lifecycle manager enabled "
                             f"({self._lifecycle.get_summary()})")

        # --- Phase 4: Multi-Population Coevolution ---
        from genetic_algorithm.core.coevolution import CoevolutionEngine
        self._coevolution = CoevolutionEngine(
            self.config,
            strategy_generator=getattr(self, 'strategy_generator', None),
            fitness_evaluator=getattr(self, 'fitness_evaluator', None),
        )
        if self._coevolution.enabled:
            self.logger.info("[COEVOLUTION] Multi-population coevolution enabled")

        # Strategy culling
        self.culler = StrategyCuller(self.config, self.logger)
        if self.culler.enabled:
            self.logger.info("[CULL] Strategy culling enabled")

        # --- Experiment Tracker (persists data for dashboard) ---
        from genetic_algorithm.core.experiment_tracker import ExperimentTracker
        self._tracker = ExperimentTracker(self.config)
        self._tracker.initialise()

        # --- Generation step delegate ---
        self._generation_step = GenerationStep(
            population_size=self.population_size,
            elite_size=self.elite_size,
            mode=self.mode,
            config=self.config,
            logger=self.logger,
            strategy_generator=self.strategy_generator,
            fitness_evaluator=self.fitness_evaluator,
            parallel_enabled=self.parallel_enabled,
            parallel_evaluator=getattr(self, 'parallel_evaluator', None),
            crossover_rate=self.crossover_rate,
            crossover_method=self.crossover_method,
            tournament_size=self.tournament_size,
            selection_method=self.selection_method,
            allow_self_crossover=self.allow_self_crossover,
            adaptive_tournament=self.adaptive_tournament,
            random_immigrants=self.random_immigrants,
            diversity_threshold=self.diversity_threshold,
            map_elites=self._map_elites,
            aos=self._aos,
            immigrant_provider=self._immigrant_provider,
            llm_enabled=self.llm_enabled,
            strategy_designer=getattr(self, 'strategy_designer', None),
            feature_tracker=self.feature_tracker,
        )

    # ------------------------------------------------------------------
    # Private setup helpers (extracted from __init__ for readability)
    # ------------------------------------------------------------------

    def _setup_evaluator(self):
        """Initialise fitness evaluator (standard or regime-aware) + DSR tracker."""
        regime_config = self.config.get('regime_aware') or {}
        self.regime_aware_enabled = regime_config.get('enabled', False)
        
        if self.regime_aware_enabled:
            self.fitness_evaluator = create_regime_aware_evaluator(
                self.config, auto_detect=True)
            self.logger.info("=" * 70)
            self.logger.info("REGIME-AWARE EVALUATION ENABLED")
            self.logger.info(f"  Detection method: {regime_config.get('method', 'sma_adx')}")
            self.logger.info(f"  Aggregation: {regime_config.get('aggregation', 'harmonic_mean')}")
            self.logger.info(f"  Holdout ratio: {regime_config.get('holdout_ratio', 0.20):.0%}")
            self.logger.info("  Strategies evaluated across multiple market regimes")
            self.logger.info("=" * 70)
        else:
            self.fitness_evaluator = FitnessEvaluator(self.config)
        
        # Log walk-forward status
        wf_config = self.config.get('walk_forward', {})
        if wf_config.get('enabled', False):
            self.logger.info("=" * 80)
            self.logger.info("WALK-FORWARD OPTIMIZATION ENABLED")
            self.logger.info(f"  Train days: {wf_config.get('train_days')}")
            self.logger.info(f"  Validation days: {wf_config.get('validation_days')}")
            self.logger.info(f"  Step days: {wf_config.get('step_days')}")
            self.logger.info(f"  Mode: {wf_config.get('mode', 'rolling')}")
            self.logger.info(f"  Aggregation: {wf_config.get('aggregation', 'mean')}")
            self.logger.info("  Fitness = aggregated validation score (NOT training score)")
            self.logger.info("=" * 80)
        else:
            self.logger.info("Using standard single-period backtesting (walk-forward disabled)")
        
        # Log NSGA-II mode status
        if self.mode == 'nsga2':
            self.logger.info("=" * 70)
            self.logger.info("NSGA-II MULTI-OBJECTIVE OPTIMIZATION ENABLED")
            self.logger.info(f"  Objectives: {[obj['name'] for obj in self.objectives_config]}")
            self.logger.info(f"  Pareto front size: {self.pareto_front_size}")
            self.logger.info("  Selection method will be overridden to 'nsga2'")
            self.logger.info("=" * 70)
            self.selection_method = 'nsga2'
        
        # Feature importance tracking
        self.feature_tracker = FeatureImportanceTracker()
        self.logger.info("Feature importance tracking enabled")
        
        # Centralized DSR tracker
        from genetic_algorithm.evaluation.deflated_sharpe import DSRTracker
        self._global_dsr_tracker = DSRTracker(self.config)

    def _setup_parallel(self):
        """Initialise parallel evaluator if enabled."""
        parallel_config = self.config.get('parallel_evaluation', {})
        self.parallel_enabled = parallel_config.get('enabled', False)
        self.parallel_evaluator = None
        
        if self.parallel_enabled:
            if is_parallel_available():
                self.parallel_evaluator = ParallelEvaluator(
                    self.config,
                    num_workers=parallel_config.get('num_workers'))
                self.logger.info("=" * 70)
                self.logger.info("PARALLEL EVALUATION ENABLED")
                self.logger.info(f"  Workers: {self.parallel_evaluator.num_workers}")
                self.logger.info("  Expect 3-6x speedup on multi-core systems")
                self.logger.info("=" * 70)
            else:
                self.logger.warning("Parallel evaluation requested but multiprocessing not available")
                self.parallel_enabled = False

    def _setup_visualization(self, visualize: bool, interactive: bool):
        """Initialise live visualiser and trade visualiser."""
        self.visualizer = None
        if visualize:
            try:
                from genetic_algorithm.visualization import GAVisualizer
                self.visualizer = GAVisualizer(
                    enabled=True, interactive=interactive, save_plots=True)
            except ImportError as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Visualization disabled: {e}. Install matplotlib to enable visualization.")
        
        self.trade_visualizer = None
        trade_vis_config = self.config.get('trade_visualization', {})
        if trade_vis_config.get('enabled', False):
            try:
                from genetic_algorithm.visualization.trade_visualizer import TradeVisualizer
                self.trade_visualizer = TradeVisualizer(self.config, enabled=True)
                self.trade_vis_mode = trade_vis_config.get('mode', 'final')
                self.trade_vis_top_n = trade_vis_config.get('top_n_strategies', 3)
                self.logger.info(f"TradeVisualizer initialized (mode={self.trade_vis_mode}, top_n={self.trade_vis_top_n})")
            except ImportError as e:
                self.logger.warning(f"Trade visualization disabled: {e}")
                self.trade_visualizer = None

    def _setup_hall_of_fame(self):
        """Initialise Hall of Fame + checkpoint settings."""
        hof_config = self.config.get('hall_of_fame', {})
        hof_dir = hof_config.get('directory', 'genetic_algorithm/data/hall_of_fame')
        hof_max = hof_config.get('max_size', 50)
        hof_min_fitness = hof_config.get('min_fitness', 0.0)
        self.hall_of_fame = HallOfFame(
            directory=hof_dir, max_size=hof_max, min_fitness=hof_min_fitness,
            strategy_generator=getattr(self, 'strategy_generator', None))
        self.hof_inject_count = hof_config.get('inject_count', 3)
        if self.hall_of_fame.entries:
            self.logger.info(
                f"Hall of Fame loaded: {len(self.hall_of_fame.entries)} entries "
                f"(best fitness: {self.hall_of_fame.entries[0].fitness:.4f})")
        
        # Checkpoint settings
        storage_config = self.config.get('storage', {})
        self.checkpoint_dir = Path(storage_config.get('checkpoint_dir', 'genetic_algorithm/data/checkpoints'))
        self.checkpoint_interval = storage_config.get('checkpoint_interval', 5)
        
        # Delegate checkpoint I/O to CheckpointManager
        self._checkpoint_mgr = CheckpointManager(
            self.checkpoint_dir, self.checkpoint_interval, self.logger
        )

    def _setup_holdout(self):
        """Initialise holdout monitoring configuration."""
        holdout_mon_config = self.config.get('holdout_monitoring', {})
        self.holdout_monitoring_enabled = holdout_mon_config.get('enabled', False)
        self.holdout_monitoring_interval = holdout_mon_config.get('interval', 5)
        self.holdout_monitoring_top_n = holdout_mon_config.get('top_n', 3)
        self.holdout_fitness_penalty = holdout_mon_config.get('fitness_penalty', False)
        self.holdout_penalty_factor = holdout_mon_config.get('penalty_factor', 0.5)
        self.holdout_early_stop = holdout_mon_config.get('early_stop', False)
        self.holdout_early_stop_threshold = holdout_mon_config.get('early_stop_threshold', 0.60)
        self.holdout_early_stop_checks = holdout_mon_config.get('early_stop_checks', 2)
        self._holdout_evaluator = None
        self._holdout_range = None
        self._holdout_consecutive_bad = 0
        self._holdout_degradation_history: list = []
        self.holdout_trend_early_stop = holdout_mon_config.get('trend_early_stop', True)
        self.holdout_trend_checks = holdout_mon_config.get('trend_checks', 3)
        self.holdout_trend_min_degradation = holdout_mon_config.get('trend_min_degradation', 0.30)
        self.holdout_trend_slope_threshold = holdout_mon_config.get('trend_slope_threshold', 0.05)
        self.holdout_trend_grace_checks = holdout_mon_config.get('trend_grace_checks', 2)
        self.generation_holdout_history: list = []

    def _setup_llm(self):
        """Initialise LLM strategy designer + progress bar settings."""
        llm_config = self.config.get('advanced', {}).get('llm', {})
        self.llm_enabled = llm_config.get('enabled', False)
        self.strategy_designer = StrategyDesigner(self.config)
        if self.llm_enabled and self.strategy_designer.enabled:
            self.logger.info("=" * 70)
            self.logger.info("LLM STRATEGY DESIGNER ENABLED")
            self.logger.info(f"  Provider: {llm_config.get('provider', 'N/A')}")
            self.logger.info(f"  Model: {llm_config.get('model', 'default')}")
            self.logger.info(f"  Seed ratio: {self.strategy_designer.seed_ratio:.0%}")
            self.logger.info(f"  Immigrant ratio: {self.strategy_designer.immigrant_ratio:.0%}")
            self.logger.info("=" * 70)
        
        # Progress bar settings
        progress_config = self.config.get('progress', {})
        self.progress_enabled = progress_config.get('enabled', False) and TQDM_AVAILABLE
        self.progress_show_fitness = progress_config.get('show_fitness', True)
        self.progress_show_profit = progress_config.get('show_profit', True)
        self.progress_update_every = progress_config.get('update_every', 1)
        if self.progress_enabled:
            self.logger.info("Progress bar enabled")
        elif progress_config.get('enabled', False) and not TQDM_AVAILABLE:
            self.logger.warning("Progress bar requested but tqdm not installed. Run: pip install tqdm")

    def _setup_sis(self):
        """Initialize SIS integrator if configured."""
        sis_config = self.config.get('sis', {})
        if not sis_config.get('enabled', False):
            return
        try:
            from genetic_algorithm.intelligence.sis_integrator import SISIntegrator
            self._sis_integrator = SISIntegrator(self.config, self.logger)
            self._sis_integrator._adaptive_weights.configure_for_run_length(self.generations)
            self.set_immigrant_provider(self._sis_integrator.immigrant_provider)
            self.logger.info("[SIS] Strategy Intelligence System initialized")
        except Exception as e:
            self.logger.warning(f"[SIS] Failed to initialize: {e}")

    def _get_common_indicators(self, population, top_n: int = 3):
        """Get the most common indicator types in top-performing strategies."""
        from collections import Counter
        evaluated = [ind for ind in population.individuals
                     if ind.fitness is not None]
        if not evaluated:
            return []
        # Use top 30% by fitness
        evaluated.sort(key=lambda x: x.fitness, reverse=True)
        top_inds = evaluated[:max(3, len(evaluated) // 3)]
        counter = Counter()
        for ind in top_inds:
            for indicator in ind.strategy_gene.indicators:
                counter[indicator.type] += 1
        return [t for t, _ in counter.most_common(top_n)]

    def _setup_diagnostics(self):
        """Initialise run diagnostics and terminal monitor."""
        output_config = self.config.get('output', {})
        output_dir = Path(output_config.get('dir', output_config.get('directory', 'genetic_algorithm/output')))
        self.diagnostics = RunDiagnostics(output_dir)
        self.monitor = create_monitor(self.config)
    
    def _save_legacy_checkpoint(self, population: Population, generation: int):
        """
        Save a quick checkpoint in the legacy format for web dashboard compatibility.
        
        This writes to the fixed 'latest_checkpoint.json' used by the web dashboard.
        The full checkpoint system (save_checkpoint/load_checkpoint) uses
        timestamped files for proper resume support.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'generation': generation,
            'population': [ind.to_dict() for ind in population.individuals],
            'population_size': self.population_size,
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'best_fitness_ever': self.best_fitness_ever,
            'no_improvement_count': self.no_improvement_count,
            'mutation_rate': self.mutation_rate,
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
                for s in self.generation_stats
            ],
            'random_seed': self.random_seed,
            'timestamp': time.time(),
        }
        
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.json'
        temp_path = self.checkpoint_dir / 'latest_checkpoint.tmp'
        
        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            temp_path.rename(checkpoint_path)
        except Exception as e:
            self.logger.error(f"[CHECKPOINT] Legacy checkpoint write failed: {e}")
            if temp_path.exists():
                temp_path.unlink()
    
    def _load_legacy_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest legacy checkpoint if available (used by web dashboard).
        
        Returns:
            Checkpoint dictionary if found, None otherwise
        """
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.json'
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            self.logger.info(f"[CHECKPOINT] Found legacy checkpoint at generation {checkpoint['generation']} "
                           f"(saved at {checkpoint.get('timestamp', 'unknown')})")
            return checkpoint
        except Exception as e:
            self.logger.error(f"[CHECKPOINT] Failed to load legacy checkpoint: {e}")
            return None
    
    def _restore_from_legacy_checkpoint(self, checkpoint: Dict[str, Any]) -> Population:
        """
        Restore evolution state from a checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary from load_checkpoint()
            
        Returns:
            Restored population
        """
        self.current_generation = checkpoint['generation']
        self.best_fitness_ever = checkpoint.get('best_fitness_ever', 0.0)
        self.no_improvement_count = checkpoint.get('no_improvement_count', 0)
        self.mutation_rate = checkpoint.get('mutation_rate', self.base_mutation_rate)
        
        # Restore best individual
        if checkpoint.get('best_individual'):
            self.best_individual = Individual.from_dict(checkpoint['best_individual'])
        
        # Restore population
        population = Population(
            size=checkpoint.get('population_size', self.population_size),
            generation=self.current_generation
        )
        for ind_dict in checkpoint['population']:
            individual = Individual.from_dict(ind_dict)
            population.add_individual(individual)
        
        # Restore generation stats (partial — only serializable fields)
        self.generation_stats = []
        for s in checkpoint.get('generation_stats', []):
            stats = PopulationStats(
                generation=s.get('generation', 0),
                size=s.get('size', self.population_size),
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
            self.generation_stats.append(stats)
        
        self.logger.info(f"[CHECKPOINT] Restored: generation={self.current_generation}, "
                        f"population={len(population.individuals)}, "
                        f"best_fitness={self.best_fitness_ever:.4f}")
        
        return population
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file, applying schema defaults."""
        try:
            from genetic_algorithm.config.schema import load_config as _schema_load
            return _schema_load(config_path)
        except Exception:
            # Fallback: raw YAML load (e.g. during tests with patched _load_config)
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        log_config = self.config.get('logging', {})
        logger = logging.getLogger('GeneticAlgorithm')
        logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
        # Prevent duplicate logs by not propagating to root logger
        # The root logger is configured by run_ga.py setup_logging()
        logger.propagate = False
        
        # Determine if terminal monitor is active
        # Default to True (matching create_monitor) so log suppression
        # activates even when the terminal_monitor section is missing.
        monitor_cfg = self.config.get('terminal_monitor', {})
        monitor_active = monitor_cfg.get('enabled', True)
        try:
            import rich  # noqa: F401
        except ImportError:
            monitor_active = False
        
        # When monitor is active, strip any existing console StreamHandlers
        # (they may have been added by a previous init or library code)
        if monitor_active:
            for h in list(logger.handlers):
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
        
        # Only add handlers if not already added (avoid duplicate handlers on re-init)
        if not logger.handlers:
            # Create formatter
            log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter(log_format)

            log_file = log_config.get('file')

            # When terminal monitor is active, suppress console StreamHandler
            # (the monitor captures logs via its own handler for the Logs view).
            # Also suppress StreamHandler when a log file is configured AND stdout
            # is redirected (not a real TTY): in that case the process is launched
            # with stdout pointing to the same log file (> logfile 2>&1), so
            # StreamHandler + FileHandler would write every line twice.
            import sys as _sys
            stdout_is_tty = _sys.stdout.isatty()
            if not monitor_active and log_config.get('console', True) and (not log_file or stdout_is_tty):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # Add file handler if log file path is specified
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                # Use unbuffered file handler to ensure real-time log visibility
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                file_handler.flush = lambda: file_handler.stream.flush()
                logger.addHandler(file_handler)
        
        return logger
    
    # ========================================================================
    # CHECKPOINT SAVE/LOAD SYSTEM
    # ========================================================================
    
    def save_checkpoint(self, population: 'Population', generation: int, 
                        filepath: Optional[str] = None) -> str:
        """
        Save current evolution state to a checkpoint file.
        
        Delegates to CheckpointManager while preserving the original API.
        """
        ga_state = {
            'best_individual': self.best_individual,
            'best_fitness_ever': self.best_fitness_ever,
            'no_improvement_count': self.no_improvement_count,
            'current_mutation_rate': self.mutation_rate,
            'base_mutation_rate': self.base_mutation_rate,
            'catastrophic_restart_needed': self._catastrophic_restart_needed,
            'total_generations': self.generations,
            'random_seed': self.random_seed,
        }
        config_snapshot = {
            'genetic_algorithm': self.config.get('genetic_algorithm', {}),
            'backtesting': self.config.get('backtesting', {}),
            'walk_forward': self.config.get('walk_forward', {}),
        }
        extras = {
            'feature_tracker': self.feature_tracker.to_dict() if hasattr(self, 'feature_tracker') else None,
            'holdout_state': {
                'consecutive_bad': self._holdout_consecutive_bad,
                'degradation_history': list(self._holdout_degradation_history),
                'generation_holdout_history': [
                    h if isinstance(h, dict) else getattr(h, '__dict__', {})
                    for h in self.generation_holdout_history
                ],
            },
            'aos_state': self._aos.to_dict() if self._aos.enabled else None,
            'surrogate_state': self._surrogate.to_dict() if self._surrogate.enabled else None,
            'map_elites_state': self._map_elites.to_dict() if self._map_elites.enabled else None,
        }
        return self._checkpoint_mgr.save(
            population, generation, ga_state, config_snapshot,
            self.generation_stats, extras=extras, filepath=filepath,
            monitor=self.monitor,
        )
    
    def load_checkpoint(self, filepath: str) -> tuple:
        """
        Load evolution state from a checkpoint file.
        
        Delegates to CheckpointManager, then restores GA-specific sub-system
        state on self.
        
        Args:
            filepath: Path to checkpoint JSON file
            
        Returns:
            Tuple of (population, start_generation)
        """
        population, start_generation, state = self._checkpoint_mgr.load(
            filepath, self.population_size
        )

        # Restore GA state onto self
        ga_state = state.get('ga_state', {})

        best_ind = ga_state.get('best_individual')
        if best_ind is not None:
            self.best_individual = best_ind

        self.best_fitness_ever = ga_state.get('best_fitness_ever', 0.0)
        self.no_improvement_count = ga_state.get('no_improvement_count', 0)
        self.mutation_rate = ga_state.get('current_mutation_rate', self.mutation_rate)
        self.base_mutation_rate = ga_state.get('base_mutation_rate', self.base_mutation_rate)
        self._catastrophic_restart_needed = ga_state.get('catastrophic_restart_needed', False)

        # Sync adaptive controller
        self._adaptive.load_state({
            'mutation_rate': self.mutation_rate,
            'base_mutation_rate': self.base_mutation_rate,
            'no_improvement_count': self.no_improvement_count,
            'catastrophic_restart_needed': self._catastrophic_restart_needed,
        })

        # Restore feature tracker
        ft_data = state.get('feature_tracker')
        if ft_data and hasattr(self, 'feature_tracker'):
            try:
                self.feature_tracker.load_from_dict(ft_data)
                self.logger.info(f"[CHECKPOINT] Restored feature tracker ({ft_data.get('total_generations', 0)} generations of data)")
            except Exception as e:
                self.logger.warning(f"[CHECKPOINT] Failed to restore feature tracker: {e}")

        # Restore holdout monitoring state
        holdout_state = state.get('holdout_state', {})
        if holdout_state:
            self._holdout_consecutive_bad = holdout_state.get('consecutive_bad', 0)
            self._holdout_degradation_history = holdout_state.get('degradation_history', [])
            self.generation_holdout_history = holdout_state.get('generation_holdout_history', [])

        # Restore adaptive modules
        aos_data = state.get('aos_state')
        if aos_data and self._aos.enabled:
            try:
                self._aos.load_from_dict(aos_data)
                self.logger.info("[CHECKPOINT] Restored AOS state")
            except Exception as e:
                self.logger.warning(f"[CHECKPOINT] Failed to restore AOS state: {e}")

        me_data = state.get('map_elites_state')
        if me_data and self._map_elites.enabled:
            try:
                self._map_elites.load_from_dict(me_data)
                self.logger.info(f"[CHECKPOINT] Restored MAP-Elites archive ({self._map_elites.filled_cells} cells)")
            except Exception as e:
                self.logger.warning(f"[CHECKPOINT] Failed to restore MAP-Elites state: {e}")

        surr_data = state.get('surrogate_state')
        if surr_data and self._surrogate.enabled:
            try:
                self._surrogate.load_from_dict(surr_data)
                self.logger.info(
                    "[CHECKPOINT] Restored surrogate state "
                    f"(trained={surr_data.get('is_trained')}, "
                    f"R²={surr_data.get('last_validation_r2')}, "
                    f"gens_since_retrain={surr_data.get('generations_since_retrain')})"
                )
            except Exception as e:
                self.logger.warning(f"[CHECKPOINT] Failed to restore surrogate state: {e}")

        # Restore generation stats
        self.generation_stats = state.get('generation_stats', [])

        return population, start_generation
    
    # ========================================================================
    # EXTERNAL IMMIGRANT INJECTION (LLM / FILE-BASED SEEDING)
    # ========================================================================
    
    def set_immigrant_provider(self, provider: Callable[['GeneticAlgorithm', int], List[Individual]]):
        """
        Register a callback that provides external immigrants each generation.
        
        The provider function receives (ga_instance, generation_number) and should
        return a list of Individual objects to inject as immigrants. These replace
        a portion of the random immigrants (configured via immigrant_source_ratio).
        
        Example:
            def llm_provider(ga, gen):
                # Generate strategies via LLM based on best performers
                best = ga.best_individual
                return [create_llm_strategy(best, gen)]
            
            ga.set_immigrant_provider(llm_provider)
        
        Args:
            provider: Callable that returns List[Individual]
        """
        self._immigrant_provider = provider
        self.logger.info("[IMMIGRANTS] External immigrant provider registered")
    
    def inject_immigrants(self, immigrants: List[Individual]):
        """
        Queue external immigrants for injection in the next generation.
        
        These are one-shot; they're consumed when the next generation is created.
        For persistent injection, use set_immigrant_provider().
        
        Args:
            immigrants: List of Individual objects to inject
        """
        self._external_immigrants.extend(immigrants)
        self.logger.info(f"[IMMIGRANTS] Queued {len(immigrants)} external immigrants for next generation")
    
    def load_seed_strategies(self, filepath: str) -> List[Individual]:
        """
        Load strategies from a JSON file and return as Individual objects.
        
        Can be used to seed initial population or inject as immigrants.
        The JSON file should contain a list of individual dicts (as saved by checkpoint).
        
        Args:
            filepath: Path to JSON file with strategy definitions
            
        Returns:
            List of Individual objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Support both full checkpoint format and plain list of individuals
        if isinstance(data, dict) and 'population' in data:
            # Checkpoint format
            individuals_data = data['population']['individuals']
        elif isinstance(data, dict) and 'individuals' in data:
            individuals_data = data['individuals']
        elif isinstance(data, list):
            individuals_data = data
        else:
            raise ValueError(f"Unrecognized format in {filepath}")
        
        individuals = [Individual.from_dict(d) for d in individuals_data]
        self.logger.info(f"[SEED] Loaded {len(individuals)} strategies from {filepath}")
        return individuals
    
    def request_shutdown(self):
        """Request graceful shutdown after current generation completes."""
        self._shutdown_requested = True
        self.logger.info("[SHUTDOWN] Graceful shutdown requested. Will save checkpoint after current generation.")
    
    def initialize_population(self) -> Population:
        """
        Create initial population with a mix of seeded archetypes and random strategies.
        
        Seeded strategies (10-20% of population) provide known-good building blocks
        for crossover, accelerating convergence. The rest are random for diversity.
        
        Returns:
            Initial population
        """
        self.logger.info(f"Initializing population with {self.population_size} individuals")
        
        population = Population(size=self.population_size, generation=0)
        
        # ── Warm-start: inject individuals from a previous experiment ──
        warm_injected = 0
        try:
            from genetic_algorithm.core.warm_start import WarmStartLoader
            ws_loader = WarmStartLoader(self.config, self.logger)
            ws_individuals = ws_loader.load()
            for ind in ws_individuals:
                if len(population) >= self.population_size:
                    break
                _enforce_min_entry_conditions(ind.strategy_gene, self.config)
                ind.strategy_gene.assign_instance_ids()
                population.add_individual(ind)
                warm_injected += 1
            if warm_injected > 0:
                self.logger.info(f"[WARM-START] Injected {warm_injected} strategies from previous experiment")
        except Exception as e:
            self.logger.warning(f"[WARM-START] Skipped: {e}")
        
        # Seed 15% of population with known-good archetype strategies
        seed_count = max(1, int(self.population_size * 0.15))
        seed_start_id = 0
        
        # Re-inject hall of fame members (up to inject_count)
        hof_injected = 0
        try:
            hof_individuals = self.hall_of_fame.get_individuals(self.hof_inject_count)
            for ind in hof_individuals:
                # Enforce min_entry_conditions on HoF strategies
                _enforce_min_entry_conditions(ind.strategy_gene, self.config)
                # Reset individual_id to stay within current population's range
                ind.strategy_gene.individual_id = self.population_size + hof_injected
                ind.strategy_gene.generation = 0
                population.add_individual(ind)
                hof_injected += 1
            if hof_injected > 0:
                self.logger.info(f"Injected {hof_injected} hall-of-fame strategies")
        except Exception as e:
            self.logger.warning(f"Hall of fame injection failed: {e}")
        
        try:
            from genetic_algorithm.core.seed_strategies import create_seed_population
            seed_genes = create_seed_population(
                generation=0,
                count=seed_count,
                config=self.config,
                start_id=seed_start_id
            )
            for gene in seed_genes:
                individual = Individual(strategy_gene=gene)
                population.add_individual(individual)
                seed_start_id += 1
            self.logger.info(f"Seeded {len(seed_genes)} strategies from known archetypes")
        except Exception as e:
            self.logger.warning(f"Failed to seed population: {e}. Using all random strategies.")
            seed_start_id = 0
        
        # Fill remaining slots: LLM-generated + random strategies
        remaining = self.population_size - len(population)
        
        # LLM seed generation (configurable ratio of remaining slots)
        llm_count = 0
        if self.llm_enabled and self.strategy_designer.enabled and remaining > 0:
            llm_count = int(remaining * self.strategy_designer.seed_ratio)
            if llm_count > 0:
                self.logger.info(f"[LLM] Generating {llm_count} seed strategies via LLM...")
                # Use batch generation when enabled and count > 1
                if self.strategy_designer.batch_enabled and llm_count > 1:
                    llm_genes = self.strategy_designer.generate_seed_strategies_batch(
                        count=llm_count,
                        generation=0,
                        start_id=seed_start_id,
                    )
                else:
                    llm_genes = self.strategy_designer.generate_seed_strategies(
                        count=llm_count,
                        generation=0,
                        start_id=seed_start_id,
                    )
                for gene in llm_genes:
                    individual = Individual(strategy_gene=gene)
                    individual.metrics['origin'] = 'llm_seed'
                    if self.strategy_designer and hasattr(self.strategy_designer, '_last_provider_used'):
                        individual.metrics['llm_provider'] = self.strategy_designer._last_provider_used
                    population.add_individual(individual)
                llm_count = len(llm_genes)
                self.logger.info(f"[LLM] Added {llm_count} LLM-generated seeds")
        
        # Fill rest with random strategies
        random_remaining = self.population_size - len(population)
        for i in range(random_remaining):
            strategy_gene = self.strategy_generator.generate_random_strategy(
                generation=0,
                individual_id=seed_start_id + llm_count + i
            )
            individual = Individual(strategy_gene=strategy_gene)
            population.add_individual(individual)
        
        self.logger.info(f"Population initialized: {hof_injected} hall-of-fame + {seed_count} seeded + {llm_count} LLM + {random_remaining} random")

        # SIS seed quality filtering
        if self._sis_integrator is not None:
            try:
                self._sis_integrator.filter_population(population, self)
            except Exception as e:
                self.logger.warning(f"[SIS] Seed filtering failed: {e}")

        return population
    
    def evaluate_population(self, population: Population):
        """
        Evaluate fitness for all unevaluated individuals.
        
        Uses parallel evaluation if enabled in config, otherwise
        evaluates sequentially.
        
        Args:
            population: Population to evaluate
        """
        unevaluated = [ind for ind in population if not ind.evaluated]
        
        if not unevaluated:
            self.logger.info("[EVAL] All individuals already evaluated (using cache)")
            return
        
        # Surrogate pre-filtering: assign predicted fitness to low-promise candidates
        if self._surrogate.enabled and self._surrogate.ready:
            unevaluated, skipped = self._surrogate.filter_candidates(unevaluated)
            if skipped:
                self.logger.info(f"[SURROGATE] Skipped {len(skipped)} candidates (surrogate-scored)")
        
        # Use parallel evaluation if enabled
        if self.parallel_enabled and self.parallel_evaluator:
            self._evaluate_population_parallel(unevaluated)
        else:
            self._evaluate_population_sequential(unevaluated)
        
        # Deferred validation: run full pair-split on top-N only
        evaluator = self.fitness_evaluator
        # Unwrap RegimeAwareEvaluator if present
        if hasattr(evaluator, 'base_evaluator'):
            evaluator = evaluator.base_evaluator
        if hasattr(evaluator, 'run_deferred_validation'):
            validated = evaluator.run_deferred_validation(population)
            if validated > 0:
                self.logger.info(f"[EVAL] Deferred validation completed for {validated} top candidates")

    # ── External-control helpers (web dashboard) ────────────────

    def _drain_injection_queue(self, population: 'Population', gen: int):
        """
        Drain the injection queue and add any injected strategies to the population.

        Called at the start of each generation when the web injection queue is set.
        Handles both strategy gene dicts and command sentinels (e.g. checkpoint requests).
        """
        import queue as _queue_mod
        injected = 0
        while True:
            try:
                item = self._web_injection_queue.get_nowait()
            except (_queue_mod.Empty, Exception):
                break

            # Handle command sentinels
            if isinstance(item, dict) and item.get("_command") == "checkpoint":
                self.logger.info("[WEB] Checkpoint requested via injection queue")
                self.save_checkpoint(population, gen)
                continue

            # Treat as strategy gene dict
            try:
                from genetic_algorithm.core.strategy_gene import StrategyGene
                gene = StrategyGene.from_dict(item)
                gene.generation = gen
                gene.individual_id = self.population_size + injected
                individual = Individual(strategy_gene=gene)
                population.add_individual(individual)
                injected += 1
                self.logger.info(f"[WEB] Injected strategy {individual.id} into population")
            except Exception as e:
                self.logger.warning(f"[WEB] Failed to inject strategy: {e}")

        if injected:
            self.logger.info(f"[WEB] Injected {injected} strategies this generation")

    def get_state_snapshot(self) -> dict:
        """
        Return a lightweight snapshot of current evolution state.

        Used by the web dashboard for real-time status without deep-copying
        the entire population.
        """
        return {
            "current_generation": self.current_generation,
            "total_generations": self.generations,
            "best_fitness_ever": self.best_fitness_ever,
            "no_improvement_count": self.no_improvement_count,
            "mutation_rate": self.mutation_rate,
            "best_individual_id": self.best_individual.id if self.best_individual else None,
            "best_profit": self.best_individual.metrics.get("profit") if self.best_individual and self.best_individual.metrics else None,
            "generation_stats_count": len(self.generation_stats),
        }

    def _post_hoc_walk_forward_validation(self, population: 'Population'):
        """
        Run walk-forward validation on elite candidates after parallel evaluation.
        
        When parallel evaluation is enabled, walk-forward is disabled inside workers
        to avoid the N×W backtest explosion. Instead, we validate the top candidates
        here — in parallel if possible, sequential otherwise.
        
        Only runs if walk_forward.enabled=True and parallel_evaluation.enabled=True.
        Re-evaluates the top `elite_size * 2` individuals with walk-forward and
        replaces their fitness scores with the walk-forward-validated scores.
        """
        wf_config = self.config.get('walk_forward', {})
        if not wf_config.get('enabled', False):
            return
        if not self.parallel_enabled:
            return  # WF already ran inside sequential evaluation
        
        # Get top candidates to validate
        n_validate = min(self.elite_size * 2, len(population.individuals))
        candidates = population.get_best(n_validate)
        
        self.logger.info(f"[WF-POSTHOC] Validating top {len(candidates)} strategies with walk-forward...")
        
        # Use flat WF parallelization (preferred) - submits individual
        # (candidate × window) tasks to the PERSISTENT pool, avoiding the
        # ephemeral WF pool that caused deadlocks and resource exhaustion.
        if self.parallel_evaluator and len(candidates) > 1:
            from genetic_algorithm.evaluation.parallel import parallel_walk_forward_flat
            
            timeout = self.config.get('parallel_evaluation', {}).get('backtest_timeout', 120)
            validated = parallel_walk_forward_flat(
                candidates=candidates,
                config=self.config,
                evaluator=self.parallel_evaluator,
                backtest_timeout=timeout,
            )
            self.logger.info(f"[WF-POSTHOC] Flat WF validation complete: {validated}/{len(candidates)}")
        else:
            # Fallback: sequential validation
            wf_evaluator = FitnessEvaluator(self.config)
            validated = 0
            for ind in candidates:
                try:
                    wf_fitness, wf_metrics = wf_evaluator.evaluate(ind.strategy_gene)
                    original_fitness = ind.fitness
                    ind.set_fitness(wf_fitness, wf_metrics)
                    validated += 1
                    self.logger.debug(
                        f"[WF-POSTHOC] {ind.id}: {original_fitness:.4f} -> {wf_fitness:.4f} "
                        f"(gap={wf_metrics.get('train_val_gap', 0):.4f})"
                    )
                except Exception as e:
                    self.logger.warning(f"[WF-POSTHOC] Failed for {ind.id}: {e}")
            self.logger.info(f"[WF-POSTHOC] Validated {validated}/{len(candidates)} strategies")
    
    def _run_holdout_test(self, population: 'Population', holdout_config: Dict):
        """
        Run holdout/out-of-sample test on top strategies after evolution completes.
        
        Tests the best strategies on a separate time period that was NOT used during
        evolution, to check for overfitting.
        
        Config:
            holdout_test:
              enabled: true
              timerange: "20250301-20250401"  # Must be outside training range
              top_n: 5  # Number of strategies to test
        """
        holdout_timerange = holdout_config.get('timerange', '')
        top_n = holdout_config.get('top_n', 5)
        
        if not holdout_timerange:
            self.logger.warning("[HOLDOUT] No holdout timerange configured, skipping")
            return
        
        candidates = population.get_best(top_n)
        if not candidates:
            return
        
        self.logger.info("")
        self.logger.info(f"{'─'*70}")
        self.logger.info(f"HOLDOUT TEST - Out-of-sample validation on {holdout_timerange}")
        self.logger.info(f"{'─'*70}")
        
        # Create a modified config with the holdout timerange
        import copy
        holdout_eval_config = copy.deepcopy(self.config)
        holdout_eval_config['backtesting']['timerange'] = holdout_timerange
        # Disable walk-forward for holdout (it's a straight backtest)
        if 'walk_forward' in holdout_eval_config:
            holdout_eval_config['walk_forward']['enabled'] = False
        
        holdout_evaluator = FitnessEvaluator(holdout_eval_config)
        
        results = []
        for ind in candidates:
            try:
                holdout_fitness, holdout_metrics = holdout_evaluator.evaluate(ind.strategy_gene)
                train_fitness = ind.fitness
                overfit_ratio = (train_fitness - holdout_fitness) / max(abs(train_fitness), 0.1)
                
                results.append({
                    'id': ind.id,
                    'train_fitness': train_fitness,
                    'holdout_fitness': holdout_fitness,
                    'overfit_ratio': overfit_ratio,
                    'holdout_profit': holdout_metrics.get('profit', 0),
                    'holdout_trades': holdout_metrics.get('total_trades', 0),
                })
                
                status = "✓" if holdout_fitness > 0 else "✗"
                self.logger.info(
                    f"  {status} {ind.id}: train={train_fitness:.4f} -> holdout={holdout_fitness:.4f} "
                    f"(overfit={overfit_ratio:+.1%}, profit={holdout_metrics.get('profit', 0):.2f}%, "
                    f"trades={holdout_metrics.get('total_trades', 0)})"
                )
            except Exception as e:
                self.logger.warning(f"  ✗ {ind.id}: holdout test failed: {e}")
        
        if results:
            avg_overfit = sum(r['overfit_ratio'] for r in results) / len(results)
            profitable = sum(1 for r in results if r['holdout_fitness'] > 0)
            self.logger.info(f"\n  Summary: {profitable}/{len(results)} profitable on holdout, "
                           f"avg overfit ratio: {avg_overfit:+.1%}")
    
    def _run_post_evolution_cpcv(self, population, cpcv_config: dict):
        """
        Run Combinatorial Purged Cross-Validation on top strategies after evolution.
        
        Splits the training timerange into N blocks, runs backtests per block for
        each top strategy, then computes PBO to quantify overfitting risk.
        
        Config:
            cpcv:
              enabled: true
              top_n: 5            # Number of strategies to validate
              n_groups: 6         # Number of time blocks
        """
        import copy
        import numpy as np
        
        top_n = cpcv_config.get('top_n', 5)
        n_groups = cpcv_config.get('n_groups', 6)
        
        candidates = population.get_best(top_n)
        if len(candidates) < 2:
            self.logger.info("[CPCV] Need at least 2 strategies, skipping")
            return
        
        self.logger.info("")
        self.logger.info(f"{'─'*70}")
        self.logger.info(f"CPCV POST-EVOLUTION VALIDATION ({len(candidates)} strategies, {n_groups} blocks)")
        self.logger.info(f"{'─'*70}")
        
        # Parse training timerange into date boundaries
        try:
            from genetic_algorithm.utils.timerange import parse_timerange
            timerange_str = self.config.get('backtesting', {}).get('timerange', '')
            if not timerange_str:
                self.logger.warning("[CPCV] No timerange configured, skipping")
                return
            start_dt, end_dt = parse_timerange(timerange_str)
            total_days = (end_dt - start_dt).days
            if total_days < n_groups * 7:
                self.logger.warning(f"[CPCV] Timerange too short ({total_days} days) for {n_groups} blocks")
                return
        except Exception as e:
            self.logger.warning(f"[CPCV] Could not parse timerange: {e}")
            return
        
        # Create time blocks
        from datetime import timedelta
        block_days = total_days / n_groups
        block_ranges = []
        for i in range(n_groups):
            b_start = start_dt + timedelta(days=int(i * block_days))
            b_end = start_dt + timedelta(days=int((i + 1) * block_days)) if i < n_groups - 1 else end_dt
            block_ranges.append(f"{b_start.strftime('%Y%m%d')}-{b_end.strftime('%Y%m%d')}")
        
        # Run backtests for each strategy on each block
        strategy_results = {}
        for ind in candidates:
            block_profits = []
            for block_idx, block_tr in enumerate(block_ranges):
                try:
                    block_config = copy.deepcopy(self.config)
                    block_config['backtesting']['timerange'] = block_tr
                    if 'walk_forward' in block_config:
                        block_config['walk_forward']['enabled'] = False
                    if 'monte_carlo' in block_config:
                        block_config['monte_carlo']['enabled'] = False
                    
                    block_evaluator = FitnessEvaluator(block_config)
                    _, block_metrics = block_evaluator.evaluate(ind.strategy_gene)
                    block_profits.append(block_metrics.get('profit', 0.0))
                except Exception as e:
                    self.logger.debug(f"[CPCV] Block {block_idx} failed for {ind.id}: {e}")
                    block_profits.append(0.0)
            
            strategy_results[ind.id] = np.array(block_profits)
            self.logger.debug(f"[CPCV] {ind.id} block profits: {block_profits}")
        
        # Run CPCV validation
        try:
            from genetic_algorithm.evaluation.cpcv import CPCVValidator
            validator = CPCVValidator(self.config)
            # Override enabled since we're calling explicitly
            validator.enabled = True
            validator.n_groups = n_groups
            result = validator.validate_strategies(strategy_results, timerange_str)
            
            pbo = result.get('pbo', 0.0)
            
            self.logger.info(f"\n  PBO (Probability of Backtest Overfitting): {pbo:.3f}")
            if pbo < 0.3:
                self.logger.info("  --> LOW overfitting risk. Strategies likely have genuine edge.")
            elif pbo < 0.6:
                self.logger.info("  --> MODERATE overfitting risk. Proceed with caution.")
            else:
                self.logger.info("  --> HIGH overfitting risk. Strategies may be curve-fitted.")
            
            # Log per-strategy OOS performance
            per_strategy = result.get('per_strategy_oos', {})
            for sid, stats in per_strategy.items():
                self.logger.info(f"  {sid}: OOS mean={stats['mean_oos']:.2f}%, "
                               f"std={stats['std_oos']:.2f}%, min={stats['min_oos']:.2f}%")
        except Exception as e:
            self.logger.warning(f"[CPCV] Validation failed: {e}")

    def _update_diversity_references(self, population: 'Population'):
        """
        Feed top strategies' monthly profit vectors into the fitness evaluator
        so future evaluations get a diversity bonus/penalty.
        """
        try:
            # Resolve base evaluator (may be wrapped by RegimeAwareEvaluator)
            evaluator = self.fitness_evaluator
            if hasattr(evaluator, 'base_evaluator'):
                evaluator = evaluator.base_evaluator
            if not getattr(evaluator, '_diversity_enabled', False):
                return

            top = population.get_best(max(self.elite_size, 5))
            vectors = []
            for ind in top:
                mp = (ind.metrics or {}).get('monthly_profits')
                if mp and len(mp) >= 2:
                    vectors.append(mp)
            evaluator.update_diversity_references(vectors)
        except Exception as e:
            self.logger.debug(f"Diversity reference update failed: {e}")

    def _run_holdout_monitoring(self, population: 'Population', generation: int):
        """
        Periodic holdout check during evolution.
        
        Evaluates top-N elites on holdout data and logs the results.
        When holdout_fitness_penalty is enabled, applies a soft multiplicative
        fitness adjustment to overfit elites so they're disfavored in selection
        while keeping their genetic material alive.
        """
        if not self.holdout_monitoring_enabled:
            return
        
        # Only run at specified intervals
        if (generation + 1) % self.holdout_monitoring_interval != 0:
            return
        
        # Get holdout split from the validation config
        holdout_config = self.config.get('holdout_validation', {})
        holdout_pct = holdout_config.get('holdout_pct', 0.15)
        original_timerange = self.config.get('backtesting', {}).get('timerange', '')
        
        if not original_timerange or not holdout_config.get('enabled', False):
            return
        
        try:
            evo_range, holdout_range = FitnessEvaluator.split_timerange_for_holdout(
                original_timerange, holdout_pct
            )
        except Exception as e:
            self.logger.debug(f"[HOLDOUT-MON] Could not split timerange: {e}")
            return
        
        candidates = population.get_best(self.holdout_monitoring_top_n)
        if not candidates:
            return
        
        self.logger.info(f"[HOLDOUT-MON] Gen {generation + 1}: evaluating top-{len(candidates)} on {holdout_range}...")
        
        # Reuse cached holdout evaluator (same holdout range every call)
        if self._holdout_evaluator is None or self._holdout_range != holdout_range:
            import copy
            holdout_eval_config = copy.deepcopy(self.config)
            holdout_eval_config['backtesting']['timerange'] = holdout_range
            if 'walk_forward' in holdout_eval_config:
                holdout_eval_config['walk_forward']['enabled'] = False
            try:
                self._holdout_evaluator = FitnessEvaluator(holdout_eval_config)
                self._holdout_range = holdout_range
                self.logger.debug(f"[HOLDOUT-MON] Created & cached evaluator for {holdout_range}")
            except Exception as e:
                self.logger.warning(f"[HOLDOUT-MON] Failed to create evaluator: {e}")
                return
        holdout_evaluator = self._holdout_evaluator
        
        degradations = []
        for ind in candidates:
            try:
                holdout_fitness, holdout_metrics = holdout_evaluator.evaluate(ind.strategy_gene)
                train_fitness = ind.raw_fitness if ind.raw_fitness is not None else ind.fitness
                # Use larger floor (0.1) to avoid massive percentages when
                # train_fitness is near-zero, and clamp to [-500%, 500%]
                degradation = (train_fitness - holdout_fitness) / max(abs(train_fitness), 0.1) * 100
                degradation = max(-500.0, min(500.0, degradation))
                degradations.append(degradation)
                
                symbol = "✓" if degradation < 30 else "⚠"
                self.logger.info(
                    f"  {symbol} {ind.id}: train={train_fitness:.4f} hold={holdout_fitness:.4f} "
                    f"(degrad={degradation:.1f}%, hold_profit={holdout_metrics.get('profit', 0):.2f}%)"
                )
                
                # Apply holdout fitness penalty when enabled
                # Penalty MUST hit raw_fitness too — elite selection sorts by
                # raw_fitness, and fitness sharing overwrites .fitness from
                # raw_fitness.  Without touching raw_fitness the penalty is a no-op.
                if self.holdout_fitness_penalty and degradation > 0:
                    degradation_frac = degradation / 100.0  # Convert back to 0-1
                    penalty_mult = max(0.3, 1.0 - degradation_frac * self.holdout_penalty_factor)
                    old_raw = ind.raw_fitness if ind.raw_fitness is not None else ind.fitness
                    old_fitness = ind.fitness
                    # Store pre-holdout raw_fitness so elite carry-over can
                    # restore the un-penalized value.  Without this, elites
                    # accumulate holdout penalties across generations because
                    # fitness sharing uses raw_fitness as its base.
                    ind._pre_holdout_raw_fitness = old_raw
                    ind.raw_fitness = old_raw * penalty_mult
                    ind.fitness = ind.fitness * penalty_mult
                    ind.metrics['holdout_penalty'] = 1.0 - penalty_mult
                    ind.metrics['holdout_degradation_monitored'] = degradation_frac
                    self.logger.info(
                        f"    → Holdout penalty applied: raw {old_raw:.4f}->{ind.raw_fitness:.4f}, "
                        f"fit {old_fitness:.4f}->{ind.fitness:.4f} "
                        f"(x{penalty_mult:.3f}, degrad={degradation:.1f}%)"
                    )
            except Exception as e:
                self.logger.debug(f"  ✗ {ind.id}: holdout monitoring failed: {e}")
        
        if degradations:
            avg_degrad = sum(degradations) / len(degradations)
            best_degrad = min(degradations)
            worst_degrad = max(degradations)
            self.logger.info(f"  [HOLDOUT-MON] Avg degradation: {avg_degrad:.1f}%")
            
            # Store in generation stats (if current gen stats exist)
            if self.generation_stats:
                latest_stats = self.generation_stats[-1]
                latest_stats.holdout_avg_degradation = avg_degrad
                latest_stats.holdout_best_degradation = best_degrad
                latest_stats.holdout_num_evaluated = len(degradations)
                latest_stats.holdout_num_profitable = sum(1 for d in degradations if d < 30)
            
            # Append to holdout history for reporting
            from genetic_algorithm.utils.overfit_analysis import GenerationHoldoutStats
            holdout_stat = GenerationHoldoutStats(
                generation=generation,
                avg_degradation=avg_degrad,
                best_degradation=best_degrad,
                worst_degradation=worst_degrad,
                num_evaluated=len(degradations),
                num_profitable=sum(1 for d in degradations if d < 30),
            )
            self.generation_holdout_history.append(holdout_stat)
            
            # Holdout-aware early stopping check
            if self.holdout_early_stop:
                threshold_pct = self.holdout_early_stop_threshold * 100
                if avg_degrad > threshold_pct:
                    self._holdout_consecutive_bad += 1
                    self.logger.warning(
                        f"  [HOLDOUT-MON] ⚠ Degradation {avg_degrad:.1f}% > {threshold_pct:.0f}% threshold "
                        f"({self._holdout_consecutive_bad}/{self.holdout_early_stop_checks} consecutive)"
                    )
                else:
                    self._holdout_consecutive_bad = 0

            # Trend-based early stopping: detect consecutive worsening
            self._holdout_degradation_history.append(avg_degrad)
            n_checks = len(self._holdout_degradation_history)
            if (
                self.holdout_trend_early_stop
                and n_checks >= self.holdout_trend_checks + 1
                and n_checks > self.holdout_trend_grace_checks
            ):
                recent = self._holdout_degradation_history[-(self.holdout_trend_checks + 1):]

                # Compute slope of degradation trend (percentage points per check)
                n = len(recent)
                x_mean = (n - 1) / 2.0
                y_mean = sum(recent) / n
                numer = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
                denom = sum((i - x_mean) ** 2 for i in range(n))
                slope = numer / denom if denom > 0 else 0.0

                # Trigger only if:
                # 1) Absolute degradation exceeds the minimum floor
                # 2) Slope exceeds the threshold (degradation worsening fast enough)
                if (
                    avg_degrad >= self.holdout_trend_min_degradation
                    and slope >= self.holdout_trend_slope_threshold
                ):
                    self.logger.warning(
                        f"  [HOLDOUT-TREND] ⚠ Degradation trend detected over {self.holdout_trend_checks} "
                        f"checks: {[f'{d:.1f}%' for d in recent]}, "
                        f"slope={slope:.3f}, degradation={avg_degrad:.1f}%. "
                        f"Triggering early stop."
                    )
                    # Force the consecutive_bad counter high enough to trigger early stop
                    self._holdout_consecutive_bad = max(
                        self._holdout_consecutive_bad, self.holdout_early_stop_checks
                    )

            return avg_degrad
        return None

    def _apply_global_dsr_penalties(self, population):
        """
        Recompute DSR penalties in the main process using the global trial count.

        Worker processes each have isolated DSRTrackers with very few trials,
        making their DSR correction negligible.  This method recalculates each
        individual's DSR penalty using the cumulative trial count across ALL
        evaluations in the GA run, ensuring meaningful multiple-testing correction.
        """
        if not self._global_dsr_tracker.enabled:
            return

        evaluated = [ind for ind in population.individuals
                     if ind.fitness is not None and ind.metrics]
        if not evaluated:
            return

        # Register all newly evaluated individuals
        for ind in evaluated:
            if not ind.metrics.get('_dsr_registered'):
                self._global_dsr_tracker.register_evaluation()
                ind.metrics['_dsr_registered'] = True

        # Need at least 2 trials before DSR applies
        if self._global_dsr_tracker.n_trials < 2:
            return

        for ind in evaluated:
            metrics = ind.metrics
            sharpe = metrics.get('sharpe_ratio', 0.0)
            n_returns = metrics.get('num_trades', 0)
            skewness = metrics.get('return_skewness', 0.0)
            kurtosis = metrics.get('return_kurtosis', 3.0)

            # Compute penalty with global trial count
            dsr_penalty, dsr_info = self._global_dsr_tracker.compute_penalty(
                observed_sharpe=sharpe,
                n_returns=n_returns,
                skewness=skewness,
                kurtosis=kurtosis,
            )

            old_dsr = metrics.get('dsr_penalty', 1.0)

            # Only update if the new penalty is different from the worker's value
            if abs(old_dsr - dsr_penalty) > 1e-6:
                # Reverse old worker penalty, apply correct global penalty
                if old_dsr > 1e-9 and ind.fitness is not None:
                    ind.fitness = ind.fitness / old_dsr * dsr_penalty
                    if ind.raw_fitness is not None:
                        ind.raw_fitness = ind.raw_fitness / old_dsr * dsr_penalty

            # Always store the authoritative values
            metrics['dsr'] = dsr_info.get('dsr', float('nan'))
            metrics['dsr_penalty'] = dsr_info.get('dsr_penalty', 1.0)
            metrics['dsr_skipped'] = dsr_info.get('dsr_skipped', False)
            metrics['dsr_n_trials'] = self._global_dsr_tracker.n_trials

    def _evaluate_population_parallel(self, unevaluated: list):
        """
        Evaluate population using parallel workers.
        
        Args:
            unevaluated: List of unevaluated individuals
        """
        self.logger.info(f"[EVAL] Parallel evaluation of {len(unevaluated)} individuals...")
        
        # Create progress callback for tqdm if enabled
        # When terminal monitor is active, skip tqdm to avoid display conflicts
        pbar = None
        if self.progress_enabled and not self.monitor.active:
            pbar = tqdm(
                total=len(unevaluated),
                desc=f"Gen {self.current_generation + 1}",
                unit="strategy",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        
        def progress_callback(completed, total):
            if pbar:
                pbar.n = completed
                pbar.refresh()
            self.monitor.on_eval_progress(completed, total)
        
        # Run parallel evaluation
        result = self.parallel_evaluator.evaluate_batch(
            unevaluated,
            progress_callback=progress_callback if self.progress_enabled else None
        )
        
        if pbar:
            pbar.close()
        
        # Calculate summary stats
        total_profit = sum(ind.metrics.get('profit', 0) for ind in unevaluated if ind.evaluated)
        avg_profit = total_profit / result.successful if result.successful > 0 else 0
        
        self.logger.info(
            f"[EVAL] Complete: {result.successful} succeeded, {result.failed} failed, "
            f"avg profit: {avg_profit:.2f}% ({result.total_time:.1f}s, ~{result.speedup_estimate:.1f}x speedup)"
        )
    
    def _evaluate_population_sequential(self, unevaluated: list):
        """
        Evaluate population sequentially (original implementation).
        
        Args:
            unevaluated: List of unevaluated individuals
        """
        self.logger.info(f"[EVAL] Sequential evaluation of {len(unevaluated)} individuals...")
        
        successful = 0
        failed = 0
        total_profit = 0.0
        best_fitness = 0.0
        best_profit = 0.0
        
        # Create progress bar if enabled
        # When terminal monitor is active, skip tqdm to avoid display conflicts
        if self.progress_enabled and not self.monitor.active:
            pbar = tqdm(
                enumerate(unevaluated),
                total=len(unevaluated),
                desc=f"Gen {self.current_generation + 1}",
                unit="strategy",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            )
            iterator = pbar
        else:
            iterator = enumerate(unevaluated)
            pbar = None
        
        for i, individual in iterator:
            strategy_name = individual.id
            
            # Show progress every 5 individuals or at start/end (when no progress bar)
            if not self.progress_enabled:
                if i == 0 or (i + 1) % 5 == 0 or i == len(unevaluated) - 1:
                    self.logger.info(f"[EVAL] Progress: {i+1}/{len(unevaluated)} ({((i+1)/len(unevaluated)*100):.0f}%)")
            
            try:
                # Evaluate fitness
                fitness, metrics = self.fitness_evaluator.evaluate(
                    individual.strategy_gene,
                    strategy_name=strategy_name
                )
                individual.set_fitness(fitness, metrics)
                
                # For NSGA-II: also set objectives
                if self.mode == 'nsga2':
                    nsga2_min_trades = self.nsga2_config.get('min_trades', 0)
                    objectives = extract_objectives_from_metrics(
                        metrics, self.objectives_config, min_trades=nsga2_min_trades
                    )
                    individual.set_objectives(objectives, metrics)
                
                successful += 1
                profit = metrics.get('profit', 0)
                total_profit += profit
                
                # Track best for progress bar
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_profit = profit
                
                # Update progress bar postfix
                if pbar and (i + 1) % self.progress_update_every == 0:
                    postfix = {}
                    if self.progress_show_fitness:
                        postfix['best_fit'] = f"{best_fitness:.3f}"
                    if self.progress_show_profit:
                        postfix['best_pft'] = f"{best_profit:.1f}%"
                    pbar.set_postfix(postfix)
                
                self.logger.debug(f"  {strategy_name}: fitness={fitness:.4f}, profit={metrics.get('profit', 0):.2f}%")
                
                # Update terminal monitor eval progress
                self.monitor.on_eval_progress(i + 1, len(unevaluated))
                
            except Exception as e:
                # Handle evaluation errors gracefully
                self.logger.warning(f"[EVAL] Failed {strategy_name}: {e}")
                failed += 1
                # Set zero fitness for failed evaluation
                individual.set_fitness(0.0, {
                    'profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'error': str(e)
                })
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Summary
        avg_profit = total_profit / successful if successful > 0 else 0
        self.logger.info(f"[EVAL] Complete: {successful} succeeded, {failed} failed, avg profit: {avg_profit:.2f}%")
    
    def _should_update_best_individual(self, candidate: Individual) -> bool:
        """
        Determine if candidate should replace current best individual.
        
        Uses raw_fitness (not shared_fitness) for true best strategy comparison.
        
        Handles None fitness values correctly:
        - Candidate must have valid (non-None) raw_fitness
        - Updates if no best individual exists yet
        - Updates if current best has None raw_fitness
        - Updates if candidate raw_fitness is higher than current best
        
        Args:
            candidate: Individual to consider as new best
            
        Returns:
            True if candidate should become new best individual
        """
        # Candidate must have valid raw_fitness
        if candidate.raw_fitness is None:
            return False
        
        # Update if no best exists yet
        if self.best_individual is None:
            return True
        
        # Update if current best has invalid raw_fitness
        if self.best_individual.raw_fitness is None:
            return True
        
        # Compare against best_fitness_ever (an immutable scalar snapshot) rather than
        # self.best_individual.raw_fitness, which holdout monitoring can reduce in-place
        # after we store the reference. That mutation caused non-monotonic [NEW BEST] logs
        # (e.g. Gen15 reporting 0.1871 after Gen8's 0.2738 had already been accepted).
        return candidate.raw_fitness > self.best_fitness_ever
    
    def _visualize_strategy_trades(self, individual: Individual, generation: int, individual_idx: int):
        """
        Generate trade visualization for a specific individual.
        
        Args:
            individual: Individual to visualize
            generation: Current generation number
            individual_idx: Index of individual in ranking
        """
        if not self.trade_visualizer:
            return
        
        try:
            from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
            
            # Generate strategy code
            strategy_code = self.strategy_generator.generate_strategy_code(individual.strategy_gene)
            # Use the full strategy name with GAStrategy_ prefix (same as generator.py)
            strategy_name = f"GAStrategy_Gen{individual.strategy_gene.generation}_Ind{individual.strategy_gene.individual_id}"
            
            # Run backtest with trade collection
            backtester = DirectBacktester(self.config)
            result = backtester.backtest_strategy_with_trades(strategy_code, strategy_name)
            
            if not result.success:
                self.logger.warning(f"[TRADE VIS] Backtest failed for {strategy_name}: {result.error_message}")
                return
            
            # Generate trade chart
            saved_files = self.trade_visualizer.visualize_strategy_from_backtest(
                strategy_name=strategy_name,
                backtest_result=result,
                generation=generation,
                individual_idx=individual_idx
            )
            
            if saved_files:
                self.logger.info(f"[TRADE VIS] Generated {len(saved_files)} chart(s) for {strategy_name}")
            
        except Exception as e:
            self.logger.warning(f"[TRADE VIS] Failed to visualize {individual.id}: {e}")
            import traceback
            traceback.print_exc()
    
    def create_next_generation(self, population: Population) -> Population:
        """Create next generation through selection, crossover, and mutation.

        Delegates to :class:`GenerationStep` for the actual work.
        """
        # Expose current population on self so external hooks can read it
        self.population = population

        # Sync mutable state that may have changed since init
        if self._immigrant_provider:
            # Wrap (ga, generation) -> (generation) signature for GenerationStep
            _ga_ref = self
            _provider = self._immigrant_provider
            self._generation_step.immigrant_provider = lambda gen: _provider(_ga_ref, gen)
        else:
            self._generation_step.immigrant_provider = None
        self._generation_step.map_elites = self._map_elites
        self._generation_step.aos = self._aos
        self._generation_step.llm_enabled = self.llm_enabled
        self._generation_step.strategy_designer = getattr(self, 'strategy_designer', None)

        # Collect and clear queued immigrants
        external_immigrants = list(self._external_immigrants)
        self._external_immigrants.clear()

        next_gen, _op_stats = self._generation_step.execute(
            population,
            current_generation=self.current_generation,
            mutation_rate=self.mutation_rate,
            external_immigrants=external_immigrants,
            no_improvement_count=self.no_improvement_count,
            best_fitness_ever=self.best_fitness_ever,
            generation_stats=self.generation_stats,
        )
        return next_gen
    
    
    def check_convergence(self, stats: PopulationStats) -> bool:
        """
        Check if evolution has converged.
        
        Delegates core adaptive logic to AdaptiveController, then handles
        LLM-specific escalation that depends on self.strategy_designer.
        """
        if self.best_individual is None:
            return False
        
        # Sync GA flags → controller before check
        self._adaptive._new_best_this_gen = self._new_best_this_gen
        self._adaptive.no_improvement_count = self.no_improvement_count
        self._adaptive.mutation_rate = self.mutation_rate
        self._adaptive.base_mutation_rate = self.base_mutation_rate
        
        converged = self._adaptive.check_convergence()
        
        # Sync controller state → GA attributes
        self.no_improvement_count = self._adaptive.no_improvement_count
        self.mutation_rate = self._adaptive.mutation_rate
        self._new_best_this_gen = self._adaptive._new_best_this_gen
        self._catastrophic_restart_needed = self._adaptive._catastrophic_restart_needed
        
        if converged:
            return True
        
        # LLM stagnation escalation: boost LLM involvement when stuck
        if self.llm_enabled and self.strategy_designer.enabled:
            llm_cfg = self.config.get('advanced', {}).get('llm', {})
            escalation_threshold = llm_cfg.get('escalation_threshold', 5)
            
            if self.no_improvement_count >= escalation_threshold:
                base_ratio = llm_cfg.get('immigrant_ratio', 0.5)
                escalation_ratio = llm_cfg.get('escalation_immigrant_ratio', 0.8)
                if self.strategy_designer.immigrant_ratio < escalation_ratio:
                    self.strategy_designer.immigrant_ratio = escalation_ratio
                    self.logger.info(
                        f"[LLM ESCALATION] Stagnation {self.no_improvement_count} gens "
                        f"≥ threshold {escalation_threshold}: "
                        f"immigrant_ratio {base_ratio:.0%} → {escalation_ratio:.0%}"
                    )
            elif self.no_improvement_count == 0:
                base_ratio = llm_cfg.get('immigrant_ratio', 0.5)
                if self.strategy_designer.immigrant_ratio != base_ratio:
                    self.strategy_designer.immigrant_ratio = base_ratio
                    self.logger.info(
                        f"[LLM ESCALATION] Improvement found, "
                        f"resetting immigrant_ratio to {base_ratio:.0%}"
                    )
        
        return False
    
    def evolve(self, resume_from: Optional[str] = None) -> List[Individual]:
        """
        Run the complete evolution process.
        
        Delegates to :class:`RunEngine` which handles the loop lifecycle
        (signals, timing, checkpoints, teardown) while this class provides
        the per-generation domain logic via :meth:`process_generation` and
        :meth:`advance_generation`.
        
        Args:
            resume_from: Optional path to a checkpoint file to resume from.
                        If provided, skips population initialization and continues
                        from the saved generation.
        
        Returns:
            List of best individuals
        """
        from genetic_algorithm.engine.runner import RunEngine
        engine = RunEngine(self)
        return engine.run(resume_from)
    
    def process_generation(self, population: 'Population', gen: int,
                           pareto_archive=None):
        """Execute one generation of domain logic.

        Evaluate → post-eval hooks (AOS, surrogate, MAP-Elites, culling,
        WF validation, DSR, ranking) → stats → best tracking → holdout →
        extras computation.

        Called by :class:`~genetic_algorithm.engine.runner.RunEngine` each
        iteration.

        Returns:
            :class:`~genetic_algorithm.engine.runner.GenerationResult`
        """
        from genetic_algorithm.engine.runner import GenerationResult

        # ── Evaluation phase ──
        self.diagnostics.start_phase('eval')
        self.monitor.on_phase_start('eval')
        self.evaluate_population(population)
        self.diagnostics.end_phase('eval')
        self.monitor.on_phase_end('eval', self.diagnostics.timing._phases.get('eval', 0.0))

        # AOS credit recording
        if self._aos.enabled:
            for ind in population:
                if not ind.evaluated or ind.fitness is None:
                    continue
                m = ind.metrics or {}
                cx = m.pop('_aos_cx', None)
                pf = m.pop('_aos_parent_fit', None)
                if cx is not None and pf is not None:
                    self._aos.record_outcome('crossover', cx, pf, ind.fitness)
            self._aos.update_probabilities()

        # Surrogate model training
        if self._surrogate.enabled:
            self._surrogate.add_training_data(population)
            self._surrogate.maybe_retrain()

        # MAP-Elites exploration bonus
        if self._map_elites.enabled:
            bonus_count = 0
            for ind in population:
                if ind.evaluated and ind.fitness is not None:
                    bonus = self._map_elites.get_exploration_bonus(ind)
                    if bonus > 0:
                        ind.fitness += bonus
                        bonus_count += 1
            if bonus_count:
                self.logger.info(f"[MAP-ELITES] Applied exploration bonus to {bonus_count} individuals")
            self._map_elites.update(population)

        # Strategy culling
        if self.culler.enabled:
            self.culler.cull_population(
                population, generation=gen, elite_size=self.elite_size,
            )

        # Post-hoc walk-forward validation
        self._post_hoc_walk_forward_validation(population)

        # Recompute DSR penalties with global trial count
        self._apply_global_dsr_penalties(population)

        # ── Ranking ──
        distance_matrix = None
        if self.mode == 'nsga2':
            fronts = fast_non_dominated_sort(list(population.individuals))
            for front in fronts:
                crowding_distance_assignment(front)
            pareto_front = fronts[0] if fronts else []
            self.logger.info(f"[NSGA-II] {len(fronts)} Pareto fronts, front 1 has {len(pareto_front)} individuals")
            if pareto_archive is not None:
                pareto_archive.update(list(population.individuals), generation=gen)
        else:
            if self.fitness_sharing or len(population.individuals) >= 2:
                distance_matrix = calculate_pairwise_distances(list(population.individuals))
            if self.fitness_sharing:
                apply_fitness_sharing(population, sigma_share=self.sharing_radius,
                                    distance_matrix=distance_matrix)
                self.logger.debug("[FITNESS SHARING] Applied successfully")

        # ── Stats ──
        stats = population.get_stats(distance_matrix=distance_matrix)
        self.generation_stats.append(stats)

        summary_parts = [f"Best: {stats.best_fitness:.4f}", f"Avg: {stats.avg_fitness:.4f}"]
        if stats.genetic_diversity is not None:
            summary_parts.append(f"Diversity: {stats.genetic_diversity:.4f}")

        # Resource tracking
        try:
            from genetic_algorithm.evaluation.parallel import ParallelEvaluator
            mem_mb = ParallelEvaluator.get_memory_usage_mb()
            if mem_mb > 0:
                summary_parts.append(f"RSS: {mem_mb:.0f}MB")
                stats.memory_mb = mem_mb
        except Exception:
            pass
        try:
            import psutil
            cpu_pct = psutil.Process().cpu_percent(interval=0)
            summary_parts.append(f"CPU: {cpu_pct:.0f}%")
        except Exception:
            pass

        self.logger.info(f"[STATS] {' | '.join(summary_parts)}")

        if self.visualizer:
            self.visualizer.update(gen, stats, population)

        # ── Best individual tracking ──
        best = population.get_best(1)[0]
        if self._should_update_best_individual(best):
            self.best_individual = best
            pre_penalty_raw = best.raw_fitness
            if pre_penalty_raw is not None and pre_penalty_raw > self.best_fitness_ever:
                self.best_fitness_ever = pre_penalty_raw
                self.no_improvement_count = 0
                self._new_best_this_gen = True
                self.logger.info(f"[NEW BEST] {best.id} with fitness {best.fitness:.4f}")
                self.monitor.on_new_best(best)
                try:
                    self._tracker.record_new_best(gen, best.fitness, best.metrics)
                except Exception:
                    pass

            if self.trade_visualizer and self.trade_vis_mode == 'improvement':
                self._visualize_strategy_trades(best, gen, 0)

        if self.trade_visualizer and self.trade_vis_mode == 'each_generation':
            top_individuals = population.get_best(self.trade_vis_top_n)
            for idx, ind in enumerate(top_individuals):
                self._visualize_strategy_trades(ind, gen, idx)

        # ── Feature importance + SIS ──
        try:
            self.feature_tracker.update(population)
            if (gen + 1) % 5 == 0 or gen == self.generations - 1:
                self.feature_tracker.log_summary(top_n=5)
            indicator_weights = self.feature_tracker.get_indicator_weights()
            if indicator_weights:
                self.config['_indicator_weights'] = indicator_weights
                self.logger.debug(f"[FEATURE-IMPORTANCE] Updated indicator weights: "
                                f"{len(indicator_weights)} indicators")
                if self._sis_integrator is not None:
                    sis_weights = self._sis_integrator.get_indicator_weights(indicator_weights)
                    self.config['_indicator_weights'] = sis_weights
                    if hasattr(self._sis_integrator, 'get_synergy_weights'):
                        common_inds = self._get_common_indicators(population, top_n=3)
                        syn_w = self._sis_integrator.get_synergy_weights(common_inds)
                        if syn_w:
                            self.config['_synergy_weights'] = syn_w
                    op_w = self._sis_integrator.get_operator_weights()
                    if op_w:
                        self.config['_operator_weights'] = op_w
        except Exception as e:
            self.logger.warning(f"Feature importance update failed: {e}")
            self.monitor.on_error(f"Feature importance update failed: {e}")

        # LLM performance recording
        if self.llm_enabled and self.strategy_designer and self.strategy_designer.enabled:
            try:
                self.strategy_designer.record_llm_performance(gen, population)
            except Exception as e:
                self.logger.warning(f"LLM performance recording failed: {e}")

        # Hall of Fame
        try:
            self.hall_of_fame.update(population, gen)
        except Exception as e:
            self.logger.warning(f"Hall of fame update failed: {e}")
            self.monitor.on_error(f"Hall of fame update failed: {e}")

        # Diversity references
        self._update_diversity_references(population)

        # ── Holdout monitoring ──
        should_break = False
        break_reason = ''
        self.diagnostics.start_phase('holdout')
        self.monitor.on_phase_start('holdout')
        try:
            self._run_holdout_monitoring(population, gen)
            if (self.holdout_early_stop and
                    self._holdout_consecutive_bad >= self.holdout_early_stop_checks):
                self.logger.info(
                    f"[HOLDOUT EARLY STOP] Stopping: holdout degradation exceeded "
                    f"{self.holdout_early_stop_threshold:.0%} for "
                    f"{self._holdout_consecutive_bad} consecutive checks. "
                    f"Further evolution is likely overfitting."
                )
                self.monitor.on_log(
                    f"Holdout early stop at gen {gen+1}: degradation exceeded threshold "
                    f"for {self._holdout_consecutive_bad} consecutive checks",
                    "warning",
                )
                self.feature_tracker.log_summary()
                self.save_checkpoint(population, gen)
                should_break = True
                break_reason = 'holdout_early_stop'
        except Exception as e:
            self.logger.warning(f"Holdout monitoring failed: {e}")
            self.monitor.on_error(f"Holdout monitoring failed: {e}")
        self.diagnostics.end_phase('holdout')
        self.monitor.on_phase_end('holdout', self.diagnostics.timing._phases.get('holdout', 0.0))

        # Refresh stats after holdout (penalties may have changed fitness)
        stats = population.get_stats(distance_matrix=distance_matrix)
        self.generation_stats[-1] = stats

        # ── Extras for diagnostics CSV ──
        _extras = {'mutation_rate': self.mutation_rate}
        try:
            all_inds = population.get_all() if hasattr(population, 'get_all') else []
            penalties = [ind.metrics.get('holdout_penalty', 0) for ind in all_inds if ind.metrics]
            penalised = [p for p in penalties if p > 0]
            _extras['holdout_penalties_applied'] = len(penalised)
            _extras['avg_holdout_penalty'] = round(sum(penalised) / len(penalised), 4) if penalised else 0.0
            unused_counts = [ind.metrics.get('unused_indicators', 0) for ind in all_inds if ind.metrics]
            _extras['avg_unused_indicators'] = round(sum(unused_counts) / max(len(unused_counts), 1), 2)
            origins = [ind.metrics.get('origin', '') for ind in all_inds if ind.metrics]
            _extras['llm_seeds_count'] = sum(1 for o in origins if o == 'llm_seed')
            _extras['llm_immigrants_count'] = sum(1 for o in origins if o == 'llm_immigrant')
            dsr_penalties = [ind.metrics.get('dsr_penalty', 1.0) for ind in all_inds if ind.metrics]
            if dsr_penalties:
                _extras['avg_dsr_penalty'] = round(sum(dsr_penalties) / len(dsr_penalties), 4)
                best_inds = sorted(
                    [ind for ind in all_inds if ind.fitness is not None],
                    key=lambda x: x.fitness, reverse=True,
                )
                if best_inds:
                    _extras['best_dsr_penalty'] = round(
                        best_inds[0].metrics.get('dsr_penalty', 1.0), 4
                    )
        except Exception:
            pass

        # SIS health
        if self._sis_integrator is not None:
            try:
                sis_status = self._sis_integrator.get_status_summary()
                _extras['sis'] = sis_status
                if self._web_monitor and hasattr(self._web_monitor, 'on_sis_health_update'):
                    self._web_monitor.on_sis_health_update(sis_status)
            except Exception:
                pass
            try:
                all_inds_list = list(population.individuals) if hasattr(population, 'individuals') else list(population)
                self._sis_integrator.update_from_generation(gen, all_inds_list)
            except Exception as e:
                self.logger.debug(f"[SIS] Online learning update failed: {e}")

        return GenerationResult(
            population=population,
            stats=stats,
            should_break=should_break,
            break_reason=break_reason,
            extras=_extras,
        )

    def advance_generation(self, population: 'Population', gen: int,
                           stats) -> tuple:
        """Post-orchestration domain: convergence → catastrophic restart → next gen.

        Called by :class:`~genetic_algorithm.engine.runner.RunEngine` after
        post-generation orchestration (diagnostics, checkpoint, resource
        checks).

        Returns:
            ``(population, should_break)`` — the (possibly new) population
            and whether the loop should terminate.
        """
        # Convergence check
        if self.check_convergence(stats):
            self.logger.info("[CONVERGENCE] Evolution converged early")
            self.monitor.on_log(
                f"Evolution converged early at gen {gen+1}/{self.generations}", "warning",
            )
            self.monitor.on_convergence_warning(
                self.no_improvement_count, self.convergence_patience,
            )
            self.feature_tracker.log_summary()
            self.save_checkpoint(population, gen)
            return population, True
        elif self.no_improvement_count >= self.convergence_patience // 2:
            self.monitor.on_convergence_warning(
                self.no_improvement_count, self.convergence_patience,
            )

        # Catastrophic restart
        if self._catastrophic_restart_needed:
            self._catastrophic_restart_needed = False
            replace_count = int(self.population_size * 0.4)
            sorted_inds = sorted(
                list(population.individuals),
                key=lambda ind: (ind.raw_fitness if ind.raw_fitness is not None else 0.0),
                reverse=True,
            )
            keep_count = self.population_size - replace_count
            kept = sorted_inds[:keep_count]
            from genetic_algorithm.core.population import Population as _Pop
            new_pop = _Pop(size=self.population_size)
            for ind in kept:
                new_pop.add_individual(ind)
            for i in range(replace_count):
                rand_gene = self.strategy_generator.generate_random_strategy(
                    generation=self.current_generation + 1,
                    individual_id=keep_count + i,
                )
                new_pop.add_individual(Individual(strategy_gene=rand_gene))
            population = new_pop
            self.no_improvement_count = 0
            self.mutation_rate = self.base_mutation_rate
            self.logger.warning(
                f"[CATASTROPHIC RESTART] Replaced {replace_count}/{self.population_size} "
                f"individuals with random strategies, reset stagnation counter"
            )

        # Create next generation
        if gen < self.generations - 1:
            if self.llm_enabled and self.strategy_designer.enabled:
                self.strategy_designer.reset_generation_budget()
            self.diagnostics.start_phase('selection')
            self.monitor.on_phase_start('selection')
            population = self.create_next_generation(population)
            self.diagnostics.end_phase('selection')
            self.monitor.on_phase_end(
                'selection',
                self.diagnostics.timing._phases.get('selection', 0.0),
            )

        return population, False
