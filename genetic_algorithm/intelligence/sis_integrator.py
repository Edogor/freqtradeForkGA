"""
SIS Integrator v3

Bridges Strategy Intelligence System analysis results into the live GA evolution.
Provides 5 hooks (3 original + 2 new):

1. Seed filtering — multi-model quality scoring (not just win_rate)
2. Archetype-aware immigrants — full-profile reconstruction
3. Indicator/operator bias — synergy-aware + adaptive weights
4. [NEW] Convergence-aware behavior — modulates aggression based on evolution state
5. [NEW] Online adaptive weights — Bayesian blending of prior + live evidence

Usage:
    from genetic_algorithm.intelligence.sis_integrator import SISIntegrator
    sis = SISIntegrator(config, logger)
    ga.set_immigrant_provider(sis.immigrant_provider)
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from genetic_algorithm.intelligence.corpus import CorpusBuilder, SURROGATE_FEATURE_NAMES
from genetic_algorithm.intelligence.archetypes import ArchetypeClassifier
from genetic_algorithm.intelligence.predictors import MultiTargetPredictor, _add_interaction_features
from genetic_algorithm.core.individual import Individual

logger = logging.getLogger(__name__)

# ── Static enrichment weights from PatternMiner analysis on 4289 strategies ──

SIS_INDICATOR_ENRICHMENT: Dict[str, float] = {
    'CCI': 1.5,       # Moderated (was 3.0) — still elevated, less dominant
    'STOCH': 1.5,     # Moderated (was 2.5)
    'ATR': 1.4,       # Moderated (was 2.0) — valuable across all timeframes
    'ROC': 1.3,       # Moderated (was 2.0)
    'DONCHIAN': 1.3,  # Moderated (was 1.8)
    'ADX': 1.2,       # Moderated (was 1.5)
    'RSI': 1.0,
    'BBANDS': 1.1,    # Raised (was 1.0) — useful for 4H volatility
    'SUPERTREND': 1.1, # Raised (was 1.0) — good 4H trend indicator
    'EMA': 1.0,       # Raised (was 0.9) — restore to neutral
    'SMA': 0.9,
    'WILLR': 1.1,
    'MFI': 1.0,
    'AROON': 1.0,
    'TEMA': 1.0,
    'KAMA': 1.0,
    'VROC': 1.0,
    'PSAR': 0.9,      # Raised (was 0.8)
    'CMF': 0.9,
    'MACD': 0.9,      # Raised (was 0.5) — much less suppressed, works on 4H
    'ICHIMOKU': 0.8,  # Raised (was 0.3) — restore, good 4H trend system
}

SIS_OPERATOR_ENRICHMENT: Dict[str, float] = {
    'cross_below': 2.5,   # Favored in top strategies
    '<': 1.5,
    'decreasing': 1.3,
    '>': 1.0,
    'increasing': 0.9,
    'cross_above': 0.5,   # Avoided in top strategies
    'between': 0.8,
    'value_above_ago': 1.0,
}

# ── Synergy graph: indicator pairs that co-occur strongly in top strategies ──

SYNERGY_GRAPH: Dict[str, Dict[str, float]] = {
    'DONCHIAN': {'ROC': 21.5, 'ATR': 3.2, 'AROON': 2.8},
    'CCI': {'STOCH': 8.1, 'RSI': 4.5, 'ROC': 3.0},
    'ROC': {'DONCHIAN': 21.5, 'AROON': 6.3, 'CCI': 3.0},
    'AROON': {'ROC': 6.3, 'DONCHIAN': 2.8, 'ATR': 2.0},
    'ATR': {'CCI': 3.2, 'DONCHIAN': 3.2, 'ADX': 2.2},
    'STOCH': {'CCI': 8.1, 'RSI': 3.5},
    'ADX': {'ATR': 2.2, 'CCI': 1.8},
    'RSI': {'STOCH': 3.5, 'CCI': 4.5},
}


# ══════════════════════════════════════════════════════════════════════════════
# Evolution state detection
# ══════════════════════════════════════════════════════════════════════════════

class EvolutionState:
    """Tracks evolution dynamics for convergence-aware behavior."""

    EXPLORING = "exploring"       # Early gens, high diversity
    IMPROVING = "improving"       # Fitness increasing consistently
    STAGNATING = "stagnating"     # No improvement for several gens
    POST_RESTART = "post_restart"  # After catastrophic restart

    def __init__(self):
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.state: str = self.EXPLORING
        self.gens_since_improvement: int = 0
        self.best_ever: float = 0.0
        self._last_restart_gen: int = -100

    def update(self, generation: int, best_fitness: float,
               mean_fitness: float, population_size: int) -> str:
        """Update state based on latest generation metrics."""
        self.fitness_history.append(best_fitness)

        if best_fitness > self.best_ever + 0.001:
            self.best_ever = best_fitness
            self.gens_since_improvement = 0
        else:
            self.gens_since_improvement += 1

        # Detect post-restart (population size drops or big fitness reset)
        # Guard: don't trigger on natural early-gen variance (require gen >= 5)
        if (generation >= 5 and
                len(self.fitness_history) >= 2 and
                mean_fitness < self.fitness_history[-2] * 0.5):  # 50% drop (was 0.7)
            self._last_restart_gen = generation
            self.state = self.POST_RESTART
        elif generation - self._last_restart_gen <= 3:
            self.state = self.POST_RESTART
        elif generation <= 5:
            self.state = self.EXPLORING
        elif self.gens_since_improvement >= 3:
            self.state = self.STAGNATING
        elif self.gens_since_improvement == 0:
            self.state = self.IMPROVING
        else:
            self.state = self.EXPLORING

        return self.state

    @property
    def params(self) -> Dict[str, float]:
        """Return state-dependent parameter multipliers."""
        if self.state == self.EXPLORING:
            return {
                "filter_strength": 1.0,
                "immigrant_multiplier": 1.0,
                "weight_bias_strength": 1.0,
                "exploration_bonus": 0.0,
            }
        elif self.state == self.IMPROVING:
            return {
                "filter_strength": 0.5,     # Less filtering when things are working
                "immigrant_multiplier": 0.5,  # Fewer immigrants during improvement
                "weight_bias_strength": 1.2,  # Reinforce what's working
                "exploration_bonus": 0.0,
            }
        elif self.state == self.STAGNATING:
            return {
                "filter_strength": 0.3,     # Minimal filtering — need diversity
                "immigrant_multiplier": 2.0,  # 2x more immigrants (was 3x — capped by pop fraction)
                "weight_bias_strength": 0.5,  # Flatten weights toward uniform
                "exploration_bonus": 0.3,     # Boost underrepresented indicators
            }
        elif self.state == self.POST_RESTART:
            return {
                "filter_strength": 0.2,     # Minimal filtering after restart
                "immigrant_multiplier": 2.0,  # More immigrants
                "weight_bias_strength": 0.7,
                "exploration_bonus": 0.5,     # Maximum diversity push
            }
        return {"filter_strength": 1.0, "immigrant_multiplier": 1.0,
                "weight_bias_strength": 1.0, "exploration_bonus": 0.0}

    def notify_restart(self, generation: int) -> None:
        """Called by evolution.py when a catastrophic restart actually fires."""
        self._last_restart_gen = generation
        self.state = self.POST_RESTART


# ══════════════════════════════════════════════════════════════════════════════
# Online Adaptive Weights
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveWeightTracker:
    """Bayesian blending of prior (offline) knowledge with live evidence.

    Early generations: trust historic PatternMiner weights.
    Late generations: shift toward what's actually working in this run,
    with exponential decay weighting so recent observations matter more.

    Evidence trust schedule (adaptive):
      gen 0-10  : ramp 0 → evidence_trust_initial (default 0.4)
      gen 10-40 : ramp evidence_trust_initial → evidence_trust_max (default 0.85)
      Requires min_trust_observations total observations to start trusting.
    """

    def __init__(self, prior_indicator_weights: Dict[str, float],
                 prior_operator_weights: Dict[str, float],
                 evidence_trust_max: float = 0.6,
                 min_observations: int = 5,
                 observation_halflife: int = 15):
        self._prior_ind = dict(prior_indicator_weights)
        self._prior_op = dict(prior_operator_weights)
        self._evidence_trust_max = evidence_trust_max
        self._evidence_trust_initial: float = min(0.2, evidence_trust_max * 0.25)
        self._max_generations: Optional[int] = None  # set via configure_for_run_length()
        self._min_observations = min_observations
        self._observation_halflife = observation_halflife
        # Accumulate live evidence: indicator -> [(gen, rel_fitness), ...]
        self._ind_fitness_obs: Dict[str, List[tuple]] = {}
        self._op_fitness_obs: Dict[str, List[tuple]] = {}
        self._generation: int = 0
        self._total_observations: int = 0
        # Track per-generation aggregate stats for diagnostics
        self._gen_stats: List[Dict[str, float]] = []

    def observe_generation(self, population: List[Individual],
                           generation: int) -> None:
        """Record which indicators/operators appear in fit strategies this gen."""
        self._generation = generation
        if not population:
            return

        # Only learn from evaluated individuals with decent fitness
        evaluated = [ind for ind in population
                     if ind.fitness is not None and ind.fitness > 0]
        if not evaluated:
            return

        median_fitness = np.median([ind.fitness for ind in evaluated])
        gen_obs = 0

        for ind in evaluated:
            gene = ind.strategy_gene
            fitness = ind.fitness
            # Normalize: 1.0 = at median, >1 = above median
            rel_fitness = fitness / (median_fitness + 1e-8)

            # Track indicator presence with generation tag
            for indicator in gene.indicators:
                ind_type = indicator.type
                if ind_type not in self._ind_fitness_obs:
                    self._ind_fitness_obs[ind_type] = []
                self._ind_fitness_obs[ind_type].append((generation, rel_fitness))
                gen_obs += 1

            # Track operator usage with generation tag
            for cond in gene.entry_conditions:
                op = cond.operator
                if op not in self._op_fitness_obs:
                    self._op_fitness_obs[op] = []
                self._op_fitness_obs[op].append((generation, rel_fitness))
                gen_obs += 1

        self._total_observations += gen_obs
        self._gen_stats.append({
            'generation': generation,
            'n_evaluated': len(evaluated),
            'median_fitness': float(median_fitness),
            'n_observations': gen_obs,
            'evidence_trust': self._current_evidence_trust(),
        })

    def configure_for_run_length(self, max_generations: int) -> None:
        """Auto-scale trust ramp for short runs so weights have effect."""
        self._max_generations = max_generations
        if max_generations <= 20:
            # Short run: start with higher initial trust so adaptation
            # has real effect within the run window
            self._evidence_trust_initial = min(0.4, self._evidence_trust_max * 0.5)
        elif max_generations <= 40:
            self._evidence_trust_initial = min(0.3, self._evidence_trust_max * 0.35)
        # Note: retrain interval scaling is handled by SISIntegrator

    def _current_evidence_trust(self) -> float:
        """Adaptive evidence trust: ramps up with generation and observations."""
        if self._total_observations < self._min_observations * 3:
            return 0.0
        # Phase 1 (gen 0-10): ramp to initial trust
        ramp_phase1 = 10 if self._max_generations is None or self._max_generations > 20 else 5
        ramp_phase2 = 30 if self._max_generations is None or self._max_generations > 40 else 10
        if self._generation <= ramp_phase1:
            return self._evidence_trust_initial * (self._generation / ramp_phase1)
        # Phase 2: ramp to max trust
        progress = min(1.0, (self._generation - ramp_phase1) / ramp_phase2)
        return (self._evidence_trust_initial +
                (self._evidence_trust_max - self._evidence_trust_initial) * progress)

    def get_blended_indicator_weights(self) -> Dict[str, float]:
        """Return blended indicator weights: prior × evidence."""
        evidence_weight = self._current_evidence_trust()
        prior_weight = 1.0 - evidence_weight

        live_weights = self._compute_live_indicator_weights()
        all_keys = set(self._prior_ind.keys()) | set(live_weights.keys())

        blended = {}
        for key in all_keys:
            prior_val = self._prior_ind.get(key, 1.0)
            live_val = live_weights.get(key, 1.0)
            blended[key] = prior_weight * prior_val + evidence_weight * live_val

        return blended

    def get_blended_operator_weights(self) -> Dict[str, float]:
        """Return blended operator weights."""
        evidence_weight = self._current_evidence_trust()
        prior_weight = 1.0 - evidence_weight

        live_weights = self._compute_live_operator_weights()
        all_keys = set(self._prior_op.keys()) | set(live_weights.keys())

        blended = {}
        for key in all_keys:
            prior_val = self._prior_op.get(key, 1.0)
            live_val = live_weights.get(key, 1.0)
            blended[key] = prior_weight * prior_val + evidence_weight * live_val

        return blended

    def _decay_weight(self, obs_gen: int) -> float:
        """Exponential decay: recent obs weighted more than old ones."""
        age = self._generation - obs_gen
        return 0.5 ** (age / max(1, self._observation_halflife))

    def _compute_live_indicator_weights(self) -> Dict[str, float]:
        """Compute indicator weights from live observations with decay."""
        if not self._ind_fitness_obs:
            return {}
        weights = {}
        for ind_type, observations in self._ind_fitness_obs.items():
            if len(observations) >= self._min_observations:
                decay_weights = [self._decay_weight(g) for g, _ in observations]
                fitness_vals = [f for _, f in observations]
                total_w = sum(decay_weights)
                if total_w > 0:
                    weights[ind_type] = float(
                        sum(w * f for w, f in zip(decay_weights, fitness_vals)) / total_w
                    )
                else:
                    weights[ind_type] = 1.0
            else:
                weights[ind_type] = 1.0
        return weights

    def _compute_live_operator_weights(self) -> Dict[str, float]:
        """Compute operator weights from live observations with decay."""
        if not self._op_fitness_obs:
            return {}
        weights = {}
        for op, observations in self._op_fitness_obs.items():
            if len(observations) >= self._min_observations:
                decay_weights = [self._decay_weight(g) for g, _ in observations]
                fitness_vals = [f for _, f in observations]
                total_w = sum(decay_weights)
                if total_w > 0:
                    weights[op] = float(
                        sum(w * f for w, f in zip(decay_weights, fitness_vals)) / total_w
                    )
                else:
                    weights[op] = 1.0
            else:
                weights[op] = 1.0
        return weights

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic summary of online learning state."""
        return {
            'generation': self._generation,
            'evidence_trust': round(self._current_evidence_trust(), 3),
            'total_observations': self._total_observations,
            'n_indicators_tracked': len(self._ind_fitness_obs),
            'n_operators_tracked': len(self._op_fitness_obs),
            'observation_halflife': self._observation_halflife,
            'gen_stats_last5': self._gen_stats[-5:] if self._gen_stats else [],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Main SIS Integrator
# ══════════════════════════════════════════════════════════════════════════════

class SISIntegrator:
    """Bridges SIS analysis results into the live GA evolution (v3).

    New in v3:
    - Multi-model quality scoring for seed filtering
    - Full-profile archetype immigrant generation
    - Synergy-aware indicator weights for mutation
    - Convergence-aware adaptive behavior
    - Online weight learning with Bayesian blending
    - Regime-archetype affinity mapping
    """

    def __init__(self, config: dict, logger_inst: logging.Logger):
        self.config = config
        self.logger = logger_inst
        self._sis_config: Dict[str, Any] = config.get('sis', {})

        # ── Per-hook enable/disable toggles ──────────────────────────────
        hooks_cfg = self._sis_config.get('hooks', {})
        self._hook_enabled: Dict[str, bool] = {
            'seed_filtering': hooks_cfg.get('seed_filtering', True),
            'immigrants': hooks_cfg.get('immigrants', True),
            'indicator_weights': hooks_cfg.get('indicator_weights', True),
            'operator_weights': hooks_cfg.get('operator_weights', True),
            'synergy_weights': hooks_cfg.get('synergy_weights', True),
        }

        # ── Per-hook tracking counters ───────────────────────────────────
        self._hook_calls: Dict[str, int] = {k: 0 for k in self._hook_enabled}
        self._hook_skips: Dict[str, int] = {k: 0 for k in self._hook_enabled}

        # ── Configurable thresholds (with sensible defaults) ──────────────
        self._quality_gate_threshold: float = self._sis_config.get('quality_gate_threshold', 0.25)
        self._weight_cap: float = self._sis_config.get('weight_cap', 5.0)
        self._underrep_pct: float = self._sis_config.get('underrepresentation_pct', 0.02)
        self._evidence_trust_max: float = self._sis_config.get('evidence_trust_max', 0.85)
        self._min_observations: int = self._sis_config.get('min_observations', 5)
        self._observation_halflife: int = self._sis_config.get('observation_halflife', 15)
        self._online_retrain_interval: int = self._sis_config.get('online_retrain_interval', 10)
        self._max_immigrants_fraction: float = self._sis_config.get('max_immigrants_fraction', 0.25)

        models_dir = Path(
            self._sis_config.get('models_dir', 'genetic_algorithm/ml/models')
        )
        corpus_path = Path(
            self._sis_config.get(
                'corpus_path', 'genetic_algorithm/data/strategy_corpus.parquet'
            )
        )

        # ── Archetype classifier ──────────────────────────────────────────────
        self._classifier = ArchetypeClassifier(models_dir=models_dir)
        try:
            self._classifier.load(models_dir)
        except Exception as e:
            self.logger.warning(f"[SIS] Could not load archetype classifier: {e}")
        n_real_archetypes = sum(1 for k in self._classifier.archetype_labels if k != -1)
        self._classifier_ready = n_real_archetypes > 0
        self.logger.info(
            f"[SIS] Archetype classifier: {n_real_archetypes} real archetypes, "
            f"{len(self._classifier.archetype_labels)} total labels, "
            f"centroids={len(self._classifier._centroids)}"
        )

        # ── Predictor (v3: multi-model) ────────────────────────────────────────
        self._predictor = MultiTargetPredictor(models_dir=models_dir)
        try:
            self._predictor.load(models_dir)
        except Exception as e:
            self.logger.warning(f"[SIS] Could not load predictor: {e}")
        # v3: predictor is ready if we have ANY usable model
        self._predictor_ready = bool(
            self._predictor.models or self._predictor.classifiers
        )

        # ── Corpus (for context) ───────────────────────────────────────────────
        self._corpus_df: Optional[pd.DataFrame] = None
        if corpus_path.exists():
            try:
                self._corpus_df = pd.read_parquet(corpus_path)
                self.logger.info(
                    f"[SIS] Loaded corpus: {len(self._corpus_df)} strategies"
                )
            except Exception as e:
                self.logger.warning(f"[SIS] Failed to load corpus: {e}")

        # ── Feature extractor ─────────────────────────────────────────────────
        self._corpus_builder = CorpusBuilder()

        # ── Generation tracking ───────────────────────────────────────────────
        self._n_filtered_last: int = 0
        self._n_immigrants_last: int = 0

        # ── SIS JSONL log ─────────────────────────────────────────────────────
        log_file = self._sis_config.get('log_file')
        if log_file:
            self._log_path = Path(log_file)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(
                config.get('output', {}).get(
                    'dir',
                    config.get('output', {}).get('directory', 'genetic_algorithm/output'),
                )
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = output_dir / "sis_log.jsonl"

        # ── PatternMiner auto-load ────────────────────────────────────────────
        self._pattern_enrichment: Dict[str, float] = {}
        self._pattern_operator_weights: Dict[str, float] = {}
        self._synergy_graph: Dict[str, Dict[str, float]] = dict(SYNERGY_GRAPH)
        try:
            from genetic_algorithm.intelligence.pattern_mining import PatternMiner
            miner = PatternMiner.load(models_dir)
            if miner is not None and miner.indicator_enrichment is not None:
                enrich_df = miner.indicator_enrichment
                min_samples = self._sis_config.get('min_enrichment_samples', 10)
                self._pattern_enrichment = {
                    str(row['indicator']): float(np.log1p(max(0, float(row['enrichment']))))
                    for _, row in enrich_df.iterrows()
                    if row['enrichment'] > 0
                    and row.get('n_present', 0) >= min_samples
                }
                n_skipped = len(enrich_df) - len(self._pattern_enrichment)
                self.logger.info(
                    f"[SIS] Loaded PatternMiner enrichment for "
                    f"{len(self._pattern_enrichment)} indicators "
                    f"(log-compressed, {n_skipped} skipped for low sample count)"
                )
            if miner is not None and miner.operator_patterns is not None:
                entry_ops = miner.operator_patterns.get('entry_operators', {})
                self._pattern_operator_weights = {
                    op: max(0.1, v.get('top_mean', 1.0) / (v.get('global_mean', 1.0) + 0.01))
                    for op, v in entry_ops.items()
                    if v.get('global_mean', 0) > 0.01
                }
            # Load synergy graph from miner if available
            if miner is not None and miner.indicator_synergies is not None:
                syn_df = miner.indicator_synergies
                if not syn_df.empty:
                    self._synergy_graph = {}
                    for _, row in syn_df.iterrows():
                        a, b = row['ind_a'], row['ind_b']
                        lift = row['lift']
                        if a not in self._synergy_graph:
                            self._synergy_graph[a] = {}
                        if b not in self._synergy_graph:
                            self._synergy_graph[b] = {}
                        self._synergy_graph[a][b] = lift
                        self._synergy_graph[b][a] = lift
        except Exception as e:
            self.logger.debug(
                f"[SIS] PatternMiner load skipped, using hardcoded weights: {e}"
            )

        # ── v3: Evolution state tracker ────────────────────────────────────────
        self._evo_state = EvolutionState()

        # ── v3: Adaptive weight tracker ────────────────────────────────────────
        ind_prior = self._pattern_enrichment if self._pattern_enrichment \
            else SIS_INDICATOR_ENRICHMENT
        op_prior = self._pattern_operator_weights if self._pattern_operator_weights \
            else SIS_OPERATOR_ENRICHMENT
        self._adaptive_weights = AdaptiveWeightTracker(
            ind_prior, op_prior,
            evidence_trust_max=self._evidence_trust_max,
            min_observations=self._min_observations,
            observation_halflife=self._observation_halflife,
        )

        # ── v3: Online learning state ─────────────────────────────────────────
        self._live_corpus_records: List[Dict[str, Any]] = []
        self._last_retrain_gen: int = -1
        self._last_search_direction: Dict[str, float] = {}

        # ── v3: Regime-archetype affinity ─────────────────────────────────────
        self._regime_archetype_affinity: Dict[str, Dict[int, float]] = {}
        self._build_regime_archetype_affinity()

        self.logger.info(
            f"[SIS] Initialized v3 — classifier_ready={self._classifier_ready}, "
            f"predictor_ready={self._predictor_ready}, "
            f"synergies={len(self._synergy_graph)}, "
            f"regime_affinities={len(self._regime_archetype_affinity)}"
        )

    # ── Feature helpers ────────────────────────────────────────────────────────

    def _extract_features(self, individual: Individual) -> Optional[List[float]]:
        """Extract 65-dim feature vector from an individual."""
        try:
            return self._corpus_builder._extract_features_from_dict(
                individual.strategy_gene.to_dict()
            )
        except Exception:
            return None

    def _build_feature_df(
        self, individuals: List[Individual]
    ) -> tuple:
        """Build a feature DataFrame from individuals.

        Returns:
            (df, valid_indices) where valid_indices[i] is the index in `individuals`
            that corresponds to row i of the returned DataFrame.
        """
        rows: List[List[float]] = []
        valid_indices: List[int] = []
        for i, ind in enumerate(individuals):
            feats = self._extract_features(ind)
            if feats is not None and len(feats) == 65:
                rows.append(feats)
                valid_indices.append(i)
        if not rows:
            return pd.DataFrame(columns=SURROGATE_FEATURE_NAMES), []
        return pd.DataFrame(rows, columns=SURROGATE_FEATURE_NAMES), valid_indices

    # ── Hook 1: seed quality filtering (v3: multi-model) ──────────────────────

    def filter_population(self, population, ga) -> None:
        """Replace the lowest-predicted-quality random seeds with better ones.

        v3: Uses composite quality score from all available predictors
        (classifiers + regressors), not just win_rate.
        """
        self._hook_calls['seed_filtering'] += 1
        if not self._hook_enabled['seed_filtering']:
            self._hook_skips['seed_filtering'] += 1
            return
        if not self._predictor_ready:
            return

        # Only filter when we have at least one reliable model
        reliable = self._predictor.get_reliable_models()
        if not reliable:
            self.logger.debug("[SIS] Seed filtering skipped: no reliable models")
            return

        state_params = self._evo_state.params
        base_fraction: float = self._sis_config.get('filter_fraction', 0.15)
        filter_fraction = base_fraction * state_params["filter_strength"]

        # Identify "random" individuals
        random_indices: List[int] = []
        for i, ind in enumerate(population.individuals):
            origin = ind.metrics.get('origin') if ind.metrics else None
            if origin is None and not ind.evaluated:
                random_indices.append(i)

        if not random_indices:
            return

        n_to_replace = max(1, int(len(random_indices) * filter_fraction))

        # Score existing random individuals
        random_inds = [population.individuals[i] for i in random_indices]
        feat_df, valid_idx = self._build_feature_df(random_inds)
        if feat_df.empty:
            return

        try:
            scores = self._predictor.predict_quality_score(feat_df)
        except Exception as e:
            self.logger.warning(f"[SIS] Quality scoring failed: {e}")
            return

        scored_pairs = [
            (random_indices[valid_idx[j]], scores.iloc[j]) for j in range(len(valid_idx))
        ]
        scored_pairs.sort(key=lambda x: x[1])
        pop_indices_to_replace = [p[0] for p in scored_pairs[:n_to_replace]]

        # Generate 2x replacement candidates
        n_candidates = n_to_replace * 2
        new_candidates: List[Individual] = []
        for k in range(n_candidates):
            try:
                gene = ga.strategy_generator.generate_random_strategy(
                    generation=0,
                    individual_id=len(population.individuals) + 10000 + k,
                )
                new_candidates.append(Individual(strategy_gene=gene))
            except Exception:
                continue

        if not new_candidates:
            return

        cand_feat_df, cand_valid_idx = self._build_feature_df(new_candidates)
        if cand_feat_df.empty:
            return

        try:
            cand_scores = self._predictor.predict_quality_score(cand_feat_df)
        except Exception:
            return

        cand_scored = [
            (new_candidates[cand_valid_idx[j]], cand_scores.iloc[j])
            for j in range(len(cand_valid_idx))
        ]
        cand_scored.sort(key=lambda x: x[1], reverse=True)

        replaced = 0
        for pop_idx, (replacement, _) in zip(pop_indices_to_replace, cand_scored):
            replacement.metrics['origin'] = 'sis_filtered'
            population.individuals[pop_idx] = replacement
            replaced += 1

        self._n_filtered_last = replaced
        if replaced > 0:
            self.logger.info(
                f"[SIS] Seed filtering: replaced {replaced} low-quality seeds "
                f"(reliable models: {list(reliable.keys())})"
            )

    # ── Hook 2: archetype-aware immigrants (v3: full-profile) ────────────────

    def immigrant_provider(self, ga, generation: int) -> List[Individual]:
        """Provide archetype-aware immigrants for the current generation.

        Supports two modes via config 'sis.immigrants_mode':
        - 'simple': Top-2 archetypes by fitness, 1 immigrant each, quality-gated
        - 'full' (default): Full v3 with regime/convergence/synergy awareness
        """
        immigrants: List[Individual] = []
        self._hook_calls['immigrants'] += 1

        if not self._hook_enabled['immigrants']:
            self._hook_skips['immigrants'] += 1
            return immigrants

        if not self._classifier_ready:
            self.logger.debug("[SIS] immigrant_provider: skip — classifier_ready=False")
            return immigrants

        mode = self._sis_config.get('immigrants_mode', 'full')
        if mode == 'simple':
            return self._simple_immigrant_provider(ga, generation)

        known_archetypes = set(self._classifier.archetype_labels.keys()) - {-1}
        if not known_archetypes:
            self.logger.warning("[SIS] immigrant_provider: known_archetypes EMPTY — all labels are noise!")
            return immigrants

        pop_inds: List[Individual] = []
        if hasattr(ga, 'population') and ga.population is not None:
            pop_inds = list(ga.population.individuals)

        if not pop_inds:
            return immigrants

        # Update evolution state
        fitnesses = [ind.fitness for ind in pop_inds if ind.fitness is not None]
        if fitnesses:
            best_fit = max(fitnesses)
            mean_fit = float(np.mean(fitnesses))
            state = self._evo_state.update(
                generation, best_fit, mean_fit, len(pop_inds))
        else:
            state = self._evo_state.state

        # Update adaptive weights with live data
        self._adaptive_weights.observe_generation(pop_inds, generation)

        archetype_coverage: Dict[str, float] = {}

        feat_df, _ = self._build_feature_df(pop_inds)
        if feat_df.empty or len(feat_df) < 2:
            self.log_generation(generation, pop_inds, archetype_coverage)
            return immigrants

        try:
            pred_df = self._classifier.predict(feat_df)
            present_archetypes = set(pred_df['archetype'].unique()) - {-1}
            n_noise = int((pred_df['archetype'] == -1).sum())
            self.logger.info(
                f"[SIS] Gen {generation}: predict → {len(present_archetypes)} present archetypes, "
                f"{n_noise}/{len(pred_df)} classified as noise"
            )
        except Exception as e:
            self.logger.warning(f"[SIS] Archetype prediction failed: {e}")
            self.log_generation(generation, pop_inds, archetype_coverage)
            return immigrants

        # Update search direction from archetype gaps
        if self._sis_config.get('search_direction', True):
            try:
                self._last_search_direction = self.get_search_direction(pop_inds)
            except Exception:
                pass

        if 'archetype' in pred_df.columns:
            arch_counts = pred_df['archetype'].value_counts()
            total = len(pred_df)
            archetype_coverage = {
                str(k): round(float(v) / total, 3)
                for k, v in arch_counts.items()
                if k != -1
            }

        missing_archetypes = known_archetypes - present_archetypes

        underrepresented: set = set()
        if 'archetype' in pred_df.columns:
            threshold = max(1, len(pred_df) * self._underrep_pct)
            for arch_id in known_archetypes:
                if arch_counts.get(arch_id, 0) < threshold:
                    underrepresented.add(arch_id)

        target_archetypes = list(missing_archetypes | underrepresented)

        # v3.1: Filter out low-fitness archetypes
        arch_stats = self._classifier.archetype_stats
        all_fitness = [
            s.get('fitness_mean', 0) or 0
            for k, s in arch_stats.items() if k != -1
        ]
        global_mean_fitness = float(np.mean(all_fitness)) if all_fitness else 0
        if global_mean_fitness > 0:
            target_archetypes = [
                a for a in target_archetypes
                if (arch_stats.get(a, {}).get('fitness_mean', 0) or 0)
                   >= global_mean_fitness * 0.7
            ]

        # v3.1: Sort by fitness_mean descending for deterministic, quality-first ordering
        target_archetypes.sort(
            key=lambda a: arch_stats.get(a, {}).get('fitness_mean', 0) or 0,
            reverse=True
        )

        # v3: Regime-aware archetype prioritization (secondary sort on top)
        regime = self.config.get('_island_regime', '')
        if regime and self._regime_archetype_affinity.get(regime):
            affinity = self._regime_archetype_affinity[regime]
            target_archetypes.sort(
                key=lambda a: affinity.get(a, 0), reverse=True
            )

        if not target_archetypes:
            self.logger.info(
                f"[SIS] Gen {generation}: no missing/underrep archetypes → 0 immigrants"
            )
            self.log_generation(generation, pop_inds, archetype_coverage)
            return immigrants

        # v3: Convergence-aware immigrant count
        state_params = self._evo_state.params
        base_immigrants: int = self._sis_config.get('immigrants_per_gen', 2)
        n_immigrants = max(1, int(base_immigrants * state_params["immigrant_multiplier"]))
        # v3.1: Cap immigrants to fraction of population to prevent cold-restarts
        pop_size = len(pop_inds)
        max_immigrants = max(1, int(pop_size * self._max_immigrants_fraction))
        if n_immigrants > max_immigrants:
            self.logger.debug(
                f"[SIS] Capping immigrants {n_immigrants} → {max_immigrants} "
                f"(pop={pop_size}, cap={self._max_immigrants_fraction})"
            )
            n_immigrants = max_immigrants

        for arch_id in target_archetypes[:n_immigrants]:
            try:
                immigrant = self._generate_full_profile_immigrant(
                    ga, arch_id, generation
                )
                if immigrant is not None:
                    immigrants.append(immigrant)
            except Exception as e:
                self.logger.debug(
                    f"[SIS] Failed to generate immigrant for archetype {arch_id}: {e}"
                )

        self._n_immigrants_last = len(immigrants)

        if immigrants:
            self.logger.info(
                f"[SIS] Gen {generation} [{state}]: injecting {len(immigrants)} corpus immigrants "
                f"(missing={len(missing_archetypes)}, underrep={len(underrepresented)}, "
                f"regime={regime or 'none'})"
            )

        self.log_generation(generation, pop_inds, archetype_coverage)
        return immigrants

    def _simple_immigrant_provider(
        self, ga, generation: int
    ) -> List[Individual]:
        """Simplified immigrant injection: top-2 archetypes by fitness, 1 each.

        Much simpler than the full v3 provider — no regime awareness,
        no convergence scaling, no synergy awareness. Just picks the
        highest-fitness archetypes and generates one immigrant per archetype,
        quality-gated by win_rate prediction.
        """
        immigrants: List[Individual] = []

        # Rank archetypes by average fitness
        arch_stats = self._classifier.archetype_stats
        arch_fitness: List[tuple] = []
        for arch_id, stats in arch_stats.items():
            if arch_id == -1:
                continue
            mean_fit = stats.get('mean_fitness', 0) or 0
            if mean_fit > 0:
                arch_fitness.append((arch_id, mean_fit))

        if not arch_fitness:
            return immigrants

        arch_fitness.sort(key=lambda x: x[1], reverse=True)
        n_inject = min(2, len(arch_fitness))

        for arch_id, _ in arch_fitness[:n_inject]:
            try:
                immigrant = self._generate_full_profile_immigrant(
                    ga, arch_id, generation
                )
                if immigrant is None:
                    continue

                # Quality gate: predict win_rate if possible
                if self._predictor_ready:
                    feat_df, valid_idx = self._build_feature_df([immigrant])
                    if not feat_df.empty and 'win_rate' in self._predictor.models:
                        try:
                            from genetic_algorithm.intelligence.predictors import _add_interaction_features
                            X = self._predictor._get_feature_matrix(
                                _add_interaction_features(feat_df)
                            )
                            pred_wr = self._predictor.models['win_rate'].predict(X)[0]
                            # Only inject if predicted win_rate is above median
                            # (roughly > 0.4 for typical distributions)
                            wr_threshold = self._sis_config.get(
                                'immigrant_min_win_rate', 0.35
                            )
                            if pred_wr < wr_threshold:
                                self.logger.debug(
                                    f"[SIS] Simple immigrant for arch {arch_id} "
                                    f"rejected: predicted win_rate={pred_wr:.3f} < {wr_threshold}"
                                )
                                continue
                        except Exception:
                            pass  # No win_rate model — allow without check

                immigrants.append(immigrant)
            except Exception as e:
                self.logger.debug(
                    f"[SIS] Simple immigrant for arch {arch_id} failed: {e}"
                )

        self._n_immigrants_last = len(immigrants)
        if immigrants:
            self.logger.debug(
                f"[SIS] Gen {generation} [simple]: injecting {len(immigrants)} "
                f"immigrants from top archetypes"
            )
        return immigrants

    def _generate_full_profile_immigrant(
        self, ga, archetype_id: int, generation: int
    ) -> Optional[Individual]:
        """Generate a strategy matching the full archetype profile.

        v3: Reconstructs indicator set, operator preferences, risk params,
        and timeframe from archetype statistics. Much richer than v2's
        single-indicator swap.
        """
        stats = self._classifier.archetype_stats.get(archetype_id, {})
        top_indicators: List[str] = stats.get('top_indicators', [])
        dom_timeframe = stats.get('dominant_timeframe', '')

        unique_id = hash((archetype_id, generation, int(time.time() * 1000))) & 0xFFFF
        gene = ga.strategy_generator.generate_random_strategy(
            generation=generation,
            individual_id=unique_id,
        )

        # Override timeframe if archetype has a dominant one
        if dom_timeframe and dom_timeframe in ('1m', '3m', '5m', '15m', '30m', '1h', '4h'):
            gene.timeframe = dom_timeframe

        # Override indicators to match archetype profile (up to 3)
        # v3.1: Blend corpus archetype indicators with live indicator weights
        if top_indicators and gene.indicators:
            try:
                from genetic_algorithm.evaluation.surrogate import _INDICATOR_TYPES
                _available = set(self.config.get('indicators', {}).get('available', []))
                valid_top = [
                    ind for ind in top_indicators
                    if ind in _INDICATOR_TYPES and (not _available or ind in _available)
                ]

                # Blend with live indicator weights: as evidence_trust increases,
                # live weights get more say in which archetype indicators to use
                evidence_trust = self._adaptive_weights._current_evidence_trust()
                if evidence_trust > 0.05 and valid_top:
                    live_weights = self._adaptive_weights.get_blended_indicator_weights()
                    # Score each valid indicator by blended weight
                    scored = []
                    for i, ind in enumerate(valid_top):
                        corpus_rank = 1.0 / (i + 1)  # rank-based: 1st=1.0, 2nd=0.5, ...
                        live_w = live_weights.get(ind, 1.0)
                        blended = (1.0 - evidence_trust) * corpus_rank + evidence_trust * live_w
                        scored.append((ind, blended))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    valid_top = [ind for ind, _ in scored]

                # Replace first indicator with primary archetype indicator
                if valid_top and len(gene.indicators) >= 1:
                    old_type = gene.indicators[0].type
                    gene.indicators[0].type = valid_top[0]
                    self._update_conditions_indicator(gene, old_type, valid_top[0])

                # Replace second indicator with synergy partner
                if len(valid_top) >= 2 and len(gene.indicators) >= 2:
                    primary = valid_top[0]
                    # Pick best synergy partner from archetype's indicators
                    synergies = self._synergy_graph.get(primary, {})
                    best_partner = None
                    best_lift = 0
                    for candidate in valid_top[1:]:
                        lift = synergies.get(candidate, 0)
                        if lift > best_lift:
                            best_lift = lift
                            best_partner = candidate
                    if best_partner is None:
                        best_partner = valid_top[1]

                    old_type = gene.indicators[1].type
                    gene.indicators[1].type = best_partner
                    self._update_conditions_indicator(gene, old_type, best_partner)

            except Exception as e:
                self.logger.debug(f"[SIS] Indicator override failed: {e}")

        # Override risk parameters to match archetype distributions
        if stats.get('stoploss_mean') is not None:
            noise = np.random.normal(0, 0.005)
            gene.stoploss = np.clip(stats['stoploss_mean'] + noise, -0.15, -0.01)
        if stats.get('max_drawdown_mean') is not None:
            # Wider stoploss for archetypes that tolerate more drawdown
            if stats['max_drawdown_mean'] > 0.15:
                gene.stoploss = np.clip(gene.stoploss - 0.02, -0.15, -0.01)

        individual = Individual(strategy_gene=gene)
        individual.metrics['origin'] = 'sis_immigrant'
        individual.metrics['sis_archetype'] = archetype_id
        individual.metrics['sis_version'] = 3

        # Quality gate: reject immigrants the predictor considers low-quality
        if self._predictor_ready:
            try:
                feat_df_imm, _ = self._build_feature_df([individual])
                if not feat_df_imm.empty:
                    score = self._predictor.predict_quality_score(feat_df_imm)
                    if not score.empty and float(score.iloc[0]) < self._quality_gate_threshold:
                        self.logger.info(
                            f"[SIS] Immigrant (arch={archetype_id}) REJECTED: "
                            f"quality_score={float(score.iloc[0]):.3f} < {self._quality_gate_threshold}"
                        )
                        return None
            except Exception:
                pass  # scoring failure → let the immigrant through

        return individual

    @staticmethod
    def _update_conditions_indicator(gene, old_type: str, new_type: str):
        """Update condition references from old indicator type to new."""
        for cond in gene.entry_conditions:
            if cond.indicator == old_type or (
                hasattr(cond, 'indicator') and
                isinstance(cond.indicator, str) and
                cond.indicator.startswith(old_type + '_')
            ):
                cond.indicator = new_type
        for cond in gene.exit_conditions:
            if cond.indicator == old_type or (
                hasattr(cond, 'indicator') and
                isinstance(cond.indicator, str) and
                cond.indicator.startswith(old_type + '_')
            ):
                cond.indicator = new_type

    # ── Hook 3: indicator/operator enrichment weights (v3: synergy+adaptive) ──

    def get_indicator_weights(self, base_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Return merged indicator weights with synergy awareness and adaptive blending.

        v3 changes:
        - Uses Bayesian-blended weights (prior + live evidence) instead of static
        - Applies convergence-aware scaling
        - Returns context-aware weights via synergy graph
        """
        self._hook_calls['indicator_weights'] += 1
        if not self._hook_enabled['indicator_weights']:
            self._hook_skips['indicator_weights'] += 1
            return base_weights or {}
        scale: float = self._sis_config.get('indicator_weight_scale', 1.5)
        state_params = self._evo_state.params

        # v3: Use adaptive weights (blend of prior + live evidence)
        enrichment_source = self._adaptive_weights.get_blended_indicator_weights()
        if not enrichment_source:
            enrichment_source = self._pattern_enrichment if self._pattern_enrichment \
                else SIS_INDICATOR_ENRICHMENT

        if base_weights:
            all_keys = set(enrichment_source.keys()) | set(base_weights.keys())
            raw = {
                ind: enrichment_source.get(ind, 1.0) * base_weights.get(ind, 1.0)
                for ind in all_keys
            }
        else:
            raw = dict(enrichment_source)

        # Normalize to mean = 1.0
        if raw:
            mean_w = float(np.mean(list(raw.values())))
            if mean_w > 0:
                normalized = {ind: w / mean_w for ind, w in raw.items()}
            else:
                normalized = raw
        else:
            return {}

        # v3: Apply convergence-aware scaling
        effective_scale = scale * state_params["weight_bias_strength"]
        exploration_bonus = state_params["exploration_bonus"]

        result = {}
        for ind, w in normalized.items():
            scaled = 1.0 + (w - 1.0) * effective_scale
            # During stagnation, boost underweight indicators toward 1.0
            if exploration_bonus > 0 and scaled < 1.0:
                scaled = scaled + (1.0 - scaled) * exploration_bonus
            result[ind] = max(0.1, min(self._weight_cap, scaled))

        # Filter to only indicators available in this run's config
        available_inds = set(self.config.get('indicators', {}).get('available', []))
        if available_inds:
            result = {k: v for k, v in result.items() if k in available_inds}
            for ind in available_inds:
                if ind not in result:
                    result[ind] = 0.1  # floor weight for available-but-unlisted indicators

        # Blend search direction boosts from archetype gap analysis
        if (self._hook_enabled.get('indicator_weights', True)
                and self._sis_config.get('search_direction', True)
                and self._last_search_direction):
            for ind, boost in self._last_search_direction.items():
                if ind in result:
                    result[ind] = min(self._weight_cap, result[ind] * boost)
                elif not available_inds or ind in available_inds:
                    result[ind] = min(self._weight_cap, boost)

        return result

    def get_operator_weights(self) -> Dict[str, float]:
        """Return operator weights with adaptive blending."""
        self._hook_calls['operator_weights'] += 1
        if not self._hook_enabled['operator_weights']:
            self._hook_skips['operator_weights'] += 1
            return {}
        scale: float = self._sis_config.get('operator_weight_scale', 1.0)
        state_params = self._evo_state.params

        # v3: Use adaptive blended weights
        source = self._adaptive_weights.get_blended_operator_weights()
        if not source:
            source = self._pattern_operator_weights if self._pattern_operator_weights \
                else SIS_OPERATOR_ENRICHMENT

        effective_scale = scale * state_params["weight_bias_strength"]

        if effective_scale == 1.0:
            return dict(source)

        mean_w = float(np.mean(list(source.values()))) if source else 1.0
        if mean_w <= 0:
            return dict(source)
        normalized = {op: w / mean_w for op, w in source.items()}
        return {op: max(0.1, 1.0 + (w - 1.0) * effective_scale)
                for op, w in normalized.items()}

    def get_synergy_weights(self, existing_indicators: List[str]) -> Dict[str, float]:
        """Return indicator weights biased by synergy with existing indicators.

        When adding an indicator to a strategy that already has DONCHIAN,
        this strongly prefers ROC (21.5x lift) over random alternatives.
        """
        self._hook_calls['synergy_weights'] += 1
        if not self._hook_enabled['synergy_weights']:
            self._hook_skips['synergy_weights'] += 1
            return {}
        if not existing_indicators or not self._synergy_graph:
            return {}

        synergy_weights: Dict[str, float] = {}
        for existing in existing_indicators:
            partners = self._synergy_graph.get(existing, {})
            for partner, lift in partners.items():
                if partner not in existing_indicators:
                    current = synergy_weights.get(partner, 0)
                    synergy_weights[partner] = max(current, math.log1p(lift))

        return synergy_weights

    # ── Regime-archetype affinity ──────────────────────────────────────────────

    def _build_regime_archetype_affinity(self) -> None:
        """Build mapping of regime -> archetype -> affinity score.

        Uses corpus data: for each archetype, compute average fitness
        by timeframe (proxy for regime type).
        """
        if self._corpus_df is None or not self._classifier_ready:
            return

        df = self._corpus_df
        if 'archetype' not in df.columns:
            # Try to classify
            try:
                pred_df = self._classifier.predict(
                    df[self._predictor._feature_columns].fillna(0).head(2000)
                )
                df = df.head(2000).copy()
                df['archetype'] = pred_df['archetype'].values
            except Exception:
                return

        # Map timeframes to rough regime types
        tf_to_regime = {
            '1m': 'scalping', '3m': 'scalping', '5m': 'scalping',
            '15m': 'intraday', '30m': 'intraday',
            '1h': 'swing', '4h': 'swing',
        }

        for regime in set(tf_to_regime.values()):
            regime_tfs = [tf for tf, r in tf_to_regime.items() if r == regime]
            regime_df = df[df['timeframe'].isin(regime_tfs)]
            if regime_df.empty:
                continue

            affinity: Dict[int, float] = {}
            for arch_id in set(self._classifier.archetype_labels.keys()) - {-1}:
                arch_df = regime_df[regime_df['archetype'] == arch_id]
                if len(arch_df) >= 3:
                    affinity[arch_id] = float(arch_df['fitness'].mean())

            if affinity:
                # Normalize to [0, 1]
                max_a = max(affinity.values())
                min_a = min(affinity.values())
                rng = max_a - min_a
                if rng > 0:
                    affinity = {k: (v - min_a) / rng for k, v in affinity.items()}
                self._regime_archetype_affinity[regime] = affinity

    # ── Generation logging (v3: enhanced) ─────────────────────────────────────

    def log_generation(
        self,
        generation: int,
        population: List[Individual],
        archetype_coverage: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a JSONL entry with the current generation's SIS state."""
        try:
            fitnesses = [
                ind.fitness for ind in population if ind.fitness is not None
            ]
            best_fitness = float(max(fitnesses)) if fitnesses else None
            mean_fitness = float(np.mean(fitnesses)) if fitnesses else None

            # Collect pair-split generalization ratios from validated individuals
            gen_ratios = [
                ind.metrics['pair_generalization_ratio']
                for ind in population
                if ind.metrics and ind.metrics.get('pair_generalization_ratio') is not None
            ]

            entry = {
                'gen': generation,
                'timestamp': time.time(),
                'best_fitness': best_fitness,
                'mean_fitness': mean_fitness,
                'archetype_distribution': archetype_coverage or {},
                'n_sis_immigrants': self._n_immigrants_last,
                'n_filtered': self._n_filtered_last,
                'evolution_state': self._evo_state.state,
                'gens_since_improvement': self._evo_state.gens_since_improvement,
                'indicator_weights_used': {
                    k: round(v, 3)
                    for k, v in self.get_indicator_weights().items()
                },
                'mean_gen_ratio': round(float(np.mean(gen_ratios)), 3) if gen_ratios else None,
                'pct_catastrophic_overfit': round(
                    sum(1 for r in gen_ratios if r < 0.5) / len(gen_ratios), 3
                ) if gen_ratios else None,
                'n_validated': len(gen_ratios),
                'search_direction': {
                    k: round(v, 3) for k, v in self._last_search_direction.items()
                } if self._last_search_direction else {},
                'evidence_trust': round(
                    self._adaptive_weights._current_evidence_trust(), 3
                ),
            }
            with open(self._log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.logger.debug(f"[SIS] Log write failed: {e}")

    def get_status_summary(self) -> dict:
        """Return a lightweight status dict suitable for dashboard/events."""
        reliable = {}
        if self._predictor is not None:
            try:
                reliable = self._predictor.get_reliable_models()
            except Exception:
                pass

        n_archetypes = 0
        if self._classifier is not None:
            try:
                labels = getattr(self._classifier, 'archetype_labels', [])
                n_archetypes = len([k for k in labels if k != -1])
            except Exception:
                pass

        return {
            'evolution_state': self._evo_state.state,
            'gens_since_improvement': self._evo_state.gens_since_improvement,
            'predictor_ready': self._predictor is not None,
            'n_reliable_models': len(reliable),
            'reliable_models': list(reliable.keys()),
            'classifier_ready': self._classifier is not None,
            'n_archetypes': n_archetypes,
            'n_immigrants_last': self._n_immigrants_last,
            'n_filtered_last': self._n_filtered_last,
            'hooks_enabled': dict(self._hook_enabled),
            'hook_calls': dict(self._hook_calls),
            'hook_skips': dict(self._hook_skips),
            'online_learning': self._adaptive_weights.get_diagnostics(),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Search Direction Guidance: Archetype gap analysis
    # ══════════════════════════════════════════════════════════════════════════

    def get_archetype_gaps(
        self, population: List[Individual]
    ) -> Dict[int, Dict[str, Any]]:
        """Identify promising but underrepresented archetypes in a population.

        Returns a dict of archetype_id -> {
            'historical_fitness': mean fitness from corpus,
            'population_share': fraction in current pop (0-1),
            'expected_share': even share 1/n_archetypes,
            'gap_score': how underrepresented × how promising (higher = bigger gap),
            'top_indicators': indicators that define this archetype,
        }
        """
        if not self._classifier_ready:
            return {}

        arch_stats = self._classifier.archetype_stats
        if not arch_stats:
            return {}

        # Classify current population
        feat_df, valid_idx = self._build_feature_df(population)
        if feat_df.empty:
            return {}

        try:
            pred_df = self._classifier.predict(feat_df)
        except Exception:
            return {}

        # Compute current population archetype shares
        arch_counts = pred_df['archetype'].value_counts()
        total = len(pred_df)
        n_archetypes = len([k for k in arch_stats if k != -1])
        if n_archetypes == 0:
            return {}
        expected_share = 1.0 / n_archetypes

        # Rank archetypes by their historical fitness
        all_fitness = [
            s.get('fitness_mean', 0) or 0
            for k, s in arch_stats.items() if k != -1
        ]
        max_fitness = max(all_fitness) if all_fitness else 1.0

        gaps: Dict[int, Dict[str, Any]] = {}
        for arch_id, stats in arch_stats.items():
            if arch_id == -1:
                continue
            pop_share = arch_counts.get(arch_id, 0) / total
            hist_fitness = (stats.get('fitness_mean', 0) or 0)
            # Normalize fitness to 0-1 range
            fitness_score = hist_fitness / (max_fitness + 1e-8)
            # Gap = underrepresentation × historical quality
            underrep = max(0, expected_share - pop_share) / expected_share
            gap_score = underrep * fitness_score

            if gap_score > 0.05:  # Only report meaningful gaps
                gaps[arch_id] = {
                    'historical_fitness': round(hist_fitness, 4),
                    'population_share': round(pop_share, 3),
                    'expected_share': round(expected_share, 3),
                    'gap_score': round(gap_score, 3),
                    'top_indicators': stats.get('top_indicators', []),
                }

        return dict(sorted(gaps.items(), key=lambda x: -x[1]['gap_score']))

    def get_search_direction(
        self, population: List[Individual]
    ) -> Dict[str, float]:
        """Return indicator weight boosts based on archetype gap analysis.

        Indicators that define underrepresented high-fitness archetypes
        get a boost, encouraging mutations to explore those regions.

        Returns:
            Dict of indicator_type -> boost_multiplier (>1 means boost).
        """
        gaps = self.get_archetype_gaps(population)
        if not gaps:
            return {}

        # Aggregate: for each indicator, sum gap_score across archetypes that use it
        indicator_boost: Dict[str, float] = {}
        for arch_id, info in gaps.items():
            gap_score = info['gap_score']
            for ind_type in info['top_indicators']:
                indicator_boost[ind_type] = indicator_boost.get(ind_type, 0) + gap_score

        if not indicator_boost:
            return {}

        # Normalize to multipliers: 1.0 = no change, >1 = boost
        max_boost = max(indicator_boost.values())
        if max_boost <= 0:
            return {}

        # Scale: biggest gap gets up to 1.5x boost
        max_multiplier = self._sis_config.get('gap_boost_max', 1.5)
        return {
            ind: round(1.0 + (v / max_boost) * (max_multiplier - 1.0), 3)
            for ind, v in indicator_boost.items()
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Online Learning: Mid-run predictor updates
    # ══════════════════════════════════════════════════════════════════════════

    def update_from_generation(
        self, generation: int, population: List[Individual]
    ) -> None:
        """Feed live generation results back for online learning.

        Called once per generation from evolution.py. Does two things:
        1. Accumulates individual feature records for incremental retraining.
        2. Every ``online_retrain_interval`` generations, re-fits the predictor
           on combined (historical corpus + live) data so predictions improve
           as the run progresses.
        """
        if not population:
            return

        evaluated = [
            ind for ind in population
            if ind.fitness is not None and ind.fitness > 0
        ]
        if not evaluated:
            return

        # ── Accumulate live feature records ──────────────────────────────
        for ind in evaluated:
            try:
                feats = self._extract_features(ind)
                if feats is None or len(feats) != 65:
                    continue
                record: Dict[str, Any] = {}
                for name, val in zip(SURROGATE_FEATURE_NAMES, feats):
                    record[name] = val
                record['fitness'] = ind.fitness
                if ind.metrics:
                    for col in ('profit', 'win_rate', 'max_drawdown',
                                'sharpe_ratio', 'num_trades',
                                'pair_generalization_ratio'):
                        record[col] = ind.metrics.get(col)
                record['_live_gen'] = generation
                self._live_corpus_records.append(record)
            except Exception:
                pass

        # ── Periodic predictor retrain ───────────────────────────────────
        if (self._online_retrain_interval <= 0
                or generation - self._last_retrain_gen < self._online_retrain_interval):
            return

        n_live = len(self._live_corpus_records)
        if n_live < 30:
            return  # Not enough live data to be useful

        self._last_retrain_gen = generation
        try:
            live_df = pd.DataFrame(self._live_corpus_records)
            # Combine with historical corpus if available
            if self._corpus_df is not None and not self._corpus_df.empty:
                combined = pd.concat([self._corpus_df, live_df], ignore_index=True)
            else:
                combined = live_df

            # Lightweight retrain: fewer estimators for speed
            results = self._predictor.train(combined)
            self._predictor_ready = bool(
                self._predictor.models or self._predictor.classifiers
            )

            reliable = self._predictor.get_reliable_models()
            self.logger.info(
                f"[SIS] Online retrain at gen {generation}: "
                f"{n_live} live samples, {len(combined)} total, "
                f"reliable models: {list(reliable.keys())}"
            )

            # v3.1: Also refresh archetype_stats from live data
            self._refresh_archetype_stats(live_df, generation)
        except Exception as e:
            self.logger.debug(
                f"[SIS] Online retrain at gen {generation} failed: {e}"
            )

    def _refresh_archetype_stats(
        self, live_df: pd.DataFrame, generation: int
    ) -> None:
        """Re-classify live data and merge updated archetype stats.

        Ensures immigrants use fresh indicator priors instead of frozen
        gen-0 corpus statistics for the remainder of the run.
        """
        if not self._classifier_ready:
            return
        try:
            feature_cols = self._classifier._feature_columns
            feat_df = live_df[feature_cols].fillna(0)
            if len(feat_df) < 10:
                return
            pred_df = self._classifier.predict(feat_df)
            # Merge fitness into prediction for stats computation
            pred_df['fitness'] = live_df['fitness'].values

            updated_count = 0
            for arch_id in set(pred_df['archetype'].unique()) - {-1}:
                arch_rows = pred_df[pred_df['archetype'] == arch_id]
                if len(arch_rows) < 3:
                    continue
                live_mean = float(arch_rows['fitness'].mean())
                existing = self._classifier.archetype_stats.get(arch_id, {})
                old_mean = existing.get('fitness_mean', 0) or 0
                # Blend: 70% historical + 30% live (conservative update)
                if old_mean > 0:
                    existing['fitness_mean'] = 0.7 * old_mean + 0.3 * live_mean
                else:
                    existing['fitness_mean'] = live_mean
                updated_count += 1

            if updated_count > 0:
                self.logger.info(
                    f"[SIS] Refreshed archetype_stats for {updated_count} archetypes "
                    f"from {len(live_df)} live samples at gen {generation}"
                )
        except Exception as e:
            self.logger.debug(f"[SIS] archetype_stats refresh failed: {e}")
