"""
SIS Integrator

Bridges Strategy Intelligence System analysis results into the live GA evolution.
Provides 3 hooks:
1. Seed filtering — after population initialization, replaces low-predicted-score randoms
2. Archetype-aware immigrants — each generation, injects individuals from underrepresented archetypes
3. Indicator/operator bias — provides enrichment-derived weights for mutation

Usage:
    from genetic_algorithm.intelligence.sis_integrator import SISIntegrator
    sis = SISIntegrator(config, logger)
    ga.set_immigrant_provider(sis.immigrant_provider)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from genetic_algorithm.intelligence.corpus import CorpusBuilder, SURROGATE_FEATURE_NAMES
from genetic_algorithm.intelligence.archetypes import ArchetypeClassifier
from genetic_algorithm.intelligence.predictors import MultiTargetPredictor
from genetic_algorithm.core.individual import Individual

logger = logging.getLogger(__name__)

# ── Static enrichment weights from PatternMiner analysis on 4289 strategies ──

SIS_INDICATOR_ENRICHMENT: Dict[str, float] = {
    'CCI': 3.0,       # 26.5x lift in top strategies
    'STOCH': 2.5,     # 12.5x lift
    'ATR': 2.0,       # 6.5x lift
    'ROC': 2.0,       # 6.5x lift
    'DONCHIAN': 1.8,  # Enriched + best synergy pair with ROC
    'ADX': 1.5,       # Enriched in top archetypes
    'RSI': 1.0,
    'BBANDS': 1.0,
    'SUPERTREND': 1.0,
    'EMA': 0.9,
    'SMA': 0.9,
    'WILLR': 1.1,
    'MFI': 1.0,
    'AROON': 1.0,
    'TEMA': 1.0,
    'KAMA': 1.0,
    'VROC': 1.0,
    'PSAR': 0.8,
    'CMF': 0.9,
    'MACD': 0.5,      # Depleted in top strategies
    'ICHIMOKU': 0.3,  # Strongly depleted
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


class SISIntegrator:
    """Bridges SIS analysis results into the live GA evolution.

    Thread-safety: methods are not thread-safe; intended to be called from the
    single GA evolution thread only.
    """

    def __init__(self, config: dict, logger_inst: logging.Logger):
        self.config = config
        self.logger = logger_inst
        self._sis_config: Dict[str, Any] = config.get('sis', {})

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
        self._classifier_ready = bool(self._classifier.archetype_labels)

        # ── Predictor ─────────────────────────────────────────────────────────
        self._predictor = MultiTargetPredictor(models_dir=models_dir)
        try:
            self._predictor.load(models_dir)
        except Exception as e:
            self.logger.warning(f"[SIS] Could not load predictor: {e}")
        self._predictor_ready = 'win_rate' in self._predictor.models

        # ── Corpus (optional, for context) ───────────────────────────────────
        self._corpus_df: Optional[pd.DataFrame] = None
        if corpus_path.exists():
            try:
                self._corpus_df = pd.read_parquet(corpus_path)
                self.logger.info(
                    f"[SIS] Loaded corpus: {len(self._corpus_df)} strategies"
                )
            except Exception as e:
                self.logger.warning(f"[SIS] Failed to load corpus: {e}")

        # ── Feature extractor (reuse single instance) ─────────────────────────
        self._corpus_builder = CorpusBuilder()

        # ── Generation tracking ───────────────────────────────────────────────
        self._n_filtered_last: int = 0
        self._n_immigrants_last: int = 0

        # ── SIS JSONL log — reads configured path or falls back to output dir ──
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

        # ── PatternMiner auto-load (overrides hardcoded enrichment constants) ──
        self._pattern_enrichment: Dict[str, float] = {}
        self._pattern_operator_weights: Dict[str, float] = {}
        try:
            from genetic_algorithm.intelligence.pattern_mining import PatternMiner
            miner = PatternMiner.load(models_dir)
            if miner is not None and miner.indicator_enrichment is not None:
                enrich_df = miner.indicator_enrichment
                # Apply log1p compression: raw enrichment ratios (26x, 12x, ...) are
                # mapped to log-scale weights similar to hardcoded constants (3.0, 2.5, ...)
                # log1p(26.52)≈3.29, log1p(11.83)≈2.51, log1p(1.46)≈0.88 (ICHIMOKU fix)
                self._pattern_enrichment = {
                    str(row['indicator']): float(np.log1p(max(0, float(row['enrichment']))))
                    for _, row in enrich_df.iterrows()
                    if row['enrichment'] > 0
                }
                self.logger.info(
                    f"[SIS] Loaded PatternMiner enrichment for "
                    f"{len(self._pattern_enrichment)} indicators (log-compressed)"
                )
            if miner is not None and miner.operator_patterns is not None:
                entry_ops = miner.operator_patterns.get('entry_operators', {})
                self._pattern_operator_weights = {
                    op: max(0.1, v.get('top_mean', 1.0) / (v.get('global_mean', 1.0) + 0.01))
                    for op, v in entry_ops.items()
                    if v.get('global_mean', 0) > 0.01  # skip near-zero usage operators
                }
        except Exception as e:
            self.logger.debug(
                f"[SIS] PatternMiner load skipped, using hardcoded weights: {e}"
            )

        self.logger.info(
            f"[SIS] Initialized — classifier_ready={self._classifier_ready}, "
            f"predictor_ready={self._predictor_ready}"
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
    ) -> tuple:  # (DataFrame, List[int])
        """Build a feature DataFrame from a list of individuals.

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

    # ── Hook 1: seed quality filtering ────────────────────────────────────────

    def filter_population(self, population, ga) -> None:
        """Replace the lowest-predicted-score random seeds with better candidates.

        Called once, immediately after ``initialize_population()`` completes.
        Only acts when the win_rate predictor is loaded. Individuals with an
        explicit origin (HoF, LLM, seeded) and already-evaluated individuals
        are protected.
        """
        if not self._predictor_ready:
            return

        filter_fraction: float = self._sis_config.get('filter_fraction', 0.15)

        # Identify "random" individuals: unevaluated, no origin set
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
            scored_df = self._predictor.predict(feat_df, targets=['win_rate'])
        except Exception as e:
            self.logger.warning(f"[SIS] Predictor.predict() failed during seed filtering: {e}")
            return

        pred_col = scored_df.get('pred_win_rate', pd.Series(dtype=float))
        scores: List[float] = pred_col.tolist()

        # Map back to population indices via valid_idx
        scored_pairs = [
            (random_indices[valid_idx[j]], scores[j]) for j in range(len(valid_idx))
        ]
        # Sort ascending: lowest score first (these are the ones to replace)
        scored_pairs.sort(key=lambda x: x[1])
        pop_indices_to_replace = [p[0] for p in scored_pairs[:n_to_replace]]

        # Generate 2× replacement candidates
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

        # Score replacement candidates
        cand_feat_df, cand_valid_idx = self._build_feature_df(new_candidates)
        if cand_feat_df.empty:
            return

        try:
            cand_scored_df = self._predictor.predict(cand_feat_df, targets=['win_rate'])
        except Exception as e:
            self.logger.warning(f"[SIS] Candidate scoring failed during seed filtering: {e}")
            return

        cand_pred = cand_scored_df.get('pred_win_rate', pd.Series(dtype=float))
        cand_scores: List[float] = cand_pred.tolist()
        cand_scored = [
            (new_candidates[cand_valid_idx[j]], cand_scores[j])
            for j in range(len(cand_valid_idx))
        ]
        cand_scored.sort(key=lambda x: x[1], reverse=True)  # best first

        # Swap into population
        replaced = 0
        for pop_idx, (replacement, _) in zip(
            pop_indices_to_replace, cand_scored
        ):
            replacement.metrics['origin'] = 'sis_filtered'
            population.individuals[pop_idx] = replacement
            replaced += 1

        self._n_filtered_last = replaced
        if replaced > 0:
            self.logger.info(
                f"[SIS] Seed filtering: replaced {replaced} low-quality random seeds"
            )

    # ── Hook 2: archetype-aware immigrants ────────────────────────────────────

    def immigrant_provider(self, ga, generation: int) -> List[Individual]:
        """Provide archetype-aware immigrants for the current generation.

        Registered via ``ga.set_immigrant_provider(sis.immigrant_provider)``.
        Classifies current population, identifies missing/underrepresented
        archetypes, and generates individuals biased toward those profiles.
        Returns 0-N immigrants (typically 2-3).
        """
        immigrants: List[Individual] = []

        # Inject operator weights into config so mutation can use them
        self.config['_operator_weights'] = self.get_operator_weights()

        if not self._classifier_ready:
            return immigrants

        known_archetypes = set(self._classifier.archetype_labels.keys()) - {-1}
        if not known_archetypes:
            return immigrants

        # Try to obtain current population from GA instance
        pop_inds: List[Individual] = []
        if hasattr(ga, 'population') and ga.population is not None:
            pop_inds = list(ga.population.individuals)

        if not pop_inds:
            # Population not accessible via ga.population — return early
            # (archetype classification requires a population to examine)
            return immigrants

        feat_df, _ = self._build_feature_df(pop_inds)
        if feat_df.empty or len(feat_df) < 2:
            return immigrants

        # Classify archetypes present in population
        try:
            pred_df = self._classifier.predict(feat_df)
            present_archetypes = set(pred_df['archetype'].unique()) - {-1}
        except Exception as e:
            self.logger.debug(f"[SIS] Archetype prediction failed: {e}")
            return immigrants

        missing_archetypes = known_archetypes - present_archetypes

        # Also find archetypes present but very rare (< 2% of population)
        underrepresented: set = set()
        if 'archetype' in pred_df.columns:
            counts = pred_df['archetype'].value_counts()
            threshold = max(1, len(pred_df) * 0.02)
            for arch_id in known_archetypes:
                if counts.get(arch_id, 0) < threshold:
                    underrepresented.add(arch_id)

        target_archetypes = list(missing_archetypes | underrepresented)
        if not target_archetypes:
            return immigrants

        n_immigrants: int = self._sis_config.get('immigrants_per_gen', 2)

        for arch_id in target_archetypes[:n_immigrants]:
            try:
                immigrant = self._generate_archetype_immigrant(ga, arch_id, generation)
                if immigrant is not None:
                    immigrants.append(immigrant)
            except Exception as e:
                self.logger.debug(
                    f"[SIS] Failed to generate immigrant for archetype {arch_id}: {e}"
                )

        self._n_immigrants_last = len(immigrants)

        if immigrants:
            self.logger.debug(
                f"[SIS] Gen {generation}: injecting {len(immigrants)} archetype immigrants "
                f"(missing={len(missing_archetypes)}, underrep={len(underrepresented)})"
            )

        self.log_generation(generation, pop_inds)
        return immigrants

    def _generate_archetype_immigrant(
        self, ga, archetype_id: int, generation: int
    ) -> Optional[Individual]:
        """Generate a strategy biased toward a target archetype's indicator profile."""
        stats = self._classifier.archetype_stats.get(archetype_id, {})
        top_indicators: List[str] = stats.get('top_indicators', [])

        # Generate base random strategy
        unique_id = hash((archetype_id, generation, int(time.time() * 1000))) & 0xFFFF
        gene = ga.strategy_generator.generate_random_strategy(
            generation=generation,
            individual_id=unique_id,
        )

        # Override first indicator with archetype-preferred type if available
        if top_indicators and gene.indicators:
            try:
                from genetic_algorithm.evaluation.surrogate import _INDICATOR_TYPES
                valid_top = [ind for ind in top_indicators if ind in _INDICATOR_TYPES]
                if valid_top:
                    preferred = valid_top[0]
                    old_type = gene.indicators[0].type
                    gene.indicators[0].type = preferred
                    # Update any conditions that referenced the old indicator type
                    for cond in gene.entry_conditions:
                        if cond.indicator == old_type or cond.indicator.startswith(
                            old_type + '_'
                        ):
                            cond.indicator = preferred
                    for cond in gene.exit_conditions:
                        if cond.indicator == old_type or cond.indicator.startswith(
                            old_type + '_'
                        ):
                            cond.indicator = preferred
            except Exception as e:
                self.logger.debug(f"[SIS] Indicator override failed: {e}")

        individual = Individual(strategy_gene=gene)
        individual.metrics['origin'] = 'sis_immigrant'
        individual.metrics['sis_archetype'] = archetype_id
        return individual

    # ── Hook 3: indicator/operator enrichment weights ─────────────────────────

    def get_indicator_weights(self, base_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Return merged indicator weights from SIS enrichment + optional base weights.

        Prefers live PatternMiner enrichment over hardcoded constants when available.
        Applies ``indicator_weight_scale`` by amplifying deviations from the mean
        (scale=1.0 is a no-op; scale=2.0 doubles the spread above/below 1.0).

        Args:
            base_weights: Optional weights from FeatureImportanceTracker. When
                          provided, SIS and base weights are multiplied together.

        Returns:
            Indicator name -> weight dict, normalized so mean == 1.0.
        """
        scale: float = self._sis_config.get('indicator_weight_scale', 1.5)

        # Prefer live PatternMiner enrichment; fall back to hardcoded constants
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

        # Step 1: normalize to mean = 1.0
        if raw:
            mean_w = float(np.mean(list(raw.values())))
            if mean_w > 0:
                normalized = {ind: w / mean_w for ind, w in raw.items()}
            else:
                normalized = raw
        else:
            return {}

        # Step 2: amplify deviations from 1.0 by scale (scale=1.0 → identity)
        return {ind: max(0.1, 1.0 + (w - 1.0) * scale) for ind, w in normalized.items()}

    def get_operator_weights(self) -> Dict[str, float]:
        """Return operator -> weight dict, optionally scaled by operator_weight_scale.

        Prefers live PatternMiner operator patterns; falls back to hardcoded constants.
        Applies ``operator_weight_scale`` as a deviation amplifier (scale=1.0 is no-op).
        """
        scale: float = self._sis_config.get('operator_weight_scale', 1.0)

        # Prefer live PatternMiner operator weights; fall back to hardcoded constants
        source = self._pattern_operator_weights if self._pattern_operator_weights \
            else SIS_OPERATOR_ENRICHMENT

        if scale == 1.0:
            return dict(source)

        mean_w = float(np.mean(list(source.values()))) if source else 1.0
        if mean_w <= 0:
            return dict(source)
        normalized = {op: w / mean_w for op, w in source.items()}
        return {op: max(0.1, 1.0 + (w - 1.0) * scale) for op, w in normalized.items()}

    # ── Generation logging ────────────────────────────────────────────────────

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

            entry = {
                'gen': generation,
                'timestamp': time.time(),
                'best_fitness': best_fitness,
                'mean_fitness': mean_fitness,
                'archetype_distribution': archetype_coverage or {},
                'n_sis_immigrants': self._n_immigrants_last,
                'n_filtered': self._n_filtered_last,
                'indicator_weights_used': {
                    k: round(v, 3)
                    for k, v in self.get_indicator_weights().items()
                },
            }
            with open(self._log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.logger.debug(f"[SIS] Log write failed: {e}")
