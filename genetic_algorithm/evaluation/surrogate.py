"""
Surrogate-Assisted Pre-Filtering

Trains a lightweight ML model on accumulated (strategy_features → fitness) data
to predict fitness without running a full backtest. Used to pre-filter offspring
so only the most promising candidates undergo expensive backtesting.

Config:
    surrogate:
        enabled: true
        min_training_samples: 50    # Don't activate until this many evaluated
        filter_percentile: 60       # Only backtest top 60% by surrogate score
        retrain_interval: 3         # Retrain every N generations
        validation_fraction: 0.2    # Hold out 20% for accuracy monitoring
        model: "random_forest"      # "random_forest" or "gradient_boosting"
        acquisition: "expected_improvement"  # T3.8 — optional; default
                                              # is plain predicted-fitness ranking

Usage:
    surrogate = SurrogateModel(config)
    # After each generation:
    surrogate.add_training_data(population)
    surrogate.maybe_retrain()
    # Before expensive evaluation (plain ranking):
    to_backtest, to_skip = surrogate.filter_candidates(offspring)
    # OR with Expected Improvement (T3.8 — explores high-variance points):
    to_backtest, to_skip = surrogate.filter_candidates_ei(
        offspring, best_so_far=hof.best_fitness
    )
"""

import logging
import math
import random
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Feature extraction constants
_INDICATOR_TYPES = [
    'RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'ADX', 'ATR', 'CCI',
    'STOCH', 'SUPERTREND', 'DONCHIAN', 'ICHIMOKU', 'CMF', 'VWAP',
    'VROC', 'MFI', 'OBV', 'WILLR', 'ROC', 'TEMA', 'KAMA', 'AROON',
    'PSAR', 'CDL_ENGULFING', 'CDL_HAMMER', 'CDL_DOJI',
]

_OPERATOR_TYPES = ['<', '>', 'cross_above', 'cross_below', 'increasing',
                   'decreasing', 'between', 'value_above_ago']


def extract_features(strategy_gene) -> List[float]:
    """Extract a fixed-length feature vector from a StrategyGene.

    Features capture structural properties without running a backtest:
    - Indicator composition (one-hot + counts)
    - Condition statistics (operator distribution, threshold stats)
    - Risk parameters (stoploss, ROI, max_open_trades)
    - Structural complexity measures

    Returns:
        List of floats (fixed length regardless of strategy structure)
    """
    gene = strategy_gene
    features: List[float] = []

    # 1. Indicator type one-hot (26 features)
    ind_types = {ind.type for ind in gene.indicators}
    for t in _INDICATOR_TYPES:
        features.append(1.0 if t in ind_types else 0.0)

    # 2. Counts (4 features)
    features.append(float(len(gene.indicators)))
    features.append(float(len(gene.entry_conditions)))
    features.append(float(len(gene.exit_conditions)))
    features.append(float(len(gene.short_entry_conditions) + len(gene.short_exit_conditions)))

    # 3. Operator distribution in entry conditions (8 features)
    entry_ops = [c.operator for c in gene.entry_conditions]
    for op in _OPERATOR_TYPES:
        features.append(float(entry_ops.count(op)))

    # 4. Operator distribution in exit conditions (8 features)
    exit_ops = [c.operator for c in gene.exit_conditions]
    for op in _OPERATOR_TYPES:
        features.append(float(exit_ops.count(op)))

    # 5. Parameter statistics (6 features)
    periods = []
    for ind in gene.indicators:
        p = ind.parameters.get('period', ind.parameters.get('fast_period', 0))
        if isinstance(p, (int, float)):
            periods.append(float(p))
    features.append(sum(periods) / max(1, len(periods)))  # mean period
    features.append(max(periods) if periods else 0.0)      # max period
    features.append(min(periods) if periods else 0.0)      # min period

    # Indicator weight stats
    weights = [ind.weight for ind in gene.indicators]
    features.append(sum(weights) / max(1, len(weights)))   # mean weight
    features.append(max(weights) if weights else 0.0)       # max weight
    features.append(min(weights) if weights else 0.0)       # min weight

    # 6. Risk parameters (5 features)
    features.append(gene.stoploss)
    roi_values = list(gene.minimal_roi.values())
    features.append(max(roi_values) if roi_values else 0.0)
    features.append(min(roi_values) if roi_values else 0.0)
    features.append(float(gene.max_open_trades))
    features.append(1.0 if gene.trailing_stop else 0.0)

    # 7. Condition threshold statistics (4 features)
    all_thresholds = [c.threshold for c in gene.entry_conditions + gene.exit_conditions]
    if all_thresholds:
        features.append(sum(all_thresholds) / len(all_thresholds))  # mean
        features.append(max(all_thresholds))
        features.append(min(all_thresholds))
        features.append(float(len(set(all_thresholds))))  # unique count
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # 8. Logic composition (2 features)
    and_count = sum(1 for c in gene.entry_conditions if c.logic == 'AND')
    or_count = sum(1 for c in gene.entry_conditions if c.logic == 'OR')
    features.append(float(and_count))
    features.append(float(or_count))

    # 9. Multi-timeframe (2 features)
    features.append(float(len(gene.informative_timeframes)))
    features.append(1.0 if gene.can_short else 0.0)

    # Defensive check: this should ALWAYS be exactly 65 features
    # 26 (indicator one-hot) + 4 (counts) + 8 (entry ops) + 8 (exit ops) +
    # 6 (params) + 5 (risk) + 4 (thresholds) + 2 (logic) + 2 (mtf) = 65
    assert len(features) == 65, (
        f"Surrogate feature vector length mismatch: expected 65, got {len(features)}"
    )

    return features  # Total: 26 + 4 + 8 + 8 + 6 + 5 + 4 + 2 + 2 = 65 features


class SurrogateModel:
    """Lightweight fitness predictor trained on evaluated strategies."""

    def __init__(self, config: Dict[str, Any]):
        surr_config = config.get('surrogate', {})
        self.enabled = surr_config.get('enabled', False)
        self.min_training_samples = surr_config.get('min_training_samples', 50)
        self.filter_percentile = surr_config.get('filter_percentile', 60)
        self.retrain_interval = surr_config.get('retrain_interval', 3)
        self.validation_fraction = surr_config.get('validation_fraction', 0.2)
        self.model_type = surr_config.get('model', 'random_forest')
        self.mutate_skipped = surr_config.get('mutate_skipped', True)
        self._ga_config = config  # Keep ref for mutation calls

        # Adaptive filter: ramp percentile down as R² improves
        adaptive_cfg = surr_config.get('adaptive_filter', {})
        self._adaptive_enabled = adaptive_cfg.get('enabled', True)
        self._adaptive_min_percentile = adaptive_cfg.get('min_percentile', 35)
        self._adaptive_r2_threshold = adaptive_cfg.get('r2_threshold', 0.3)
        self._initial_filter_percentile = self.filter_percentile

        self._model = None
        self._training_X: List[List[float]] = []
        self._training_y: List[float] = []
        self._generations_since_retrain: int = 0
        self._last_validation_r2: Optional[float] = None
        self._is_trained: bool = False

    @property
    def ready(self) -> bool:
        """Whether the surrogate has enough data and a trained model."""
        return self._is_trained and self._model is not None

    def add_training_data(self, population) -> int:
        """Extract features from evaluated individuals and add to training buffer.

        Args:
            population: Evaluated Population with fitness scores

        Returns:
            Number of new samples added
        """
        if not self.enabled:
            return 0

        added = 0
        for ind in population:
            if not ind.evaluated or ind.fitness is None:
                continue
            # Skip surrogate-evaluated individuals (circular training)
            if getattr(ind, 'surrogate_evaluated', False):
                continue
            try:
                features = extract_features(ind.strategy_gene)
                fitness = float(ind.fitness)
                if math.isfinite(fitness):
                    self._training_X.append(features)
                    self._training_y.append(fitness)
                    added += 1
            except Exception as e:
                logger.debug(f"[SURROGATE] Feature extraction failed: {e}")

        self._generations_since_retrain += 1
        return added

    def maybe_retrain(self) -> bool:
        """Retrain the model if enough data and retrain interval reached.

        Returns:
            True if model was retrained
        """
        if not self.enabled:
            return False
        if len(self._training_y) < self.min_training_samples:
            return False
        if self._generations_since_retrain < self.retrain_interval and self._is_trained:
            return False

        return self._train()

    def _train(self) -> bool:
        """Fit the surrogate model on accumulated training data."""
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
        except ImportError:
            logger.warning("[SURROGATE] scikit-learn not available — surrogate disabled")
            self.enabled = False
            return False

        X = self._training_X
        y = self._training_y

        # Train/validation split
        val_size = max(5, int(len(y) * self.validation_fraction))
        if len(y) - val_size < 10:
            # Not enough data for a meaningful split
            X_train, y_train = X, y
            X_val, y_val = X[-val_size:], y[-val_size:]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=42
            )

        if self.model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, subsample=0.8
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42,
                n_jobs=-1, min_samples_leaf=3
            )

        model.fit(X_train, y_train)

        # Evaluate on validation set
        if X_val:
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            self._last_validation_r2 = r2
            logger.info(f"[SURROGATE] Retrained on {len(X_train)} samples — "
                        f"validation R²={r2:.3f} (total buffer: {len(y)})")
        else:
            logger.info(f"[SURROGATE] Retrained on {len(X_train)} samples")

        self._model = model
        self._is_trained = True
        self._generations_since_retrain = 0

        # Adaptive filter: lower the percentile as R² improves
        if self._adaptive_enabled and self._last_validation_r2 is not None:
            r2 = self._last_validation_r2
            if r2 >= self._adaptive_r2_threshold:
                # Linear interpolation: R²=threshold → initial, R²=1.0 → min
                frac = min(1.0, (r2 - self._adaptive_r2_threshold)
                           / (1.0 - self._adaptive_r2_threshold))
                new_pct = self._initial_filter_percentile - frac * (
                    self._initial_filter_percentile - self._adaptive_min_percentile
                )
                old_pct = self.filter_percentile
                self.filter_percentile = max(
                    self._adaptive_min_percentile, int(round(new_pct))
                )
                if self.filter_percentile != old_pct:
                    logger.info(
                        "[SURROGATE] Adaptive filter: R²=%.3f → percentile %d→%d",
                        r2, old_pct, self.filter_percentile,
                    )

        return True

    def predict(self, strategy_gene) -> Optional[float]:
        """Predict fitness for a single strategy gene.

        Returns None if model is not ready.
        """
        if not self.ready:
            return None
        try:
            features = extract_features(strategy_gene)
            prediction = float(self._model.predict([features])[0])
            return prediction
        except Exception:
            return None

    def filter_candidates(self, individuals: list) -> Tuple[list, list]:
        """Split individuals into those worth backtesting and those to skip.

        Args:
            individuals: List of unevaluated Individual objects

        Returns:
            (to_backtest, to_skip) — to_skip get surrogate-estimated fitness
        """
        if not self.ready or not self.enabled:
            return individuals, []

        # Score all candidates
        scored: List[Tuple[Any, float]] = []
        failed: List[Any] = []
        for ind in individuals:
            pred = self.predict(ind.strategy_gene)
            if pred is not None:
                scored.append((ind, pred))
            else:
                failed.append(ind)

        if not scored:
            return individuals, []

        # Sort by predicted fitness (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Top percentile goes to full backtest
        cutoff = max(1, int(len(scored) * self.filter_percentile / 100))
        to_backtest = [ind for ind, _ in scored[:cutoff]] + failed
        to_skip = []

        for ind, pred_fitness in scored[cutoff:]:
            # Mutate skipped individuals to prevent population collapse
            # (without this, skipped candidates are clones → diversity loss)
            if self.mutate_skipped:
                try:
                    from genetic_algorithm.core.mutation import mutate
                    mutation_rate = self._ga_config.get('mutation_rate', 0.18)
                    ind = mutate(ind, mutation_rate, self._ga_config)
                except Exception as e:
                    logger.debug(f"[SURROGATE] Mutation of skipped individual failed: {e}")
            # Assign surrogate fitness (discounted to discourage gaming)
            discount = 0.85  # Surrogate predictions are slightly pessimistic
            ind.fitness = pred_fitness * discount
            ind.raw_fitness = pred_fitness * discount
            ind.evaluated = True
            ind.surrogate_evaluated = True
            if hasattr(ind, 'metrics') and isinstance(ind.metrics, dict):
                ind.metrics['surrogate_fitness'] = pred_fitness
                ind.metrics['surrogate_evaluated'] = True
                ind.metrics['surrogate_mutated'] = self.mutate_skipped
            to_skip.append(ind)

        logger.info(f"[SURROGATE] Filtered {len(individuals)} candidates: "
                    f"{len(to_backtest)} to backtest, {len(to_skip)} surrogate-scored "
                    f"(R²={self._last_validation_r2:.3f})" if self._last_validation_r2 else "")
        return to_backtest, to_skip

    def predict_with_uncertainty(
        self, strategy_gene
    ) -> Optional[Tuple[float, float]]:
        """T3.8 — Predict (mean, std) for a single gene.

        Standard deviation is estimated from individual tree predictions
        when the underlying model is a RandomForest.  For other model
        types (e.g. GradientBoosting), uncertainty is returned as ``0.0``
        which makes Expected Improvement degenerate to plain ranking by
        predicted mean — still safe.

        Returns ``None`` if the model is not ready or feature extraction
        fails.
        """
        if not self.ready:
            return None
        try:
            features = extract_features(strategy_gene)
        except Exception:
            return None

        try:
            mean = float(self._model.predict([features])[0])
        except Exception:
            return None

        std = 0.0
        estimators = getattr(self._model, "estimators_", None)
        if estimators:
            try:
                preds = []
                for est in estimators:
                    # GradientBoosting nests estimators in a 2D ndarray
                    if hasattr(est, "predict"):
                        preds.append(float(est.predict([features])[0]))
                if len(preds) >= 2:
                    mean_p = sum(preds) / len(preds)
                    var = sum((p - mean_p) ** 2 for p in preds) / (len(preds) - 1)
                    std = math.sqrt(max(0.0, var))
            except Exception:
                std = 0.0
        return mean, std

    def expected_improvement(
        self, strategy_gene, best_so_far: float, xi: float = 0.01
    ) -> Optional[float]:
        """T3.8 — Expected Improvement acquisition.

        EI(x) = (μ - f* - ξ) · Φ(z) + σ · φ(z)   if σ > 0
              = max(0, μ - f* - ξ)                if σ = 0

        Where μ, σ are surrogate mean and std at x, f* is the incumbent
        best fitness, ξ is an exploration knob (higher ξ → more
        exploration).  We *maximize* fitness so the formula matches the
        Mockus EI for maximization problems.

        Returns ``None`` if surrogate is not ready.
        """
        result = self.predict_with_uncertainty(strategy_gene)
        if result is None:
            return None
        mean, std = result
        improvement = mean - best_so_far - xi
        if std <= 1e-12:
            return max(0.0, improvement)
        try:
            from statistics import NormalDist

            z = improvement / std
            normal = NormalDist()
            cdf = normal.cdf(z)
            pdf = normal.pdf(z)
        except Exception:
            return max(0.0, improvement)
        return improvement * cdf + std * pdf

    def filter_candidates_ei(
        self,
        individuals: list,
        best_so_far: float,
        xi: float = 0.01,
    ) -> Tuple[list, list]:
        """Variant of :meth:`filter_candidates` that ranks by Expected
        Improvement instead of raw predicted fitness.

        EI prefers candidates whose *uncertainty* gives them a real
        chance of beating the incumbent, which prevents the surrogate
        from greedily exploiting a single high-fitness basin and
        starving exploration.

        Falls back to the standard filter when the surrogate is not
        ready.
        """
        if not self.ready or not self.enabled:
            return individuals, []

        scored: List[Tuple[Any, float]] = []
        failed: List[Any] = []
        for ind in individuals:
            ei = self.expected_improvement(ind.strategy_gene, best_so_far, xi=xi)
            if ei is not None:
                scored.append((ind, ei))
            else:
                failed.append(ind)

        if not scored:
            return individuals, []

        scored.sort(key=lambda x: x[1], reverse=True)
        cutoff = max(1, int(len(scored) * self.filter_percentile / 100))
        to_backtest = [ind for ind, _ in scored[:cutoff]] + failed
        to_skip: list = []

        for ind, ei_score in scored[cutoff:]:
            # We still need a fitness number for skipped individuals.
            # Use the surrogate mean rather than EI (EI can be ~0 for
            # high-mean low-variance points which are still fine).
            pred = self.predict(ind.strategy_gene)
            if pred is None:
                # Safety net: leave un-evaluated so the runner can keep
                # them in the backtest queue.
                continue
            if self.mutate_skipped:
                try:
                    from genetic_algorithm.core.mutation import mutate

                    mutation_rate = self._ga_config.get("mutation_rate", 0.18)
                    ind = mutate(ind, mutation_rate, self._ga_config)
                except Exception as e:
                    logger.debug(f"[SURROGATE] EI mutation failed: {e}")
            discount = 0.85
            ind.fitness = pred * discount
            ind.raw_fitness = pred * discount
            ind.evaluated = True
            ind.surrogate_evaluated = True
            if hasattr(ind, "metrics") and isinstance(ind.metrics, dict):
                ind.metrics["surrogate_fitness"] = pred
                ind.metrics["surrogate_ei"] = ei_score
                ind.metrics["surrogate_evaluated"] = True
                ind.metrics["acquisition"] = "expected_improvement"
            to_skip.append(ind)

        logger.info(
            "[SURROGATE] EI filter: %d candidates → %d backtest, %d skipped "
            "(R²=%s, best_so_far=%.4f)",
            len(individuals),
            len(to_backtest),
            len(to_skip),
            f"{self._last_validation_r2:.3f}" if self._last_validation_r2 else "n/a",
            best_so_far,
        )
        return to_backtest, to_skip

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state (excluding the sklearn model — retrain on restore)."""
        return {
            'training_size': len(self._training_y),
            'is_trained': self._is_trained,
            'last_validation_r2': self._last_validation_r2,
            'generations_since_retrain': self._generations_since_retrain,
            # Don't serialize training data — too large. Model retrains from scratch.
        }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore counters from checkpoint. Model itself retrains from scratch."""
        self._is_trained = data.get('is_trained', False)
        self._last_validation_r2 = data.get('last_validation_r2')
        self._generations_since_retrain = data.get('generations_since_retrain', 0)

    def get_report(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'ready': self.ready,
            'training_samples': len(self._training_y),
            'is_trained': self._is_trained,
            'last_validation_r2': self._last_validation_r2,
            'model_type': self.model_type,
        }
