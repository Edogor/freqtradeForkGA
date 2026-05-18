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

Usage:
    surrogate = SurrogateModel(config)
    # After each generation:
    surrogate.add_training_data(population)
    surrogate.maybe_retrain()
    # Before expensive evaluation:
    to_backtest, to_skip = surrogate.filter_candidates(offspring)
    # to_skip individuals get surrogate-estimated fitness
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
