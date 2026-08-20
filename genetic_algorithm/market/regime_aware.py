"""
Regime-Aware Fitness Evaluator

Extends the FitnessEvaluator to support regime-balanced evaluation.
Strategies are evaluated across multiple market regime segments (bullish, 
bearish, sideways) to prevent overfitting to a single market condition.

This module connects the regime detection pipeline to the GA fitness evaluation.
"""

import logging
import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.evaluation.direct_backtester import BacktestResult
from genetic_algorithm.utils.regime_detector import (
    RegimeDetector,
    RegimeSegment,
    load_ohlcv_data,
)

if TYPE_CHECKING:
    from genetic_algorithm.utils.dataset_policy import DatasetPolicy

logger = logging.getLogger(__name__)


@dataclass
class RegimeEvaluationResult:
    """Container for per-regime evaluation results."""
    segment: RegimeSegment
    fitness: float
    metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

    def trade_count(self) -> int:
        """Return a non-negative trade count for the segment contract."""
        if not self.success:
            return 0
        try:
            return max(0, int(self.metrics.get('num_trades', 0)))
        except (TypeError, ValueError, OverflowError):
            return 0

    def outcome_status(self, min_segment_trades: int) -> str:
        """Classify the result without allowing missing evidence to disappear."""
        if not self.success:
            return 'failed'
        try:
            raw_fitness = float(self.fitness)
        except (TypeError, ValueError, OverflowError):
            return 'invalid_fitness'
        if not math.isfinite(raw_fitness):
            return 'invalid_fitness'
        trades = self.trade_count()
        if trades == 0:
            return 'zero_trades'
        if trades < min_segment_trades:
            return 'low_trades'
        return 'ok'

    def effective_fitness(self, min_segment_trades: int) -> float:
        """Fitness used by aggregation; weak/missing evidence can never help."""
        status = self.outcome_status(min_segment_trades)
        if status in {'failed', 'invalid_fitness'}:
            return 0.0
        raw_fitness = float(self.fitness)
        if status in {'zero_trades', 'low_trades'}:
            return min(raw_fitness, 0.0)
        return raw_fitness


class RegimeAwareEvaluator:
    """
    Evaluates strategies across multiple regime segments.
    
    This evaluator wraps the standard FitnessEvaluator and extends it to:
    1. Run backtests on specific regime segments (using timeranges)
    2. Aggregate fitness across regimes using configurable methods
    3. Track holdout segments separately for final validation
    4. Cache segment-level results for efficiency
    
    The goal is to produce strategies robust to different market conditions,
    not just strategies that perform well in one type of market (e.g., bull run).
    
    Usage:
        evaluator = RegimeAwareEvaluator(config, segments={'optimization': [...], 'holdout': [...]})
        fitness, metrics = evaluator.evaluate(strategy_gene)
    """
    
    # Supported aggregation methods
    AGGREGATION_METHODS = ['mean', 'min', 'harmonic_mean', 'cvar']
    
    def __init__(
        self,
        config: Dict[str, Any],
        segments: Optional[Dict[str, List[RegimeSegment]]] = None,
    ):
        """
        Initialize regime-aware evaluator.

        Args:
            config: Configuration dictionary (includes regime_aware section)
            segments: Optional pre-computed segments dict with keys:
                      'optimization', 'model_selection', 'holdout'
                      If not provided, will auto-detect from data.
        """
        self.config = config
        self.regime_config = config.get('regime_aware', {})
        
        # Initialize base fitness evaluator
        self.base_evaluator = FitnessEvaluator(config)
        
        # Store segments
        self.segments = segments or {}
        self._optimization_segments = self.segments.get('optimization', [])
        self._holdout_segments = self.segments.get('holdout', [])
        self._model_selection_segments = self.segments.get('model_selection', [])
        
        # Aggregation method
        self.aggregation_method = self.regime_config.get('aggregation', 'harmonic_mean')
        if self.aggregation_method not in self.AGGREGATION_METHODS:
            logger.warning(f"Unknown aggregation method '{self.aggregation_method}', using 'harmonic_mean'")
            self.aggregation_method = 'harmonic_mean'
        
        # CVaR parameters (for 'cvar' aggregation)
        self.cvar_alpha = self.regime_config.get('cvar_alpha', 0.2)  # Bottom 20%
        if (isinstance(self.cvar_alpha, bool)
                or not isinstance(self.cvar_alpha, (int, float))
                or not math.isfinite(float(self.cvar_alpha))
                or not 0.0 < float(self.cvar_alpha) <= 1.0):
            raise ValueError("regime_aware.cvar_alpha must be finite and in (0, 1]")
        self.cvar_alpha = float(self.cvar_alpha)

        # Minimum trades per segment. Low-trade segments remain part of coverage,
        # but positive scores are capped at zero because the evidence is weak.
        self.min_segment_trades = self.regime_config.get('min_segment_trades', 0)
        if (isinstance(self.min_segment_trades, bool)
                or not isinstance(self.min_segment_trades, int)
                or self.min_segment_trades < 0):
            raise ValueError("regime_aware.min_segment_trades must be a non-negative integer")
        
        # Cache for segment-level results: (strategy_hash, segment_id) -> RegimeEvaluationResult
        self._segment_cache: Dict[Tuple[str, str], RegimeEvaluationResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Holdout protection - prevents access during evolution
        self._holdout_locked = True  # Locked by default
        self._holdout_access_attempts = 0
        
        # Regime weights (optional - for weighted aggregation by regime type)
        self.regime_weights = self.regime_config.get('regime_weights', {
            'bullish': 1.0,
            'bearish': 1.0,
            'sideways': 1.0,
        })
        if not isinstance(self.regime_weights, dict):
            raise ValueError("regime_aware.regime_weights must be a mapping")
        for regime_name, weight in self.regime_weights.items():
            if (isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(float(weight))
                    or float(weight) <= 0.0):
                raise ValueError(
                    f"regime_aware.regime_weights.{regime_name} must be finite and > 0"
                )
        
        logger.info(
            f"RegimeAwareEvaluator initialized with {len(self._optimization_segments)} "
            f"optimization segments, {len(self._holdout_segments)} holdout segments, "
            f"aggregation={self.aggregation_method}"
        )
        if self._holdout_segments:
            logger.info(
                "[HOLDOUT PROTECTION] Holdout segments are LOCKED - "
                "call unlock_holdout() for final validation"
            )
    
    def lock_holdout(self) -> None:
        """
        Lock holdout segments to prevent access during evolution.
        
        This is the default state. Use after final validation to re-lock.
        """
        self._holdout_locked = True
        logger.info("[HOLDOUT PROTECTION] Holdout segments LOCKED")
    
    def unlock_holdout(self) -> None:
        """
        Unlock holdout segments for final validation.
        
        Call this ONLY when evolution is complete and you want to
        evaluate the best strategy on holdout data.
        
        WARNING: After calling this, ensure holdout is re-locked if
        you continue evolution.
        """
        self._holdout_locked = False
        logger.info("[HOLDOUT PROTECTION] Holdout segments UNLOCKED for final validation")
    
    def is_holdout_locked(self) -> bool:
        """Check if holdout segments are currently locked."""
        return self._holdout_locked
    
    def evaluate(
        self,
        strategy_gene: StrategyGene,
        strategy_name: Optional[str] = None,
        use_holdout: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate strategy across multiple regime segments.
        
        During evolution, use_holdout=False to evaluate only on optimization segments.
        For final validation, use_holdout=True to get true out-of-sample performance.
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name (auto-generated if not provided)
            use_holdout: If True, evaluate on holdout segments instead of optimization
            
        Returns:
            Tuple of (aggregated_fitness, aggregated_metrics)
        """
        # HOLDOUT PROTECTION: Prevent access to holdout during evolution
        if use_holdout and self._holdout_locked:
            self._holdout_access_attempts += 1
            logger.error(
                f"[HOLDOUT PROTECTION] Attempted to access locked holdout segments! "
                f"(attempt #{self._holdout_access_attempts}). "
                f"Call unlock_holdout() only after evolution is complete."
            )
            raise RuntimeError(
                "Holdout segments are locked. This protects against data leakage during evolution. "
                "Call evaluator.unlock_holdout() only after GA evolution is complete."
            )
        
        # Select which segments to use
        if use_holdout:
            segments = self._holdout_segments
            segment_type = 'holdout'
        else:
            segments = self._optimization_segments
            segment_type = 'optimization'
        
        # An enabled evaluator without segments must not masquerade as a valid
        # regime-aware run. The disabled factory path retains legacy fallback.
        if not segments:
            if self.regime_config.get('enabled', False):
                raise RuntimeError(
                    f"Regime-aware evaluation is enabled but no {segment_type} "
                    "segments are available"
                )
            logger.warning(
                f"No {segment_type} segments available, falling back to standard evaluation"
            )
            return self.base_evaluator.evaluate(strategy_gene, strategy_name)
        
        # Generate strategy code and hash for caching
        strategy_code = self.base_evaluator.strategy_generator.generate_strategy_code(strategy_gene)
        strategy_hash = hashlib.sha256(strategy_code.encode()).hexdigest()[:16]
        generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
        
        # Evaluate on each segment
        segment_results: List[RegimeEvaluationResult] = []
        
        for segment in segments:
            result = self._evaluate_segment(
                strategy_gene=strategy_gene,
                strategy_code=strategy_code,
                strategy_hash=strategy_hash,
                generated_name=generated_name,
                segment=segment,
            )
            segment_results.append(result)
        
        # Aggregate results
        aggregated_fitness, aggregated_metrics = self._aggregate_results(
            segment_results, strategy_gene
        )
        
        # Add regime-aware metadata
        aggregated_metrics['regime_aware'] = True
        aggregated_metrics['segment_type'] = segment_type
        aggregated_metrics['num_segments'] = len(segments)
        aggregated_metrics['aggregation'] = self.aggregation_method
        
        # Log per-regime summary
        regime_summary = self._get_regime_summary(segment_results)
        for regime_type, summary in regime_summary.items():
            aggregated_metrics[f'{regime_type}_avg_fitness'] = summary['avg_fitness']
            aggregated_metrics[f'{regime_type}_segment_count'] = summary['count']
        
        logger.debug(
            f"Regime-aware evaluation for {generated_name}: "
            f"fitness={aggregated_fitness:.4f}, segments={len(segment_results)}, "
            f"regime_summary={regime_summary}"
        )
        
        return aggregated_fitness, aggregated_metrics
    
    def _evaluate_segment(
        self,
        strategy_gene: StrategyGene,
        strategy_code: str,
        strategy_hash: str,
        generated_name: str,
        segment: RegimeSegment,
    ) -> RegimeEvaluationResult:
        """
        Evaluate strategy on a single regime segment.
        
        Uses caching to avoid re-evaluating same strategy on same segment.
        
        Args:
            strategy_gene: Strategy being evaluated
            strategy_code: Generated Python code
            strategy_hash: Hash of strategy code for caching
            generated_name: Strategy name for logging
            segment: Regime segment to evaluate on
            
        Returns:
            RegimeEvaluationResult with fitness and metrics
        """
        # Check cache
        cache_key = (strategy_hash, segment.segment_id)
        if cache_key in self._segment_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for {generated_name} on segment {segment.segment_id}")
            return self._segment_cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            # Run backtest with segment's timerange
            backtest_result = self._backtest_with_segment(
                strategy_code=strategy_code,
                strategy_name=generated_name,
                segment=segment,
                strategy_max_open_trades=strategy_gene.max_open_trades,
            )
            
            if not backtest_result.success:
                logger.warning(
                    f"Backtest failed for {generated_name} on segment {segment.segment_id}: "
                    f"{backtest_result.error_message}"
                )
                result = RegimeEvaluationResult(
                    segment=segment,
                    fitness=0.0,
                    metrics=self._failed_segment_metrics(
                        segment, backtest_result.error_message
                    ),
                    success=False,
                    error_message=backtest_result.error_message,
                )
            else:
                # Calculate metrics and fitness
                metrics = self.base_evaluator._backtest_result_to_metrics(backtest_result)
                metrics['complexity'] = strategy_gene.calculate_complexity()
                metrics['regime'] = segment.regime.value
                metrics['segment_id'] = segment.segment_id
                metrics['segment_confidence'] = segment.confidence
                
                fitness = self.base_evaluator.calculate_fitness(metrics, strategy_gene)
                
                result = RegimeEvaluationResult(
                    segment=segment,
                    fitness=fitness,
                    metrics=metrics,
                    success=True,
                )
                
                logger.debug(
                    f"Segment {segment.segment_id} ({segment.regime.value}): "
                    f"fitness={fitness:.4f}, trades={metrics['num_trades']}"
                )
        
        except Exception as e:
            logger.error(f"Error evaluating segment {segment.segment_id}: {e}", exc_info=True)
            result = RegimeEvaluationResult(
                segment=segment,
                fitness=0.0,
                metrics=self._failed_segment_metrics(segment, str(e)),
                success=False,
                error_message=str(e),
            )
        
        # Cache result
        self._segment_cache[cache_key] = result
        
        return result

    @staticmethod
    def _failed_segment_metrics(
        segment: RegimeSegment,
        error_message: Optional[str],
    ) -> Dict[str, Any]:
        """Create a complete, conservative metric record for a failed segment."""
        return {
            'profit': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 1.0,
            'win_rate': 0.0,
            'num_trades': 0,
            'regime': segment.regime.value,
            'segment_id': segment.segment_id,
            'segment_confidence': segment.confidence,
            'error': error_message,
        }
    
    def _backtest_with_segment(
        self,
        strategy_code: str,
        strategy_name: str,
        segment: RegimeSegment,
        strategy_max_open_trades: Optional[int] = None,
    ) -> BacktestResult:
        """
        Run backtest restricted to a specific regime segment's timerange.
        
        Args:
            strategy_code: Strategy Python code
            strategy_name: Strategy name
            segment: Regime segment with timerange
            strategy_max_open_trades: Optional max trades override
            
        Returns:
            BacktestResult
        """
        # Temporarily modify backtester config with segment's timerange
        original_timerange = self.base_evaluator.backtester.backtest_config.get('timerange', '')
        self.base_evaluator.backtester.backtest_config['timerange'] = segment.timerange
        
        try:
            result = self.base_evaluator.backtester.backtest_strategy(
                strategy_code,
                strategy_name,
                strategy_max_open_trades=strategy_max_open_trades,
            )
            return result
        finally:
            # Restore original timerange
            self.base_evaluator.backtester.backtest_config['timerange'] = original_timerange
    
    def _aggregate_results(
        self,
        results: List[RegimeEvaluationResult],
        strategy_gene: StrategyGene,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate fitness scores and metrics across segments.
        
        Now uses confidence weighting: segments with higher detection confidence
        contribute more to the final score. This prevents low-confidence segments
        (e.g., mixed-regime periods) from having equal influence.
        
        Args:
            results: List of per-segment evaluation results
            strategy_gene: Strategy being evaluated (for complexity)
            
        Returns:
            Tuple of (aggregated_fitness, aggregated_metrics)
        """
        # Every declared segment contributes an effective score. A failed,
        # invalid, zero-trade, or low-trade segment is evidence about missing
        # robustness and must never disappear from an aggregate.
        weighted_scores: List[Tuple[float, float]] = []
        regime_scores: Dict[str, List[float]] = {}
        use_confidence = self.regime_config.get('confidence_weighting', True)
        
        # Phase 1B: regime specialization settings
        spec_config = self.regime_config.get('regime_specialization', {})
        spec_enabled = spec_config.get('enabled', False)
        specialist_boost = spec_config.get('specialist_boost', 1.5)
        diversity_weight = spec_config.get('diversity_weight', 0.05)
        if (isinstance(specialist_boost, bool)
                or not isinstance(specialist_boost, (int, float))
                or not math.isfinite(float(specialist_boost))
                or float(specialist_boost) <= 0.0):
            raise ValueError("regime_specialization.specialist_boost must be finite and > 0")
        if (isinstance(diversity_weight, bool)
                or not isinstance(diversity_weight, (int, float))
                or not math.isfinite(float(diversity_weight))
                or float(diversity_weight) < 0.0):
            raise ValueError("regime_specialization.diversity_weight must be finite and >= 0")
        specialist_boost = float(specialist_boost)
        diversity_weight = float(diversity_weight)
        
        preferred_regime = getattr(strategy_gene, 'preferred_regime', None)
        regime_mode = getattr(strategy_gene, 'regime_mode', 'generalist')
        
        for result in results:
            regime_type = result.segment.regime.value
            effective_fitness = result.effective_fitness(self.min_segment_trades)

            # Apply regime weight. ``exclusive`` deliberately does not remove
            # non-preferred regimes: the gene currently changes evaluation only,
            # not runtime trading behavior, so all phases remain in scope.
            regime_weight = float(self.regime_weights.get(regime_type, 1.0))

            # Specialist mode may emphasize its target but cannot erase others.
            if (spec_enabled and regime_mode == 'specialist'
                    and preferred_regime is not None
                    and regime_type == preferred_regime):
                regime_weight *= specialist_boost

            # Apply confidence weight: scale from 0.5 (low conf) to 1.0 (high conf).
            # Invalid confidence is conservatively treated as zero confidence.
            if use_confidence:
                try:
                    confidence = float(result.segment.confidence)
                except (TypeError, ValueError, OverflowError):
                    confidence = 0.0
                if not math.isfinite(confidence):
                    confidence = 0.0
                conf_weight = 0.5 + 0.5 * max(0.0, min(1.0, confidence))
            else:
                conf_weight = 1.0

            combined_weight = regime_weight * conf_weight
            weighted_scores.append((effective_fitness, combined_weight))
            regime_scores.setdefault(regime_type, []).append(effective_fitness)
        
        if not weighted_scores:
            logger.warning("No regime segment results, returning fail-closed metrics")
            aggregated_metrics = self._aggregate_metrics(results)
            aggregated_metrics['complexity'] = strategy_gene.calculate_complexity()
            return 0.0, aggregated_metrics
        
        # Calculate aggregated fitness
        fitness_values = [score for score, _ in weighted_scores]
        
        if self.aggregation_method == 'mean':
            # Weighted mean
            total_weight = sum(w for _, w in weighted_scores)
            aggregated_fitness = sum(s * w for s, w in weighted_scores) / total_weight
        
        elif self.aggregation_method == 'min':
            # Worst-case performance
            aggregated_fitness = min(fitness_values)
        
        elif self.aggregation_method == 'harmonic_mean':
            # Harmonic mean is defined here only when every segment is positive.
            # A zero/negative outcome is the worst-case result, not a value to
            # filter out. Positive inputs use the actual weighted harmonic mean.
            if any(score <= 0.0 for score, _ in weighted_scores):
                aggregated_fitness = min(fitness_values)
            else:
                total_weight = sum(weight for _, weight in weighted_scores)
                aggregated_fitness = total_weight / sum(
                    weight / score for score, weight in weighted_scores
                )
        
        elif self.aggregation_method == 'cvar':
            # Conditional Value at Risk: average of worst alpha% outcomes
            sorted_scores = sorted(fitness_values)
            n_worst = max(1, int(len(sorted_scores) * self.cvar_alpha))
            aggregated_fitness = sum(sorted_scores[:n_worst]) / n_worst
        
        else:
            # Fallback to mean
            aggregated_fitness = sum(fitness_values) / len(fitness_values)
        
        # Phase 1B: regime diversity bonus/penalty
        # Low fitness variance across regimes = consistent = bonus
        # High variance = mono-regime specialist = slight penalty for generalists
        regime_fitness_variance = 0.0
        regime_diversity_score = 0.0
        if spec_enabled and len(regime_scores) > 1 and diversity_weight > 0:
            # Compute variance of mean fitness across regime types
            regime_means = [
                sum(scores) / len(scores)
                for scores in regime_scores.values()
                if scores
            ]
            if len(regime_means) > 1:
                mean_of_means = sum(regime_means) / len(regime_means)
                regime_fitness_variance = sum(
                    (m - mean_of_means) ** 2 for m in regime_means
                ) / len(regime_means)
                # Diversity score: 1.0 = perfectly consistent, 0.0 = very inconsistent
                # Use sigmoid-like mapping: variance of 0 → score 1.0, variance of 0.1+ → score ~0
                regime_diversity_score = 1.0 / (1.0 + 10.0 * regime_fitness_variance)

                # Apply as bonus/penalty for generalists only
                # Specialists get no diversity penalty (they're supposed to focus)
                coverage_complete = all(
                    result.outcome_status(self.min_segment_trades) == 'ok'
                    for result in results
                )
                if (regime_mode == 'generalist'
                        and coverage_complete
                        and aggregated_fitness > 0.0):
                    diversity_adjustment = diversity_weight * (regime_diversity_score - 0.5) * 2
                    aggregated_fitness *= (1.0 + diversity_adjustment)
                    logger.debug(
                        f"Regime diversity: variance={regime_fitness_variance:.4f}, "
                        f"score={regime_diversity_score:.4f}, "
                        f"adjustment={diversity_adjustment:+.4f}"
                    )
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(results)
        aggregated_metrics['complexity'] = strategy_gene.calculate_complexity()
        
        # Phase 1B: add regime specialization metadata
        if spec_enabled:
            aggregated_metrics['preferred_regime'] = preferred_regime
            aggregated_metrics['regime_mode'] = regime_mode
            aggregated_metrics['regime_fitness_variance'] = regime_fitness_variance
            aggregated_metrics['regime_diversity_score'] = regime_diversity_score
        
        return aggregated_fitness, aggregated_metrics
    
    def _aggregate_metrics(
        self,
        results: List[RegimeEvaluationResult],
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across all segment results.
        
        Args:
            results: List of segment evaluation results
            
        Returns:
            Aggregated metrics dictionary
        """
        statuses = [
            result.outcome_status(self.min_segment_trades)
            for result in results
        ]
        expected_count = len(results)
        successful_count = sum(1 for result in results if result.success)
        eligible_count = statuses.count('ok')

        # Average metrics over every expected segment. Failed segments therefore
        # contribute conservative zeroes instead of shrinking the denominator.
        aggregated: Dict[str, Any] = {}
        numeric_keys = ['profit', 'sharpe_ratio', 'sortino_ratio', 'profit_factor', 
                        'win_rate']

        for key in numeric_keys:
            values = [self._finite_metric(result.metrics.get(key), 0.0) for result in results]
            aggregated[key] = sum(values) / expected_count if expected_count else 0.0

        aggregated['num_trades'] = sum(result.trade_count() for result in results)

        # Missing/weak evidence cannot look like perfect zero drawdown.
        drawdowns = []
        for result, status in zip(results, statuses):
            if status != 'ok':
                drawdowns.append(1.0)
            else:
                drawdowns.append(
                    max(0.0, self._finite_metric(result.metrics.get('max_drawdown'), 1.0))
                )
        aggregated['max_drawdown'] = max(drawdowns) if drawdowns else 1.0

        outcomes = []
        raw_fitness_values: List[Optional[float]] = []
        effective_fitness_values: List[float] = []
        for result, status in zip(results, statuses):
            try:
                raw_fitness = float(result.fitness)
            except (TypeError, ValueError, OverflowError):
                raw_fitness = None
            if raw_fitness is not None and not math.isfinite(raw_fitness):
                raw_fitness = None
            effective_fitness = result.effective_fitness(self.min_segment_trades)
            raw_fitness_values.append(raw_fitness)
            effective_fitness_values.append(effective_fitness)
            outcomes.append({
                'segment_id': result.segment.segment_id,
                'regime': result.segment.regime.value,
                'status': status,
                'num_trades': result.trade_count(),
                'raw_fitness': raw_fitness,
                'effective_fitness': effective_fitness,
                'success': result.success,
                'error': result.error_message or result.metrics.get('error'),
            })

        aggregated['segment_fitness_values'] = raw_fitness_values
        aggregated['segment_effective_fitness_values'] = effective_fitness_values
        aggregated['segment_outcomes'] = outcomes
        aggregated['expected_segment_count'] = expected_count
        aggregated['successful_segment_count'] = successful_count
        aggregated['failed_segment_count'] = statuses.count('failed')
        aggregated['invalid_fitness_segment_count'] = statuses.count('invalid_fitness')
        aggregated['zero_trade_segment_count'] = statuses.count('zero_trades')
        aggregated['low_trade_segment_count'] = statuses.count('low_trades')
        aggregated['eligible_segment_count'] = eligible_count
        aggregated['segment_success_rate'] = (
            successful_count / expected_count if expected_count else 0.0
        )
        aggregated['segment_evidence_rate'] = (
            eligible_count / expected_count if expected_count else 0.0
        )
        aggregated['segment_coverage_complete'] = (
            expected_count > 0 and eligible_count == expected_count
        )
        # Legacy key retained truthfully: no low-trade segment is skipped now.
        aggregated['skipped_low_trade_segments'] = 0
        aggregated['penalized_low_trade_segments'] = (
            statuses.count('zero_trades') + statuses.count('low_trades')
        )
        
        return aggregated

    @staticmethod
    def _finite_metric(value: Any, default: float) -> float:
        """Coerce a metric to a finite float without propagating NaN/inf."""
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return result if math.isfinite(result) else default
    
    def _get_regime_summary(
        self,
        results: List[RegimeEvaluationResult],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Summarize results by regime type.
        
        Args:
            results: List of segment evaluation results
            
        Returns:
            Dict mapping regime type to summary stats
        """
        summary: Dict[str, Dict[str, Any]] = {}
        
        for result in results:
            regime_type = result.segment.regime.value
            if regime_type not in summary:
                summary[regime_type] = {
                    'fitness_values': [],
                    'raw_fitness_values': [],
                    'eligible_count': 0,
                    'count': 0,
                }
            
            effective_fitness = result.effective_fitness(self.min_segment_trades)
            summary[regime_type]['fitness_values'].append(effective_fitness)
            try:
                raw_fitness = float(result.fitness)
            except (TypeError, ValueError, OverflowError):
                raw_fitness = 0.0
            if not math.isfinite(raw_fitness):
                raw_fitness = 0.0
            summary[regime_type]['raw_fitness_values'].append(raw_fitness)
            if result.outcome_status(self.min_segment_trades) == 'ok':
                summary[regime_type]['eligible_count'] += 1
            summary[regime_type]['count'] += 1
        
        # Calculate averages
        for regime_type, data in summary.items():
            values = data['fitness_values']
            data['avg_fitness'] = sum(values) / len(values) if values else 0.0
            data['min_fitness'] = min(values) if values else 0.0
            data['max_fitness'] = max(values) if values else 0.0
            raw_values = data['raw_fitness_values']
            data['raw_avg_fitness'] = (
                sum(raw_values) / len(raw_values) if raw_values else 0.0
            )
            data['evidence_rate'] = (
                data['eligible_count'] / data['count'] if data['count'] else 0.0
            )
            del data['fitness_values']  # Remove raw values from summary
            del data['raw_fitness_values']
        
        return summary
    
    def evaluate_holdout(
        self,
        strategy_gene: StrategyGene,
        strategy_name: Optional[str] = None,
        auto_unlock: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Convenience method to evaluate on holdout segments.
        
        Should only be called once at the end of evolution for final validation.
        By default, automatically unlocks holdout, evaluates, and re-locks.
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name
            auto_unlock: If True (default), temporarily unlock holdout for evaluation
            
        Returns:
            Tuple of (holdout_fitness, holdout_metrics)
        """
        if not self._holdout_segments:
            raise ValueError("No holdout segments configured")
        
        was_locked = self._holdout_locked
        
        try:
            if auto_unlock and was_locked:
                self.unlock_holdout()
            
            return self.evaluate(strategy_gene, strategy_name, use_holdout=True)
        finally:
            # Re-lock if it was locked before
            if auto_unlock and was_locked:
                self.lock_holdout()
    
    def get_holdout_protection_stats(self) -> Dict[str, Any]:
        """Get holdout protection statistics."""
        return {
            'holdout_locked': self._holdout_locked,
            'holdout_access_attempts': self._holdout_access_attempts,
            'holdout_segments_count': len(self._holdout_segments),
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_size': len(self._segment_cache),
            'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
        }
    
    def clear_cache(self):
        """Clear segment cache."""
        self._segment_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Segment cache cleared")


def create_regime_aware_evaluator(
    config: Dict[str, Any],
    auto_detect: bool = True,
    data_path: Optional[str] = None,
    policy: Optional["DatasetPolicy"] = None,
) -> RegimeAwareEvaluator:
    """
    Factory function to create a RegimeAwareEvaluator with auto-detected segments.
    
    Args:
        config: GA configuration dictionary
        auto_detect: If True, auto-detect regimes from data (legacy behavior)
        data_path: Optional path to data directory (uses config if not provided)
        policy: Optional DatasetPolicy to use for segment creation
        
    Returns:
        Configured RegimeAwareEvaluator
    """
    from pathlib import Path
    from genetic_algorithm.utils.dataset_policy import create_policy_from_config
    
    regime_config = config.get('regime_aware', {})
    
    if not regime_config.get('enabled', False):
        logger.info("Regime-aware evaluation disabled in config")
        # Return evaluator with no segments (will fall back to standard evaluation)
        return RegimeAwareEvaluator(config, segments={})
    
    segments = {}
    
    # Use provided policy, or create from config
    if policy is not None:
        logger.info(f"Using provided policy: {policy.describe()}")
        segments = policy.build_segments(
            config,
            data_path=Path(data_path) if data_path else None,
        )
    elif auto_detect:
        # Use DatasetPolicy for consistent behavior
        detected_policy = create_policy_from_config(config)
        logger.info(f"Using auto-detected policy: {detected_policy.describe()}")
        segments = detected_policy.build_segments(
            config,
            data_path=Path(data_path) if data_path else None,
        )
    
    return RegimeAwareEvaluator(config, segments=segments)


def _auto_detect_segments(
    config: Dict[str, Any],
    data_path: Optional[str] = None,
) -> Dict[str, List[RegimeSegment]]:
    """
    Auto-detect regime segments from historical data.
    
    Args:
        config: Configuration dictionary
        data_path: Optional data directory path
        
    Returns:
        Dict with 'optimization', 'model_selection', 'holdout' segment lists
    """
    from pathlib import Path
    
    regime_config = config.get('regime_aware', {})
    backtest_config = config.get('backtesting', {})
    
    # Get data path
    if data_path:
        datadir = Path(data_path)
    else:
        datadir = Path(backtest_config.get('datadir', 'user_data/data/binance'))
    
    # Get primary pair for regime detection (use first pair or benchmark pair)
    pairs = backtest_config.get('pairs', [])
    benchmark_pair = regime_config.get('benchmark_pair')
    if benchmark_pair is None:  # Handle explicit null in YAML
        benchmark_pair = pairs[0] if pairs else 'BTC/USDT'
    
    # Get timeframe (prefer 1h or 4h for regime detection, even if trading on 5m)
    timeframe = regime_config.get('detection_timeframe', '1h')
    
    # Get timerange
    timerange = backtest_config.get('timerange', '')
    
    logger.info(f"Auto-detecting regimes from {benchmark_pair} {timeframe} in {datadir}")
    
    try:
        # Load data
        df = load_ohlcv_data(
            pair=benchmark_pair,
            timeframe=timeframe,
            datadir=datadir,
            timerange=timerange,
        )
        
        if df.empty:
            logger.warning(f"No data loaded for {benchmark_pair} {timeframe}, no segments created")
            return {}
        
        # Create detector.  The legacy fallback remains readable for schema-v1
        # configs; schema-v2 accepts only the canonical ``method`` key.
        method = regime_config.get(
            'method',
            regime_config.get('detection_method', 'adx_di_hysteresis'),
        )
        detector = RegimeDetector(method=method)
        
        # Classify periods
        period_days = regime_config.get('period_days', 90)
        min_period_days = regime_config.get('min_period_days', 60)
        embargo_days = regime_config.get('embargo_days', 5)
        
        segments = detector.classify_periods(
            df=df,
            period_days=period_days,
            min_period_days=min_period_days,
            embargo_days=embargo_days,
        )
        
        if not segments:
            logger.warning("No segments created from regime detection")
            return {}
        
        # Get balanced segments
        segments_per_regime = regime_config.get('segments_per_regime', 3)
        balanced = detector.get_balanced_segments(
            segments,
            segments_per_regime=segments_per_regime,
        )
        
        # Split into train/holdout
        holdout_ratio = regime_config.get('holdout_ratio', 0.20)
        splits = detector.split_segments_by_role(
            balanced,
            optimization_ratio=1.0 - holdout_ratio,
            model_selection_ratio=0.0,
            holdout_ratio=holdout_ratio,
        )
        
        logger.info(
            f"Auto-detected segments: {len(splits['optimization'])} optimization, "
            f"{len(splits['holdout'])} holdout"
        )
        
        return splits
        
    except Exception as e:
        logger.error(f"Failed to auto-detect segments: {e}", exc_info=True)
        return {}
