"""
Fitness Evaluator

Evaluates the fitness of trading strategies through backtesting
and calculating performance metrics. Supports both standard backtesting
and walk-forward optimization for preventing overfitting.
"""

import logging
import hashlib
import math
from collections import OrderedDict
from typing import Tuple, Dict, Any, List, Optional

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester, BacktestResult
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.utils.timerange import (
    create_walk_forward_windows,
    validate_walk_forward_config,
    aggregate_validation_scores,
    parse_timerange,
    format_date,
)

logger = logging.getLogger(__name__)

# Canonical fitness weight keys used by calculate_fitness()
VALID_FITNESS_WEIGHT_KEYS = frozenset({
    'profit', 'sharpe_ratio', 'sortino_ratio', 'profit_factor',
    'drawdown', 'win_rate', 'trade_frequency',
    'monthly_stability', 'cross_pair', 'drawdown_duration', 'consecutive_losses',
})

# Deprecated key aliases from older configs
FITNESS_WEIGHT_KEY_ALIASES = {
    'consistency': 'monthly_stability',
    'exposure_time': 'cross_pair',
}


def resolve_fitness_weight_aliases(weights: dict) -> dict:
    """
    Resolve deprecated key aliases and warn about unknown keys.

    Returns a new dict with aliases replaced by canonical keys.
    """
    resolved = {}
    for key, val in weights.items():
        if key in FITNESS_WEIGHT_KEY_ALIASES:
            canonical = FITNESS_WEIGHT_KEY_ALIASES[key]
            logger.warning(
                "[FITNESS] Deprecated weight key '%s' — use '%s' instead. "
                "Auto-resolved for this run.", key, canonical,
            )
            resolved[canonical] = val
        elif key not in VALID_FITNESS_WEIGHT_KEYS:
            logger.warning(
                "[FITNESS] Unknown fitness_weight key '%s' (ignored). "
                "Valid keys: %s", key, ', '.join(sorted(VALID_FITNESS_WEIGHT_KEYS)),
            )
        else:
            resolved[key] = val
    return resolved


class FitnessEvaluator:
    """
    Evaluates strategy fitness through backtesting.
    
    Responsible for:
    - Running FreqTrade backtests
    - Parsing backtest results
    - Calculating fitness score
    - Computing performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fitness evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fitness_weights = resolve_fitness_weight_aliases(
            config.get('fitness_weights', {})
        )
        self.fitness_penalties = config.get('fitness_penalties', {})
        self.backtest_config = config.get('backtesting', {})
        self.walk_forward_config = config.get('walk_forward', {})
        self.monte_carlo_config = config.get('monte_carlo', {})
        
        # Pair-split validation config (train on some pairs, validate on others)
        self.pair_validation_config = config.get('pair_validation', {})
        self.pair_validation_enabled = self.pair_validation_config.get('enabled', False)
        self.validate_top_n_only = self.pair_validation_config.get('validate_top_n_only', 0)
        
        # Fitness bounds for clamping extreme values
        fitness_bounds = config.get('fitness_bounds', {})
        self.profit_min = fitness_bounds.get('profit_min', -10)
        self.profit_max = fitness_bounds.get('profit_max', 10)
        self.sharpe_min = fitness_bounds.get('sharpe_min', -5)
        self.sharpe_max = fitness_bounds.get('sharpe_max', 10)
        self.sortino_min = fitness_bounds.get('sortino_min', -5)
        self.sortino_max = fitness_bounds.get('sortino_max', 12)
        self.profit_factor_max = fitness_bounds.get('profit_factor_max', 10)
        self.profit_factor_norm = fitness_bounds.get('profit_factor_normalization', 3.0)
        
        # Trade frequency thresholds
        tf_config = config.get('trade_frequency_thresholds', {})
        # Auto-scale ONLY the built-in defaults by number of pairs.
        # If the user explicitly set a value in config, use it as-is
        # (the user already accounts for pair count when setting thresholds).
        num_pairs = len(self.backtest_config.get('pairs', ['BTC/USDT']))
        pair_scale = max(1.0, num_pairs / 1.0)  # 1.0 for single pair baseline
        auto_scale = tf_config.get('auto_scale_by_pairs', True)

        # Timeframe-based scaling: shorter timeframes naturally produce more trades.
        # Reference baseline is 1h; multipliers scale thresholds accordingly.
        _TF_TRADE_MULTIPLIERS = {
            '1m': 60.0, '3m': 20.0, '5m': 12.0, '15m': 4.0, '30m': 2.0,
            '1h': 1.0, '2h': 0.5, '4h': 0.25, '6h': 0.17, '8h': 0.125,
            '12h': 0.083, '1d': 0.042,
        }
        auto_scale_tf = tf_config.get('auto_scale_by_timeframe', True)
        strategy_constraints = config.get('strategy_constraints', {})
        base_timeframes = strategy_constraints.get('timeframes', ['5m'])
        # Use the shortest configured timeframe as reference
        if base_timeframes and auto_scale_tf:
            shortest_tf = min(base_timeframes,
                              key=lambda t: _TF_TRADE_MULTIPLIERS.get(t, 1.0))
            tf_scale = _TF_TRADE_MULTIPLIERS.get(shortest_tf, 1.0)
        else:
            tf_scale = 1.0

        def _tf(key: str, default: float) -> int:
            if key in tf_config:
                raw = tf_config[key]
                if auto_scale:
                    return int(raw * pair_scale)
                return int(raw)
            return int(default * pair_scale * tf_scale)

        self.tf_very_few = _tf('very_few', 5)
        self.tf_few = _tf('few', 10)
        self.tf_ideal_min = _tf('ideal_min', 10)
        self.tf_ideal_max = _tf('ideal_max', 50)
        self.tf_moderate_excess = _tf('moderate_excess', 100)
        self.tf_shape = tf_config.get('shape', 'gaussian')  # 'gaussian' or 'asymmetric'

        if tf_config and not auto_scale:
            logger.info(f"[FITNESS] Trade freq thresholds (no auto-scale): "
                       f"ideal_min={self.tf_ideal_min}, ideal_max={self.tf_ideal_max}")
        elif tf_config:
            logger.info(f"[FITNESS] Trade freq thresholds (auto-scaled ×{pair_scale:.0f} pairs): "
                       f"ideal_min={self.tf_ideal_min}, ideal_max={self.tf_ideal_max}")
        else:
            logger.info(f"[FITNESS] Trade freq thresholds (auto-scaled ×{pair_scale:.0f} pairs, "
                       f"×{tf_scale:.1f} timeframe): "
                       f"ideal_min={self.tf_ideal_min}, ideal_max={self.tf_ideal_max}")
        
        # Trade frequency per month hard penalty
        self.min_trades_per_month = self.fitness_penalties.get('min_trades_per_month', 0)
        self.min_tpm_penalty = self.fitness_penalties.get('min_trades_per_month_penalty', 0.0)
        self.timerange_months = self._calculate_timerange_months(
            self.backtest_config.get('timerange', ''))
        if self.min_trades_per_month > 0:
            logger.info(f"[FITNESS] Trade frequency floor: {self.min_trades_per_month} trades/month "
                       f"(penalty={self.min_tpm_penalty}, timerange={self.timerange_months:.1f} months)")
        
        # Resolve walk-forward preset if specified
        _WF_PRESETS = {
            'scalping':  {'train_days': 30,  'validation_days': 7,  'step_days': 3},
            'intraday':  {'train_days': 60,  'validation_days': 15, 'step_days': 7},
            'swing':     {'train_days': 120, 'validation_days': 30, 'step_days': 15},
        }
        wf_preset = self.walk_forward_config.get('preset')
        if wf_preset and wf_preset in _WF_PRESETS:
            preset_vals = _WF_PRESETS[wf_preset]
            for k, v in preset_vals.items():
                if k not in self.walk_forward_config:
                    self.walk_forward_config[k] = v
            logger.info(f"[WF] Applied preset '{wf_preset}': {preset_vals}")

        # Validate walk-forward config if enabled
        if self.walk_forward_config.get('enabled', False):
            validate_walk_forward_config(self.walk_forward_config)
            logger.info("Walk-forward optimization enabled")
        
        # Initialize direct backtester and strategy generator
        self.backtester = DirectBacktester(config)
        self.strategy_generator = StrategyGenerator(config)
        
        # Walk-forward cache: (strategy_hash, train_timerange) -> BacktestResult
        # Uses OrderedDict for LRU eviction — most-recently-used entries at the end.
        self._wf_cache: OrderedDict[Tuple[str, str], BacktestResult] = OrderedDict()
        self._wf_cache_hits = 0
        self._wf_cache_misses = 0
        self._wf_cache_max_size = self.walk_forward_config.get('cache_max_size', 10000)
        
        # Deflated Sharpe Ratio tracker (anti-overfitting)
        from genetic_algorithm.evaluation.deflated_sharpe import DSRTracker
        self._dsr_tracker = DSRTracker(config)

        # Portfolio diversity: reward strategies uncorrelated with existing elites
        diversity_cfg = config.get('portfolio_diversity', {})
        self._diversity_enabled = diversity_cfg.get('enabled', False)
        self._diversity_max_bonus = diversity_cfg.get('max_bonus', 0.10)
        self._diversity_max_penalty = diversity_cfg.get('max_penalty', 0.10)
        self._diversity_corr_threshold = diversity_cfg.get('correlation_threshold', 0.5)
        self._diversity_references: list = []  # list of monthly-profit vectors

    # ------------------------------------------------------------------
    # Portfolio diversity helpers
    # ------------------------------------------------------------------

    def update_diversity_references(self, monthly_profit_vectors: list) -> None:
        """
        Set reference monthly-profit vectors for diversity scoring.

        Called by the evolution engine after each generation with the elite
        strategies' monthly profit series.

        Args:
            monthly_profit_vectors: list of lists, each inner list is a
                strategy's monthly profit percentages.
        """
        self._diversity_references = [
            v for v in monthly_profit_vectors
            if v and len(v) >= 2
        ]
        if self._diversity_references:
            logger.debug(
                f"[DIVERSITY] Updated references: {len(self._diversity_references)} "
                f"vectors (len {len(self._diversity_references[0])})"
            )

    def _compute_diversity_bonus(self, metrics: Dict[str, Any]) -> float:
        """
        Compute diversity multiplier from monthly profit correlation.

        Returns a multiplier in [1 - max_penalty, 1 + max_bonus].
        Low average correlation → bonus (> 1.0).
        High average correlation → penalty (< 1.0).
        """
        if not self._diversity_enabled or not self._diversity_references:
            return 1.0

        monthly = metrics.get('monthly_profits')
        if not monthly or len(monthly) < 2:
            return 1.0

        # Compute Pearson correlation with each reference
        correlations = []
        for ref in self._diversity_references:
            # Align lengths: use the shorter of the two
            min_len = min(len(monthly), len(ref))
            if min_len < 2:
                continue
            a = monthly[:min_len]
            b = ref[:min_len]
            mean_a = sum(a) / min_len
            mean_b = sum(b) / min_len
            cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(min_len))
            var_a = sum((x - mean_a) ** 2 for x in a)
            var_b = sum((x - mean_b) ** 2 for x in b)
            denom = (var_a * var_b) ** 0.5
            if denom > 1e-12:
                correlations.append(cov / denom)

        if not correlations:
            return 1.0

        avg_corr = sum(correlations) / len(correlations)
        metrics['diversity_avg_correlation'] = avg_corr

        threshold = self._diversity_corr_threshold
        if avg_corr < threshold:
            # Bonus: scale linearly from 0 at threshold to max_bonus at corr = -1
            bonus_frac = (threshold - avg_corr) / (threshold + 1.0)
            return 1.0 + self._diversity_max_bonus * bonus_frac
        else:
            # Penalty: scale linearly from 0 at threshold to max_penalty at corr = 1
            penalty_frac = (avg_corr - threshold) / (1.0 - threshold) if threshold < 1.0 else 0.0
            return 1.0 - self._diversity_max_penalty * penalty_frac

    def evaluate(self, strategy_gene: StrategyGene, strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a strategy's fitness through backtesting.
        
        Routing priority:
        1. Walk-forward optimization (if enabled)
        2. Pair-split validation (if enabled) — train on some pairs, validate on others
        3. Standard single-period backtesting
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy (auto-generated if not provided)
            
        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        # Check if walk-forward is enabled
        if self.walk_forward_config.get('enabled', False):
            return self.evaluate_walk_forward(strategy_gene, strategy_name)
        
        # Check if pair-split validation is enabled
        if self.pair_validation_enabled:
            # When validate_top_n_only > 0, initial evaluation is training-only;
            # validation backtest is deferred to run_deferred_validation()
            skip_val = self.validate_top_n_only > 0
            return self.evaluate_pair_split(strategy_gene, strategy_name, skip_validation=skip_val)
        
        # Standard single-period evaluation
        return self._evaluate_standard(strategy_gene, strategy_name)
    
    def _evaluate_standard(self, strategy_gene: StrategyGene, strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Standard single-period evaluation (original evaluate logic).
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy
            
        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        try:
            # Generate strategy code (strategy name is auto-generated from gene info)
            strategy_code = self.strategy_generator.generate_strategy_code(strategy_gene)
            
            # Use generated name from the gene for consistency
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            
            # Run backtest with strategy-specific max_open_trades
            backtest_result = self.backtester.backtest_strategy(
                strategy_code, 
                generated_name,
                strategy_max_open_trades=strategy_gene.max_open_trades
            )
            
            # Check if backtest was successful
            if not backtest_result.success:
                logger.warning(f"Backtest failed for {generated_name}: {backtest_result.error_message}")
                # Return very low fitness for failed strategies
                return 0.0, {
                    'profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'complexity': strategy_gene.calculate_complexity(),
                    'error': backtest_result.error_message
                }
            
            # Convert backtest result to metrics dictionary
            metrics = self._backtest_result_to_metrics(backtest_result)
            
            # Flag zero-trade results so classification can detect them
            if backtest_result.total_trades == 0:
                metrics['no_trades'] = True
                logger.warning(f"{generated_name}: zero trades — strategy produces no signal")
            
            # Add complexity to metrics
            metrics['complexity'] = strategy_gene.calculate_complexity()
            
            # Calculate fitness (includes complexity penalty)
            fitness = self.calculate_fitness(metrics, strategy_gene)
            
            # Monte Carlo robustness adjustment (optional, config-driven)
            mc_cfg = self.monte_carlo_config
            if mc_cfg.get('enabled', False) and backtest_result.trade_profit_ratios:
                mc_min_fitness = mc_cfg.get('min_fitness_threshold', 0.0)
                if fitness >= mc_min_fitness:
                    try:
                        from genetic_algorithm.evaluation.monte_carlo import run_monte_carlo
                        trades_for_mc = [{'profit_ratio': p} for p in backtest_result.trade_profit_ratios]
                        mc_result = run_monte_carlo(trades_for_mc, mc_cfg)
                        # Apply robustness score as a multiplier: (0.5 + 0.5 * score)
                        # Score of 1.0 → no penalty; score of 0.0 → 50% penalty
                        mc_multiplier = 0.5 + 0.5 * mc_result.robustness_score
                        fitness *= mc_multiplier
                        metrics['mc_robustness'] = mc_result.robustness_score
                        metrics['mc_multiplier'] = mc_multiplier
                        metrics['mc_profit_p5'] = mc_result.profit_p5
                        logger.debug(f"{generated_name}: MC robustness={mc_result.robustness_score:.2%}, "
                                   f"multiplier={mc_multiplier:.3f}")
                    except Exception as e:
                        logger.debug(f"Monte Carlo evaluation failed: {e}")
            
            # Log at debug level - summary is logged by evolution.py
            logger.debug(f"{generated_name}: fitness={fitness:.4f}, profit={metrics['profit']:.2f}%, trades={metrics['num_trades']}")
            
            # Portfolio diversity adjustment (optional, config-driven)
            diversity_mult = self._compute_diversity_bonus(metrics)
            if diversity_mult != 1.0:
                fitness *= diversity_mult
                metrics['diversity_multiplier'] = diversity_mult
                logger.debug(f"{generated_name}: diversity multiplier={diversity_mult:.3f}")

            return fitness, metrics
            
        except Exception as e:
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            logger.error(f"Error evaluating strategy {generated_name}: {e}", exc_info=True)
            # Return zero fitness on error
            return 0.0, {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'complexity': strategy_gene.calculate_complexity(),
                'error': str(e)
            }
    
    def evaluate_pair_split(self, strategy_gene: StrategyGene, strategy_name: str = None, skip_validation: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a strategy using pair-split validation.
        
        Runs the same strategy on training pairs and validation pairs separately,
        then blends the fitnesses with configurable weights. This measures
        overfitting by checking how well a strategy generalizes to unseen pairs
        *without* walk-forward or holdout time splits.
        
        Config:
            pair_validation:
              enabled: true
              training_pairs: ["BTC/USDT", "BNB/USDT", "XRP/USDT"]
              validation_pairs: ["ETH/USDT", "SOL/USDT"]
              weight_train: 0.6
              weight_val: 0.4
              min_val_fitness: 0.0
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy
            
        Returns:
            Tuple of (composite_fitness, metrics_dict)
        """
        pv = self.pair_validation_config
        training_pairs = pv.get('training_pairs', [])
        validation_pairs = pv.get('validation_pairs', [])
        weight_train = pv.get('weight_train', 0.6)
        weight_val = pv.get('weight_val', 0.4)
        min_val_fitness = pv.get('min_val_fitness', 0.0)
        
        # Normalize weights to sum to 1.0 (prevents silent scaling errors)
        _w_total = weight_train + weight_val
        if _w_total > 0 and abs(_w_total - 1.0) > 1e-6:
            logger.debug(f"[PAIR-SPLIT] Normalizing weights: {weight_train}+{weight_val}={_w_total}")
            weight_train /= _w_total
            weight_val /= _w_total
        
        # Always use GAStrategy_ prefix to match the class name in generated code
        generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
        
        try:
            strategy_code = self.strategy_generator.generate_strategy_code(strategy_gene)
            
            # === Training pairs backtest ===
            train_result = self.backtester.backtest_strategy(
                strategy_code, generated_name,
                strategy_max_open_trades=strategy_gene.max_open_trades,
                pairs_override=training_pairs,
            )
            
            if not train_result.success:
                logger.warning(f"[PAIR-SPLIT] {generated_name}: training backtest failed")
                return 0.0, {'profit': 0.0, 'num_trades': 0, 'error': 'train_backtest_failed'}
            
            train_metrics = self._backtest_result_to_metrics(train_result)
            train_metrics['complexity'] = strategy_gene.calculate_complexity()
            train_fitness = self.calculate_fitness(train_metrics, strategy_gene)
            
            # When skip_validation is set, return training-only fitness (discounted)
            # Used by validate_top_n_only to defer validation backtest to top-N only
            if skip_validation:
                discount = weight_train + weight_val * 0.8  # ~0.92 discount
                metrics = {
                    'profit': train_metrics.get('profit', 0.0),
                    'sharpe_ratio': train_metrics.get('sharpe_ratio', 0.0),
                    'sortino_ratio': train_metrics.get('sortino_ratio', 0.0),
                    'max_drawdown': train_metrics.get('max_drawdown', 1.0),
                    'win_rate': train_metrics.get('win_rate', 0.0),
                    'num_trades': train_metrics.get('num_trades', 0),
                    'profit_factor': train_metrics.get('profit_factor', 0.0),
                    'complexity': train_metrics.get('complexity', 0),
                    'train_fitness': train_fitness,
                    'val_fitness': None,
                    'training_only': True,
                    'training_pairs': ','.join(training_pairs),
                    'validation_pairs': ','.join(validation_pairs),
                }
                logger.info(f"[PAIR-SPLIT] {generated_name}: train_only={train_fitness:.4f} "
                           f"(deferred validation)")
                return train_fitness * discount, metrics
            
            # === Validation pairs backtest ===
            val_result = self.backtester.backtest_strategy(
                strategy_code, generated_name,
                strategy_max_open_trades=strategy_gene.max_open_trades,
                pairs_override=validation_pairs,
            )
            
            if not val_result.success:
                logger.warning(f"[PAIR-SPLIT] {generated_name}: validation backtest failed")
                return 0.0, {'profit': 0.0, 'num_trades': 0, 'error': 'val_backtest_failed'}
            
            val_metrics = self._backtest_result_to_metrics(val_result)
            val_metrics['complexity'] = strategy_gene.calculate_complexity()
            val_fitness = self.calculate_fitness(val_metrics, strategy_gene)
            
            # === Composite fitness ===
            composite_fitness = train_fitness * weight_train + val_fitness * weight_val
            
            # Generalization ratio: how well does validation fitness track training
            gen_ratio = val_fitness / (train_fitness + 1e-8)
            
            # Enforce minimum validation fitness — penalise overfitted strategies
            if min_val_fitness > 0 and val_fitness < min_val_fitness:
                penalty_ratio = max(0.0, val_fitness / min_val_fitness)
                logger.warning(
                    f"[PAIR-SPLIT] {generated_name}: val_fitness={val_fitness:.4f} below "
                    f"threshold {min_val_fitness}. Applying penalty ratio {penalty_ratio:.3f}"
                )
                composite_fitness *= penalty_ratio
            
            # Build combined metrics
            metrics = {
                # Use training metrics as the primary display values
                'profit': train_metrics.get('profit', 0.0),
                'sharpe_ratio': train_metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': train_metrics.get('sortino_ratio', 0.0),
                'max_drawdown': train_metrics.get('max_drawdown', 1.0),
                'win_rate': train_metrics.get('win_rate', 0.0),
                'num_trades': train_metrics.get('num_trades', 0),
                'profit_factor': train_metrics.get('profit_factor', 0.0),
                'complexity': train_metrics.get('complexity', 0),
                # Pair-split specific metrics
                'train_fitness': train_fitness,
                'val_fitness': val_fitness,
                'pair_generalization_ratio': gen_ratio,
                'val_profit': val_metrics.get('profit', 0.0),
                'val_sharpe': val_metrics.get('sharpe_ratio', 0.0),
                'val_trades': val_metrics.get('num_trades', 0),
                'val_max_drawdown': val_metrics.get('max_drawdown', 1.0),
                'val_win_rate': val_metrics.get('win_rate', 0.0),
                'training_pairs': ','.join(training_pairs),
                'validation_pairs': ','.join(validation_pairs),
                # Map pair-split validation to holdout-compatible fields so
                # overfit_analysis.classify_overfitting() can classify as
                # SAFE/WARNING/OVERFIT instead of UNKNOWN.
                # Pair-split measures "spatial overfitting" (cross-pair generalization)
                # analogous to holdout's "temporal overfitting".
                'holdout_fitness': val_fitness,
                'holdout_degradation': (
                    (train_fitness - val_fitness) / max(abs(train_fitness), 1e-4)
                    if train_fitness > 1e-8 else 0.0
                ),
                'holdout_profit': val_metrics.get('profit', 0.0),
                'holdout_trades': val_metrics.get('num_trades', 0),
                'holdout_drawdown': val_metrics.get('max_drawdown', 1.0),
                # Also provide train_val_gap for the WF signal path
                'train_val_gap': (
                    (train_fitness - val_fitness) / max(abs(train_fitness), 1e-4)
                    if train_fitness > 1e-4 else 0.0
                ),
            }
            
            # Flag zero-trade results
            if train_result.total_trades == 0 or val_result.total_trades == 0:
                metrics['no_trades'] = True
            
            logger.info(f"[PAIR-SPLIT] {generated_name}: train={train_fitness:.4f} val={val_fitness:.4f} "
                       f"composite={composite_fitness:.4f} gen_ratio={gen_ratio:.2f}")
            
            return composite_fitness, metrics
            
        except Exception as e:
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            logger.error(f"[PAIR-SPLIT] Error evaluating {generated_name}: {e}", exc_info=True)
            return 0.0, {
                'profit': 0.0, 'num_trades': 0,
                'complexity': strategy_gene.calculate_complexity(),
                'error': str(e)
            }
    
    def run_deferred_validation(self, population) -> int:
        """Run validation backtests on the top-N individuals (by training fitness).
        
        Called after the initial evaluation pass when validate_top_n_only > 0.
        Re-evaluates top-N with full pair-split (training + validation).
        Non-top individuals keep their discounted training-only fitness.
        
        Returns:
            Number of individuals that received full validation
        """
        if self.validate_top_n_only <= 0 or not self.pair_validation_enabled:
            return 0
        
        # Collect evaluated individuals with training-only results
        candidates = [
            ind for ind in population
            if ind.evaluated and ind.fitness is not None
            and ind.metrics and ind.metrics.get('training_only', False)
        ]
        
        if not candidates:
            return 0
        
        # Sort by training fitness descending
        candidates.sort(key=lambda x: x.metrics.get('train_fitness', 0), reverse=True)
        top_n = candidates[:self.validate_top_n_only]
        
        validated = 0
        for ind in top_n:
            try:
                fitness, metrics = self.evaluate_pair_split(
                    ind.strategy_gene,
                    skip_validation=False,
                )
                ind.set_fitness(fitness, metrics)
                validated += 1
            except Exception as e:
                logger.warning(f"[PAIR-SPLIT] Deferred validation failed: {e}")
        
        logger.info(f"[PAIR-SPLIT] Deferred validation: {validated}/{len(top_n)} top candidates validated, "
                    f"{len(candidates) - len(top_n)} kept training-only")
        return validated
    
    def _auto_adjust_walk_forward_params(
        self, 
        timerange: str
    ) -> Optional[Dict[str, int]]:
        """
        Auto-adjust walk-forward parameters to fit available data range.
        
        When the available data is shorter than the configured train_days + validation_days,
        this method reduces the parameters proportionally so that at least one window can
        be created.
        
        Args:
            timerange: Effective timerange string (YYYYMMDD-YYYYMMDD)
            
        Returns:
            Adjusted parameters dict with 'train_days', 'validation_days', 'step_days',
            or None if no valid adjustment is possible (data too short).
        """
        from genetic_algorithm.utils.timerange import parse_timerange
        
        start, end = parse_timerange(timerange)
        available_days = (end - start).days
        
        train_days = self.walk_forward_config['train_days']
        validation_days = self.walk_forward_config['validation_days']
        step_days = self.walk_forward_config['step_days']
        required_days = train_days + validation_days
        
        if available_days >= required_days:
            return {
                'train_days': train_days,
                'validation_days': validation_days,
                'step_days': step_days,
            }
        
        # Need to shrink parameters to fit.
        # Keep the train/validation ratio the same, but scale down.
        # Reserve at least 5 days for validation and 7 days for training.
        MIN_TRAIN_DAYS = 7
        MIN_VAL_DAYS = 5
        min_total = MIN_TRAIN_DAYS + MIN_VAL_DAYS
        
        if available_days < min_total:
            logger.warning(
                f"Available data ({available_days} days) is too short for walk-forward "
                f"(minimum {min_total} days needed). Cannot auto-adjust.")
            return None
        
        # Scale proportionally, ensuring both minimums are met
        ratio = train_days / required_days
        adjusted_train = min(
            available_days - MIN_VAL_DAYS,
            max(MIN_TRAIN_DAYS, int(available_days * ratio))
        )
        adjusted_val = max(MIN_VAL_DAYS, available_days - adjusted_train)
        
        # Make sure they actually fit
        if adjusted_train + adjusted_val > available_days:
            adjusted_train = available_days - adjusted_val
        
        if adjusted_train < MIN_TRAIN_DAYS:
            return None
        
        adjusted_step = max(1, adjusted_val)
        
        logger.warning(
            f"⚠️  Walk-forward auto-adjusted: available data is only {available_days} days "
            f"(need {required_days} for configured train={train_days}+val={validation_days}). "
            f"Adjusted to train={adjusted_train}, val={adjusted_val}, step={adjusted_step}.")
        
        return {
            'train_days': adjusted_train,
            'validation_days': adjusted_val,
            'step_days': adjusted_step,
        }
    
    def evaluate_walk_forward(
        self, 
        strategy_gene: StrategyGene, 
        strategy_name: str = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate strategy using walk-forward optimization.
        
        Trains on multiple windows and validates on out-of-sample data.
        Final fitness is based on aggregated validation performance, not training performance.
        
        If the available data is too short for the configured walk-forward parameters,
        the parameters are auto-adjusted. If even that is not possible, it falls back
        to standard single-period evaluation with a warning.
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy
            progress_callback: Optional callback(window_idx, total_windows) for progress tracking
            
        Returns:
            Tuple of (aggregated_validation_fitness, aggregated_metrics)
        """
        try:
            # Generate strategy code once (reused for all windows)
            strategy_code = self.strategy_generator.generate_strategy_code(strategy_gene)
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            
            # Create a hash for caching (using SHA-256 for robustness)
            strategy_hash = hashlib.sha256(strategy_code.encode()).hexdigest()[:16]
            
            # Create walk-forward windows using actual data range
            original_timerange = self.backtest_config.get('timerange', '')
            
            # Detect actual data range to avoid creating windows outside available data
            effective_timerange = self.backtester.get_available_data_range()
            if effective_timerange and effective_timerange != original_timerange:
                logger.info(f"Adjusted timerange from config ({original_timerange}) "
                           f"to effective data range ({effective_timerange})")
            timerange_for_windows = effective_timerange or original_timerange
            
            # Auto-adjust walk-forward parameters if data is too short
            adjusted = self._auto_adjust_walk_forward_params(timerange_for_windows)
            if adjusted is None:
                logger.warning(
                    f"⚠️  Walk-forward optimization disabled for this run: insufficient data. "
                    f"Falling back to standard single-period evaluation. "
                    f"To use walk-forward, download more historical data.")
                return self._evaluate_standard(strategy_gene, strategy_name)
            
            wf_train_days = adjusted['train_days']
            wf_val_days = adjusted['validation_days']
            wf_step_days = adjusted['step_days']
            
            try:
                windows = create_walk_forward_windows(
                    timerange=timerange_for_windows,
                    train_days=wf_train_days,
                    validation_days=wf_val_days,
                    step_days=wf_step_days,
                    mode=self.walk_forward_config.get('mode', 'rolling'),
                    embargo_days=self.walk_forward_config.get('embargo_days', 0),
                    max_windows=self.walk_forward_config.get('max_windows', None)
                )
            except ValueError as e:
                logger.warning(
                    f"⚠️  Walk-forward window creation failed even after auto-adjust "
                    f"(train={wf_train_days}, val={wf_val_days}, step={wf_step_days}, "
                    f"timerange={timerange_for_windows}): {e}. "
                    f"Falling back to standard single-period evaluation.")
                return self._evaluate_standard(strategy_gene, strategy_name)
            
            logger.info(f"Evaluating {generated_name} with {len(windows)} walk-forward windows")
            
            validation_fitness_scores = []
            train_fitness_scores = []  # For comparison/debugging
            all_window_metrics = []
            failed_windows = 0  # Track failed windows separately
            consecutive_zero_trade = 0  # Early exit after repeated zero-trade windows
            max_consecutive_zero = self.walk_forward_config.get('max_consecutive_zero_windows', 3)
            
            for window in windows:
                if progress_callback:
                    progress_callback(window.window_index, len(windows))
                
                # Check cache first
                cache_key = (strategy_hash, window.train_timerange)
                if cache_key in self._wf_cache:
                    self._wf_cache_hits += 1
                    # Promote to end for LRU ordering
                    self._wf_cache.move_to_end(cache_key)
                    train_result = self._wf_cache[cache_key]
                    logger.debug(f"Cache hit for window {window.window_index + 1}/{len(windows)}")
                else:
                    self._wf_cache_misses += 1
                    # Run backtest on training window
                    train_result = self._backtest_with_timerange(
                        strategy_code, 
                        generated_name, 
                        window.train_timerange,
                        strategy_max_open_trades=strategy_gene.max_open_trades
                    )
                    # Cache the training result (LRU eviction if over limit)
                    self._wf_cache[cache_key] = train_result
                    while len(self._wf_cache) > self._wf_cache_max_size:
                        self._wf_cache.popitem(last=False)  # evict oldest
                
                # Skip validation if training backtest completely failed
                if not train_result.success:
                    logger.warning(f"Window {window.window_index + 1}/{len(windows)}: Training backtest failed "
                                 f"({train_result.error_message}). Skipping window.")
                    failed_windows += 1
                    continue
                
                # Adaptive min_train_trades: scale based on window size relative to expected
                base_min_train_trades = self.walk_forward_config.get('min_train_trades', 10)
                expected_train_days = self.walk_forward_config.get('train_days', 90)
                # Calculate actual window days from timerange
                try:
                    w_start, w_end = parse_timerange(window.train_timerange)
                    actual_window_days = (w_end - w_start).days
                except Exception:
                    actual_window_days = expected_train_days
                adaptive_min_trades = max(3, int(base_min_train_trades * (actual_window_days / max(expected_train_days, 1))))
                
                # Partial credit system: instead of binary skip, compute a trade-count
                # confidence factor. Windows with few trades get reduced weight.
                train_trade_credit = 1.0
                if train_result.total_trades == 0:
                    consecutive_zero_trade += 1
                    # Rate-limit zero-trade warnings (only log first 3 per evaluation)
                    if consecutive_zero_trade <= 3:
                        logger.warning(f"Window {window.window_index + 1}/{len(windows)}: Zero training trades. "
                                     f"Skipping window. ({consecutive_zero_trade} consecutive)")
                    failed_windows += 1
                    if consecutive_zero_trade >= max_consecutive_zero:
                        logger.warning(f"Early exit: {consecutive_zero_trade} consecutive zero-trade windows. "
                                     f"Skipping remaining {len(windows) - window.window_index - 1} windows.")
                        break
                    continue
                else:
                    consecutive_zero_trade = 0  # Reset on successful window
                if train_result.total_trades < adaptive_min_trades:
                    # Partial credit: scale from 0.3 (1 trade) to 1.0 (at adaptive_min_trades)
                    train_trade_credit = 0.3 + 0.7 * (train_result.total_trades / adaptive_min_trades)
                    logger.info(f"Window {window.window_index + 1}/{len(windows)}: Low training trades "
                               f"({train_result.total_trades} < {adaptive_min_trades}). "
                               f"Applying partial credit: {train_trade_credit:.2f}")
                
                # Run backtest on validation window (never cached - validation is key metric)
                val_result = self._backtest_with_timerange(
                    strategy_code,
                    generated_name,
                    window.val_timerange,
                    strategy_max_open_trades=strategy_gene.max_open_trades
                )
                
                # Calculate fitness for validation data
                if val_result.success and val_result.total_trades > 0:
                    val_metrics = self._backtest_result_to_metrics(val_result)
                    val_metrics['complexity'] = strategy_gene.calculate_complexity()
                    val_fitness = self.calculate_fitness(val_metrics, strategy_gene)
                    # Apply confidence factor: use the LOWER of training and validation
                    # trade credits so both halves must demonstrate sufficient activity.
                    val_trade_credit = 1.0
                    if val_result.total_trades < adaptive_min_trades:
                        val_trade_credit = 0.3 + 0.7 * (val_result.total_trades / adaptive_min_trades)
                    combined_credit = min(train_trade_credit, val_trade_credit)
                    val_fitness *= combined_credit
                else:
                    val_fitness = 0.0
                    val_metrics = {
                        'profit': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,  # Zero trades = no drawdown (not 100%)
                        'win_rate': 0.0,
                        'num_trades': 0,
                        'complexity': strategy_gene.calculate_complexity()
                    }
                
                validation_fitness_scores.append(val_fitness)
                
                # Calculate training fitness for logging
                if train_result.success:
                    train_metrics = self._backtest_result_to_metrics(train_result)
                    train_fitness = self.calculate_fitness(train_metrics, strategy_gene)
                else:
                    train_fitness = 0.0
                
                train_fitness_scores.append(train_fitness)
                
                # Store metrics for this window
                all_window_metrics.append({
                    'window_index': window.window_index,
                    'train_fitness': train_fitness,
                    'val_fitness': val_fitness,
                    'train_trades': train_result.total_trades,
                    'val_trades': val_result.total_trades,
                    **val_metrics
                })
                
                logger.debug(f"Window {window.window_index + 1}/{len(windows)}: "
                          f"Train fitness={train_fitness:.4f} ({train_result.total_trades} trades), "
                          f"Val fitness={val_fitness:.4f} ({val_result.total_trades} trades)")
            
            # Aggregate validation scores
            aggregation_method = self.walk_forward_config.get('aggregation', 'mean')
            
            # If all windows failed, fall back to standard eval with heavy penalty
            # instead of returning 0.0 — this keeps genetic material alive but strongly disfavored
            if not validation_fitness_scores:
                logger.warning(f"All {len(windows)} walk-forward windows failed for {generated_name}. "
                              f"Falling back to standard eval with overfitting penalty.")
                fallback_fitness, fallback_metrics = self._evaluate_standard(strategy_gene, strategy_name)
                # Apply heavy penalty: strategy couldn't survive walk-forward at all
                wf_fallback_penalty = 0.3
                fallback_fitness *= wf_fallback_penalty
                fallback_metrics['walk_forward'] = True
                fallback_metrics['walk_forward_fallback'] = True
                fallback_metrics['num_windows'] = len(windows)
                fallback_metrics['failed_windows'] = failed_windows
                fallback_metrics['wf_fallback_penalty'] = wf_fallback_penalty
                logger.info(f"Walk-forward fallback for {generated_name}: "
                           f"standard fitness={fallback_fitness / wf_fallback_penalty:.4f} -> "
                           f"penalized={fallback_fitness:.4f} (x{wf_fallback_penalty})")
                return fallback_fitness, fallback_metrics
            
            # For weighted aggregation, auto-generate recency weights (later windows weighted more)
            if aggregation_method == 'weighted' and validation_fitness_scores:
                n = len(validation_fitness_scores)
                # Linear recency weights: [1, 2, 3, ..., n] normalized to sum to 1
                weights = [i / sum(range(1, n + 1)) for i in range(1, n + 1)]
                final_fitness = aggregate_validation_scores(validation_fitness_scores, method=aggregation_method, weights=weights)
            else:
                final_fitness = aggregate_validation_scores(validation_fitness_scores, method=aggregation_method)
            
            # Calculate average metrics across validation windows
            avg_metrics = self._aggregate_window_metrics(all_window_metrics)
            avg_metrics['walk_forward'] = True
            avg_metrics['num_windows'] = len(windows)
            avg_metrics['failed_windows'] = failed_windows
            avg_metrics['successful_windows'] = len(validation_fitness_scores)
            
            # Apply proportional penalty for failed windows
            # If e.g. 2 out of 5 windows failed, penalty = (5-2)/5 = 0.6 multiplier
            if failed_windows > 0:
                total_windows = len(windows)
                success_ratio = len(validation_fitness_scores) / total_windows
                final_fitness *= success_ratio
                logger.info(f"Walk-forward window failure penalty: {failed_windows}/{total_windows} failed, "
                           f"fitness scaled by {success_ratio:.2f}")
            
            avg_metrics['avg_train_fitness'] = sum(train_fitness_scores) / len(train_fitness_scores) if train_fitness_scores else 0.0
            avg_metrics['avg_val_fitness'] = sum(validation_fitness_scores) / len(validation_fitness_scores) if validation_fitness_scores else 0.0
            # Train-val gap: Positive = training better (potential overfit), Negative = validation better (rare but good)
            avg_metrics['train_val_gap'] = avg_metrics['avg_train_fitness'] - avg_metrics['avg_val_fitness']
            
            # Apply train-validation gap penalty to discourage overfitting
            # A strategy that performs much better on training than validation is likely overfit
            gap_penalty_config = self.walk_forward_config.get('gap_penalty', {})
            gap_penalty_enabled = gap_penalty_config.get('enabled', False)
            gap_penalty_threshold = gap_penalty_config.get('threshold', 0.1)
            gap_penalty_max = gap_penalty_config.get('max_penalty', 0.5)
            
            if gap_penalty_enabled and avg_metrics['train_val_gap'] > gap_penalty_threshold:
                excess_gap = avg_metrics['train_val_gap'] - gap_penalty_threshold
                # Progressive penalty: larger gap = harsher penalty, capped at max_penalty
                gap_penalty_factor = max(1.0 - gap_penalty_max, 1.0 - excess_gap * 2.0)
                final_fitness *= gap_penalty_factor
                avg_metrics['gap_penalty_applied'] = 1.0 - gap_penalty_factor
                logger.info(f"Walk-forward gap penalty: gap={avg_metrics['train_val_gap']:.4f}, "
                           f"penalty={1.0 - gap_penalty_factor:.2%} applied to {generated_name}")
            
            # Log summary only
            logger.debug(f"Walk-forward {generated_name}: fitness={final_fitness:.4f}, gap={avg_metrics['train_val_gap']:.4f}")
            
            return final_fitness, avg_metrics
            
        except Exception as e:
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            logger.error(f"Error in walk-forward evaluation for {generated_name}: {e}", exc_info=True)
            return 0.0, {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'complexity': strategy_gene.calculate_complexity(),
                'error': str(e),
                'walk_forward': True
            }
    
    def get_wf_cache_stats(self) -> Dict[str, Any]:
        """Get walk-forward cache statistics."""
        total = self._wf_cache_hits + self._wf_cache_misses
        hit_rate = self._wf_cache_hits / total if total > 0 else 0.0
        return {
            'hits': self._wf_cache_hits,
            'misses': self._wf_cache_misses,
            'total': total,
            'hit_rate': hit_rate,
            'cache_size': len(self._wf_cache),
        }
    
    def log_wf_cache_stats(self):
        """Log walk-forward cache statistics at INFO level."""
        stats = self.get_wf_cache_stats()
        if stats['total'] > 0:
            logger.info(
                f"[WF-CACHE] hits={stats['hits']}, misses={stats['misses']}, "
                f"hit_rate={stats['hit_rate']:.1%}, cache_size={stats['cache_size']}"
            )
    
    def _backtest_with_timerange(
        self, 
        strategy_code: str, 
        strategy_name: str, 
        timerange: str,
        strategy_max_open_trades: Optional[int] = None
    ) -> BacktestResult:
        """
        Run backtest with a specific timerange (helper for walk-forward).
        
        Args:
            strategy_code: Strategy Python code
            strategy_name: Strategy name
            timerange: Timerange string (e.g., '20230101-20230201')
            strategy_max_open_trades: Optional max open trades for this strategy
            
        Returns:
            BacktestResult
        """
        # Temporarily modify backtester config
        original_timerange = self.backtester.backtest_config.get('timerange', '')
        self.backtester.backtest_config['timerange'] = timerange
        
        try:
            result = self.backtester.backtest_strategy(
                strategy_code, 
                strategy_name,
                strategy_max_open_trades=strategy_max_open_trades
            )
            return result
        finally:
            # Restore original timerange
            self.backtester.backtest_config['timerange'] = original_timerange
    
    def _aggregate_window_metrics(self, window_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate metrics across all validation windows.
        
        Args:
            window_metrics: List of metric dictionaries, one per window
            
        Returns:
            Aggregated metrics dictionary
        """
        if not window_metrics:
            return {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'complexity': 0
            }
        
        # Average most metrics
        avg_metrics = {}
        numeric_keys = ['profit', 'sharpe_ratio', 'sortino_ratio', 'win_rate', 'num_trades', 
                       'profit_factor', 'complexity', 'val_trades', 'train_trades',
                       'dsr_penalty', 'dsr']
        
        for key in numeric_keys:
            values = [m.get(key, 0) for m in window_metrics if key in m]
            avg_metrics[key] = sum(values) / len(values) if values else 0.0
        
        # Max drawdown: take the worst (highest) across windows
        drawdowns = [m.get('max_drawdown', 0) for m in window_metrics if 'max_drawdown' in m]
        avg_metrics['max_drawdown'] = max(drawdowns) if drawdowns else 1.0
        
        return avg_metrics
    
    def _backtest_result_to_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """
        Convert BacktestResult to metrics dictionary for fitness calculation.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'profit': result.profit_percent,
            'sharpe_ratio': max(-10.0, min(50.0, result.sharpe_ratio)),  # Clamp to sane display range
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'num_trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'sortino_ratio': max(-10.0, min(50.0, result.sortino_ratio)),  # Clamp to sane display range
            # FreqTrade's ``profit_mean`` is already a decimal ratio.
            'avg_profit': result.avg_profit,
            'avg_duration': result.avg_duration,
            'trades': result.trades or [],
        }
        
        # Include per-pair profits for robustness analysis
        if result.per_pair_profit:
            metrics['per_pair_profit'] = result.per_pair_profit
            # Worst pair profit (most negative)
            metrics['worst_pair_profit'] = min(result.per_pair_profit.values())
            # Pair consistency: std deviation of per-pair profits
            pair_profits = list(result.per_pair_profit.values())
            if len(pair_profits) > 1:
                mean_pp = sum(pair_profits) / len(pair_profits)
                metrics['pair_profit_std'] = (sum((p - mean_pp) ** 2 for p in pair_profits) / len(pair_profits)) ** 0.5
        
        # Include tail-risk metrics. Missing values stay absent instead of
        # becoming a perfect zero-risk observation.
        if result.max_consecutive_losses is not None:
            metrics['max_consecutive_losses'] = result.max_consecutive_losses
        if result.max_drawdown_duration_days is not None:
            metrics['max_drawdown_duration_days'] = result.max_drawdown_duration_days

        # V2 daily-equity evidence is shadow data for now.  It is deliberately
        # namespaced and does not replace the legacy search-score inputs until
        # the policy has passed shadow calibration.
        v2_fields = {
            'v2_equity_method': result.equity_method,
            'v2_daily_net_returns': result.daily_net_returns,
            'v2_equity_curve': result.equity_curve,
            'v2_annualized_net_return': result.annualized_net_return,
            'v2_daily_sharpe_ratio': result.daily_sharpe_ratio,
            'v2_daily_sortino_ratio': result.daily_sortino_ratio,
            'v2_daily_expected_shortfall_5': result.daily_expected_shortfall_5,
            'v2_calmar_ratio': result.calmar_ratio,
            'v2_ulcer_index': result.ulcer_index,
            'v2_time_under_water_ratio': result.time_under_water_ratio,
            'v2_daily_max_drawdown': result.daily_max_drawdown,
            'v2_risk_metric_contract_version': result.risk_metric_contract_version,
            'v2_risk_periods_per_year': result.risk_periods_per_year,
            'v2_annual_risk_free_rate': result.annual_risk_free_rate,
            'v2_periodic_risk_free_rate': result.periodic_risk_free_rate,
            'v2_equity_error': result.equity_error_message,
            'v2_mark_to_market_error': result.mark_to_market_error_message,
        }
        metrics.update({key: value for key, value in v2_fields.items() if value is not None})
        if result.trade_profit_ratios is not None:
            metrics['trade_profit_ratios'] = result.trade_profit_ratios

        # Include monthly profits for stability analysis
        if result.monthly_profits and len(result.monthly_profits) > 1:
            metrics['monthly_profits'] = result.monthly_profits
            if (
                result.monthly_periods
                and len(result.monthly_periods) == len(result.monthly_profits)
            ):
                metrics['monthly_periods'] = result.monthly_periods
            monthly = result.monthly_profits
            mean_monthly = sum(monthly) / len(monthly)
            metrics['monthly_return_std'] = (sum((m - mean_monthly) ** 2 for m in monthly) / len(monthly)) ** 0.5
            # Positive months ratio
            metrics['positive_months_ratio'] = sum(1 for m in monthly if m > 0) / len(monthly)
        
        return metrics
    
    def calculate_fitness(self, metrics: Dict[str, float], strategy_gene: StrategyGene = None) -> float:
        """
        Calculate overall fitness score from metrics.
        
        Uses weighted combination of metrics with penalties and robustness scoring.
        
        Args:
            metrics: Dictionary of performance metrics
            strategy_gene: Optional StrategyGene for complexity penalty calculation
            
        Returns:
            Fitness score (higher is better)
        """
        import math
        
        # Extract and normalize metrics
        profit = metrics.get('profit', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)  # New: downside risk focus
        profit_factor = metrics.get('profit_factor', 0)  # New: win/loss ratio
        drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        trades = metrics.get('num_trades', 0)
        
        # NaN/Inf protection: replace invalid values with minimum bounds
        # Using minimum bounds (not 0) prevents NaN Sharpe from scoring as 0.33
        # after normalization: (0 - (-5)) / 15 = 0.33 is wrong for a failed strategy
        if math.isnan(profit) or math.isinf(profit):
            profit = 0
        if math.isnan(sharpe) or math.isinf(sharpe):
            sharpe = self.sharpe_min
        if math.isnan(sortino) or math.isinf(sortino):
            sortino = self.sortino_min
        if math.isnan(profit_factor) or math.isinf(profit_factor):
            profit_factor = 0
        if math.isnan(drawdown) or math.isinf(drawdown):
            drawdown = 1.0  # Assume worst case
        if math.isnan(win_rate) or math.isinf(win_rate):
            win_rate = 0
        
        # Clamp values to reasonable ranges to avoid extreme outliers
        # Use configurable bounds from self.profit_min, self.profit_max, etc.
        profit = max(self.profit_min, min(profit, self.profit_max))
        sharpe = max(self.sharpe_min, min(sharpe, self.sharpe_max))
        sortino = max(self.sortino_min, min(sortino, self.sortino_max))
        profit_factor = max(0, min(profit_factor, self.profit_factor_max))
        drawdown = min(drawdown, 1.0)  # 0 to 100%
        win_rate = max(0, min(win_rate, 1.0))  # 0 to 100%
        
        # Normalize to 0-1 range with configurable scaling
        profit_range = self.profit_max - self.profit_min
        norm_profit = (profit - self.profit_min) / profit_range if profit_range > 0 else 0
        sharpe_range = self.sharpe_max - self.sharpe_min
        norm_sharpe = (sharpe - self.sharpe_min) / sharpe_range if sharpe_range > 0 else 0
        sortino_range = self.sortino_max - self.sortino_min
        norm_sortino = (sortino - self.sortino_min) / sortino_range if sortino_range > 0 else 0
        norm_profit_factor = min(1.0, profit_factor / self.profit_factor_norm)  # configurable via fitness_bounds.profit_factor_normalization
        norm_drawdown = 1 - drawdown  # Lower drawdown is better
        norm_win_rate = win_rate  # Already 0-1
        norm_trades = self._normalize_trade_frequency(trades)
        
        # Clamp normalized values
        norm_profit = max(0, min(norm_profit, 1))
        norm_sharpe = max(0, min(norm_sharpe, 1))
        norm_sortino = max(0, min(norm_sortino, 1))
        norm_profit_factor = max(0, min(norm_profit_factor, 1))
        norm_drawdown = max(0, min(norm_drawdown, 1))
        
        # Get weights with defaults (adjusted to include new metrics)
        # Profit is heavily weighted to ensure fitness correlates with actual profitability.
        # Benchmark analysis (2026-03-12) showed high-fitness strategies with negative
        # profit when profit weight was 0.22 — raised to 0.35 to fix this misalignment.
        w = self.fitness_weights
        w_profit = w.get('profit', 0.35)
        w_sharpe = w.get('sharpe_ratio', 0.12)
        w_sortino = w.get('sortino_ratio', 0.10)
        w_profit_factor = w.get('profit_factor', 0.10)
        w_drawdown = w.get('drawdown', 0.13)
        w_win_rate = w.get('win_rate', 0.05)
        w_trades = w.get('trade_frequency', 0.07)
        w_stability = w.get('monthly_stability', 0.06)
        w_cross_pair = w.get('cross_pair', 0.06)
        # Optional metrics receive weight only when the resolved policy declares
        # it.  Hidden default weights diluted the configured 1.0 weight sum and
        # rewarded unavailable tail-risk fields in legacy results.
        w_dd_duration = w.get('drawdown_duration', 0.0)
        w_consec_losses = w.get('consecutive_losses', 0.0)
        
        # === Tail-risk scores ===
        # Drawdown duration: 0 days = 1.0, 90+ days = 0.0 (linear)
        dd_duration = metrics.get('max_drawdown_duration_days')
        norm_dd_duration = (
            max(0.0, 1.0 - dd_duration / 90.0)
            if dd_duration is not None else 0.0
        )
        # Consecutive losses: 0 = 1.0, 10+ = 0.0 (linear)
        consec_losses = metrics.get('max_consecutive_losses')
        norm_consec_losses = (
            max(0.0, 1.0 - consec_losses / 10.0)
            if consec_losses is not None else 0.0
        )

        # === Monthly stability score ===
        # Lower monthly return std = higher stability = better
        monthly_return_std = metrics.get('monthly_return_std')
        positive_months = metrics.get('positive_months_ratio')
        if monthly_return_std is None or positive_months is None:
            norm_stability = 0.0
        elif monthly_return_std > 0:
            # Normalize: std of 0 gets 1.0, std of 20+ gets ~0
            norm_stability = max(0, 1.0 - monthly_return_std / 20.0)
            # Bonus for high positive months ratio
            norm_stability = norm_stability * 0.7 + positive_months * 0.3
        else:
            norm_stability = positive_months  # Measured zero std; retain positive-month evidence
        
        # === Cross-pair consistency score ===
        # Penalize strategies that only work on 1-2 pairs
        pair_profit_std = metrics.get('pair_profit_std', 0)
        per_pair_profit = metrics.get('per_pair_profit', {})
        if per_pair_profit and len(per_pair_profit) > 1:
            # Count pairs with positive profit
            positive_pairs = sum(1 for v in per_pair_profit.values() if v > 0)
            pair_consistency_ratio = positive_pairs / len(per_pair_profit)
            # Low std across pairs = consistent = good
            norm_cross_pair = max(0, 1.0 - pair_profit_std / 30.0) * 0.5 + pair_consistency_ratio * 0.5
        else:
            norm_cross_pair = 0.0  # Missing/single-pair evidence must not earn robustness credit
        
        # Normalize weights to sum to 1.0 (handles missing or extra weights in configs)
        weights_dict = {
            'profit': w_profit,
            'sharpe_ratio': w_sharpe,
            'sortino_ratio': w_sortino,
            'profit_factor': w_profit_factor,
            'drawdown': w_drawdown,
            'win_rate': w_win_rate,
            'trade_frequency': w_trades,
            'monthly_stability': w_stability,
            'cross_pair': w_cross_pair,
            'drawdown_duration': w_dd_duration,
            'consecutive_losses': w_consec_losses
        }
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            w_profit = weights_dict['profit'] / total_weight
            w_sharpe = weights_dict['sharpe_ratio'] / total_weight
            w_sortino = weights_dict['sortino_ratio'] / total_weight
            w_profit_factor = weights_dict['profit_factor'] / total_weight
            w_drawdown = weights_dict['drawdown'] / total_weight
            w_win_rate = weights_dict['win_rate'] / total_weight
            w_trades = weights_dict['trade_frequency'] / total_weight
            w_stability = weights_dict['monthly_stability'] / total_weight
            w_cross_pair = weights_dict['cross_pair'] / total_weight
            w_dd_duration = weights_dict['drawdown_duration'] / total_weight
            w_consec_losses = weights_dict['consecutive_losses'] / total_weight
        
        # Calculate weighted fitness
        fitness = (
            w_profit * norm_profit + 
            w_sharpe * norm_sharpe + 
            w_sortino * norm_sortino +
            w_profit_factor * norm_profit_factor +
            w_drawdown * norm_drawdown + 
            w_win_rate * norm_win_rate + 
            w_trades * norm_trades +
            w_stability * norm_stability +
            w_cross_pair * norm_cross_pair +
            w_dd_duration * norm_dd_duration +
            w_consec_losses * norm_consec_losses
        )
        
        # ==================================================================================
        # BONUS STACKING STRATEGY:
        # Smooth sigmoid-based bonuses replace hard step functions to eliminate
        # fitness landscape discontinuities that mislead the GA optimizer.
        # Maximum total bonus: ~1.3x (soft cap via tanh saturation).
        # ==================================================================================
        
        def _sigmoid_bonus(value, threshold, max_bonus, steepness=5.0):
            """Smooth logistic bonus: 0 far below threshold, max_bonus far above."""
            try:
                return max_bonus / (1.0 + math.exp(-steepness * (value - threshold)))
            except OverflowError:
                return 0.0 if value < threshold else max_bonus
        
        total_bonus = 1.0
        
        # Robustness bonus: reward consistency (smooth blend of Sortino + profit factor)
        sortino_bonus = _sigmoid_bonus(sortino, 1.0, 0.10, steepness=3.0)
        pf_bonus = _sigmoid_bonus(profit_factor, 1.5, 0.05, steepness=3.0)
        total_bonus += sortino_bonus * pf_bonus / 0.05  # Scales 0-0.10 when both good
        
        # Profit bonus: smooth ramp centred at break-even (0% profit).
        # Threshold moved from 5.0→0.0 and steepness raised 0.3→0.5 so the
        # gradient is strongest exactly where strategies cross into positive territory.
        profit_bonus = _sigmoid_bonus(profit, 0.0, 0.15, steepness=0.5)
        total_bonus += profit_bonus
        
        # Risk-adjusted excellence: smooth product of Sharpe and low-drawdown sigmoids
        sharpe_factor = _sigmoid_bonus(sharpe, 2.0, 1.0, steepness=2.0)
        dd_factor = _sigmoid_bonus(-drawdown, -0.15, 1.0, steepness=20.0)
        total_bonus += sharpe_factor * dd_factor * 0.10  # Up to 10% when both excellent
        
        # Soft cap via tanh saturation instead of hard min()
        excess = total_bonus - 1.0
        total_bonus = 1.0 + 0.3 * math.tanh(excess / 0.3)  # Saturates near 1.3x

        # Hard floor: negative-profit strategies cannot use bonus amplification.
        # This defunds the ~30% of the population that survives on Sharpe/drawdown
        # metrics alone while producing negative returns — they score at base fitness
        # only, leaving more selection pressure available for profitable strategies.
        if profit < 0:
            total_bonus = min(total_bonus, 1.0)

        fitness *= total_bonus
        
        # ==================================================================================
        # DEFLATED SHARPE RATIO PENALTY
        # Corrects for selection bias (multiple testing) and non-normal return distributions.
        # A low DSR means the observed Sharpe is likely a statistical artifact.
        # ==================================================================================
        dsr_penalty, dsr_info = self._dsr_tracker.compute_penalty(
            observed_sharpe=sharpe,
            n_returns=int(trades),
            skewness=metrics.get('return_skewness', 0.0),
            kurtosis=metrics.get('return_kurtosis', 3.0),
        )
        fitness *= dsr_penalty
        
        # Store DSR info in metrics for downstream reporting
        metrics['dsr'] = dsr_info.get('dsr', float('nan'))
        metrics['dsr_penalty'] = dsr_info.get('dsr_penalty', 1.0)
        
        # Register this evaluation for future DSR calculations
        # Use a structural hash so re-evaluations of the same strategy
        # (e.g. across walk-forward windows) count as ONE trial.
        strategy_hash = None
        if strategy_gene is not None:
            import hashlib, json as _json
            gene_fp = _json.dumps(strategy_gene.to_dict(), sort_keys=True, default=str)
            strategy_hash = hashlib.sha256(gene_fp.encode()).hexdigest()[:16]
        self._dsr_tracker.register_evaluation(strategy_hash=strategy_hash)
        
        # Apply penalties and return
        penalized_fitness = self._apply_penalties(fitness, metrics, strategy_gene)
        
        # Ensure non-negative
        return max(0, penalized_fitness)
    
    @staticmethod
    def _calculate_timerange_months(timerange_str: str) -> float:
        """Parse FreqTrade timerange string (YYYYMMDD-YYYYMMDD) and return duration in months."""
        if not timerange_str or '-' not in timerange_str:
            return 0.0
        try:
            parts = timerange_str.split('-')
            start_str, end_str = parts[0].strip(), parts[1].strip()
            from datetime import datetime
            start = datetime.strptime(start_str, '%Y%m%d')
            end = datetime.strptime(end_str, '%Y%m%d')
            days = (end - start).days
            return days / 30.44  # average days per month
        except (ValueError, IndexError):
            return 0.0

    def _normalize_trade_frequency(self, num_trades: int) -> float:
        """
        Normalize trade frequency to 0-1 range.
        
        Supports two modes:
        - 'gaussian': Symmetric bell curve centred on ideal range (original)
        - 'asymmetric': Strong penalty below ideal_min, gentle falloff above ideal_max.
          Better for experiments that want to encourage high trade counts.
        """
        if num_trades <= 0:
            return 0.0
        
        ideal_mean = (self.tf_ideal_min + self.tf_ideal_max) / 2.0
        sigma = max((self.tf_ideal_max - self.tf_ideal_min) / 2.0, 1.0)
        
        if self.tf_shape == 'asymmetric':
            # Asymmetric: steep penalty below ideal_min, gentle above ideal_max
            if num_trades < self.tf_ideal_min:
                # Below minimum: steep Gaussian penalty (same as symmetric)
                z = (num_trades - self.tf_ideal_min) / sigma
                score = math.exp(-z * z / 2.0)
            elif num_trades <= self.tf_ideal_max:
                # In ideal range: perfect score
                score = 1.0
            else:
                # Above maximum: gentle linear decay, floor at 0.3
                excess = num_trades - self.tf_ideal_max
                score = max(0.3, 1.0 - excess / (self.tf_ideal_max * 3))
        else:
            # Original symmetric Gaussian
            z = (num_trades - ideal_mean) / sigma
            score = math.exp(-z * z / 2.0)
        
        return max(0.15, min(1.0, score))
    
    def _apply_penalties(self, fitness: float, metrics: Dict[str, float], strategy_gene: StrategyGene = None) -> float:
        """
        Apply penalties for constraint violations.
        
        Penalties are applied multiplicatively to reduce fitness for strategies
        that violate important constraints.
        
        Args:
            fitness: Base fitness score
            metrics: Performance metrics
            strategy_gene: Optional StrategyGene for complexity penalty
            
        Returns:
            Fitness with penalties applied
        """
        penalties = self.fitness_penalties
        original_fitness = fitness
        
        num_trades = metrics.get('num_trades', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # Smooth penalty for low trade count using logistic S-curve.
        # Replaces the hard 0.01 cliff at 0 trades with a continuous ramp.
        min_trades = penalties.get('min_trades', 10 * max(1, len(self.backtest_config.get('pairs', ['BTC/USDT']))))
        if num_trades < min_trades:
            if num_trades <= 0:
                fitness *= 0.01  # Zero trades: near-zero fitness
            else:
                # Logistic S-curve: midpoint at min_trades/2, smooth ramp 0→1
                try:
                    trade_penalty = 1.0 / (1.0 + math.exp(-8.0 * (num_trades - min_trades / 2.0) / min_trades))
                except OverflowError:
                    trade_penalty = 1.0 if num_trades >= min_trades else 0.0
                # Floor at 5% to avoid near-zero for strategies with very few trades
                trade_penalty = max(0.05, trade_penalty)
                fitness *= trade_penalty
        
        # Hard penalty for minimum trades per month
        # Unlike the S-curve above, this enforces a strict floor on trade frequency.
        # When min_trades_per_month_penalty=0.0, strategies below threshold get zero fitness.
        if self.min_trades_per_month > 0 and self.timerange_months > 0:
            actual_tpm = num_trades / self.timerange_months
            if actual_tpm < self.min_trades_per_month:
                fitness *= self.min_tpm_penalty  # 0.0 = hard kill
                if self.min_tpm_penalty == 0.0:
                    logger.debug(f"[FITNESS] Hard penalty: {actual_tpm:.1f} trades/month "
                               f"< {self.min_trades_per_month} minimum → fitness=0")
                    return 0.0  # Early return, skip remaining penalties
        
        # Profit factor penalty: penalise strategies below minimum PF (< 1.0 = losing money).
        # Uses a steep sigmoid so PF ≥ min_pf is unaffected, PF → 0 is heavily penalised.
        min_pf = penalties.get('min_profit_factor', 0.0)
        if min_pf > 0:
            pf = metrics.get('profit_factor', 0)
            if pf <= 0:
                fitness *= 0.01  # No winning trades at all → near-zero fitness
            elif pf < min_pf:
                # Linear ramp: 0 at PF=0, 1.0 at PF=min_pf
                pf_penalty = max(0.01, pf / min_pf)
                fitness *= pf_penalty
                logger.debug(f"[FITNESS] profit_factor penalty: PF={pf:.3f} < {min_pf} → x{pf_penalty:.3f}")

        # Smooth penalty for excessive drawdown (sigmoid onset around threshold).
        # Gives a gentle signal even slightly below threshold instead of a hard gate.
        max_dd_threshold = penalties.get('max_drawdown', 0.30)
        try:
            dd_penalty_value = 0.7 / (1.0 + math.exp(-20.0 * (max_drawdown - max_dd_threshold)))
        except OverflowError:
            dd_penalty_value = 0.7 if max_drawdown > max_dd_threshold else 0.0
        dd_penalty = 1.0 - dd_penalty_value  # 1.0 when DD is low, ~0.3 when DD is extreme
        fitness *= dd_penalty
        
        # Smooth win rate penalty (replaces binary gate at num_trades >= 5).
        # Penalty strength scales with trade confidence (more trades = more trust).
        min_win_rate = penalties.get('min_win_rate', 0.30)
        if num_trades >= 2:
            # Confidence factor: full confidence at 10+ trades, partial below
            confidence = min(1.0, (num_trades - 1) / 9.0)
            # Sigmoid penalty centred at min_win_rate
            try:
                wr_raw = 0.4 / (1.0 + math.exp(15.0 * (win_rate - min_win_rate)))
            except OverflowError:
                wr_raw = 0.0 if win_rate > min_win_rate else 0.4
            wr_penalty = 1.0 - wr_raw * confidence  # 0.6–1.0 range, scaled by confidence
            fitness *= wr_penalty
        
        # Complexity penalty: penalize overly complex strategies
        # Applied multiplicatively for consistency with other penalties
        if strategy_gene is not None:
            complexity_weight = penalties.get('complexity_weight', 0.01)
            if complexity_weight > 0:
                complexity = strategy_gene.calculate_complexity()
                # Cap penalty at 30% reduction to avoid crushing low-fitness strategies
                complexity_mult = max(0.7, 1.0 - complexity_weight * complexity)
                fitness *= complexity_mult
                logger.debug(f"Applied complexity penalty: x{complexity_mult:.3f} "
                           f"(complexity={complexity}, weight={complexity_weight})")
        
        # Per-pair robustness penalty: penalize strategies with large losses on any single pair
        # This prevents pair-concentration risk where aggregate profit masks individual pair losses
        worst_pair_profit = metrics.get('worst_pair_profit')
        pair_loss_threshold = penalties.get('pair_loss_threshold', -10.0)  # Max acceptable loss on any pair
        if worst_pair_profit is not None and worst_pair_profit < pair_loss_threshold:
            excess_loss = abs(worst_pair_profit - pair_loss_threshold)
            pair_penalty = max(0.5, 1.0 - excess_loss / 100.0)  # Cap at 50% penalty
            fitness *= pair_penalty
            logger.debug(f"Applied per-pair penalty: worst_pair={worst_pair_profit:.2f}%, "
                        f"penalty={1.0 - pair_penalty:.2%}")
        
        # Unused-indicator penalty: penalize indicators that don't contribute to any condition
        # Unused indicators add noise and computational overhead without improving signal quality
        if strategy_gene is not None:
            unused_penalty_weight = penalties.get('unused_indicator_weight', 0.02)
            if unused_penalty_weight > 0:
                total_indicators = len(strategy_gene.indicators)
                if total_indicators > 0:
                    # Collect all indicator references from conditions
                    used_indicators = set()
                    for cond in strategy_gene.entry_conditions:
                        used_indicators.add(cond.indicator)
                    for cond in strategy_gene.exit_conditions:
                        used_indicators.add(cond.indicator)
                    
                    # Count indicators that are actually used
                    used_count = 0
                    for ind in strategy_gene.indicators:
                        ind_ref = ind.instance_id or ind.type
                        if ind_ref in used_indicators:
                            used_count += 1
                    
                    unused_count = total_indicators - used_count
                    if unused_count > 0:
                        unused_ratio = unused_count / total_indicators
                        # Cap at 20% reduction for consistency with other multiplicative penalties
                        unused_mult = max(0.8, 1.0 - unused_ratio * unused_penalty_weight * total_indicators)
                        fitness *= unused_mult
                        logger.debug(f"Applied unused-indicator penalty: x{unused_mult:.3f} "
                                   f"({unused_count}/{total_indicators} unused)")
        
        # Dead exit condition penalty: penalize strategies where ALL exit
        # conditions use impossible thresholds (e.g. RSI < 0) — forcing exits
        # to rely entirely on ROI/stoploss, which overfits to training data.
        if strategy_gene is not None and strategy_gene.exit_conditions:
            _BOUNDED = {'RSI': (0, 100), 'STOCH': (0, 100), 'CCI': (-300, 300),
                        'CMF': (-1, 1), 'ADX': (0, 100)}
            dead_count = 0
            bounded_count = 0
            for cond in strategy_gene.exit_conditions:
                base_type = cond.indicator.split('_')[0] if '_' in cond.indicator else cond.indicator
                bounds = _BOUNDED.get(base_type.upper())
                if bounds:
                    bounded_count += 1
                    lo, hi = bounds
                    # Condition is "dead" if threshold is outside the indicator's range
                    if cond.operator in ('<', 'less_than') and cond.threshold <= lo:
                        dead_count += 1
                    elif cond.operator in ('>', 'greater_than') and cond.threshold >= hi:
                        dead_count += 1
            if bounded_count > 0 and dead_count > 0:
                # Proportional penalty based on ratio of dead conditions
                dead_ratio = dead_count / bounded_count
                dead_penalty = 1.0 - dead_ratio * 0.3  # 0.7 when all dead, 1.0 when none
                fitness *= dead_penalty
                logger.debug(f"Applied dead-exit penalty: {dead_count}/{bounded_count} bounded exit "
                           f"conditions use impossible thresholds (fitness x{dead_penalty:.3f})")
        
        # ── Trade duration penalty ──────────────────────────────────────────
        # Penalize strategies that hold positions too long for their timeframe.
        # Scalping strategies on 5m should exit within ~12 candles, not hold for hours.
        max_avg_candles = penalties.get('max_avg_trade_candles', 0)
        duration_penalty_weight = penalties.get('trade_duration_penalty_weight', 0.3)
        if max_avg_candles > 0 and strategy_gene is not None:
            avg_duration_str = metrics.get('avg_duration', '')
            if avg_duration_str:
                # Parse duration string (e.g. "2:30:00" or "0 days 02:30:00" or minutes float)
                _dur_minutes = self._parse_duration_minutes(avg_duration_str)
                if _dur_minutes is not None and _dur_minutes > 0:
                    from genetic_algorithm.core.strategy_gene import timeframe_to_minutes
                    candle_min = timeframe_to_minutes(strategy_gene.timeframe) or 60
                    avg_candles = _dur_minutes / candle_min
                    if avg_candles > max_avg_candles:
                        excess = (avg_candles - max_avg_candles) / max_avg_candles
                        try:
                            dur_penalty = max(1.0 - duration_penalty_weight,
                                              1.0 / (1.0 + math.exp(3.0 * excess)))
                        except OverflowError:
                            dur_penalty = 1.0 - duration_penalty_weight
                        fitness *= dur_penalty
                        logger.debug(f"Applied trade duration penalty: avg={avg_candles:.1f} candles "
                                   f"(max={max_avg_candles}), penalty x{dur_penalty:.3f}")

        # ── Trade clustering penalty ──────────────────────────────────────────
        # Penalize strategies that spam-open multiple trades within a narrow window.
        # This prevents the GA from evolving "always enter" strategies.
        clustering_enabled = penalties.get('trade_clustering_enabled', False)
        if clustering_enabled and num_trades > 10:
            max_per_window = penalties.get('trade_clustering_max_per_window', 3)
            window_candles = penalties.get('trade_clustering_window_candles', 5)
            trades_list = metrics.get('trades', [])
            if trades_list and len(trades_list) > 10:
                cluster_ratio = self._compute_cluster_ratio(
                    trades_list, max_per_window, window_candles,
                    strategy_gene.timeframe if strategy_gene else '1h')
                if cluster_ratio > 0.3:
                    cluster_penalty = max(0.7, 1.0 - cluster_ratio * 0.3)
                    fitness *= cluster_penalty
                    logger.debug(f"Applied trade clustering penalty: ratio={cluster_ratio:.2f}, "
                               f"penalty x{cluster_penalty:.3f}")

        # ── Spread-aware fitness penalty ──────────────────────────────────────
        # Penalize strategies whose average profit per trade is barely above costs.
        # If the edge is thinner than 2× round-trip costs, it won't survive live.
        spread_aware = penalties.get('spread_aware_enabled', False)
        if spread_aware and num_trades > 0:
            fee = self.backtest_config.get('fee', 0.001)
            slippage = self.backtest_config.get('slippage_pct', 0.0)
            round_trip_cost = (fee + slippage) * 2  # entry + exit
            # ``avg_profit`` is the decimal FreqTrade ``profit_mean`` value.
            # Dividing it by 100 again made this penalty effectively trigger on
            # almost every positive strategy.
            avg_profit_ratio = metrics.get('avg_profit', 0.0)
            min_edge = penalties.get('spread_aware_min_edge_multiplier', 2.0)
            if avg_profit_ratio > 0 and avg_profit_ratio < round_trip_cost * min_edge:
                edge_ratio = avg_profit_ratio / (round_trip_cost * min_edge) if round_trip_cost > 0 else 1.0
                spread_penalty = max(0.5, edge_ratio)
                fitness *= spread_penalty
                logger.debug(f"Applied spread-aware penalty: avg_profit={avg_profit_ratio:.4f}, "
                           f"min_edge={round_trip_cost * min_edge:.4f}, penalty x{spread_penalty:.3f}")

        # Combined penalty floor: prevent penalty compounding from destroying
        # viable strategies. With N multiplicative penalties, the product can
        # approach zero even for decent strategies. Floor at 10% of original.
        min_penalized = penalties.get('min_penalty_floor', 0.10)
        if min_penalized > 0 and original_fitness > 0:
            fitness = max(fitness, original_fitness * min_penalized)
        
        return fitness

    @staticmethod
    def _parse_duration_minutes(duration_str) -> float:
        """Parse a duration string into minutes.

        Handles formats like:
        - ``"2:30:00"`` (H:M:S)
        - ``"0 days 02:30:00"`` (pandas Timedelta str)
        - numeric (already minutes)
        Returns None on parse failure.
        """
        if isinstance(duration_str, (int, float)):
            return float(duration_str)
        if not isinstance(duration_str, str) or not duration_str.strip():
            return None
        try:
            s = duration_str.strip()
            # Strip "X days " prefix
            if 'days' in s or 'day' in s:
                parts = s.split(' ', 2)
                days = int(parts[0])
                s = parts[-1] if len(parts) > 2 else '0:00:00'
                day_minutes = days * 1440
            else:
                day_minutes = 0
            # Parse H:M:S
            hms = s.split(':')
            h = int(hms[0]) if len(hms) > 0 else 0
            m = int(hms[1]) if len(hms) > 1 else 0
            sec = int(float(hms[2])) if len(hms) > 2 else 0
            return day_minutes + h * 60 + m + sec / 60.0
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _compute_cluster_ratio(trades_list, max_per_window: int,
                               window_candles: int, timeframe: str) -> float:
        """Compute fraction of trades that are part of dense clusters.

        A cluster is a window of ``window_candles`` candles where more than
        ``max_per_window`` trades were opened.  Returns ratio in [0, 1].
        """
        if len(trades_list) < 5:
            return 0.0
        try:
            from genetic_algorithm.core.strategy_gene import timeframe_to_minutes
            candle_min = timeframe_to_minutes(timeframe) or 60
            window_minutes = candle_min * window_candles

            # Extract open timestamps — trade dicts may have different key names
            stamps = []
            for t in trades_list:
                ts = t.get('open_date') or t.get('open_timestamp')
                if ts is None:
                    continue
                if isinstance(ts, str):
                    from datetime import datetime
                    try:
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        continue
                stamps.append(ts)

            if len(stamps) < 5:
                return 0.0

            stamps.sort()
            clustered = 0
            for i, ts in enumerate(stamps):
                count = 0
                for j in range(i, len(stamps)):
                    diff = stamps[j] - ts
                    diff_min = diff.total_seconds() / 60.0 if hasattr(diff, 'total_seconds') else float(diff)
                    if diff_min <= window_minutes:
                        count += 1
                    else:
                        break
                if count > max_per_window:
                    clustered += 1
            return clustered / len(stamps)
        except Exception:
            return 0.0
    
    def evaluate_holdout(self, strategy_gene: StrategyGene, holdout_timerange: str,
                         strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a strategy on a completely unseen holdout period.
        
        This method is designed to be called ONLY ONCE after evolution is complete,
        on the final top-N strategies. The holdout period should never be seen during
        evolution to provide a true out-of-sample performance estimate.
        
        Args:
            strategy_gene: Strategy to evaluate
            holdout_timerange: Timerange string for holdout period (YYYYMMDD-YYYYMMDD)
            strategy_name: Optional name for the strategy
            
        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        try:
            strategy_code = self.strategy_generator.generate_strategy_code(strategy_gene)
            generated_name = strategy_name or f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            
            logger.info(f"[HOLDOUT] Evaluating {generated_name} on holdout period: {holdout_timerange}")
            
            result = self._backtest_with_timerange(
                strategy_code, generated_name, holdout_timerange,
                strategy_max_open_trades=strategy_gene.max_open_trades
            )
            
            if not result.success or result.total_trades == 0:
                logger.warning(f"[HOLDOUT] {generated_name}: backtest failed or zero trades")
                return 0.0, {
                    'profit': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 1.0,
                    'win_rate': 0.0, 'num_trades': 0, 'holdout': True,
                    'error': result.error_message if not result.success else 'zero trades'
                }
            
            metrics = self._backtest_result_to_metrics(result)
            metrics['complexity'] = strategy_gene.calculate_complexity()
            metrics['holdout'] = True
            fitness = self.calculate_fitness(metrics, strategy_gene)
            
            logger.info(f"[HOLDOUT] {generated_name}: fitness={fitness:.4f}, "
                       f"profit={metrics['profit']:.2f}%, trades={metrics['num_trades']}")
            
            return fitness, metrics
            
        except Exception as e:
            logger.error(f"[HOLDOUT] Error evaluating {strategy_name}: {e}", exc_info=True)
            return 0.0, {'profit': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 1.0,
                        'win_rate': 0.0, 'num_trades': 0, 'holdout': True, 'error': str(e)}
    
    @staticmethod
    def split_timerange_for_holdout(timerange: str, holdout_pct: float = 0.15) -> Tuple[str, str]:
        """
        Split a timerange into evolution and holdout periods.
        
        The holdout period is taken from the END of the timerange (most recent data),
        since we want to validate forward generalization.
        
        Args:
            timerange: Full timerange string (YYYYMMDD-YYYYMMDD)
            holdout_pct: Fraction of data to reserve as holdout (default: 15%)
            
        Returns:
            Tuple of (evolution_timerange, holdout_timerange)
        """
        from datetime import timedelta
        start, end = parse_timerange(timerange)
        total_days = (end - start).days
        holdout_days = max(7, int(total_days * holdout_pct))  # minimum 7 days
        
        split_date = end - timedelta(days=holdout_days)
        
        evolution_tr = f"{format_date(start)}-{format_date(split_date)}"
        holdout_tr = f"{format_date(split_date)}-{format_date(end)}"
        
        logger.info(f"[HOLDOUT] Split timerange: evolution={evolution_tr} ({total_days - holdout_days}d), "
                    f"holdout={holdout_tr} ({holdout_days}d)")
        
        return evolution_tr, holdout_tr
