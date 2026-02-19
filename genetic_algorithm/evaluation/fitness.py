"""
Fitness Evaluator

Evaluates the fitness of trading strategies through backtesting
and calculating performance metrics.
"""

import logging
from typing import Tuple, Dict, Any, List, Optional

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester, BacktestResult
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.utils.timerange import (
    create_walk_forward_windows,
    aggregate_validation_scores,
    validate_walk_forward_config,
    get_walk_forward_summary,
    WalkForwardWindow
)

logger = logging.getLogger(__name__)


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
        self.fitness_weights = config.get('fitness_weights', {})
        self.fitness_penalties = config.get('fitness_penalties', {})
        self.backtest_config = config.get('backtesting', {})
        
        # Walk-forward configuration
        self.walk_forward_config = config.get('walk_forward', {})
        self.walk_forward_enabled = self.walk_forward_config.get('enabled', False)
        
        # Initialize direct backtester and strategy generator
        self.backtester = DirectBacktester(config)
        self.strategy_generator = StrategyGenerator(config)
        
        # Initialize walk-forward windows if enabled
        self.walk_forward_windows: List[WalkForwardWindow] = []
        if self.walk_forward_enabled:
            self._initialize_walk_forward()
    
    def _initialize_walk_forward(self):
        """
        Initialize walk-forward windows from configuration.
        """
        try:
            # Validate configuration
            validate_walk_forward_config(self.config)
            
            # Get timerange from backtesting config
            timerange = self.backtest_config.get('timerange', '')
            if not timerange:
                logger.error("Cannot initialize walk-forward: no timerange specified in backtesting config")
                self.walk_forward_enabled = False
                return
            
            # Create walk-forward windows
            wf_config = self.walk_forward_config
            self.walk_forward_windows = create_walk_forward_windows(
                timerange=timerange,
                train_days=wf_config['train_days'],
                validation_days=wf_config['validation_days'],
                step_days=wf_config['step_days'],
                mode=wf_config.get('mode', 'rolling'),
                min_train_trades=wf_config.get('min_train_trades'),
                max_windows=wf_config.get('max_windows')
            )
            
            # Log summary
            summary = get_walk_forward_summary(self.walk_forward_windows)
            logger.info("=" * 80)
            logger.info("Walk-Forward Optimization Enabled")
            logger.info(f"  Number of windows: {summary['num_windows']}")
            logger.info(f"  Average train days: {summary['avg_train_days']:.1f}")
            logger.info(f"  Average validation days: {summary['avg_validate_days']:.1f}")
            logger.info(f"  First train start: {summary['first_train_start']}")
            logger.info(f"  Last validate end: {summary['last_validate_end']}")
            logger.info(f"  Aggregation method: {wf_config.get('aggregation', 'mean')}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to initialize walk-forward: {e}")
            logger.warning("Disabling walk-forward optimization and falling back to standard evaluation")
            self.walk_forward_enabled = False
            self.walk_forward_windows = []
    
    def evaluate(self, strategy_gene: StrategyGene, strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a strategy's fitness through backtesting.
        
        Uses walk-forward optimization if enabled, otherwise uses standard evaluation.
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy (auto-generated if not provided)
            
        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        if self.walk_forward_enabled and self.walk_forward_windows:
            return self._evaluate_walk_forward(strategy_gene, strategy_name)
        else:
            return self._evaluate_standard(strategy_gene, strategy_name)
    
    def _evaluate_standard(self, strategy_gene: StrategyGene, strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a strategy using standard backtesting (no walk-forward).
        
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
            
            # Run backtest
            backtest_result = self.backtester.backtest_strategy(strategy_code, generated_name)
            
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
            
            # Add complexity to metrics
            metrics['complexity'] = strategy_gene.calculate_complexity()
            
            # Calculate fitness (includes complexity penalty)
            fitness = self.calculate_fitness(metrics, strategy_gene)
            
            logger.info(f"Strategy {generated_name}: fitness={fitness:.4f}, "
                       f"profit={metrics['profit']:.2f}%, trades={metrics['num_trades']}, "
                       f"complexity={metrics['complexity']}")
            
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
    
    def _evaluate_walk_forward(self, strategy_gene: StrategyGene, strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a strategy using walk-forward optimization.
        
        For each walk-forward window:
        1. Train on training window (not used for fitness, but for consistency)
        2. Evaluate on validation window
        3. Aggregate validation scores
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy
            
        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        try:
            # Generate strategy code
            strategy_code = self.strategy_generator.generate_strategy_code(strategy_gene)
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            
            validation_scores: List[float] = []
            validation_metrics_list: List[Dict[str, float]] = []
            
            logger.info(f"Evaluating {generated_name} using walk-forward with {len(self.walk_forward_windows)} windows")
            
            # Evaluate on each window
            for i, wf_window in enumerate(self.walk_forward_windows):
                logger.info(f"  [{i+1}/{len(self.walk_forward_windows)}] Window {i+1}: "
                           f"Val={wf_window.validation_window.to_freqtrade_format()}")
                
                # Run backtest on validation window
                val_timerange = wf_window.validation_window.to_freqtrade_format()
                val_result = self.backtester.backtest_strategy(
                    strategy_code,
                    f"{generated_name}_W{i}_Val",
                    timerange=val_timerange
                )
                
                # Check if validation backtest was successful
                if not val_result.success:
                    logger.warning(f"  Window {i+1} validation failed: {val_result.error_message}")
                    validation_scores.append(0.0)
                    continue
                
                # Convert validation result to metrics
                val_metrics = self._backtest_result_to_metrics(val_result)
                val_metrics['complexity'] = strategy_gene.calculate_complexity()
                
                # Calculate fitness for this validation window
                val_fitness = self.calculate_fitness(val_metrics, strategy_gene)
                validation_scores.append(val_fitness)
                validation_metrics_list.append(val_metrics)
                
                logger.debug(f"    Validation fitness: {val_fitness:.4f}, "
                           f"profit: {val_metrics['profit']:.2f}%, "
                           f"trades: {val_metrics['num_trades']}")
            
            # Aggregate validation scores
            aggregation_method = self.walk_forward_config.get('aggregation', 'mean')
            aggregated_fitness = aggregate_validation_scores(validation_scores, method=aggregation_method)
            
            # Aggregate metrics (use mean for all metrics)
            if validation_metrics_list:
                aggregated_metrics = self._aggregate_metrics(validation_metrics_list)
                aggregated_metrics['complexity'] = strategy_gene.calculate_complexity()
                aggregated_metrics['walk_forward_windows'] = len(self.walk_forward_windows)
                aggregated_metrics['validation_scores'] = validation_scores
                aggregated_metrics['aggregation_method'] = aggregation_method
            else:
                aggregated_metrics = {
                    'profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'complexity': strategy_gene.calculate_complexity(),
                    'walk_forward_windows': len(self.walk_forward_windows),
                    'error': 'All validation windows failed'
                }
            
            logger.info(f"Strategy {generated_name}: walk-forward fitness={aggregated_fitness:.4f} "
                       f"({aggregation_method} of {len(validation_scores)} windows), "
                       f"profit={aggregated_metrics.get('profit', 0):.2f}%, "
                       f"trades={aggregated_metrics.get('num_trades', 0)}")
            
            return aggregated_fitness, aggregated_metrics
            
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
                'error': str(e)
            }
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics from multiple validation windows.
        
        Uses arithmetic mean for all metrics.
        
        Args:
            metrics_list: List of metrics dictionaries
            
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics_list:
            return {}
        
        # Get all metric keys from first dict
        metric_keys = [k for k in metrics_list[0].keys() if k not in ['error', 'complexity']]
        
        aggregated = {}
        for key in metric_keys:
            values = [m.get(key, 0) for m in metrics_list if m.get(key) is not None]
            if values:
                aggregated[key] = sum(values) / len(values)
            else:
                aggregated[key] = 0.0
        
        return aggregated
    
    def _backtest_result_to_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """
        Convert BacktestResult to metrics dictionary for fitness calculation.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dictionary of metrics
        """
        return {
            'profit': result.profit_percent,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'num_trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'sortino_ratio': result.sortino_ratio,
        }
    
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
        # Extract and normalize metrics
        profit = metrics.get('profit', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)  # New: downside risk focus
        profit_factor = metrics.get('profit_factor', 0)  # New: win/loss ratio
        drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        trades = metrics.get('num_trades', 0)
        
        # Clamp values to reasonable ranges to avoid extreme outliers
        profit = max(-50, min(profit, 200))  # -50% to +200%
        sharpe = max(-5, min(sharpe, 10))  # -5 to 10
        sortino = max(-5, min(sortino, 12))  # Sortino often higher than Sharpe
        profit_factor = max(0, min(profit_factor, 10))  # 0 to 10
        drawdown = min(drawdown, 1.0)  # 0 to 100%
        win_rate = max(0, min(win_rate, 1.0))  # 0 to 100%
        
        # Normalize to 0-1 range with better scaling
        norm_profit = (profit + 50) / 250  # -50% to +200%
        norm_sharpe = (sharpe + 5) / 15  # -5 to 10
        norm_sortino = (sortino + 5) / 17  # -5 to 12
        norm_profit_factor = min(1.0, profit_factor / 3.0)  # >3.0 is excellent
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
        w = self.fitness_weights
        w_profit = w.get('profit', 0.25)
        w_sharpe = w.get('sharpe_ratio', 0.15)
        w_sortino = w.get('sortino_ratio', 0.15)  # New weight
        w_profit_factor = w.get('profit_factor', 0.10)  # New weight
        w_drawdown = w.get('drawdown', 0.15)
        w_win_rate = w.get('win_rate', 0.10)
        w_trades = w.get('trade_frequency', 0.10)
        
        # Normalize weights to sum to 1.0 (handles missing or extra weights in configs)
        weights_dict = {
            'profit': w_profit,
            'sharpe_ratio': w_sharpe,
            'sortino_ratio': w_sortino,
            'profit_factor': w_profit_factor,
            'drawdown': w_drawdown,
            'win_rate': w_win_rate,
            'trade_frequency': w_trades
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
        
        # Calculate weighted fitness
        fitness = (
            w_profit * norm_profit + 
            w_sharpe * norm_sharpe + 
            w_sortino * norm_sortino +
            w_profit_factor * norm_profit_factor +
            w_drawdown * norm_drawdown + 
            w_win_rate * norm_win_rate + 
            w_trades * norm_trades
        )
        
        # Robustness bonus: reward consistency (good Sortino and profit factor together)
        if sortino > 1.0 and profit_factor > 1.5:
            robustness_bonus = 1.0 + (0.05 * min(sortino, 3.0))  # Up to 15% bonus
            fitness *= robustness_bonus
        
        # Bonus for positive profit (encourage profitable strategies)
        if profit > 0:
            fitness *= 1.1  # 10% bonus for any positive profit
        
        # Extra bonus for significantly profitable strategies
        # Note: This is cumulative with above, so total bonus is 32% (1.1 * 1.2) for >10% profit
        if profit > 10:
            fitness *= 1.2  # Additional 20% bonus (32% total with previous bonus)
        
        # Risk-adjusted excellence bonus: reward exceptional risk-adjusted returns
        if sharpe > 2.0 and drawdown < 0.15:
            fitness *= 1.15  # 15% bonus for excellent risk management
        
        # Apply penalties and return
        penalized_fitness = self._apply_penalties(fitness, metrics, strategy_gene)
        
        # Ensure non-negative
        return max(0, penalized_fitness)
    
    def _normalize_trade_frequency(self, num_trades: int) -> float:
        """
        Normalize trade frequency to 0-1 range.
        
        Prefers 10-50 trades for most strategies. Too few trades = unreliable,
        too many trades = overtrading and high fees.
        """
        if num_trades == 0:
            return 0.0
        elif num_trades < 5:
            # Very few trades - heavily penalized
            return num_trades / 10
        elif 5 <= num_trades < 10:
            # Few trades - some penalty
            return 0.5 + (num_trades - 5) / 10
        elif 10 <= num_trades <= 50:
            # Ideal range - full score
            return 1.0
        elif 50 < num_trades <= 100:
            # Moderate overtrading - slight penalty
            return 1.0 - (num_trades - 50) / 100
        else:
            # Excessive trading - significant penalty
            return max(0.3, 1.0 - (num_trades - 50) / 200)
    
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
        
        num_trades = metrics.get('num_trades', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # Soft penalty for low trade count (gradual instead of harsh)
        min_trades = penalties.get('min_trades', 5)
        if num_trades < min_trades:
            if num_trades == 0:
                fitness *= 0.1  # Very low fitness for no trades
            else:
                # Gradual penalty: 50% at 1 trade, increasing to full at min_trades
                # Formula: 0.5 + (num_trades / min_trades) * 0.5
                # E.g., with min_trades=5: 1 trade=60%, 2=70%, 3=80%, 4=90%, 5+=100%
                trade_penalty = 0.5 + (num_trades / min_trades) * 0.5
                fitness *= trade_penalty
        
        # Penalty for excessive drawdown
        max_dd_threshold = penalties.get('max_drawdown', 0.30)
        if max_drawdown > max_dd_threshold:
            # Progressive penalty: worse drawdown = worse penalty
            dd_excess = max_drawdown - max_dd_threshold
            dd_penalty = max(0.3, 1.0 - dd_excess * 2)
            fitness *= dd_penalty
        
        # Penalty for low win rate (but not too harsh)
        min_win_rate = penalties.get('min_win_rate', 0.30)
        if win_rate < min_win_rate and num_trades >= 5:  # Only penalize if enough trades
            # Gradual penalty for low win rate
            wr_penalty = max(0.6, win_rate / min_win_rate)
            fitness *= wr_penalty
        
        # Complexity penalty: penalize overly complex strategies
        # Applied additively (after multiplicative penalties) to allow fine-tuning
        # Additive approach chosen because:
        # - Complexity is a count (discrete), not a rate
        # - Easier to interpret and tune (linear relationship)
        # - Avoids compound effects with other multiplicative penalties
        if strategy_gene is not None:
            complexity_weight = penalties.get('complexity_weight', 0.01)
            if complexity_weight > 0:
                complexity = strategy_gene.calculate_complexity()
                complexity_penalty = complexity_weight * complexity
                # Subtract penalty from fitness
                fitness = max(0, fitness - complexity_penalty)
                logger.debug(f"Applied complexity penalty: {complexity_penalty:.4f} "
                           f"(complexity={complexity}, weight={complexity_weight})")
        
        return fitness
    

