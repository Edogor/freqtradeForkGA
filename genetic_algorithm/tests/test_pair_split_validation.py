"""
Tests for Pair-Split Validation Mode

Tests the pair-split overfitting evaluation pipeline:
- FitnessEvaluator config detection when pair_validation.enabled = true/false
- evaluate() routing to evaluate_pair_split()
- Composite fitness blending (weight_train * train + weight_val * val)
- Pair generalization ratio computation
- Zero-trade flagging on either split
- Backward compatibility when pair_validation disabled
- Backtester pairs_override parameter propagation
"""

import copy
import math
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import replace

from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.evaluation.direct_backtester import BacktestResult
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_gene(generation=0, individual_id=0):
    """Create a minimal StrategyGene for testing."""
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30.0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70.0, logic='AND'),
        ],
        stoploss=-0.10,
        timeframe='15m',
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
        max_open_trades=3,
    )


def _base_config(**overrides):
    """Build a minimal config dict for FitnessEvaluator."""
    cfg = {
        'fitness_weights': {
            'profit': 0.3,
            'sharpe_ratio': 0.2,
            'sortino_ratio': 0.1,
            'profit_factor': 0.1,
            'drawdown': 0.1,
            'win_rate': 0.1,
            'trade_frequency': 0.1,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.30,
        },
        'backtesting': {
            'pairs': ['BTC/USDT'],
            'timerange': '20210301-20260301',
        },
        'walk_forward': {'enabled': False},
    }
    cfg.update(overrides)
    return cfg


def _make_backtest_result(success=True, total_profit=10.0, profit_percent=5.0,
                          total_trades=50, sharpe=1.5, max_dd=0.10,
                          win_rate=0.55, sortino=2.0, profit_factor=1.5,
                          no_trades=False, error_message=None):
    """Create a BacktestResult with sensible defaults."""
    return BacktestResult(
        success=success,
        strategy_name='TestStrat',
        total_profit=total_profit,
        profit_percent=profit_percent,
        total_trades=total_trades,
        wins=int(total_trades * win_rate),
        losses=total_trades - int(total_trades * win_rate),
        win_rate=win_rate,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        profit_factor=profit_factor,
        avg_profit=profit_percent / max(total_trades, 1),
        error_message=error_message,
        no_trades=no_trades,
    )


# ═══════════════════════════════════════════════════════════════════
# Config Detection Tests
# ═══════════════════════════════════════════════════════════════════

class TestPairSplitConfigDetection:
    """Tests that FitnessEvaluator correctly detects pair-split config."""

    def test_pair_split_disabled_by_default(self):
        evaluator = FitnessEvaluator(_base_config())
        assert not evaluator.pair_validation_enabled

    def test_pair_split_disabled_explicitly(self):
        cfg = _base_config(pair_validation={'enabled': False})
        evaluator = FitnessEvaluator(cfg)
        assert not evaluator.pair_validation_enabled

    def test_pair_split_enabled(self):
        cfg = _base_config(pair_validation={
            'enabled': True,
            'training_pairs': ['BTC/USDT', 'BNB/USDT'],
            'validation_pairs': ['ETH/USDT'],
            'weight_train': 0.6,
            'weight_val': 0.4,
        })
        evaluator = FitnessEvaluator(cfg)
        assert evaluator.pair_validation_enabled
        assert evaluator.pair_validation_config['training_pairs'] == ['BTC/USDT', 'BNB/USDT']
        assert evaluator.pair_validation_config['validation_pairs'] == ['ETH/USDT']

    def test_pair_split_config_defaults(self):
        cfg = _base_config(pair_validation={'enabled': True})
        evaluator = FitnessEvaluator(cfg)
        pv = evaluator.pair_validation_config
        assert pv.get('weight_train', 0.6) == 0.6
        assert pv.get('weight_val', 0.4) == 0.4


# ═══════════════════════════════════════════════════════════════════
# Routing Tests
# ═══════════════════════════════════════════════════════════════════

class TestEvaluateRouting:
    """Tests that evaluate() routes to the correct method."""

    @patch.object(FitnessEvaluator, '_evaluate_standard', return_value=(0.5, {'profit': 1.0}))
    def test_routes_to_standard_when_disabled(self, mock_std):
        cfg = _base_config(pair_validation={'enabled': False})
        evaluator = FitnessEvaluator(cfg)
        fitness, metrics = evaluator.evaluate(_make_gene())
        mock_std.assert_called_once()

    @patch.object(FitnessEvaluator, 'evaluate_pair_split', return_value=(0.5, {'profit': 1.0}))
    def test_routes_to_pair_split_when_enabled(self, mock_ps):
        cfg = _base_config(pair_validation={
            'enabled': True,
            'training_pairs': ['BTC/USDT'],
            'validation_pairs': ['ETH/USDT'],
        })
        evaluator = FitnessEvaluator(cfg)
        fitness, metrics = evaluator.evaluate(_make_gene())
        mock_ps.assert_called_once()

    @patch.object(FitnessEvaluator, 'evaluate_walk_forward', return_value=(0.5, {'profit': 1.0}))
    def test_walk_forward_takes_priority_over_pair_split(self, mock_wf):
        cfg = _base_config(
            walk_forward={'enabled': True, 'train_days': 60, 'validation_days': 15,
                          'step_days': 15, 'mode': 'rolling'},
            pair_validation={'enabled': True, 'training_pairs': ['BTC/USDT'],
                             'validation_pairs': ['ETH/USDT']},
        )
        evaluator = FitnessEvaluator(cfg)
        evaluator.evaluate(_make_gene())
        mock_wf.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# Composite Fitness Tests
# ═══════════════════════════════════════════════════════════════════

class TestPairSplitFitness:
    """Tests that evaluate_pair_split computes fitness correctly."""

    def _make_evaluator(self, weight_train=0.6, weight_val=0.4, min_val_fitness=0.0):
        cfg = _base_config(pair_validation={
            'enabled': True,
            'training_pairs': ['BTC/USDT', 'BNB/USDT', 'XRP/USDT'],
            'validation_pairs': ['ETH/USDT', 'SOL/USDT'],
            'weight_train': weight_train,
            'weight_val': weight_val,
            'min_val_fitness': min_val_fitness,
        })
        evaluator = FitnessEvaluator(cfg)
        return evaluator

    @patch('genetic_algorithm.evaluation.fitness.StrategyGenerator')
    def test_composite_fitness_blending(self, MockGen):
        """Composite = train_fitness * weight_train + val_fitness * weight_val."""
        evaluator = self._make_evaluator(weight_train=0.6, weight_val=0.4)

        train_result = _make_backtest_result(total_profit=20.0, profit_percent=10.0, total_trades=60, sharpe=2.0)
        val_result = _make_backtest_result(total_profit=5.0, profit_percent=3.0, total_trades=40, sharpe=1.0)

        # Mock the backtester to return different results for different pair sets
        mock_bt = MagicMock()
        call_count = [0]

        def backtest_side_effect(code, name, strategy_max_open_trades=None, pairs_override=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return train_result  # First call = training
            return val_result  # Second call = validation

        mock_bt.backtest_strategy.side_effect = backtest_side_effect
        evaluator.backtester = mock_bt

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_strategy_code.return_value = "class TestStrat: pass"
        evaluator.strategy_generator = mock_gen_instance

        gene = _make_gene()
        composite, metrics = evaluator.evaluate_pair_split(gene)

        # Verify both calls were made with correct pairs
        calls = mock_bt.backtest_strategy.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs.get('pairs_override') == ['BTC/USDT', 'BNB/USDT', 'XRP/USDT']
        assert calls[1].kwargs.get('pairs_override') == ['ETH/USDT', 'SOL/USDT']

        # Verify composite blending
        train_fitness = metrics['train_fitness']
        val_fitness = metrics['val_fitness']
        expected_composite = train_fitness * 0.6 + val_fitness * 0.4
        assert abs(composite - expected_composite) < 1e-6, \
            f"Composite {composite} != expected {expected_composite}"

    @patch('genetic_algorithm.evaluation.fitness.StrategyGenerator')
    def test_generalization_ratio(self, MockGen):
        """gen_ratio = val_fitness / (train_fitness + epsilon)."""
        evaluator = self._make_evaluator()

        # Use same result for both to get gen_ratio near 1.0
        result = _make_backtest_result(total_trades=50, sharpe=1.5)
        mock_bt = MagicMock()
        mock_bt.backtest_strategy.return_value = result
        evaluator.backtester = mock_bt

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_strategy_code.return_value = "class TestStrat: pass"
        evaluator.strategy_generator = mock_gen_instance

        gene = _make_gene()
        _, metrics = evaluator.evaluate_pair_split(gene)

        gen_ratio = metrics['pair_generalization_ratio']
        # Same backtest result → train_fitness == val_fitness → ratio ≈ 1.0
        assert abs(gen_ratio - 1.0) < 0.01, f"Expected gen_ratio ≈ 1.0, got {gen_ratio}"

    @patch('genetic_algorithm.evaluation.fitness.StrategyGenerator')
    def test_zero_trades_flagged(self, MockGen):
        """no_trades should be set when train or val produces zero trades."""
        evaluator = self._make_evaluator()

        train_result = _make_backtest_result(total_trades=50)
        val_result = _make_backtest_result(total_trades=0, no_trades=True)

        mock_bt = MagicMock()
        call_count = [0]

        def side_effect(code, name, strategy_max_open_trades=None, pairs_override=None):
            call_count[0] += 1
            return train_result if call_count[0] == 1 else val_result

        mock_bt.backtest_strategy.side_effect = side_effect
        evaluator.backtester = mock_bt

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_strategy_code.return_value = "class TestStrat: pass"
        evaluator.strategy_generator = mock_gen_instance

        _, metrics = evaluator.evaluate_pair_split(_make_gene())
        assert metrics.get('no_trades') is True

    @patch('genetic_algorithm.evaluation.fitness.StrategyGenerator')
    def test_train_backtest_failure_returns_zero(self, MockGen):
        """If training backtest fails, fitness = 0."""
        evaluator = self._make_evaluator()

        failed_result = _make_backtest_result(success=False, error_message='data missing')
        mock_bt = MagicMock()
        mock_bt.backtest_strategy.return_value = failed_result
        evaluator.backtester = mock_bt

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_strategy_code.return_value = "class TestStrat: pass"
        evaluator.strategy_generator = mock_gen_instance

        fitness, metrics = evaluator.evaluate_pair_split(_make_gene())
        assert fitness == 0.0
        assert 'error' in metrics

    @patch('genetic_algorithm.evaluation.fitness.StrategyGenerator')
    def test_val_backtest_failure_returns_zero(self, MockGen):
        """If validation backtest fails, fitness = 0."""
        evaluator = self._make_evaluator()

        train_result = _make_backtest_result(total_trades=50)
        val_failed = _make_backtest_result(success=False, error_message='data missing')

        mock_bt = MagicMock()
        call_count = [0]

        def side_effect(code, name, strategy_max_open_trades=None, pairs_override=None):
            call_count[0] += 1
            return train_result if call_count[0] == 1 else val_failed

        mock_bt.backtest_strategy.side_effect = side_effect
        evaluator.backtester = mock_bt

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_strategy_code.return_value = "class TestStrat: pass"
        evaluator.strategy_generator = mock_gen_instance

        fitness, metrics = evaluator.evaluate_pair_split(_make_gene())
        assert fitness == 0.0
        assert 'error' in metrics

    @patch('genetic_algorithm.evaluation.fitness.StrategyGenerator')
    def test_equal_weights(self, MockGen):
        """50/50 weighting should average train and val fitness."""
        evaluator = self._make_evaluator(weight_train=0.5, weight_val=0.5)

        train_result = _make_backtest_result(total_profit=20.0, profit_percent=10.0, total_trades=60, sharpe=2.0)
        val_result = _make_backtest_result(total_profit=5.0, profit_percent=3.0, total_trades=40, sharpe=1.0)

        mock_bt = MagicMock()
        call_count = [0]

        def side_effect(code, name, strategy_max_open_trades=None, pairs_override=None):
            call_count[0] += 1
            return train_result if call_count[0] == 1 else val_result

        mock_bt.backtest_strategy.side_effect = side_effect
        evaluator.backtester = mock_bt

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_strategy_code.return_value = "class TestStrat: pass"
        evaluator.strategy_generator = mock_gen_instance

        composite, metrics = evaluator.evaluate_pair_split(_make_gene())
        t = metrics['train_fitness']
        v = metrics['val_fitness']
        expected = (t + v) / 2.0
        assert abs(composite - expected) < 1e-6

    @patch('genetic_algorithm.evaluation.fitness.StrategyGenerator')
    def test_metrics_contain_pair_split_fields(self, MockGen):
        """Verify metrics dict contains all pair-split specific fields."""
        evaluator = self._make_evaluator()

        result = _make_backtest_result(total_trades=50, sharpe=1.5)
        mock_bt = MagicMock()
        mock_bt.backtest_strategy.return_value = result
        evaluator.backtester = mock_bt

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_strategy_code.return_value = "class TestStrat: pass"
        evaluator.strategy_generator = mock_gen_instance

        _, metrics = evaluator.evaluate_pair_split(_make_gene())

        required_keys = [
            'train_fitness', 'val_fitness', 'pair_generalization_ratio',
            'val_profit', 'val_sharpe', 'val_trades', 'val_max_drawdown',
            'val_win_rate', 'training_pairs', 'validation_pairs',
        ]
        for key in required_keys:
            assert key in metrics, f"Missing pair-split metric: {key}"


# ═══════════════════════════════════════════════════════════════════
# Backward Compatibility Tests
# ═══════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Ensure existing behavior is unchanged when pair-split is disabled."""

    @patch('genetic_algorithm.evaluation.fitness.StrategyGenerator')
    def test_standard_path_unaffected(self, MockGen):
        """Standard evaluation should work identically when pair_validation not in config."""
        cfg = _base_config()
        evaluator = FitnessEvaluator(cfg)

        result = _make_backtest_result(total_trades=50, sharpe=1.5)
        mock_bt = MagicMock()
        mock_bt.backtest_strategy.return_value = result
        evaluator.backtester = mock_bt

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_strategy_code.return_value = "class TestStrat: pass"
        evaluator.strategy_generator = mock_gen_instance

        fitness, metrics = evaluator.evaluate(_make_gene())

        # Should not contain pair-split keys
        assert 'train_fitness' not in metrics
        assert 'val_fitness' not in metrics
        assert 'pair_generalization_ratio' not in metrics

        # Should still produce valid fitness
        assert fitness > 0.0
        assert 'num_trades' in metrics


# ═══════════════════════════════════════════════════════════════════
# Config Validator Tests
# ═══════════════════════════════════════════════════════════════════

class TestPairSplitConfigValidator:
    """Tests for pair_validation section in config_validator."""

    def test_valid_pair_validation_config(self):
        from genetic_algorithm.utils.config_validator import validate_ga_config
        config = _base_config(pair_validation={
            'enabled': True,
            'training_pairs': ['BTC/USDT', 'BNB/USDT', 'XRP/USDT'],
            'validation_pairs': ['ETH/USDT', 'SOL/USDT'],
            'weight_train': 0.6,
            'weight_val': 0.4,
        })
        # Should not raise
        errors = validate_ga_config(config)
        # Filter pair-validation specific errors only
        pv_errors = [e for e in (errors or []) if 'pair' in str(e).lower()]
        assert len(pv_errors) == 0, f"Unexpected pair-validation errors: {pv_errors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
