"""
Tests for LLM Subsystem — Parser, Diagnostics, and Router

Mock-based tests that don't require API keys or network access.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from genetic_algorithm.llm.diagnostics import (
    diagnose_failure_mode,
    diagnose_all_failure_modes,
    select_mutation_objective,
)
from genetic_algorithm.llm.parser import StrategyParser
from genetic_algorithm.llm.router import LLMProviderRouter, create_provider_or_router


# =============================================================================
# DIAGNOSTICS TESTS
# =============================================================================


class TestDiagnoseFailureMode:

    def test_zero_trades(self):
        metrics = {'num_trades': 0}
        result = diagnose_failure_mode(metrics)
        assert result is not None
        assert "ZERO TRADES" in result

    def test_too_few_trades(self):
        metrics = {'num_trades': 3}
        result = diagnose_failure_mode(metrics)
        assert result is not None
        assert "TOO FEW TRADES" in result

    def test_excessive_drawdown(self):
        metrics = {'num_trades': 20, 'max_drawdown': 0.35, 'win_rate': 60, 'sharpe_ratio': 1.5}
        result = diagnose_failure_mode(metrics)
        assert result is not None
        assert "DRAWDOWN" in result

    def test_low_win_rate(self):
        metrics = {'num_trades': 20, 'max_drawdown': 0.10, 'win_rate': 15, 'sharpe_ratio': 1.0}
        result = diagnose_failure_mode(metrics)
        assert result is not None
        assert "WIN RATE" in result

    def test_poor_sharpe(self):
        metrics = {'num_trades': 20, 'max_drawdown': 0.10, 'win_rate': 60, 'sharpe_ratio': 0.1}
        result = diagnose_failure_mode(metrics)
        assert result is not None
        assert "RISK-ADJUSTED" in result

    def test_overfitting(self):
        metrics = {
            'num_trades': 20, 'max_drawdown': 0.05, 'win_rate': 70,
            'sharpe_ratio': 2.0, 'wf_gap': 0.30,
        }
        result = diagnose_failure_mode(metrics)
        assert result is not None
        assert "OVERFITTING" in result

    def test_excessive_complexity(self):
        metrics = {
            'num_trades': 20, 'max_drawdown': 0.05, 'win_rate': 70,
            'sharpe_ratio': 2.0, 'indicator_count': 6, 'condition_count': 5,
        }
        result = diagnose_failure_mode(metrics)
        assert result is not None
        assert "COMPLEXITY" in result

    def test_large_loss(self):
        metrics = {
            'num_trades': 20, 'max_drawdown': 0.10, 'win_rate': 60,
            'sharpe_ratio': 1.0, 'profit': -15.0,
        }
        result = diagnose_failure_mode(metrics)
        assert result is not None
        assert "LOSS" in result

    def test_healthy_strategy_returns_none(self):
        metrics = {
            'num_trades': 50, 'max_drawdown': 0.08, 'win_rate': 65,
            'sharpe_ratio': 1.8, 'profit': 25.0,
        }
        result = diagnose_failure_mode(metrics)
        assert result is None

    def test_custom_thresholds_from_config(self):
        """Config overrides should adjust failure thresholds."""
        metrics = {'num_trades': 8}
        config = {'fitness': {'penalties': {'min_trades': 10}}}
        result = diagnose_failure_mode(metrics, config)
        assert result is not None
        assert "TOO FEW" in result

    def test_priority_zero_trades_before_drawdown(self):
        metrics = {'num_trades': 0, 'max_drawdown': 0.50}
        result = diagnose_failure_mode(metrics)
        assert "ZERO TRADES" in result  # most critical wins


class TestDiagnoseAllFailureModes:

    def test_multiple_failures(self):
        metrics = {
            'num_trades': 3, 'max_drawdown': 0.30, 'win_rate': 20,
            'sharpe_ratio': 0.1, 'profit': -10.0,
        }
        failures = diagnose_all_failure_modes(metrics)
        assert len(failures) > 3
        assert "too_few_trades" in failures
        assert "excessive_drawdown" in failures
        assert "low_win_rate" in failures

    def test_no_failures(self):
        metrics = {
            'num_trades': 50, 'max_drawdown': 0.08, 'win_rate': 65,
            'sharpe_ratio': 1.8, 'profit': 25.0,
        }
        assert diagnose_all_failure_modes(metrics) == []


class TestSelectMutationObjective:

    def test_zero_trades(self):
        metrics = {'num_trades': 0}
        assert select_mutation_objective(metrics) == "increase_trades"

    def test_drawdown(self):
        metrics = {'num_trades': 20, 'max_drawdown': 0.35, 'win_rate': 60, 'sharpe_ratio': 1.5}
        assert select_mutation_objective(metrics) == "reduce_drawdown"

    def test_low_win_rate(self):
        metrics = {'num_trades': 20, 'max_drawdown': 0.10, 'win_rate': 15, 'sharpe_ratio': 1.0}
        assert select_mutation_objective(metrics) == "improve_entries"

    def test_poor_sharpe(self):
        metrics = {'num_trades': 20, 'max_drawdown': 0.10, 'win_rate': 60, 'sharpe_ratio': 0.2}
        assert select_mutation_objective(metrics) == "improve_risk_adjusted"

    def test_complexity(self):
        metrics = {
            'num_trades': 20, 'max_drawdown': 0.05, 'win_rate': 70,
            'sharpe_ratio': 2.0, 'indicator_count': 6, 'condition_count': 5,
        }
        assert select_mutation_objective(metrics) == "simplify"

    def test_healthy_returns_general(self):
        metrics = {
            'num_trades': 50, 'max_drawdown': 0.08, 'win_rate': 65,
            'sharpe_ratio': 1.8, 'profit': 25.0,
        }
        assert select_mutation_objective(metrics) == "general_improvement"


# =============================================================================
# PARSER TESTS
# =============================================================================


class TestStrategyParser:

    @pytest.fixture
    def parser(self):
        config = {
            'indicators': {
                'available': ['RSI', 'EMA', 'SMA', 'MACD', 'BBANDS', 'ATR', 'ADX'],
                'min_entry_conditions': 2,
                'min_exit_conditions': 1,
            },
            'strategy_constraints': {
                'stoploss_range': [-0.20, -0.05],
                'timeframes': ['5m', '15m', '1h'],
            },
        }
        return StrategyParser(config)

    def test_valid_json_parses(self, parser):
        data = {
            'indicators': [
                {'type': 'RSI', 'parameters': {'period': 14}},
                {'type': 'EMA', 'parameters': {'period': 20}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
                {'indicator': 'EMA_0', 'operator': '>', 'threshold': 0},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70},
            ],
            'timeframe': '15m',
            'stoploss': -0.10,
        }
        gene = parser.parse(data, generation=1, individual_id=1)
        assert gene is not None
        assert len(gene.indicators) == 2
        assert gene.timeframe == '15m'

    def test_unknown_indicator_removed(self, parser):
        data = {
            'indicators': [
                {'type': 'RSI', 'parameters': {'period': 14}},
                {'type': 'FAKE_INDICATOR', 'parameters': {}},
                {'type': 'EMA', 'parameters': {'period': 20}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
                {'indicator': 'EMA_0', 'operator': '>', 'threshold': 0},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70},
            ],
        }
        gene = parser.parse(data, generation=1, individual_id=1)
        assert gene is not None
        indicator_types = [ind.type for ind in gene.indicators]
        assert 'FAKE_INDICATOR' not in indicator_types

    def test_stoploss_clamped(self, parser):
        data = {
            'indicators': [
                {'type': 'RSI', 'parameters': {'period': 14}},
                {'type': 'EMA', 'parameters': {'period': 20}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
                {'indicator': 'EMA_0', 'operator': '>', 'threshold': 0},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70},
            ],
            'stoploss': -0.50,  # too aggressive
        }
        gene = parser.parse(data, generation=1, individual_id=1)
        assert gene is not None
        assert gene.stoploss >= -0.20
        assert gene.stoploss <= -0.05

    def test_invalid_timeframe_replaced(self, parser):
        data = {
            'indicators': [
                {'type': 'RSI', 'parameters': {'period': 14}},
                {'type': 'EMA', 'parameters': {'period': 20}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
                {'indicator': 'EMA_0', 'operator': '>', 'threshold': 0},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70},
            ],
            'timeframe': '3m',  # not in allowed list
        }
        gene = parser.parse(data, generation=1, individual_id=1)
        assert gene is not None
        assert gene.timeframe in ['5m', '15m', '1h']

    def test_no_indicators_returns_none_with_feedback(self, parser):
        data = {
            'indicators': [],
            'entry_conditions': [],
            'exit_conditions': [],
        }
        gene, error = parser.parse_with_feedback(data, generation=1, individual_id=1)
        assert gene is None
        assert error is not None
        assert "indicator" in error.lower()

    def test_default_conditions_added_when_too_few(self, parser):
        """Parser should auto-add default conditions if LLM under-specifies."""
        data = {
            'indicators': [
                {'type': 'RSI', 'parameters': {'period': 14}},
                {'type': 'EMA', 'parameters': {'period': 20}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
            ],
            'exit_conditions': [],
        }
        gene = parser.parse(data, generation=1, individual_id=1)
        assert gene is not None
        assert len(gene.entry_conditions) >= 2
        assert len(gene.exit_conditions) >= 1

    def test_max_open_trades_clamped(self, parser):
        data = {
            'indicators': [
                {'type': 'RSI', 'parameters': {'period': 14}},
                {'type': 'EMA', 'parameters': {'period': 20}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
                {'indicator': 'EMA_0', 'operator': '>', 'threshold': 0},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70},
            ],
            'max_open_trades': 50,
        }
        gene = parser.parse(data, generation=1, individual_id=1)
        assert gene is not None
        assert gene.max_open_trades <= 10

    def test_minimal_roi_coerced(self, parser):
        data = {
            'indicators': [
                {'type': 'RSI', 'parameters': {'period': 14}},
                {'type': 'EMA', 'parameters': {'period': 20}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
                {'indicator': 'EMA_0', 'operator': '>', 'threshold': 0},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70},
            ],
            'minimal_roi': {0: 0.05, 30: 0.03, 60: 0.01},  # int keys
        }
        gene = parser.parse(data, generation=1, individual_id=1)
        assert gene is not None
        assert all(isinstance(k, str) for k in gene.minimal_roi.keys())


# =============================================================================
# ROUTER TESTS
# =============================================================================


class TestLLMProviderRouter:

    def _mock_provider(self, name, response="mock response", should_fail=False):
        provider = MagicMock()
        provider.provider_name = name
        provider.model = f"{name}-model"
        if should_fail:
            provider.generate.side_effect = RuntimeError(f"{name} failed")
        else:
            provider.generate.return_value = response
        return provider

    @patch('genetic_algorithm.llm.router.LLMProviderFactory')
    def test_single_provider_success(self, mock_factory):
        mock_factory.create.return_value = self._mock_provider("groq", "result")
        config = {
            'providers_list': [{'provider': 'groq'}],
            'cooldown_seconds': 5,
        }
        router = LLMProviderRouter(config)
        result = router.generate("test prompt")
        assert result == "result"
        assert router.last_used_provider == "groq"

    @patch('genetic_algorithm.llm.router.LLMProviderFactory')
    def test_failover_to_second_provider(self, mock_factory):
        providers = {
            'groq': self._mock_provider("groq", should_fail=True),
            'grok': self._mock_provider("grok", "grok result"),
        }
        mock_factory.create.side_effect = lambda c: providers[c['provider']]
        config = {
            'providers_list': [
                {'provider': 'groq'},
                {'provider': 'grok'},
            ],
            'cooldown_seconds': 5,
        }
        router = LLMProviderRouter(config)
        result = router.generate("test prompt")
        assert result == "grok result"
        assert router.last_used_provider == "grok"

    @patch('genetic_algorithm.llm.router.LLMProviderFactory')
    def test_all_providers_fail_raises(self, mock_factory):
        mock_factory.create.return_value = self._mock_provider("fail", should_fail=True)
        config = {
            'providers_list': [{'provider': 'fail'}],
            'cooldown_seconds': 5,
        }
        router = LLMProviderRouter(config)
        with pytest.raises(RuntimeError, match="All .* providers failed"):
            router.generate("test prompt")

    @patch('genetic_algorithm.llm.router.LLMProviderFactory')
    def test_cooldown_skips_failed_provider(self, mock_factory):
        groq = self._mock_provider("groq", should_fail=True)
        grok = self._mock_provider("grok", "grok result")
        call_count = [0]

        def create_side_effect(c):
            name = c['provider']
            return groq if name == 'groq' else grok

        mock_factory.create.side_effect = create_side_effect
        config = {
            'providers_list': [
                {'provider': 'groq'},
                {'provider': 'grok'},
            ],
            'cooldown_seconds': 60,
        }
        router = LLMProviderRouter(config)
        # First call: groq fails, grok succeeds
        router.generate("test 1")
        # Second call: groq should be skipped (cooling down)
        router.generate("test 2")
        stats = router.get_router_stats()
        # groq attempted once, grok attempted twice
        assert stats['stats']['groq']['attempts'] == 1
        assert stats['stats']['grok']['attempts'] == 2

    @patch('genetic_algorithm.llm.router.LLMProviderFactory')
    def test_router_stats(self, mock_factory):
        mock_factory.create.return_value = self._mock_provider("groq", "ok")
        config = {
            'providers_list': [{'provider': 'groq'}],
            'cooldown_seconds': 5,
        }
        router = LLMProviderRouter(config)
        router.generate("test")
        stats = router.get_router_stats()
        assert 'groq' in stats['providers']
        assert stats['stats']['groq']['successes'] == 1
        assert stats['stats']['groq']['failures'] == 0

    def test_no_providers_raises(self):
        config = {'providers_list': [], 'cooldown_seconds': 5}
        with pytest.raises(ValueError, match="no valid providers"):
            LLMProviderRouter(config)

    @patch('genetic_algorithm.llm.router.LLMProviderFactory')
    def test_provider_name_property(self, mock_factory):
        mock_factory.create.return_value = self._mock_provider("test")
        config = {'providers_list': [{'provider': 'test'}], 'cooldown_seconds': 5}
        router = LLMProviderRouter(config)
        assert router.provider_name == "LLMProviderRouter"


class TestCreateProviderOrRouter:

    @patch('genetic_algorithm.llm.router.LLMProviderFactory')
    def test_returns_router_with_providers_list(self, mock_factory):
        mock_factory.create.return_value = MagicMock()
        config = {'providers_list': [{'provider': 'groq'}], 'cooldown_seconds': 5}
        result = create_provider_or_router(config)
        assert isinstance(result, LLMProviderRouter)

    @patch('genetic_algorithm.llm.router.LLMProviderFactory')
    def test_returns_single_provider_without_list(self, mock_factory):
        mock_provider = MagicMock()
        mock_factory.create.return_value = mock_provider
        config = {'provider': 'groq', 'api_key': 'test'}
        result = create_provider_or_router(config)
        assert result is mock_provider
