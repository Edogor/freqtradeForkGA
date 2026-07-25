"""Regression tests for versioned right-censored profit-factor evidence."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from genetic_algorithm.evaluation.profit_factor_v2 import (
    PROFIT_FACTOR_CENSORED_CAP,
    normalize_profit_factor,
    profit_factor_for_scoring,
)
from genetic_algorithm.orchestration.result_contract import ScenarioMetricsV2
from genetic_algorithm.tests.v2_metric_fixtures import expectancy_metrics


def test_zero_loss_profit_factor_is_finite_and_explicitly_censored():
    value, censored = normalize_profit_factor(
        0.0,
        total_trades=5,
        losses=0,
        net_profit=4.0,
    )

    assert value == PROFIT_FACTOR_CENSORED_CAP
    assert censored is True


def test_mixed_and_empty_samples_keep_fail_closed_semantics():
    assert normalize_profit_factor(
        1.7,
        total_trades=20,
        losses=4,
        net_profit=2.0,
    ) == (1.7, False)
    assert normalize_profit_factor(
        0.0,
        total_trades=0,
        losses=0,
        net_profit=0.0,
    ) == (0.0, False)


def test_tiny_censored_sample_cannot_outrank_adequate_mixed_sample_on_pf():
    tiny_censored = profit_factor_for_scoring(
        PROFIT_FACTOR_CENSORED_CAP,
        censored=True,
        trade_count=1,
        normalization_cap=3.0,
    )
    adequate_mixed = profit_factor_for_scoring(
        1.5,
        censored=False,
        trade_count=60,
        normalization_cap=3.0,
    )
    adequate_censored = profit_factor_for_scoring(
        PROFIT_FACTOR_CENSORED_CAP,
        censored=True,
        trade_count=30,
        normalization_cap=3.0,
    )

    assert tiny_censored < adequate_mixed < adequate_censored


def test_v2_rejects_censored_profit_factor_without_versioned_cap():
    payload = {
        "scenario_id": "pf-contract",
        "pair": "BTC/USDT",
        "timeframe": "1h",
        "role": "TRAIN",
        "period_start": "2025-01-01",
        "period_end": "2025-02-01",
        "cost_multiplier": 1.0,
        "status": "VALID",
        "success": True,
        "net_return": 0.08,
        "annualized_net_return": 0.12,
        "annualized_net_return_lcb": 0.04,
        "max_drawdown": 0.10,
        "max_drawdown_ucb": 0.15,
        "daily_expected_shortfall_5": 0.01,
        "daily_expected_shortfall_5_ucb": 0.02,
        "net_expectancy": 0.003,
        "net_expectancy_lcb": 0.001,
        "profit_factor": 2.0,
        "profit_factor_censored": True,
        "profit_factor_contract_version": "right-censored-profit-factor-v1",
        "win_rate": 1.0,
        "trade_count": 40,
        "effective_sample_size": 30.0,
        **expectancy_metrics(
            trade_count=40,
            mean_return=0.003,
            lower_confidence_bound=0.001,
            effective_sample_size=30.0,
        ),
        "active_months": 5,
        "max_consecutive_losses": 0,
        "max_drawdown_duration_days": 2.0,
    }

    with pytest.raises(ValidationError, match="versioned finite cap"):
        ScenarioMetricsV2(**payload)
