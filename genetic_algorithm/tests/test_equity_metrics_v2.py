"""Reference tests for calendar-aligned V2 equity and downside metrics."""

from __future__ import annotations

import math
from datetime import date
from statistics import fmean, stdev

import pandas as pd
import pytest
from genetic_algorithm.tests.v2_metric_fixtures import expectancy_metrics

from genetic_algorithm.evaluation.confidence_metrics_v2 import (
    effective_sample_size,
    moving_block_expectancy_lcb,
)
from genetic_algorithm.evaluation.equity_metrics_v2 import (
    EquityDataError,
    build_equity_from_freqtrade_stats,
    build_mark_to_market_equity,
    build_realized_close_equity,
    calculate_equity_risk_metrics,
    moving_block_bootstrap_bounds,
)


def _spot_long_trade(**overrides):
    trade = {
        "pair": "BTC/USDT",
        "stake_amount": 100.0,
        "max_stake_amount": 100.0,
        "amount": 1.0,
        "open_date": "2025-01-01 10:00:00+00:00",
        "close_date": "2025-01-03 12:00:00+00:00",
        "open_rate": 100.0,
        "close_rate": 110.0,
        "fee_open": 0.001,
        "fee_close": 0.001,
        "profit_abs": 9.79,
        "leverage": 1.0,
        "is_short": False,
        "is_open": False,
        "funding_fees": 0.0,
        "orders": [
            {
                "ft_is_entry": True,
                "amount": 1.0,
                "safe_price": 100.0,
                "order_filled_timestamp": 1735725600000,
            },
            {
                "ft_is_entry": False,
                "amount": 1.0,
                "safe_price": 110.0,
                "order_filled_timestamp": 1735905600000,
            },
        ],
    }
    trade.update(overrides)
    return trade


def _daily_closes(*values):
    return {
        "BTC/USDT": pd.DataFrame({
            "date": pd.date_range("2025-01-01 23:00:00", periods=len(values), freq="D", tz="UTC"),
            "close": values,
        })
    }


def test_realized_equity_is_calendar_filled_and_capital_weighted():
    series = build_realized_close_equity(
        period_start="2025-01-01",
        period_end="2025-01-04",
        starting_balance=100.0,
        daily_profit_abs=[
            ("2025-01-01", 10.0),
            ("2025-01-04", -5.0),
        ],
        expected_final_balance=105.0,
    )

    assert series.dates == (
        date(2025, 1, 1),
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 4),
    )
    assert series.daily_profit_abs == (10.0, 0.0, 0.0, -5.0)
    assert series.daily_net_returns == pytest.approx((0.1, 0.0, 0.0, -5.0 / 110.0))
    assert series.equity_curve == (100.0, 110.0, 110.0, 110.0, 105.0)
    assert series.final_balance == 105.0
    assert series.equity_method == "REALIZED_CLOSE"


def test_mark_to_market_equity_exposes_open_loss_and_reconciles_final_wallet():
    series = build_mark_to_market_equity(
        period_start="2025-01-01",
        period_end="2025-01-03",
        starting_balance=1000.0,
        trades=[_spot_long_trade()],
        ohlcv_by_pair=_daily_closes(90.0, 105.0, 110.0),
        expected_final_balance=1009.79,
    )

    # Open value includes entry fee; marks include the hypothetical exit fee.
    assert series.equity_curve == pytest.approx([1000.0, 989.81, 1004.795, 1009.79])
    assert series.daily_profit_abs == pytest.approx([-10.19, 14.985, 4.995])
    assert series.equity_method == "MARK_TO_MARKET"
    risk = calculate_equity_risk_metrics(series)
    assert risk.max_drawdown == pytest.approx(10.19 / 1000.0)


def test_mark_to_market_fails_closed_on_missing_open_day_candle():
    candles = _daily_closes(90.0, 105.0, 110.0)
    candles["BTC/USDT"] = candles["BTC/USDT"].iloc[[0, 2]]

    with pytest.raises(EquityDataError, match="missing end-of-day close"):
        build_mark_to_market_equity(
            period_start="2025-01-01",
            period_end="2025-01-03",
            starting_balance=1000.0,
            trades=[_spot_long_trade()],
            ohlcv_by_pair=candles,
            expected_final_balance=1009.79,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"funding_fees": -0.1}, "no timestamped funding ledger"),
        ({"is_open": True}, "open trades"),
        ({"leverage": 0.0}, "non-positive leverage"),
        ({"orders": []}, "entry and exit"),
    ],
)
def test_mark_to_market_rejects_unsupported_wallet_semantics(override, message):
    with pytest.raises(EquityDataError, match=message):
        build_mark_to_market_equity(
            period_start="2025-01-01",
            period_end="2025-01-03",
            starting_balance=1000.0,
            trades=[_spot_long_trade(**override)],
            ohlcv_by_pair=_daily_closes(90.0, 105.0, 110.0),
        )


def test_mark_to_market_supports_short_and_leverage_without_double_counting():
    trade = _spot_long_trade(
        amount=2.0,
        stake_amount=100.0,
        max_stake_amount=100.0,
        close_rate=80.0,
        profit_abs=39.64,
        leverage=2.0,
        is_short=True,
        orders=[
            {
                "ft_is_entry": True,
                "amount": 2.0,
                "safe_price": 100.0,
                "order_filled_timestamp": 1735725600000,
            },
            {
                "ft_is_entry": False,
                "amount": 2.0,
                "safe_price": 80.0,
                "order_filled_timestamp": 1735905600000,
            },
        ],
    )

    series = build_mark_to_market_equity(
        period_start="2025-01-01",
        period_end="2025-01-03",
        starting_balance=1000.0,
        trades=[trade],
        ohlcv_by_pair=_daily_closes(90.0, 85.0, 80.0),
        expected_final_balance=1039.64,
    )

    assert series.equity_curve == pytest.approx([1000.0, 1019.62, 1029.63, 1039.64])


def test_mark_to_market_books_timestamped_funding_on_actual_event_days():
    trade = _spot_long_trade(
        amount=2.0,
        stake_amount=100.0,
        max_stake_amount=100.0,
        profit_abs=18.83,
        leverage=2.0,
        funding_fees=-0.75,
        funding_fee_events=[
            {"timestamp": 1735761600000, "amount": -1.0},
            {"timestamp": 1735848000000, "amount": 0.25},
        ],
        orders=[
            {
                "ft_is_entry": True,
                "amount": 2.0,
                "safe_price": 100.0,
                "order_filled_timestamp": 1735725600000,
            },
            {
                "ft_is_entry": False,
                "amount": 2.0,
                "safe_price": 110.0,
                "order_filled_timestamp": 1735905600000,
            },
        ],
    )

    series = build_mark_to_market_equity(
        period_start="2025-01-01",
        period_end="2025-01-03",
        starting_balance=1000.0,
        trades=[trade],
        ohlcv_by_pair=_daily_closes(100.0, 105.0, 110.0),
        expected_final_balance=1018.83,
    )

    assert series.equity_curve == pytest.approx([1000.0, 998.6, 1008.84, 1018.83])


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([{"timestamp": 1735761600000, "amount": -0.5}], "funding ledger sums"),
        ([{"timestamp": 1735686000000, "amount": -0.75}], "outside its filled position"),
        (
            [
                {"timestamp": 1735848000000, "amount": -0.5},
                {"timestamp": 1735761600000, "amount": -0.25},
            ],
            "not chronological",
        ),
    ],
)
def test_mark_to_market_rejects_corrupt_funding_provenance(events, message):
    with pytest.raises(EquityDataError, match=message):
        build_mark_to_market_equity(
            period_start="2025-01-01",
            period_end="2025-01-03",
            starting_balance=1000.0,
            trades=[
                _spot_long_trade(
                    funding_fees=-0.75,
                    funding_fee_events=events,
                    profit_abs=9.04,
                )
            ],
            ohlcv_by_pair=_daily_closes(100.0, 105.0, 110.0),
        )


def test_mark_to_market_replays_dca_and_partial_exits_in_fill_order():
    trade = _spot_long_trade(
        amount=1.0,
        stake_amount=90.0,
        max_stake_amount=180.0,
        open_rate=90.0,
        close_rate=120.0,
        profit_abs=49.59,
        orders=[
            {
                "ft_is_entry": True,
                "amount": 1.0,
                "safe_price": 100.0,
                "order_filled_timestamp": 1735725600000,
            },
            {
                "ft_is_entry": True,
                "amount": 1.0,
                "safe_price": 80.0,
                "order_filled_timestamp": 1735808400000,
            },
            {
                "ft_is_entry": False,
                "amount": 1.0,
                "safe_price": 110.0,
                "order_filled_timestamp": 1735812000000,
            },
            {
                "ft_is_entry": False,
                "amount": 1.0,
                "safe_price": 120.0,
                "order_filled_timestamp": 1735905600000,
            },
        ],
    )

    series = build_mark_to_market_equity(
        period_start="2025-01-01",
        period_end="2025-01-03",
        starting_balance=1000.0,
        trades=[trade],
        ohlcv_by_pair=_daily_closes(90.0, 85.0, 120.0),
        expected_final_balance=1049.59,
    )

    # Day 2: +19.80 realised on the partial exit and -5.175 unrealised.
    assert series.equity_curve == pytest.approx([1000.0, 989.81, 1014.625, 1049.59])


def test_mark_to_market_rejects_non_reconciling_order_ledger():
    with pytest.raises(EquityDataError, match="order ledger yields"):
        build_mark_to_market_equity(
            period_start="2025-01-01",
            period_end="2025-01-03",
            starting_balance=1000.0,
            trades=[_spot_long_trade(profit_abs=999.0)],
            ohlcv_by_pair=_daily_closes(90.0, 105.0, 110.0),
        )


def test_mark_to_market_requires_complete_order_fill_provenance():
    trade = _spot_long_trade()
    del trade["orders"][0]["safe_price"]

    with pytest.raises(EquityDataError, match="order 0 missing: safe_price"):
        build_mark_to_market_equity(
            period_start="2025-01-01",
            period_end="2025-01-03",
            starting_balance=1000.0,
            trades=[trade],
            ohlcv_by_pair=_daily_closes(90.0, 105.0, 110.0),
        )


def test_equity_risk_metrics_match_hand_calculated_reference():
    series = build_realized_close_equity(
        period_start="2025-01-01",
        period_end="2025-01-04",
        starting_balance=100.0,
        daily_profit_abs=[
            ("2025-01-01", 10.0),
            ("2025-01-02", -22.0),
            ("2025-01-04", 12.0),
        ],
        expected_final_balance=100.0,
    )
    metrics = calculate_equity_risk_metrics(series)
    returns = [0.1, -0.2, 0.0, 12.0 / 88.0]
    expected_sharpe = fmean(returns) / stdev(returns) * math.sqrt(365.0)
    downside = math.sqrt(fmean(min(value, 0.0) ** 2 for value in returns))
    expected_sortino = fmean(returns) / downside * math.sqrt(365.0)

    assert metrics.observation_days == 4
    assert metrics.total_net_return == pytest.approx(0.0)
    assert metrics.annualized_net_return == pytest.approx(0.0)
    assert metrics.sharpe_ratio == pytest.approx(expected_sharpe)
    assert metrics.sortino_ratio == pytest.approx(expected_sortino)
    assert metrics.max_drawdown == pytest.approx(0.2)
    assert metrics.max_drawdown_duration_days == 3
    assert metrics.time_under_water_ratio == pytest.approx(0.75)
    expected_ulcer = math.sqrt(fmean([0.0, 0.2**2, 0.2**2, (10.0 / 110.0) ** 2]))
    assert metrics.ulcer_index == pytest.approx(expected_ulcer)
    assert metrics.daily_expected_shortfall_5 == pytest.approx(0.2)
    assert metrics.calmar_ratio == pytest.approx(0.0)


def test_undefined_ratios_are_none_instead_of_magic_scores():
    series = build_realized_close_equity(
        period_start="2025-01-01",
        period_end="2025-02-01",
        starting_balance=100.0,
        daily_profit_abs=[],
        expected_final_balance=100.0,
    )
    metrics = calculate_equity_risk_metrics(series)

    assert metrics.total_net_return == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.daily_expected_shortfall_5 == 0.0
    assert metrics.sharpe_ratio is None
    assert metrics.sortino_ratio is None
    assert metrics.calmar_ratio is None
    assert metrics.has_defined_risk_ratios is False


def test_freqtrade_stats_require_daily_profit_when_trades_exist():
    stats = {
        "backtest_start": "2025-01-01 00:00:00",
        "backtest_end": "2025-01-31 00:00:00",
        "starting_balance": 100.0,
        "final_balance": 105.0,
        "total_trades": 3,
    }

    with pytest.raises(EquityDataError, match="missing daily_profit"):
        build_equity_from_freqtrade_stats(stats)


def test_balance_mismatch_fails_closed():
    with pytest.raises(EquityDataError, match="expected final balance"):
        build_realized_close_equity(
            period_start="2025-01-01",
            period_end="2025-01-02",
            starting_balance=100.0,
            daily_profit_abs=[("2025-01-01", 5.0)],
            expected_final_balance=106.0,
        )


def test_moving_block_bootstrap_is_deterministic_and_adverse_sensitive():
    benign = [0.001] * 120
    adverse = ([0.001] * 9 + [-0.02]) * 12

    first = moving_block_bootstrap_bounds(
        adverse,
        samples=200,
        block_length_days=10,
        confidence=0.95,
        seed=123,
    )
    second = moving_block_bootstrap_bounds(
        adverse,
        samples=200,
        block_length_days=10,
        confidence=0.95,
        seed=123,
    )
    benign_bounds = moving_block_bootstrap_bounds(
        benign,
        samples=200,
        block_length_days=10,
        confidence=0.95,
        seed=123,
    )

    assert first == second
    assert first.annualized_net_return_lcb < benign_bounds.annualized_net_return_lcb
    assert first.max_drawdown_ucb > benign_bounds.max_drawdown_ucb
    assert first.daily_expected_shortfall_5_ucb > benign_bounds.daily_expected_shortfall_5_ucb


def test_bootstrap_rejects_insufficient_evidence():
    with pytest.raises(EquityDataError, match="at least"):
        moving_block_bootstrap_bounds([0.001] * 20, samples=100, block_length_days=10)


def test_trade_effective_sample_size_penalizes_serial_clustering():
    clustered = ([0.01] * 10 + [-0.01] * 10) * 3
    alternating = [0.01, -0.01] * 30

    assert effective_sample_size(clustered) < effective_sample_size(alternating)
    assert effective_sample_size(alternating) == len(alternating)


def test_trade_expectancy_lcb_is_deterministic_and_below_sample_mean():
    returns = [0.012, 0.012, 0.012, -0.02, 0.003, 0.01, -0.015] * 6
    first = moving_block_expectancy_lcb(
        returns, samples=200, block_length_trades=5, confidence=0.95, seed=7
    )
    second = moving_block_expectancy_lcb(
        returns, samples=200, block_length_trades=5, confidence=0.95, seed=7
    )

    assert first == second
    assert first.lower_confidence_bound < first.mean
    assert 1.0 <= first.effective_sample_size <= len(returns)


def test_trade_confidence_rejects_constant_or_tiny_samples():
    with pytest.raises(EquityDataError, match="near-constant"):
        moving_block_expectancy_lcb([0.01] * 30, samples=100)
    with pytest.raises(EquityDataError, match="at least"):
        moving_block_expectancy_lcb([0.01, -0.01] * 5, samples=100)


def test_direct_backtest_parser_exposes_v2_shadow_metrics():
    from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

    stats = {
        "profit_total": 0.05,
        "profit_total_abs": 5.0,
        "profit_total_pct": 5.0,
        "total_trades": 2,
        "wins": 1,
        "losses": 1,
        "winrate": 0.5,
        "max_drawdown_account": 0.01,
        "max_drawdown_abs": 1.0,
        "sharpe": 99.0,  # legacy value stays separate
        "sortino": 88.0,
        "profit_factor": 2.0,
        "profit_mean": 0.025,
        "profit_median": 0.025,
        "duration_avg": "1:00:00",
        "backtest_start": "2025-01-01 00:00:00",
        "backtest_end": "2025-01-03 00:00:00",
        "starting_balance": 100.0,
        "final_balance": 105.0,
        "daily_profit": [["2025-01-01", 10.0], ["2025-01-02", -5.0]],
    }

    parser = object.__new__(DirectBacktester)
    result = parser._parse_stats(stats, "ParsedStrategy")

    assert result.equity_method == "REALIZED_CLOSE"
    assert result.equity_error_message is None
    assert result.daily_profit_abs == [
        ["2025-01-01", 10.0],
        ["2025-01-02", -5.0],
        ["2025-01-03", 0.0],
    ]
    assert result.daily_net_returns == pytest.approx([0.1, -5.0 / 110.0, 0.0])
    assert result.equity_curve == [100.0, 110.0, 105.0, 105.0]
    assert result.daily_max_drawdown == pytest.approx(5.0 / 110.0)
    assert result.daily_expected_shortfall_5 == pytest.approx(5.0 / 110.0)
    assert result.sharpe_ratio == 99.0
    assert result.daily_sharpe_ratio != result.sharpe_ratio


def test_direct_backtest_parser_keeps_missing_equity_as_none():
    from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

    parser = object.__new__(DirectBacktester)
    result = parser._parse_stats(
        {
            "profit_total": 0.01,
            "profit_total_abs": 1.0,
            "total_trades": 1,
        },
        "IncompleteStrategy",
    )

    assert result.daily_net_returns is None
    assert result.equity_curve is None
    assert result.daily_sharpe_ratio is None
    assert result.daily_expected_shortfall_5 is None
    assert result.equity_error_message is not None


def test_result_contract_rejects_realized_close_equity_as_valid():
    from pydantic import ValidationError

    from genetic_algorithm.orchestration.result_contract import BacktestRecordV2

    with pytest.raises(ValidationError, match="MARK_TO_MARKET"):
        BacktestRecordV2(
            attempt_id="attempt-1",
            wave_id="wave-1",
            experiment_id="experiment-1",
            candidate_id="candidate-1",
            config_hash="config-hash-1",
            phenotype_hash="phenotype-hash-1",
            code_version="commit-1",
            data_manifest_hash="manifest-hash-1",
            fitness_policy_version="policy-1",
            seed=42,
            worker_count=1,
            fee_rate=0.001,
            slippage_rate=0.0005,
            spread_rate=0.0,
            funding_rate=0.0,
            equity_method="REALIZED_CLOSE",
            metrics={
                "scenario_id": "btc-inner-1",
                "pair": "BTC/USDT",
                "timeframe": "1h",
                "role": "INNER_VALIDATION",
                "period_start": "2025-01-01",
                "period_end": "2025-02-01",
                "cost_multiplier": 1.0,
                "status": "VALID",
                "success": True,
                "trade_count": 1,
                "active_months": 1,
                "net_return": 0.01,
                "annualized_net_return": 0.12,
                "annualized_net_return_lcb": 0.01,
                "max_drawdown": 0.02,
                "max_drawdown_ucb": 0.03,
                "daily_expected_shortfall_5": 0.01,
                "daily_expected_shortfall_5_ucb": 0.015,
                "net_expectancy": 0.01,
                "net_expectancy_lcb": 0.001,
                "profit_factor": 1.5,
                "win_rate": 0.6,
                "effective_sample_size": 1.0,
                **expectancy_metrics(
                    trade_count=1,
                    mean_return=0.01,
                    lower_confidence_bound=0.001,
                    effective_sample_size=1.0,
                ),
                "max_consecutive_losses": 0,
                "max_drawdown_duration_days": 2.0,
            },
            daily_net_returns=[0.01],
            equity_curve=[100.0, 101.0],
            trades=[{"profit_ratio": 0.01}],
        )
