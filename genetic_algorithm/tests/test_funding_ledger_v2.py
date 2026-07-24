"""Contract tests for timestamped funding cashflows produced by backtesting."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from freqtrade.enums import TradingMode
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import LocalTrade


def _futures_trade() -> LocalTrade:
    return LocalTrade(
        pair="BTC/USDT:USDT",
        stake_amount=100.0,
        max_stake_amount=100.0,
        amount=2.0,
        open_rate=100.0,
        open_date=datetime(2025, 1, 1, tzinfo=UTC),
        fee_open=0.001,
        fee_close=0.001,
        exchange="binance",
        is_short=False,
        leverage=2.0,
        trading_mode=TradingMode.FUTURES,
    )


def test_trade_records_incremental_wallet_funding_cashflows():
    trade = _futures_trade()
    first = datetime(2025, 1, 1, 8, tzinfo=UTC)
    second = datetime(2025, 1, 1, 16, tzinfo=UTC)
    after_fill = datetime(2025, 1, 2, tzinfo=UTC)

    trade.set_funding_fees(-0.1, event_time=first)
    trade.set_funding_fees(-0.3, event_time=second)
    trade.set_funding_fees(-0.3, event_time=second)

    assert trade.funding_fees == pytest.approx(-0.3)
    assert trade.to_json(True)["funding_fee_events"] == [
        {"timestamp": 1735718400000, "amount": pytest.approx(-0.1)},
        {"timestamp": 1735747200000, "amount": pytest.approx(-0.2)},
    ]

    # Simulate an unobserved bookkeeping transfer/reset at a fill. The next
    # update reconciles against the recorded ledger, not the mutable fields.
    trade.funding_fees = 0.0
    trade.funding_fee_running = 0.0
    trade.set_funding_fees(-0.25, event_time=after_fill)

    assert trade.funding_fees == pytest.approx(-0.25)
    assert trade.to_json(True)["funding_fee_events"][-1] == {
        "timestamp": 1735776000000,
        "amount": pytest.approx(0.05),
    }
    assert sum(
        event["amount"] for event in trade.to_json(True)["funding_fee_events"]
    ) == pytest.approx(trade.funding_fees)


def test_backtester_passes_funding_event_time_into_trade_ledger():
    backtesting = object.__new__(Backtesting)
    backtesting.trading_mode = TradingMode.FUTURES
    backtesting.funding_fee_timeframe_secs = 8 * 60 * 60
    backtesting.futures_data = {"BTC/USDT:USDT": object()}
    backtesting.exchange = MagicMock()
    backtesting.exchange.calculate_funding_fees.return_value = -0.125

    trade = MagicMock()
    trade.pair = "BTC/USDT:USDT"
    trade.amount = 2.0
    trade.is_short = False
    trade.date_last_filled_utc = datetime(2025, 1, 1, tzinfo=UTC)
    event_time = datetime(2025, 1, 1, 8, tzinfo=UTC)

    backtesting._run_funding_fees(trade, event_time)

    trade.set_funding_fees.assert_called_once_with(-0.125, event_time=event_time)
