from __future__ import annotations

from genetic_algorithm.evaluation.activity_metrics import count_active_trade_months


def test_trade_close_months_are_primary_activity_evidence():
    trades = [
        {"close_date": "2025-01-31T23:00:00Z"},
        {"close_date": "2025-02-01T01:00:00Z"},
        # Same-day realized profits can cancel; the close ledger must still
        # retain February as an active month.
        {"close_timestamp": 1_738_458_000_000},
    ]

    assert count_active_trade_months(
        trades,
        daily_profit_abs=[["2025-01-31", 1.0], ["2025-02-01", 0.0]],
    ) == 2


def test_close_dates_are_normalized_to_utc_months():
    assert count_active_trade_months(
        [{"close_date": "2025-01-31T23:30:00-02:00"}]
    ) == 1


def test_legacy_daily_profit_fallback_remains_available():
    assert count_active_trade_months(
        [{"pair": "BTC/USDT", "profit_ratio": 0.01}],
        daily_profit_abs=[
            ["2025-01-01", 1.0],
            ["2025-02-01", 0.0],
            ["2025-03-01", -1.0],
        ],
    ) == 2
