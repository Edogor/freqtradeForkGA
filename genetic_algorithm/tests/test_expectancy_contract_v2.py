"""References for clustered, committed-capital net expectancy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from genetic_algorithm.evaluation.confidence_metrics_v2 import (
    EXPECTANCY_CONTRACT_VERSION,
    InsufficientTradeEvidenceError,
    clustered_trade_expectancy_lcb,
)
from genetic_algorithm.evaluation.equity_metrics_v2 import EquityDataError
from genetic_algorithm.orchestration.result_contract import ScenarioMetricsV2
from genetic_algorithm.tests.v2_metric_fixtures import expectancy_metrics


def _trade(
    index: int,
    *,
    pair: str = "BTC/USDT",
    committed_before_fee: float = 100.0,
    net_return: float = 0.01,
    is_short: bool = False,
    fee_open: float = 0.001,
    day_offset: int | None = None,
    leverage: float = 1.0,
    funding_fees: float = 0.0,
) -> dict:
    day = index if day_offset is None else day_offset
    opened = datetime(2025, 1, 1, 8, tzinfo=UTC) + timedelta(days=day)
    closed = opened + timedelta(hours=4)
    fee_factor = 1.0 - fee_open if is_short else 1.0 + fee_open
    return {
        "trade_id": index,
        "pair": pair,
        "is_open": False,
        "is_short": is_short,
        "open_date": opened.isoformat(),
        "close_date": closed.isoformat(),
        "max_stake_amount": committed_before_fee,
        "stake_amount": committed_before_fee,
        "fee_open": fee_open,
        "leverage": leverage,
        "profit_abs": net_return * committed_before_fee * fee_factor,
        "profit_ratio": net_return,
        "funding_fees": funding_fees,
    }


def test_equal_trade_edge_can_disagree_with_committed_capital_edge():
    trades = [
        _trade(index, committed_before_fee=100.0, net_return=0.01)
        for index in range(19)
    ]
    trades.append(
        _trade(19, committed_before_fee=10_000.0, net_return=-0.01)
    )

    result = clustered_trade_expectancy_lcb(
        trades,
        samples=300,
        block_length_clusters=5,
        seed=7,
    )

    expected_capital = sum(
        trade["max_stake_amount"] * (1.0 + trade["fee_open"])
        for trade in trades
    )
    expected_profit = sum(trade["profit_abs"] for trade in trades)
    expected_trade_mean = sum(trade["profit_ratio"] for trade in trades) / len(trades)

    assert result.expectancy_contract_version == EXPECTANCY_CONTRACT_VERSION
    assert result.mean_trade_return == pytest.approx(expected_trade_mean)
    assert result.mean_trade_return > 0
    assert result.total_committed_capital == pytest.approx(expected_capital)
    assert result.total_net_profit_abs == pytest.approx(expected_profit)
    assert result.return_on_committed_capital == pytest.approx(
        expected_profit / expected_capital
    )
    assert result.return_on_committed_capital < 0
    assert result.capital_effective_sample_size < 2
    assert result.effective_sample_size <= result.capital_effective_sample_size


def test_temporal_clusters_keep_contemporaneous_pairs_together():
    trades = []
    for day in range(20):
        trades.append(
            _trade(
                day * 2,
                pair="BTC/USDT",
                day_offset=day,
                net_return=0.012 if day % 2 else -0.008,
            )
        )
        trades.append(
            _trade(
                day * 2 + 1,
                pair="ETH/USDT",
                day_offset=day,
                net_return=0.006 if day % 2 else -0.004,
                is_short=True,
                leverage=3.0,
                funding_fees=-0.02,
            )
        )

    first = clustered_trade_expectancy_lcb(
        trades,
        samples=300,
        block_length_clusters=3,
        seed=123,
    )
    second = clustered_trade_expectancy_lcb(
        list(reversed(trades)),
        samples=300,
        block_length_clusters=3,
        seed=123,
    )

    assert first == second
    assert first.observations == 40
    assert first.temporal_clusters == 20
    assert first.pair_count == 2
    assert first.effective_pair_count == pytest.approx(2.0)
    assert first.pair_capital_hhi == pytest.approx(0.5)
    assert first.effective_sample_size <= 20
    assert first.mean_trade_return_lcb < first.mean_trade_return
    assert (
        first.return_on_committed_capital_lcb
        < first.return_on_committed_capital
    )


def test_single_pair_concentration_stays_inside_contract_bounds():
    trades = [
        _trade(
            index,
            committed_before_fee=123.4567,
            net_return=0.01 if index % 3 else -0.005,
        )
        for index in range(73)
    ]

    result = clustered_trade_expectancy_lcb(
        trades,
        samples=200,
        block_length_clusters=5,
    )

    assert result.pair_count == 1
    assert result.pair_capital_hhi == 1.0
    assert result.effective_pair_count == pytest.approx(1.0)
    assert 0.0 < result.max_trade_capital_share <= 1.0
    assert 0.0 < result.max_cluster_capital_share <= 1.0


def test_losses_below_committed_margin_are_finite_economic_observations():
    trades = [
        _trade(
            index,
            net_return=-1.2 if index == 19 else (0.02 if index % 2 else -0.01),
            leverage=5.0,
        )
        for index in range(20)
    ]

    result = clustered_trade_expectancy_lcb(
        trades,
        samples=200,
        block_length_clusters=5,
    )

    assert result.mean_trade_return < 0
    assert result.return_on_committed_capital < 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"profit_ratio": 0.5}, "does not reconcile"),
        ({"max_stake_amount": None}, "must be numeric"),
        ({"fee_open": 1.0}, r"must be in \[0, 1\)"),
        ({"is_open": True}, "must be a closed trade"),
        ({"is_short": None}, "must be boolean"),
    ],
)
def test_corrupt_trade_evidence_fails_closed(mutation, message):
    trades = [
        _trade(index, net_return=0.01 if index % 2 else -0.005)
        for index in range(20)
    ]
    trades[5].update(mutation)

    with pytest.raises(EquityDataError, match=message):
        clustered_trade_expectancy_lcb(
            trades,
            samples=100,
            block_length_clusters=5,
        )


def test_mixed_quote_currency_profit_cannot_be_aggregated():
    trades = [
        _trade(
            index,
            pair="BTC/USDC" if index == 19 else "BTC/USDT",
            net_return=0.01 if index % 2 else -0.005,
        )
        for index in range(20)
    ]

    with pytest.raises(EquityDataError, match="multiple quote currencies"):
        clustered_trade_expectancy_lcb(
            trades,
            samples=100,
            block_length_clusters=5,
        )


def test_many_same_day_trades_do_not_fake_temporal_evidence():
    trades = [
        _trade(
            index,
            day_offset=index // 5,
            net_return=0.01 if index % 2 else -0.005,
        )
        for index in range(50)
    ]

    with pytest.raises(
        InsufficientTradeEvidenceError,
        match="temporal trade clusters",
    ):
        clustered_trade_expectancy_lcb(
            trades,
            samples=100,
            block_length_clusters=6,
        )


def _valid_scenario_payload() -> dict:
    return {
        "scenario_id": "btc-temporal",
        "pair": "BTC/USDT",
        "timeframe": "1h",
        "role": "TEMPORAL_VALIDATION",
        "period_start": "2025-01-01",
        "period_end": "2025-06-01",
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
        "profit_factor": 1.5,
        "profit_factor_censored": False,
        "profit_factor_contract_version": "right-censored-profit-factor-v1",
        "win_rate": 0.6,
        "trade_count": 40,
        "effective_sample_size": 30.0,
        **expectancy_metrics(
            trade_count=40,
            mean_return=0.003,
            lower_confidence_bound=0.001,
            effective_sample_size=30.0,
        ),
        "active_months": 5,
        "max_consecutive_losses": 3,
        "max_drawdown_duration_days": 20.0,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_committed_capital", 999.0, "does not reconcile"),
        ("effective_sample_size", 31.0, "does not match cluster components"),
        ("expectancy_effective_pair_count", 1.5, "pair count does not match"),
        ("expectancy_max_trade_capital_share", 0.001, "feasible minimum"),
        (
            "expectancy_serial_effective_sample_size",
            31.0,
            "exceeds temporal clusters",
        ),
        (
            "expectancy_capital_effective_sample_size",
            41.0,
            "exceeds measured trades",
        ),
        (
            "expectancy_temporal_effective_sample_size",
            31.0,
            "exceeds temporal clusters",
        ),
    ],
)
def test_scenario_contract_rejects_inconsistent_expectancy_evidence(
    field,
    value,
    message,
):
    payload = _valid_scenario_payload()
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        ScenarioMetricsV2(**payload)


def test_versioned_expectancy_evidence_cannot_be_partially_persisted():
    payload = _valid_scenario_payload()
    payload["status"] = "INVALID"
    payload["success"] = False
    payload["error_code"] = "UPSTREAM_MISMATCH"
    del payload["total_committed_capital"]

    with pytest.raises(ValidationError, match="incomplete expectancy evidence"):
        ScenarioMetricsV2(**payload)
