"""Consistent measured-metric payloads for strict V2 contract tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta


def expectancy_metrics(
    *,
    trade_count: int,
    mean_return: float,
    lower_confidence_bound: float,
    effective_sample_size: float,
    temporal_clusters: int | None = None,
    pair_count: int = 1,
    quote_currency: str = "USDT",
    capital_per_trade: float = 100.0,
) -> dict:
    """Build internally consistent synthetic clustered-expectancy evidence."""

    clusters = temporal_clusters or min(trade_count, max(10, int(effective_sample_size)))
    total_capital = trade_count * capital_per_trade
    total_profit = mean_return * total_capital
    serial_n_eff = max(effective_sample_size, min(float(clusters), float(trade_count)))
    temporal_n_eff = max(effective_sample_size, min(float(clusters), float(trade_count)))
    capital_n_eff = max(effective_sample_size, float(trade_count))
    return {
        "expectancy_contract_version": "net-expectancy-clustered-v1",
        "expectancy_quote_currency": quote_currency,
        "net_expectancy_on_committed_capital": mean_return,
        "net_expectancy_on_committed_capital_lcb": lower_confidence_bound,
        "mean_net_profit_abs_per_trade": total_profit / trade_count,
        "total_net_profit_abs": total_profit,
        "total_committed_capital": total_capital,
        "expectancy_temporal_clusters": clusters,
        "expectancy_cluster_days": 1,
        "expectancy_pair_count": pair_count,
        "expectancy_effective_pair_count": float(pair_count),
        "expectancy_pair_capital_hhi": 1.0 / pair_count,
        "expectancy_max_trade_capital_share": 1.0 / trade_count,
        "expectancy_max_cluster_capital_share": 1.0 / clusters,
        "expectancy_serial_effective_sample_size": serial_n_eff,
        "expectancy_capital_effective_sample_size": capital_n_eff,
        "expectancy_temporal_effective_sample_size": temporal_n_eff,
    }


def trade_evidence(
    returns: list[float],
    *,
    pair: str = "BTC/USDT",
    start: date = date(2025, 1, 1),
    committed_capital: float = 100.0,
    fee_open: float = 0.001,
) -> list[dict]:
    """Create reconciled, one-UTC-cluster-per-trade evidence for adapter tests."""

    trades = []
    for index, net_return in enumerate(returns):
        opened = datetime.combine(
            start + timedelta(days=index),
            time(hour=8),
            tzinfo=UTC,
        )
        closed = opened + timedelta(hours=4)
        trades.append(
            {
                "trade_id": index,
                "pair": pair,
                "is_open": False,
                "is_short": False,
                "open_date": opened.isoformat(),
                "close_date": closed.isoformat(),
                "max_stake_amount": committed_capital,
                "stake_amount": committed_capital,
                "fee_open": fee_open,
                "leverage": 1.0,
                "profit_abs": net_return * committed_capital * (1.0 + fee_open),
                "profit_ratio": net_return,
                "funding_fees": 0.0,
            }
        )
    return trades
