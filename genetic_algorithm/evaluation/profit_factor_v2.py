"""Versioned, finite semantics for zero-loss profit-factor samples."""

from __future__ import annotations

import math


PROFIT_FACTOR_CONTRACT_VERSION = "right-censored-profit-factor-v1"
PROFIT_FACTOR_CENSORED_CAP = 10.0
PROFIT_FACTOR_FULL_CREDIT_TRADES = 30


def normalize_profit_factor(
    raw_profit_factor: object,
    *,
    total_trades: int,
    losses: int,
    net_profit: float,
) -> tuple[float, bool]:
    """Return a finite profit factor and whether it is right-censored.

    Freqtrade reports ``0.0`` when a profitable sample has no loss trade,
    although gross-profit / gross-loss is then undefined/infinite.  Persist a
    stable finite cap and an explicit censor flag instead of interpreting that
    sentinel as the worst possible profit factor.
    """

    if total_trades <= 0:
        return 0.0, False
    if losses == 0 and net_profit > 0.0:
        return PROFIT_FACTOR_CENSORED_CAP, True
    try:
        value = float(raw_profit_factor)
    except (TypeError, ValueError):
        return 0.0, False
    if math.isnan(value) or value < 0.0:
        return 0.0, False
    if math.isinf(value):
        return PROFIT_FACTOR_CENSORED_CAP, True
    return value, False


def profit_factor_for_scoring(
    profit_factor: float,
    *,
    censored: bool,
    trade_count: int,
    normalization_cap: float,
) -> float:
    """Discount a censored PF until it has a minimally useful sample size."""

    if not censored:
        return profit_factor
    if trade_count <= 0:
        return 0.0
    upper = min(PROFIT_FACTOR_CENSORED_CAP, normalization_cap, profit_factor)
    if upper <= 1.0:
        return upper
    evidence = min(1.0, trade_count / PROFIT_FACTOR_FULL_CREDIT_TRADES)
    return 1.0 + (upper - 1.0) * evidence
