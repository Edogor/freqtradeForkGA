"""
Portfolio-Level Backtester

Combines per-strategy backtest results into a portfolio equity curve and
computes aggregate metrics (Sharpe, drawdown, correlation matrix) WITHOUT
re-running FreqTrade.

Usage:
    from genetic_algorithm.evaluation.portfolio_backtester import PortfolioBacktester

    pb = PortfolioBacktester()
    result = pb.evaluate_portfolio(strategies, weights)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise

from genetic_algorithm.evaluation.equity_metrics_v2 import (
    DEFAULT_ANNUAL_RISK_FREE_RATE,
    MONTHS_PER_YEAR,
    RISK_METRIC_CONTRACT_VERSION,
    EquityDataError,
    calculate_periodic_risk_ratios,
)


@dataclass
class PortfolioResult:
    """Aggregated results for a portfolio of strategies."""

    valid: bool = False
    error_code: str | None = None
    error_message: str | None = None
    total_profit_pct: float = 0.0
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    total_trades: int = 0
    strategy_count: int = 0
    correlation_matrix: dict[str, dict[str, float | None]] | None = None
    monthly_profits: list[float] | None = None
    monthly_periods: list[str] | None = None
    per_strategy_weights: dict[str, float] | None = None
    per_strategy_profits: dict[str, float] | None = None
    risk_metric_contract_version: str = RISK_METRIC_CONTRACT_VERSION
    periods_per_year: int = MONTHS_PER_YEAR
    annual_risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE
    periodic_risk_free_rate: float | None = None


@dataclass
class StrategySlot:
    """One strategy in a portfolio with its capital weight and results."""
    strategy_id: str
    weight: float  # capital allocation fraction (sums to 1)
    monthly_profits: list[float] = field(default_factory=list)
    monthly_periods: list[str] = field(default_factory=list)
    trade_profit_ratios: list[float] = field(default_factory=list)
    total_profit_pct: float = 0.0
    total_trades: int = 0


class PortfolioBacktester:
    """
    Combine per-strategy results into a portfolio and compute aggregate metrics.

    No backtests are re-run — this uses the monthly_profits and
    trade_profit_ratios already stored on evaluated individuals.
    """

    @staticmethod
    def evaluate_portfolio(
        slots: list[StrategySlot],
        risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE,
    ) -> PortfolioResult:
        """
        Evaluate a portfolio of strategy slots.

        Args:
            slots: list of StrategySlot with weights and per-strategy results.
            risk_free_rate: annual risk-free rate for Sharpe/Sortino (default 0).

        Returns:
            PortfolioResult with aggregate metrics.
        """
        try:
            active_slots, normalized_weights, periods = _validate_portfolio_inputs(slots)
        except EquityDataError as exc:
            return PortfolioResult(
                error_code="INVALID_PORTFOLIO_EVIDENCE",
                error_message=str(exc),
            )

        # ─── Weighted monthly returns ───────────────────────────────
        portfolio_monthly: list[float] = []
        for month_index in range(len(periods)):
            month_return_pct = sum(
                normalized_weights[slot.strategy_id] * slot.monthly_profits[month_index]
                for slot in active_slots
            )
            portfolio_monthly.append(month_return_pct)

        # ─── Aggregate profit ───────────────────────────────────────
        monthly_decimals = [value / 100.0 for value in portfolio_monthly]
        growth = math.prod(1.0 + value for value in monthly_decimals)
        total_profit = (growth - 1.0) * 100.0
        total_trades = sum(s.total_trades for s in active_slots)

        # ─── Sharpe & Sortino (monthly → annualised) ───────────────
        try:
            ratios = calculate_periodic_risk_ratios(
                monthly_decimals,
                periods_per_year=MONTHS_PER_YEAR,
                annual_risk_free_rate=risk_free_rate,
            )
        except EquityDataError as exc:
            return PortfolioResult(
                error_code="INVALID_PORTFOLIO_RETURN",
                error_message=str(exc),
            )

        # ─── Max drawdown from monthly equity curve ─────────────────
        max_dd = _max_drawdown_from_periodic_returns(monthly_decimals)

        # ─── Pairwise correlation matrix ────────────────────────────
        corr_matrix = _correlation_matrix(active_slots)

        per_strategy_profits = {s.strategy_id: s.total_profit_pct for s in active_slots}

        return PortfolioResult(
            valid=True,
            total_profit_pct=total_profit,
            sharpe_ratio=ratios.sharpe_ratio,
            sortino_ratio=ratios.sortino_ratio,
            max_drawdown=max_dd,
            total_trades=total_trades,
            strategy_count=len(active_slots),
            correlation_matrix=corr_matrix,
            monthly_profits=portfolio_monthly,
            monthly_periods=periods,
            per_strategy_weights=normalized_weights,
            per_strategy_profits=per_strategy_profits,
            annual_risk_free_rate=ratios.annual_risk_free_rate,
            periodic_risk_free_rate=ratios.periodic_risk_free_rate,
        )

    @staticmethod
    def build_slots_from_individuals(
        individuals: list,
        weights: dict[str, float] | None = None,
    ) -> list[StrategySlot]:
        """
        Build StrategySlot list from evaluated Individual objects.

        Args:
            individuals: list of Individual objects with .metrics populated.
            weights: optional {individual.id: weight} map. Equal weight if None.

        Returns:
            list of StrategySlot ready for evaluate_portfolio.
        """
        slots: list[StrategySlot] = []
        for ind in individuals:
            metrics = ind.metrics or {}
            mp = metrics.get('monthly_profits', [])
            if not mp:
                continue
            w = 1.0
            if weights and ind.id in weights:
                w = weights[ind.id]
            slots.append(StrategySlot(
                strategy_id=ind.id,
                weight=w,
                monthly_profits=list(mp),
                monthly_periods=list(metrics.get('monthly_periods', [])),
                trade_profit_ratios=list(metrics.get('trade_profit_ratios', [])),
                total_profit_pct=metrics.get('profit', 0.0),
                total_trades=metrics.get('num_trades', 0),
            ))
        return slots


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

def _parse_month(value: object, field_name: str) -> tuple[int, int]:
    if not isinstance(value, str) or len(value) != 7 or value[4] != "-":
        raise EquityDataError(f"{field_name} must use YYYY-MM")
    try:
        year = int(value[:4])
        month = int(value[5:])
        date(year, month, 1)
    except (TypeError, ValueError) as exc:
        raise EquityDataError(f"{field_name} must use YYYY-MM") from exc
    return year, month


def _validated_slot_weight(slot: StrategySlot, index: int) -> float:
    if isinstance(slot.weight, bool):
        raise EquityDataError(f"slots[{index}].weight must be numeric")
    try:
        weight = float(slot.weight)
    except (TypeError, ValueError) as exc:
        raise EquityDataError(f"slots[{index}].weight must be numeric") from exc
    if not math.isfinite(weight) or weight < 0:
        raise EquityDataError(f"slots[{index}].weight must be finite and non-negative")
    return weight


def _validated_monthly_profits(slot: StrategySlot, index: int) -> list[float]:
    if len(slot.monthly_profits) < 2:
        raise EquityDataError(
            f"slots[{index}] requires at least two monthly return observations"
        )
    values: list[float] = []
    for month_index, raw_value in enumerate(slot.monthly_profits):
        if isinstance(raw_value, bool):
            raise EquityDataError(
                f"slots[{index}].monthly_profits[{month_index}] must be numeric"
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise EquityDataError(
                f"slots[{index}].monthly_profits[{month_index}] must be numeric"
            ) from exc
        if not math.isfinite(value) or value < -100.0:
            raise EquityDataError(
                f"slots[{index}].monthly_profits[{month_index}] is invalid"
            )
        values.append(value)
    return values


def _validated_monthly_periods(slot: StrategySlot, index: int) -> list[str]:
    if len(slot.monthly_periods) != len(slot.monthly_profits):
        raise EquityDataError(
            f"slots[{index}] requires one dated period per monthly return"
        )
    periods = list(slot.monthly_periods)
    parsed = [
        _parse_month(value, f"slots[{index}].monthly_periods[{month_index}]")
        for month_index, value in enumerate(periods)
    ]
    for previous, current in pairwise(parsed):
        expected = (
            (previous[0] + 1, 1)
            if previous[1] == 12
            else (previous[0], previous[1] + 1)
        )
        if current != expected:
            raise EquityDataError(f"slots[{index}].monthly_periods must be consecutive")
    return periods


def _validated_slot_totals(slot: StrategySlot, index: int) -> tuple[float, int]:
    if isinstance(slot.total_profit_pct, bool):
        raise EquityDataError(f"slots[{index}].total_profit_pct must be numeric")
    try:
        total_profit_pct = float(slot.total_profit_pct)
    except (TypeError, ValueError) as exc:
        raise EquityDataError(f"slots[{index}].total_profit_pct must be numeric") from exc
    if not math.isfinite(total_profit_pct):
        raise EquityDataError(f"slots[{index}].total_profit_pct must be finite")
    if (
        not isinstance(slot.total_trades, int)
        or isinstance(slot.total_trades, bool)
        or slot.total_trades < 0
    ):
        raise EquityDataError(
            f"slots[{index}].total_trades must be a non-negative integer"
        )
    return total_profit_pct, slot.total_trades


def _validate_portfolio_inputs(
    slots: Sequence[StrategySlot],
) -> tuple[list[StrategySlot], dict[str, float], list[str]]:
    if not slots:
        raise EquityDataError("portfolio requires at least one strategy")

    active: list[StrategySlot] = []
    raw_weights: dict[str, float] = {}
    seen_ids: set[str] = set()
    reference_periods: list[str] | None = None

    for index, slot in enumerate(slots):
        if not isinstance(slot.strategy_id, str) or not slot.strategy_id:
            raise EquityDataError(f"slots[{index}].strategy_id must be non-empty")
        if slot.strategy_id in seen_ids:
            raise EquityDataError(f"duplicate strategy_id: {slot.strategy_id}")
        seen_ids.add(slot.strategy_id)

        weight = _validated_slot_weight(slot, index)
        if weight == 0:
            continue

        values = _validated_monthly_profits(slot, index)
        slot_periods = _validated_monthly_periods(slot, index)
        total_profit_pct, total_trades = _validated_slot_totals(slot, index)
        if reference_periods is None:
            reference_periods = slot_periods
        elif slot_periods != reference_periods:
            raise EquityDataError("strategy monthly periods are not calendar-aligned")

        active.append(
            StrategySlot(
                strategy_id=slot.strategy_id,
                weight=weight,
                monthly_profits=values,
                monthly_periods=slot_periods,
                trade_profit_ratios=list(slot.trade_profit_ratios),
                total_profit_pct=total_profit_pct,
                total_trades=total_trades,
            )
        )
        raw_weights[slot.strategy_id] = weight

    total_weight = sum(raw_weights.values())
    if total_weight <= 0 or not active:
        raise EquityDataError("portfolio requires at least one positive strategy weight")
    normalized = {
        strategy_id: weight / total_weight for strategy_id, weight in raw_weights.items()
    }
    return active, normalized, list(reference_periods or [])


def _annualised_sharpe(
    monthly_returns: list[float],
    risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE,
) -> float | None:
    """Compatibility wrapper for percentage-point monthly returns."""

    return calculate_periodic_risk_ratios(
        [value / 100.0 for value in monthly_returns],
        periods_per_year=MONTHS_PER_YEAR,
        annual_risk_free_rate=risk_free_rate,
    ).sharpe_ratio


def _annualised_sortino(
    monthly_returns: list[float],
    risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE,
) -> float | None:
    """Compatibility wrapper for percentage-point monthly returns."""

    return calculate_periodic_risk_ratios(
        [value / 100.0 for value in monthly_returns],
        periods_per_year=MONTHS_PER_YEAR,
        annual_risk_free_rate=risk_free_rate,
    ).sortino_ratio


def _max_drawdown_from_periodic_returns(periodic_returns: Sequence[float]) -> float:
    """Compute peak-to-trough drawdown from decimal periodic returns."""

    if not periodic_returns:
        return 0.0
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in periodic_returns:
        cumulative *= 1.0 + value
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _max_drawdown_from_monthly(monthly_returns: list[float]) -> float:
    """Compatibility wrapper for percentage-point monthly returns."""

    return _max_drawdown_from_periodic_returns([value / 100.0 for value in monthly_returns])


def _pearson_corr(a: list[float], b: list[float]) -> float | None:
    """Pearson correlation between two equal-length series."""
    if len(a) != len(b):
        return None
    n = len(a)
    if n < 2:
        return None
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    denom = (var_a * var_b) ** 0.5
    if denom < 1e-15:
        return None
    return cov / denom


def _correlation_matrix(
    slots: list[StrategySlot],
) -> dict[str, dict[str, float | None]]:
    """Build pairwise correlation matrix from monthly profit vectors."""
    matrix: dict[str, dict[str, float | None]] = {}
    for i, si in enumerate(slots):
        row: dict[str, float | None] = {}
        for j, sj in enumerate(slots):
            if i == j:
                row[sj.strategy_id] = 1.0
            else:
                row[sj.strategy_id] = _pearson_corr(si.monthly_profits, sj.monthly_profits)
        matrix[si.strategy_id] = row
    return matrix
