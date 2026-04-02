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

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    """Aggregated results for a portfolio of strategies."""
    total_profit_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    strategy_count: int = 0
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    monthly_profits: Optional[List[float]] = None
    per_strategy_weights: Optional[Dict[str, float]] = None
    per_strategy_profits: Optional[Dict[str, float]] = None


@dataclass
class StrategySlot:
    """One strategy in a portfolio with its capital weight and results."""
    strategy_id: str
    weight: float  # capital allocation fraction (sums to 1)
    monthly_profits: List[float] = field(default_factory=list)
    trade_profit_ratios: List[float] = field(default_factory=list)
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
        slots: List[StrategySlot],
        risk_free_rate: float = 0.0,
    ) -> PortfolioResult:
        """
        Evaluate a portfolio of strategy slots.

        Args:
            slots: list of StrategySlot with weights and per-strategy results.
            risk_free_rate: annual risk-free rate for Sharpe/Sortino (default 0).

        Returns:
            PortfolioResult with aggregate metrics.
        """
        if not slots:
            return PortfolioResult()

        # Normalise weights to sum to 1
        total_weight = sum(s.weight for s in slots)
        if total_weight <= 0:
            return PortfolioResult()
        for s in slots:
            s.weight /= total_weight

        # ─── Weighted monthly returns ───────────────────────────────
        max_months = max((len(s.monthly_profits) for s in slots), default=0)
        portfolio_monthly: List[float] = []
        for m in range(max_months):
            month_ret = 0.0
            for s in slots:
                if m < len(s.monthly_profits):
                    month_ret += s.weight * s.monthly_profits[m]
            portfolio_monthly.append(month_ret)

        # ─── Aggregate profit ───────────────────────────────────────
        total_profit = sum(s.weight * s.total_profit_pct for s in slots)
        total_trades = sum(s.total_trades for s in slots)

        # ─── Sharpe & Sortino (monthly → annualised) ───────────────
        sharpe = _annualised_sharpe(portfolio_monthly, risk_free_rate)
        sortino = _annualised_sortino(portfolio_monthly, risk_free_rate)

        # ─── Max drawdown from monthly equity curve ─────────────────
        max_dd = _max_drawdown_from_monthly(portfolio_monthly)

        # ─── Pairwise correlation matrix ────────────────────────────
        corr_matrix = _correlation_matrix(slots)

        per_strategy_weights = {s.strategy_id: s.weight for s in slots}
        per_strategy_profits = {s.strategy_id: s.total_profit_pct for s in slots}

        return PortfolioResult(
            total_profit_pct=total_profit,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            total_trades=total_trades,
            strategy_count=len(slots),
            correlation_matrix=corr_matrix,
            monthly_profits=portfolio_monthly,
            per_strategy_weights=per_strategy_weights,
            per_strategy_profits=per_strategy_profits,
        )

    @staticmethod
    def build_slots_from_individuals(
        individuals: list,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[StrategySlot]:
        """
        Build StrategySlot list from evaluated Individual objects.

        Args:
            individuals: list of Individual objects with .metrics populated.
            weights: optional {individual.id: weight} map. Equal weight if None.

        Returns:
            list of StrategySlot ready for evaluate_portfolio.
        """
        slots: List[StrategySlot] = []
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
                trade_profit_ratios=list(metrics.get('trade_profit_ratios', [])),
                total_profit_pct=metrics.get('profit', 0.0),
                total_trades=metrics.get('num_trades', 0),
            ))
        return slots


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

def _annualised_sharpe(monthly_returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio from monthly returns."""
    if len(monthly_returns) < 2:
        return 0.0
    monthly_rf = risk_free_rate / 12.0
    excess = [r - monthly_rf for r in monthly_returns]
    mean_e = sum(excess) / len(excess)
    var = sum((x - mean_e) ** 2 for x in excess) / (len(excess) - 1)
    std = var ** 0.5
    if std < 1e-12:
        return 0.0
    return (mean_e / std) * (12 ** 0.5)


def _annualised_sortino(monthly_returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino ratio from monthly returns."""
    if len(monthly_returns) < 2:
        return 0.0
    monthly_rf = risk_free_rate / 12.0
    excess = [r - monthly_rf for r in monthly_returns]
    mean_e = sum(excess) / len(excess)
    downside = [x for x in excess if x < 0]
    if not downside:
        return 0.0 if mean_e <= 0 else 10.0  # cap at 10 if no downside
    dd_var = sum(x ** 2 for x in downside) / len(downside)
    dd_std = dd_var ** 0.5
    if dd_std < 1e-12:
        return 0.0
    return (mean_e / dd_std) * (12 ** 0.5)


def _max_drawdown_from_monthly(monthly_returns: List[float]) -> float:
    """Compute max drawdown from a series of monthly return percentages."""
    if not monthly_returns:
        return 0.0
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in monthly_returns:
        cumulative *= (1.0 + r / 100.0)
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _pearson_corr(a: List[float], b: List[float]) -> float:
    """Pearson correlation between two equal-length series."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a = a[:n]
    b = b[:n]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    denom = (var_a * var_b) ** 0.5
    if denom < 1e-12:
        return 0.0
    return cov / denom


def _correlation_matrix(slots: List[StrategySlot]) -> Dict[str, Dict[str, float]]:
    """Build pairwise correlation matrix from monthly profit vectors."""
    matrix: Dict[str, Dict[str, float]] = {}
    for i, si in enumerate(slots):
        row: Dict[str, float] = {}
        for j, sj in enumerate(slots):
            if i == j:
                row[sj.strategy_id] = 1.0
            else:
                row[sj.strategy_id] = _pearson_corr(si.monthly_profits, sj.monthly_profits)
        matrix[si.strategy_id] = row
    return matrix
