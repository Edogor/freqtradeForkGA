"""Independent references for the versioned Sharpe/Sortino/Calmar contract."""

from __future__ import annotations

import math
from datetime import date
from statistics import fmean, stdev

import pytest

from genetic_algorithm.config.schema import resolve_config_data
from genetic_algorithm.evaluation.direct_backtester import (
    _monthly_returns_from_daily_equity,
)
from genetic_algorithm.evaluation.equity_metrics_v2 import (
    CALENDAR_DAYS_PER_YEAR,
    MONTHS_PER_YEAR,
    RISK_METRIC_CONTRACT_VERSION,
    DailyEquitySeriesV2,
    EquityDataError,
    build_realized_close_equity,
    calculate_equity_risk_metrics,
    calculate_periodic_risk_ratios,
)
from genetic_algorithm.evaluation.portfolio_backtester import (
    PortfolioBacktester,
    StrategySlot,
    _annualised_sortino,
)


def test_daily_sharpe_and_sortino_match_independent_effective_rate_reference():
    returns = (0.012, -0.007, 0.003, -0.002, 0.009)
    annual_risk_free_rate = 0.05
    daily_target = math.expm1(
        math.log1p(annual_risk_free_rate) / CALENDAR_DAYS_PER_YEAR
    )
    excess = [value - daily_target for value in returns]
    expected_sharpe = (
        fmean(excess) / stdev(returns) * math.sqrt(CALENDAR_DAYS_PER_YEAR)
    )
    expected_downside = math.sqrt(
        sum(min(value, 0.0) ** 2 for value in excess) / len(excess)
    )
    expected_sortino = (
        fmean(excess) / expected_downside * math.sqrt(CALENDAR_DAYS_PER_YEAR)
    )

    actual = calculate_periodic_risk_ratios(
        returns,
        periods_per_year=CALENDAR_DAYS_PER_YEAR,
        annual_risk_free_rate=annual_risk_free_rate,
    )

    assert actual.risk_metric_contract_version == RISK_METRIC_CONTRACT_VERSION
    assert actual.periodic_risk_free_rate == pytest.approx(daily_target)
    assert actual.periodic_return_volatility == pytest.approx(stdev(returns))
    assert actual.periodic_downside_deviation == pytest.approx(expected_downside)
    assert actual.sharpe_ratio == pytest.approx(expected_sharpe)
    assert actual.sortino_ratio == pytest.approx(expected_sortino)


def test_sortino_uses_all_observations_and_never_invents_no_downside_score():
    returns_pct = [1.0, 2.0, 3.0, 4.0]

    ratios = calculate_periodic_risk_ratios(
        [value / 100.0 for value in returns_pct],
        periods_per_year=MONTHS_PER_YEAR,
    )

    assert ratios.sharpe_ratio is not None
    assert ratios.sortino_ratio is None
    assert _annualised_sortino(returns_pct) is None


@pytest.mark.parametrize("annual_rate", [-1.0, -1.01, math.inf, math.nan, True])
def test_invalid_annual_risk_free_rates_fail_closed(annual_rate):
    with pytest.raises(EquityDataError, match="annual_risk_free_rate"):
        calculate_periodic_risk_ratios(
            [0.01, -0.01],
            periods_per_year=CALENDAR_DAYS_PER_YEAR,
            annual_risk_free_rate=annual_rate,
        )


def test_calmar_uses_geometric_calendar_cagr_and_peak_to_trough_drawdown():
    series = build_realized_close_equity(
        period_start="2025-01-01",
        period_end="2025-01-03",
        starting_balance=100.0,
        daily_profit_abs=[
            ("2025-01-01", 10.0),
            ("2025-01-02", -11.0),
            ("2025-01-03", 9.9),
        ],
        expected_final_balance=108.9,
    )
    metrics = calculate_equity_risk_metrics(series)
    expected_cagr = math.expm1(math.log(1.089) * CALENDAR_DAYS_PER_YEAR / 3)

    assert metrics.max_drawdown == pytest.approx(0.10)
    assert metrics.annualized_net_return == pytest.approx(expected_cagr)
    assert metrics.calmar_ratio == pytest.approx(expected_cagr / 0.10)


def test_terminal_total_loss_has_negative_one_cagr_and_calmar():
    series = build_realized_close_equity(
        period_start="2025-01-01",
        period_end="2025-01-01",
        starting_balance=100.0,
        daily_profit_abs=[("2025-01-01", -100.0)],
        expected_final_balance=0.0,
    )

    metrics = calculate_equity_risk_metrics(series)

    assert metrics.annualized_net_return == -1.0
    assert metrics.max_drawdown == 1.0
    assert metrics.calmar_ratio == -1.0


def test_point_metrics_reject_an_internally_inconsistent_equity_series():
    inconsistent = DailyEquitySeriesV2(
        dates=(date(2025, 1, 1), date(2025, 1, 2)),
        daily_profit_abs=(10.0, 0.0),
        daily_net_returns=(0.10, 0.25),
        equity_curve=(100.0, 110.0, 110.0),
        starting_balance=100.0,
        final_balance=110.0,
    )

    with pytest.raises(EquityDataError, match="daily return does not reconcile"):
        calculate_equity_risk_metrics(inconsistent)


def test_monthly_returns_compound_daily_wallet_returns_and_keep_calendar_labels():
    periods, returns_pct = _monthly_returns_from_daily_equity(
        [
            ["2025-01-30", 10.0],
            ["2025-01-31", -11.0],
            ["2025-02-01", 4.95],
        ],
        [0.10, -0.10, 0.05],
    )

    assert periods == ["2025-01", "2025-02"]
    assert returns_pct == pytest.approx([-1.0, 5.0])


def test_direct_parser_applies_and_persists_configured_risk_free_rate():
    from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

    parser = object.__new__(DirectBacktester)
    parser.config = {"evaluation_v2": {"annual_risk_free_rate": 0.05}}
    result = parser._parse_stats(
        {
            "profit_total": 0.01,
            "profit_total_abs": 1.0,
            "profit_total_pct": 1.0,
            "total_trades": 2,
            "wins": 1,
            "losses": 1,
            "backtest_start": "2025-01-01",
            "backtest_end": "2025-01-03",
            "starting_balance": 100.0,
            "final_balance": 101.0,
            "daily_profit": [["2025-01-01", 2.0], ["2025-01-02", -1.0]],
        },
        "ConfiguredRiskFree",
    )
    expected_daily_rate = math.expm1(math.log1p(0.05) / CALENDAR_DAYS_PER_YEAR)

    assert result.risk_metric_contract_version == RISK_METRIC_CONTRACT_VERSION
    assert result.risk_periods_per_year == CALENDAR_DAYS_PER_YEAR
    assert result.annual_risk_free_rate == 0.05
    assert result.periodic_risk_free_rate == pytest.approx(expected_daily_rate)
    assert result.to_dict()["annual_risk_free_rate"] == 0.05


def test_portfolio_uses_dated_aligned_returns_without_mutating_weights():
    first = StrategySlot(
        strategy_id="first",
        weight=1.0,
        monthly_periods=["2025-01", "2025-02", "2025-03"],
        monthly_profits=[10.0, -10.0, 5.0],
        total_profit_pct=3.95,
        total_trades=12,
    )
    second = StrategySlot(
        strategy_id="second",
        weight=3.0,
        monthly_periods=["2025-01", "2025-02", "2025-03"],
        monthly_profits=[0.0, 10.0, -5.0],
        total_profit_pct=4.5,
        total_trades=20,
    )

    result = PortfolioBacktester.evaluate_portfolio(
        [first, second],
        risk_free_rate=0.05,
    )

    expected_monthly_pct = [2.5, 5.0, -2.5]
    expected_total_pct = (
        math.prod(1.0 + value / 100.0 for value in expected_monthly_pct) - 1.0
    ) * 100.0
    monthly_target = math.expm1(math.log1p(0.05) / MONTHS_PER_YEAR)
    excess = [value / 100.0 - monthly_target for value in expected_monthly_pct]
    expected_sharpe = (
        fmean(excess)
        / stdev(value / 100.0 for value in expected_monthly_pct)
        * math.sqrt(MONTHS_PER_YEAR)
    )

    assert result.valid is True
    assert result.monthly_periods == ["2025-01", "2025-02", "2025-03"]
    assert result.monthly_profits == pytest.approx(expected_monthly_pct)
    assert result.total_profit_pct == pytest.approx(expected_total_pct)
    assert result.sharpe_ratio == pytest.approx(expected_sharpe)
    assert result.per_strategy_weights == {"first": 0.25, "second": 0.75}
    assert first.weight == 1.0
    assert second.weight == 3.0


def test_constant_portfolio_returns_keep_ratios_and_correlation_undefined():
    periods = ["2025-01", "2025-02", "2025-03"]
    result = PortfolioBacktester.evaluate_portfolio(
        [
            StrategySlot(
                strategy_id="first",
                weight=1.0,
                monthly_periods=periods,
                monthly_profits=[1.0, 1.0, 1.0],
            ),
            StrategySlot(
                strategy_id="second",
                weight=1.0,
                monthly_periods=periods,
                monthly_profits=[2.0, 2.0, 2.0],
            ),
        ]
    )

    assert result.valid is True
    assert result.sharpe_ratio is None
    assert result.sortino_ratio is None
    assert result.correlation_matrix["first"]["second"] is None


@pytest.mark.parametrize(
    "second_periods",
    [
        [],
        ["2025-02", "2025-03"],
        ["2025-01", "2025-03"],
    ],
)
def test_portfolio_rejects_missing_shifted_or_gapped_month_evidence(second_periods):
    slots = [
        StrategySlot(
            strategy_id="first",
            weight=1.0,
            monthly_periods=["2025-01", "2025-02"],
            monthly_profits=[1.0, -1.0],
        ),
        StrategySlot(
            strategy_id="second",
            weight=1.0,
            monthly_periods=second_periods,
            monthly_profits=[2.0, -2.0],
        ),
    ]

    result = PortfolioBacktester.evaluate_portfolio(slots)

    assert result.valid is False
    assert result.error_code == "INVALID_PORTFOLIO_EVIDENCE"
    assert result.sharpe_ratio is None
    assert result.sortino_ratio is None


@pytest.mark.parametrize("annual_rate", [-1.0, math.inf, math.nan, True])
def test_v2_config_rejects_invalid_risk_free_convention(annual_rate):
    resolution = resolve_config_data(
        {
            "config_schema_version": 2,
            "evaluation_v2": {"annual_risk_free_rate": annual_rate},
        }
    )

    assert any("annual_risk_free_rate" in error for error in resolution.errors)
