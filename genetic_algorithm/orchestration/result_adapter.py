"""Adapters from measured backtests into strict V2 shadow records."""

from __future__ import annotations

import math
from datetime import date, datetime

from pydantic import Field

from genetic_algorithm.evaluation.activity_metrics import count_active_trade_months
from genetic_algorithm.evaluation.confidence_metrics_v2 import (
    InsufficientTradeEvidenceError,
    clustered_trade_expectancy_lcb,
)
from genetic_algorithm.evaluation.direct_backtester import BacktestResult
from genetic_algorithm.evaluation.equity_metrics_v2 import (
    MARK_TO_MARKET_EQUITY,
    REALIZED_CLOSE_EQUITY,
    EquityDataError,
    moving_block_bootstrap_bounds,
)
from genetic_algorithm.orchestration.result_contract import (
    BacktestRecordV2,
    EvaluationStatus,
    ScenarioMetricsV2,
    ScenarioRole,
    StrictV2Model,
)


class BacktestContextV2(StrictV2Model):
    """Immutable provenance supplied by a runner/replay, never inferred."""

    attempt_id: str = Field(min_length=1)
    wave_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    config_hash: str = Field(min_length=8)
    phenotype_hash: str = Field(min_length=8)
    code_version: str = Field(min_length=1)
    data_manifest_hash: str = Field(min_length=8)
    fitness_policy_version: str = Field(min_length=1)
    seed: int
    worker_count: int = Field(ge=1)

    scenario_id: str = Field(min_length=1)
    pair: str = Field(min_length=3)
    timeframe: str = Field(min_length=1)
    role: ScenarioRole
    period_start: date
    period_end: date
    cost_multiplier: float = Field(gt=0)
    fee_rate: float = Field(ge=0)
    slippage_rate: float = Field(ge=0)
    spread_rate: float = Field(ge=0)
    funding_rate: float = Field(ge=0)


def _active_months(result: BacktestResult) -> int:
    return count_active_trade_months(
        result.trades,
        daily_profit_abs=result.daily_profit_abs,
    )


def _result_period(result: BacktestResult) -> tuple[date, date] | None:
    if result.backtest_start is None or result.backtest_end is None:
        return None

    def parse(value: str) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])

    try:
        return parse(result.backtest_start), parse(result.backtest_end)
    except (TypeError, ValueError):
        return None


def adapt_shadow_backtest_result(
    result: BacktestResult,
    context: BacktestContextV2,
    *,
    bootstrap_samples: int = 1000,
    bootstrap_block_days: int = 10,
    trade_bootstrap_block: int = 5,
    expectancy_cluster_days: int = 1,
    bootstrap_confidence: float = 0.95,
) -> BacktestRecordV2:
    """Create a versioned shadow record without granting premature validity.

    Daily and trade confidence bounds are deterministic for the manifest seed.
    A scenario becomes ``VALID`` only with reconciled mark-to-market equity,
    sufficient daily/trade evidence, and all required tail measurements.
    """

    bounds = None
    expectancy = None
    error_detail = None
    if not result.success:
        status = EvaluationStatus.FAIL
        error_code = "BACKTEST_FAILED"
        error_detail = result.error_message
    elif result.no_trades or result.total_trades == 0:
        status = EvaluationStatus.INCONCLUSIVE
        error_code = "NO_TRADES"
    elif not result.daily_net_returns or not result.equity_curve or not result.equity_method:
        status = EvaluationStatus.INVALID
        error_code = "MISSING_EQUITY_EVIDENCE"
        error_detail = result.equity_error_message
    elif _result_period(result) is None:
        status = EvaluationStatus.INVALID
        error_code = "MISSING_PERIOD_EVIDENCE"
        error_detail = "backtest result has no parseable start/end period"
    elif _result_period(result) != (context.period_start, context.period_end):
        status = EvaluationStatus.INVALID
        error_code = "PERIOD_COVERAGE_MISMATCH"
        error_detail = (
            f"measured={_result_period(result)}, "
            f"declared={(context.period_start, context.period_end)}"
        )
    else:
        try:
            bounds = moving_block_bootstrap_bounds(
                result.daily_net_returns,
                samples=bootstrap_samples,
                block_length_days=bootstrap_block_days,
                confidence=bootstrap_confidence,
                seed=context.seed,
            )
        except EquityDataError as exc:
            status = EvaluationStatus.INCONCLUSIVE
            error_code = "INSUFFICIENT_BOOTSTRAP_EVIDENCE"
            error_detail = str(exc)
        else:
            if result.equity_method == REALIZED_CLOSE_EQUITY:
                status = EvaluationStatus.INCONCLUSIVE
                error_code = "MARK_TO_MARKET_REPLAY_REQUIRED"
                error_detail = result.mark_to_market_error_message
            else:
                try:
                    expectancy = clustered_trade_expectancy_lcb(
                        result.trades or [],
                        samples=bootstrap_samples,
                        cluster_days=expectancy_cluster_days,
                        block_length_clusters=trade_bootstrap_block,
                        confidence=bootstrap_confidence,
                        seed=context.seed,
                    )
                except InsufficientTradeEvidenceError as exc:
                    status = EvaluationStatus.INCONCLUSIVE
                    error_code = "INSUFFICIENT_TRADE_EVIDENCE"
                    error_detail = str(exc)
                except EquityDataError as exc:
                    status = EvaluationStatus.INVALID
                    error_code = "INVALID_TRADE_EVIDENCE"
                    error_detail = str(exc)
                else:
                    if expectancy.observations != result.total_trades:
                        status = EvaluationStatus.INVALID
                        error_code = "TRADE_COUNT_MISMATCH"
                        error_detail = (
                            f"measured={expectancy.observations}, "
                            f"reported={result.total_trades}"
                        )
                    elif not math.isclose(
                        expectancy.total_net_profit_abs,
                        result.total_profit,
                        rel_tol=1e-8,
                        abs_tol=max(1e-7, abs(result.total_profit) * 1e-8),
                    ):
                        status = EvaluationStatus.INVALID
                        error_code = "TRADE_PROFIT_MISMATCH"
                        error_detail = (
                            f"trade_ledger={expectancy.total_net_profit_abs:.12g}, "
                            f"reported={result.total_profit:.12g}"
                        )
                    elif (
                        result.starting_balance is not None
                        and result.final_balance is not None
                        and not math.isclose(
                            expectancy.total_net_profit_abs,
                            result.final_balance - result.starting_balance,
                            rel_tol=1e-8,
                            abs_tol=max(
                                1e-7,
                                abs(result.final_balance) * 1e-8,
                            ),
                        )
                    ):
                        status = EvaluationStatus.INVALID
                        error_code = "TRADE_WALLET_PROFIT_MISMATCH"
                        error_detail = (
                            f"trade_ledger={expectancy.total_net_profit_abs:.12g}, "
                            "wallet_delta="
                            f"{result.final_balance - result.starting_balance:.12g}"
                        )
                    elif (
                        result.max_consecutive_losses is None
                        or result.max_drawdown_duration_days is None
                    ):
                        status = EvaluationStatus.INCONCLUSIVE
                        error_code = "MISSING_TAIL_METRICS"
                    else:
                        status = EvaluationStatus.VALID
                        error_code = None

    net_return = None
    if result.starting_balance and result.final_balance is not None:
        net_return = result.final_balance / result.starting_balance - 1.0
    elif result.success:
        net_return = result.profit_percent / 100.0

    metrics = ScenarioMetricsV2(
        scenario_id=context.scenario_id,
        pair=context.pair,
        timeframe=context.timeframe,
        role=context.role,
        period_start=context.period_start,
        period_end=context.period_end,
        cost_multiplier=context.cost_multiplier,
        status=status,
        success=result.success,
        no_trades=result.no_trades or result.total_trades == 0,
        error_code=error_code,
        error_detail=error_detail,
        net_return=net_return,
        annualized_net_return=result.annualized_net_return,
        annualized_net_return_lcb=(
            bounds.annualized_net_return_lcb if bounds is not None else None
        ),
        max_drawdown=result.daily_max_drawdown,
        max_drawdown_ucb=bounds.max_drawdown_ucb if bounds is not None else None,
        daily_expected_shortfall_5=result.daily_expected_shortfall_5,
        daily_sharpe_ratio=result.daily_sharpe_ratio,
        daily_sortino_ratio=result.daily_sortino_ratio,
        calmar_ratio=result.calmar_ratio,
        daily_expected_shortfall_5_ucb=(
            bounds.daily_expected_shortfall_5_ucb if bounds is not None else None
        ),
        net_expectancy=(
            expectancy.mean_trade_return
            if expectancy is not None
            else (result.avg_profit if result.total_trades > 0 else None)
        ),
        net_expectancy_lcb=(
            expectancy.mean_trade_return_lcb if expectancy is not None else None
        ),
        expectancy_contract_version=(
            expectancy.expectancy_contract_version if expectancy is not None else None
        ),
        expectancy_quote_currency=(
            expectancy.quote_currency if expectancy is not None else None
        ),
        net_expectancy_on_committed_capital=(
            expectancy.return_on_committed_capital if expectancy is not None else None
        ),
        net_expectancy_on_committed_capital_lcb=(
            expectancy.return_on_committed_capital_lcb
            if expectancy is not None
            else None
        ),
        mean_net_profit_abs_per_trade=(
            expectancy.mean_net_profit_abs_per_trade if expectancy is not None else None
        ),
        total_net_profit_abs=(
            expectancy.total_net_profit_abs if expectancy is not None else None
        ),
        total_committed_capital=(
            expectancy.total_committed_capital if expectancy is not None else None
        ),
        expectancy_temporal_clusters=(
            expectancy.temporal_clusters if expectancy is not None else None
        ),
        expectancy_cluster_days=(
            expectancy.cluster_days if expectancy is not None else None
        ),
        expectancy_pair_count=(
            expectancy.pair_count if expectancy is not None else None
        ),
        expectancy_effective_pair_count=(
            expectancy.effective_pair_count if expectancy is not None else None
        ),
        expectancy_pair_capital_hhi=(
            expectancy.pair_capital_hhi if expectancy is not None else None
        ),
        expectancy_max_trade_capital_share=(
            expectancy.max_trade_capital_share if expectancy is not None else None
        ),
        expectancy_max_cluster_capital_share=(
            expectancy.max_cluster_capital_share if expectancy is not None else None
        ),
        expectancy_serial_effective_sample_size=(
            expectancy.serial_effective_sample_size if expectancy is not None else None
        ),
        expectancy_capital_effective_sample_size=(
            expectancy.capital_effective_sample_size if expectancy is not None else None
        ),
        expectancy_temporal_effective_sample_size=(
            expectancy.temporal_effective_sample_size
            if expectancy is not None
            else None
        ),
        profit_factor=result.profit_factor if result.total_trades > 0 else None,
        profit_factor_censored=(
            result.profit_factor_censored if result.total_trades > 0 else None
        ),
        profit_factor_contract_version=(
            result.profit_factor_contract_version
            if result.total_trades > 0
            else None
        ),
        win_rate=result.win_rate if result.total_trades > 0 else None,
        trade_count=max(0, result.total_trades),
        effective_sample_size=(
            expectancy.effective_sample_size if expectancy is not None else None
        ),
        active_months=_active_months(result),
        max_consecutive_losses=result.max_consecutive_losses,
        max_drawdown_duration_days=result.max_drawdown_duration_days,
    )

    equity_method = (
        result.equity_method
        if result.equity_method in {REALIZED_CLOSE_EQUITY, MARK_TO_MARKET_EQUITY}
        else None
    )
    return BacktestRecordV2(
        attempt_id=context.attempt_id,
        wave_id=context.wave_id,
        experiment_id=context.experiment_id,
        candidate_id=context.candidate_id,
        config_hash=context.config_hash,
        phenotype_hash=context.phenotype_hash,
        code_version=context.code_version,
        data_manifest_hash=context.data_manifest_hash,
        fitness_policy_version=context.fitness_policy_version,
        seed=context.seed,
        worker_count=context.worker_count,
        fee_rate=context.fee_rate,
        slippage_rate=context.slippage_rate,
        spread_rate=context.spread_rate,
        funding_rate=context.funding_rate,
        risk_metric_contract_version=result.risk_metric_contract_version,
        risk_periods_per_year=result.risk_periods_per_year,
        annual_risk_free_rate=result.annual_risk_free_rate,
        periodic_risk_free_rate=result.periodic_risk_free_rate,
        equity_method=equity_method,
        metrics=metrics,
        daily_net_returns=list(result.daily_net_returns or []),
        equity_curve=list(result.equity_curve or []),
        trades=list(result.trades or []),
    )
