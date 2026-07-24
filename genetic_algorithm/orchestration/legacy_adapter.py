"""Fail-closed adapters from legacy GA outputs into V2 contracts.

Legacy results are useful evidence, but they lack calendar-aligned equity and
the full provenance required by the V2 promotion policy.  Successful imports
therefore remain ``INCONCLUSIVE`` until a common-panel V2 replay replaces them.
"""

from __future__ import annotations

from genetic_algorithm.evaluation.direct_backtester import BacktestResult
from genetic_algorithm.orchestration.result_adapter import BacktestContextV2
from genetic_algorithm.orchestration.result_contract import (
    BacktestRecordV2,
    EvaluationStatus,
    ScenarioMetricsV2,
)


class LegacyBacktestContextV2(BacktestContextV2):
    """Named compatibility context for explicit legacy imports."""


def adapt_legacy_backtest_result(
    result: BacktestResult,
    context: LegacyBacktestContextV2,
) -> BacktestRecordV2:
    """Preserve legacy measurements without claiming V2 validity.

    Even a successful legacy result is ``INCONCLUSIVE`` because it cannot
    provide daily portfolio returns, a capital-weighted equity curve, uncertainty
    bounds, or an effective sample size.  A failed legacy backtest remains
    ``FAIL``.  The only path to ``VALID`` is a real V2 replay.
    """

    if not result.success:
        status = EvaluationStatus.FAIL
        error_code = "LEGACY_BACKTEST_FAILED"
    elif result.no_trades or result.total_trades == 0:
        status = EvaluationStatus.INCONCLUSIVE
        error_code = "LEGACY_NO_TRADES"
    else:
        status = EvaluationStatus.INCONCLUSIVE
        error_code = "LEGACY_REPLAY_REQUIRED"

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
        net_return=(result.profit_percent / 100.0) if result.success else None,
        max_drawdown=result.max_drawdown if result.success else None,
        net_expectancy=result.avg_profit if result.total_trades > 0 else None,
        profit_factor=result.profit_factor if result.total_trades > 0 else None,
        win_rate=result.win_rate if result.total_trades > 0 else None,
        trade_count=max(0, result.total_trades),
        active_months=0,
        max_consecutive_losses=result.max_consecutive_losses,
        max_drawdown_duration_days=(
            result.max_drawdown_duration_days if result.total_trades > 0 else None
        ),
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
        metrics=metrics,
        # Missing V2 raw evidence is intentionally represented by empty arrays.
        daily_net_returns=[],
        equity_curve=[],
        trades=result.trades or [],
    )
