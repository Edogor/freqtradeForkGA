"""Fail-closed contract tests for regime-aware segment aggregation."""

from datetime import datetime, timedelta

import pytest

from genetic_algorithm.evaluation.regime_aware import (
    RegimeAwareEvaluator,
    RegimeEvaluationResult,
)
from genetic_algorithm.utils.regime_detector import RegimeSegment, RegimeType


class _StrategyGeneStub:
    preferred_regime = None
    regime_mode = "generalist"

    @staticmethod
    def calculate_complexity() -> float:
        return 1.0


def _config(**regime_overrides):
    regime_config = {
        "enabled": True,
        "aggregation": "harmonic_mean",
        "confidence_weighting": False,
        "min_segment_trades": 3,
    }
    regime_config.update(regime_overrides)
    return {
        "backtesting": {
            "pairs": ["BTC/USDT"],
            "timeframe": "1h",
            "timerange": "20230101-20230401",
        },
        "regime_aware": regime_config,
    }


def _segment(index: int, regime: RegimeType = RegimeType.BULLISH) -> RegimeSegment:
    start = datetime(2023, 1, 1) + timedelta(days=index * 10)
    return RegimeSegment(
        segment_id=f"segment-{index}",
        start_date=start,
        end_date=start + timedelta(days=9),
        regime=regime,
        confidence=1.0,
        role="optimization",
    )


def _result(
    index: int,
    fitness: float,
    trades: int,
    *,
    success: bool = True,
    regime: RegimeType = RegimeType.BULLISH,
) -> RegimeEvaluationResult:
    return RegimeEvaluationResult(
        segment=_segment(index, regime),
        fitness=fitness,
        metrics={
            "profit": fitness * 10,
            "sharpe_ratio": fitness,
            "sortino_ratio": fitness,
            "profit_factor": max(0.0, fitness),
            "max_drawdown": 0.1,
            "win_rate": 0.5,
            "num_trades": trades,
        },
        success=success,
        error_message=None if success else "backtest failed",
    )


def test_harmonic_mean_does_not_remove_zero_trade_segment():
    evaluator = RegimeAwareEvaluator(_config(), segments={})
    results = [
        _result(0, 0.8, 5),
        _result(1, 0.5, 4, regime=RegimeType.BEARISH),
        _result(2, 0.9, 0, regime=RegimeType.SIDEWAYS),
    ]

    fitness, metrics = evaluator._aggregate_results(results, _StrategyGeneStub())

    assert fitness == 0.0
    assert metrics["zero_trade_segment_count"] == 1
    assert metrics["eligible_segment_count"] == 2
    assert metrics["segment_evidence_rate"] == pytest.approx(2 / 3)
    assert metrics["segment_coverage_complete"] is False
    assert metrics["segment_effective_fitness_values"] == [0.8, 0.5, 0.0]
    assert metrics["max_drawdown"] == 1.0


def test_failed_segment_remains_in_weighted_mean_and_metric_denominator():
    evaluator = RegimeAwareEvaluator(
        _config(aggregation="mean", min_segment_trades=1), segments={}
    )
    results = [_result(0, 0.8, 5), _result(1, 0.0, 0, success=False)]

    fitness, metrics = evaluator._aggregate_results(results, _StrategyGeneStub())

    assert fitness == pytest.approx(0.4)
    assert metrics["profit"] == pytest.approx(4.0)
    assert metrics["failed_segment_count"] == 1
    assert metrics["successful_segment_count"] == 1
    assert metrics["expected_segment_count"] == 2
    assert metrics["segment_success_rate"] == 0.5
    assert [outcome["segment_id"] for outcome in metrics["segment_outcomes"]] == [
        "segment-0",
        "segment-1",
    ]


@pytest.mark.parametrize(
    ("aggregation", "extra_config", "expected"),
    [
        ("min", {}, 0.0),
        ("cvar", {"cvar_alpha": 0.67}, 0.3),
    ],
)
def test_failed_segment_is_part_of_worst_case_aggregators(
    aggregation, extra_config, expected
):
    evaluator = RegimeAwareEvaluator(
        _config(
            aggregation=aggregation,
            min_segment_trades=1,
            **extra_config,
        ),
        segments={},
    )
    results = [
        _result(0, 0.8, 5),
        _result(1, 0.6, 5),
        _result(2, 0.0, 0, success=False),
    ]

    fitness, metrics = evaluator._aggregate_results(results, _StrategyGeneStub())

    assert fitness == pytest.approx(expected)
    assert metrics["segment_effective_fitness_values"] == [0.8, 0.6, 0.0]


def test_low_trade_positive_is_capped_at_zero_but_loss_is_preserved():
    evaluator = RegimeAwareEvaluator(_config(min_segment_trades=5), segments={})

    positive = _result(0, 0.7, 2)
    loss = _result(1, -0.3, 2)

    assert positive.outcome_status(5) == "low_trades"
    assert positive.effective_fitness(5) == 0.0
    assert loss.effective_fitness(5) == -0.3

    fitness, metrics = evaluator._aggregate_results(
        [positive, loss], _StrategyGeneStub()
    )
    assert fitness == -0.3
    assert metrics["low_trade_segment_count"] == 2
    assert metrics["penalized_low_trade_segments"] == 2
    assert metrics["skipped_low_trade_segments"] == 0


def test_exclusive_mode_cannot_hide_non_preferred_runtime_regime():
    evaluator = RegimeAwareEvaluator(
        _config(
            min_segment_trades=1,
            regime_specialization={
                "enabled": True,
                "specialist_boost": 2.0,
                "diversity_weight": 0.0,
            },
        ),
        segments={},
    )
    gene = _StrategyGeneStub()
    gene.preferred_regime = "bullish"
    gene.regime_mode = "exclusive"

    fitness, metrics = evaluator._aggregate_results(
        [
            _result(0, 0.8, 5, regime=RegimeType.BULLISH),
            _result(1, 0.2, 5, regime=RegimeType.BEARISH),
        ],
        gene,
    )

    assert fitness == pytest.approx(0.32)
    assert metrics["expected_segment_count"] == 2
    assert metrics["segment_coverage_complete"] is True


def test_enabled_evaluator_without_segments_fails_closed():
    evaluator = RegimeAwareEvaluator(_config(), segments={})

    with pytest.raises(RuntimeError, match="enabled but no optimization segments"):
        evaluator.evaluate(_StrategyGeneStub())


@pytest.mark.parametrize(
    "overrides",
    [
        {"regime_weights": {"bullish": 0.0}},
        {"regime_weights": {"bullish": float("nan")}},
        {"min_segment_trades": -1},
        {"cvar_alpha": 0.0},
    ],
)
def test_invalid_regime_aggregation_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        RegimeAwareEvaluator(_config(**overrides), segments={})
