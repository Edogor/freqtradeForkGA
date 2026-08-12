"""Integration boundary between six independent replays and raw fitness."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from genetic_algorithm.evaluation.direct_backtester import BacktestResult
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.evaluation.raw_multipair_score import RawMultiPairPanel
from genetic_algorithm.engine.population import calculate_behavioral_distance


def _pair_metrics(*, trades: int = 10) -> dict:
    if trades == 0:
        return {
            "profit": 0.0,
            "avg_profit": 0.0,
            "period_start": "2023-05-09T00:00:00+00:00",
            "period_end": "2026-03-26T23:00:00+00:00",
            "num_trades": 0,
            "active_months": 0,
            "profit_factor": 0.0,
            "profit_factor_censored": False,
            "median_holding_hours": None,
            "p90_holding_hours": None,
            "max_drawdown": 0.0,
            "max_drawdown_duration_days": None,
            "max_consecutive_losses": 0,
        }
    return {
        "profit": 1.0,
        "avg_profit": 0.001,
        "period_start": "2023-05-09T00:00:00+00:00",
        "period_end": "2026-03-26T23:00:00+00:00",
        "num_trades": trades,
        "active_months": 5,
        "profit_factor": 1.2,
        "profit_factor_censored": False,
        "median_holding_hours": 12.0,
        "p90_holding_hours": 24.0,
        "max_drawdown": 0.05,
        "max_drawdown_duration_days": 10.0,
        "max_consecutive_losses": 2,
        # Forbidden legacy/statistical metrics may be present in reporting,
        # but are intentionally not consumed by the raw scorer.
        "sharpe_ratio": -999.0,
        "expectancy_lcb": 999.0,
        "effective_sample_size": 0.0,
    }


def _evaluator() -> FitnessEvaluator:
    evaluator = object.__new__(FitnessEvaluator)
    evaluator.raw_multipair_panel = RawMultiPairPanel(timeframe="1h")
    return evaluator


def _summary(pairs: tuple[str, ...], *, trades: int = 10) -> dict:
    return {
        "profit": 1.0,
        "num_trades": trades * len(pairs),
        "independent_pair_metrics": {
            pair: _pair_metrics(trades=trades) for pair in pairs
        },
    }


def _gene() -> SimpleNamespace:
    return SimpleNamespace(individual_id=7, calculate_complexity=lambda: 3)


def test_complete_panel_uses_one_raw_score_and_ignores_statistical_metrics():
    evaluator = _evaluator()
    panel = evaluator.raw_multipair_panel

    score, metrics = evaluator._score_raw_complete_panel(
        train_metrics=_summary(panel.development_pairs),
        val_metrics=_summary(panel.validation_pairs),
        strategy_gene=_gene(),
    )

    assert score == pytest.approx(metrics["raw_multipair_score"])
    assert score != 0.0
    assert metrics["raw_multipair_status"] == "VALID"
    assert metrics["train_fitness"] == metrics["val_fitness"] == score
    assert set(metrics["raw_pair_metrics"]) == {
        *panel.development_pairs,
        *panel.validation_pairs,
    }
    assert set(metrics["per_pair_profit"]) == set(metrics["raw_pair_metrics"])


def test_raw_pair_outcomes_drive_behavioral_diversity():
    evaluator = _evaluator()
    panel = evaluator.raw_multipair_panel
    first_train = _summary(panel.development_pairs)
    first_val = _summary(panel.validation_pairs)
    second_train = _summary(panel.development_pairs)
    second_val = _summary(panel.validation_pairs)
    second_train["independent_pair_metrics"]["BTC/USDT"]["profit"] = -3.0
    second_val["independent_pair_metrics"]["ETH/USDT"]["profit"] = 4.0

    _, first_metrics = evaluator._score_raw_complete_panel(
        train_metrics=first_train,
        val_metrics=first_val,
        strategy_gene=_gene(),
    )
    _, second_metrics = evaluator._score_raw_complete_panel(
        train_metrics=second_train,
        val_metrics=second_val,
        strategy_gene=_gene(),
    )

    distance = calculate_behavioral_distance(
        SimpleNamespace(metrics=first_metrics),
        SimpleNamespace(metrics=second_metrics),
    )
    assert distance is not None
    assert distance > 0.0


def test_missing_raw_metric_marks_candidate_as_technical_evidence_failure():
    evaluator = _evaluator()
    panel = evaluator.raw_multipair_panel
    train = _summary(panel.development_pairs)
    train["independent_pair_metrics"][panel.development_pairs[0]][
        "p90_holding_hours"
    ] = None

    score, metrics = evaluator._score_raw_complete_panel(
        train_metrics=train,
        val_metrics=_summary(panel.validation_pairs),
        strategy_gene=_gene(),
    )

    assert score == 0.0
    assert metrics["raw_multipair_status"] == "INVALID"
    assert metrics["error"].startswith("raw_multipair_invalid:MISSING_METRIC")


def test_six_zero_trade_pairs_are_valid_negative_fitness():
    evaluator = _evaluator()
    panel = evaluator.raw_multipair_panel

    score, metrics = evaluator._score_raw_complete_panel(
        train_metrics=_summary(panel.development_pairs, trades=0),
        val_metrics=_summary(panel.validation_pairs, trades=0),
        strategy_gene=_gene(),
    )

    assert score == pytest.approx(-55.0)
    assert metrics["raw_multipair_status"] == "VALID"
    assert metrics["no_trades"] is True
    assert "error" not in metrics
    assert {
        (item["period_start"], item["period_end"])
        for item in metrics["raw_pair_metrics"].values()
    } == {
        (
            "2023-05-09T00:00:00+00:00",
            "2026-03-26T23:00:00+00:00",
        )
    }


def test_raw_search_rejects_one_pair_with_mismatched_measured_period():
    evaluator = _evaluator()
    panel = evaluator.raw_multipair_panel
    train = _summary(panel.development_pairs)
    train["independent_pair_metrics"]["BTC/USDT"]["period_start"] = (
        "2023-05-09T01:00:00+00:00"
    )

    score, metrics = evaluator._score_raw_complete_panel(
        train_metrics=train,
        val_metrics=_summary(panel.validation_pairs),
        strategy_gene=_gene(),
    )

    assert score == 0.0
    assert metrics["raw_multipair_status"] == "INVALID"
    assert "PERIOD_COVERAGE_MISMATCH" in metrics["error"]
    assert (
        "PERIOD_COVERAGE_MISMATCH"
        in metrics["raw_pair_metrics"]["BTC/USDT"]["technical_error"]
    )


def test_holding_quantiles_survive_lightweight_trade_payload_and_timestamp_fallback():
    evaluator = object.__new__(FitnessEvaluator)
    result = BacktestResult(
        success=True,
        strategy_name="HoldingEvidence",
        total_trades=2,
        trades=[
            {
                "trade_duration": 60,
                "close_date": "2025-01-01T01:00:00Z",
            },
            {
                "open_timestamp": 1_735_689_600_000,
                "close_timestamp": 1_735_696_800_000,
                "close_date": "2025-01-01T02:00:00Z",
            },
        ],
    )

    metrics = evaluator._backtest_result_to_metrics(result)

    assert metrics["median_holding_hours"] == pytest.approx(1.5)
    assert metrics["p90_holding_hours"] == pytest.approx(1.9)


def test_raw_search_uses_same_daily_drawdown_as_strict_replay():
    evaluator = object.__new__(FitnessEvaluator)
    evaluator.raw_multipair_score_enabled = True
    result = BacktestResult(
        success=True,
        strategy_name="DrawdownParity",
        max_drawdown=0.11,
        daily_max_drawdown=0.37,
    )

    raw_metrics = evaluator._backtest_result_to_metrics(result)

    assert raw_metrics["max_drawdown"] == pytest.approx(0.37)
    # Strict replay's ScenarioMetricsV2 maps this same source field.
    assert raw_metrics["max_drawdown"] == result.daily_max_drawdown


def test_raw_search_and_strict_replay_share_exact_wallet_net_return():
    evaluator = object.__new__(FitnessEvaluator)
    evaluator.raw_multipair_score_enabled = True
    result = BacktestResult(
        success=True,
        strategy_name="ReturnParity",
        profit_percent=1.23,
        total_profit=123.456,
        starting_balance=10_000.0,
        final_balance=10_123.456,
    )

    raw_metrics = evaluator._backtest_result_to_metrics(result)

    assert raw_metrics["net_return"] == pytest.approx(0.0123456)
    assert raw_metrics["net_return"] != result.profit_percent / 100.0
