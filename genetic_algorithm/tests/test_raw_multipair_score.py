"""Focused contract tests for raw-multipair-score-v2."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from genetic_algorithm.evaluation.raw_multipair_score import (
    COMPONENT_NAMES,
    HARDCORE_DEVELOPMENT_PAIRS,
    HARDCORE_VALIDATION_PAIRS,
    RAW_MULTIPAIR_SCORE_VERSION,
    PairScenario,
    RawMultiPairPanel,
    RawMultiPairStatus,
    is_material_improvement,
    material_improvement_threshold,
    score_raw_multipair,
)


ALL_PAIRS = HARDCORE_DEVELOPMENT_PAIRS + HARDCORE_VALIDATION_PAIRS


def _panel(timeframe: str = "15m") -> RawMultiPairPanel:
    return RawMultiPairPanel(timeframe=timeframe)


def _scenario(pair: str, timeframe: str = "15m", **overrides) -> PairScenario:
    values = {
        "pair": pair,
        "timeframe": timeframe,
        "success": True,
        # The immutable panel has 35 inclusive calendar-month buckets.
        # Full 15m activity credit: 17 trades/month across 35 months.
        "trade_count": 595,
        "active_months": 27,
        "net_return": 0.10,
        "net_expectancy": 0.005,
        "profit_factor": 1.5,
        "profit_factor_censored": False,
        "median_holding_hours": 24.0,
        "p90_holding_hours": 72.0,
        "max_drawdown": 0.20,
        "max_drawdown_duration_days": 120.0,
        "max_consecutive_losses": 10,
    }
    values.update(overrides)
    return PairScenario(**values)


def _scenarios(timeframe: str = "15m") -> list[PairScenario]:
    return [_scenario(pair, timeframe) for pair in ALL_PAIRS]


def _replace_pair(scenarios: list[PairScenario], pair: str, **changes) -> list[PairScenario]:
    return [
        replace(scenario, **changes) if scenario.pair == pair else scenario
        for scenario in scenarios
    ]


def test_exact_balanced_formula_for_identical_pairs():
    result = score_raw_multipair(_panel(), _scenarios())

    unit = math.tanh(1.0)
    expected = 100.0 * (0.27 * unit + 0.25)
    assert result.status is RawMultiPairStatus.VALID
    assert result.score_version == RAW_MULTIPAIR_SCORE_VERSION
    assert result.score == pytest.approx(expected)
    assert result.aggregate_components is not None
    assert result.aggregate_components.return_score == pytest.approx(unit)
    assert result.aggregate_components.expectancy_score == pytest.approx(unit)
    assert result.aggregate_components.profit_factor_score == pytest.approx(unit)
    assert result.aggregate_components.activity_score == pytest.approx(1.0)
    assert result.aggregate_components.holding_score == pytest.approx(1.0)
    assert result.aggregate_components.drawdown_risk == pytest.approx(unit)
    assert result.aggregate_components.drawdown_duration_risk == pytest.approx(unit)
    assert result.aggregate_components.loss_streak_risk == pytest.approx(unit)
    assert result.aggregate_components.overtrading_risk == 0.0


def test_break_even_edge_is_neutral_and_forbidden_metrics_are_not_components():
    scenarios = [
        replace(
            scenario,
            net_return=0.0,
            net_expectancy=0.0,
            profit_factor=1.0,
        )
        for scenario in _scenarios()
    ]
    result = score_raw_multipair(_panel(), scenarios)

    assert result.is_valid
    assert result.aggregate_components is not None
    assert result.aggregate_components.return_score == 0.0
    assert result.aggregate_components.expectancy_score == 0.0
    assert result.aggregate_components.profit_factor_score == 0.0
    assert not {
        "sharpe",
        "sortino",
        "lcb",
        "ucb",
        "ess",
        "gate_status",
    } & set(COMPONENT_NAMES)
    assert not {
        "sharpe",
        "sortino",
        "annualized_net_return_lcb",
        "effective_sample_size",
    } & set(PairScenario.__dataclass_fields__)


def test_validation_worst_pair_has_more_influence_than_development_worst_pair():
    development_bad = score_raw_multipair(
        _panel(),
        _replace_pair(_scenarios(), "BTC/USDT", net_return=-0.10),
    )
    validation_bad = score_raw_multipair(
        _panel(),
        _replace_pair(_scenarios(), "ETH/USDT", net_return=-0.10),
    )

    assert development_bad.is_valid and validation_bad.is_valid
    assert development_bad.score is not None and validation_bad.score is not None
    assert validation_bad.score < development_bad.score
    # Difference from 0.4/0.6 group weighting, with the same 0.6 worst-pair term.
    assert development_bad.score - validation_bad.score == pytest.approx(
        100.0 * 0.25 * 0.24 * math.tanh(1.0)
    )


def test_group_and_scenario_order_do_not_change_identity_or_score():
    standard = _panel()
    permuted = RawMultiPairPanel(
        timeframe="15m",
        development_pairs=tuple(reversed(HARDCORE_DEVELOPMENT_PAIRS)),
        validation_pairs=tuple(reversed(HARDCORE_VALIDATION_PAIRS)),
    )

    first = score_raw_multipair(standard, _scenarios())
    second = score_raw_multipair(permuted, reversed(_scenarios()))
    assert standard.panel_id == permuted.panel_id
    assert first.score == second.score
    assert first.to_dict() == second.to_dict()


def test_panel_identity_separates_timeframes_and_mixed_input_fails_closed():
    panel_15m = _panel("15m")
    panel_1h = _panel("1h")
    assert panel_15m.panel_id != panel_1h.panel_id

    result = score_raw_multipair(
        panel_15m,
        _replace_pair(_scenarios(), "BTC/USDT", timeframe="1h"),
    )
    assert result.status is RawMultiPairStatus.INVALID
    assert result.score is None
    assert result.reason_code == "TIMEFRAME_MISMATCH"


def test_panel_contract_is_fixed_and_serializable():
    panel = _panel()
    assert panel.calendar_months == 35
    payload = panel.to_dict()
    assert payload["timeframe"] == "15m"
    assert payload["fee_rate"] == 0.001
    assert payload["slippage_rate"] == 0.0005
    json.dumps(payload)

    with pytest.raises(ValueError, match="fee_rate"):
        RawMultiPairPanel(timeframe="15m", fee_rate=0.002)
    with pytest.raises(ValueError, match="supports only"):
        RawMultiPairPanel(timeframe="5m")


def test_zero_trade_scenarios_are_valid_but_harsh():
    zero = [
        PairScenario(
            pair=pair,
            timeframe="15m",
            success=True,
            trade_count=0,
            active_months=0,
        )
        for pair in ALL_PAIRS
    ]
    result = score_raw_multipair(_panel(), zero)

    assert result.status is RawMultiPairStatus.VALID
    assert result.score == pytest.approx(-70.0)
    assert all(item.zero_trades for item in result.pair_components)
    assert all(item.components.activity_score == -1.0 for item in result.pair_components)
    assert all(item.components.return_score == -1.0 for item in result.pair_components)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"p90_holding_hours": None}, "MISSING_METRIC"),
        ({"net_return": math.nan}, "NONFINITE_METRIC"),
        ({"profit_factor": math.inf}, "NONFINITE_METRIC"),
        ({"active_months": 0}, "INVALID_METRIC"),
        ({"p90_holding_hours": 12.0}, "INVALID_METRIC"),
    ],
)
def test_missing_nonfinite_and_inconsistent_evidence_is_invalid(changes, reason):
    result = score_raw_multipair(
        _panel(),
        _replace_pair(_scenarios(), "BTC/USDT", **changes),
    )
    assert result.status is RawMultiPairStatus.INVALID
    assert result.score is None
    assert result.reason_code == reason
    assert not result.pair_components


def test_technical_worker_failure_is_distinct_from_negative_fitness():
    negative = score_raw_multipair(
        _panel(),
        [
            replace(
                scenario,
                net_return=-0.10,
                net_expectancy=-0.005,
                profit_factor=0.5,
            )
            for scenario in _scenarios()
        ],
    )
    failed = score_raw_multipair(
        _panel(),
        _replace_pair(
            _scenarios(),
            "BTC/USDT",
            success=False,
            technical_error="worker exited 137",
        ),
    )

    assert negative.is_valid and negative.score is not None and negative.score < 0
    assert failed.status is RawMultiPairStatus.INVALID
    assert failed.reason_code == "TECHNICAL_ERROR"
    assert "worker exited 137" in (failed.reason_detail or "")


def test_profit_factor_is_capped_and_discounted_by_trade_evidence_even_when_finite():
    low_evidence = score_raw_multipair(
        _panel(),
        _replace_pair(
            _scenarios(),
            "BTC/USDT",
            trade_count=3,
            active_months=1,
            profit_factor=10.0,
            profit_factor_censored=True,
            max_consecutive_losses=0,
        ),
    )
    full_evidence = score_raw_multipair(
        _panel(),
        _replace_pair(
            _scenarios(),
            "BTC/USDT",
            trade_count=30,
            active_months=10,
            profit_factor=10.0,
            profit_factor_censored=True,
            max_consecutive_losses=0,
        ),
    )
    uncensored_low_evidence = score_raw_multipair(
        _panel(),
        _replace_pair(
            _scenarios(),
            "BTC/USDT",
            trade_count=3,
            active_months=1,
            profit_factor=10.0,
            profit_factor_censored=False,
            max_consecutive_losses=0,
        ),
    )

    low = low_evidence.pair_component_map()["BTC/USDT"]
    full = full_evidence.pair_component_map()["BTC/USDT"]
    uncensored = uncensored_low_evidence.pair_component_map()["BTC/USDT"]
    assert low.effective_profit_factor == pytest.approx(1.2)
    assert full.effective_profit_factor == pytest.approx(3.0)
    assert uncensored.effective_profit_factor == pytest.approx(1.2)
    assert low.components.profit_factor_score < full.components.profit_factor_score


def test_activity_requires_trade_rate_and_active_month_coverage():
    full = score_raw_multipair(_panel(), _scenarios())
    sparse_months = score_raw_multipair(
        _panel(),
        _replace_pair(_scenarios(), "BTC/USDT", active_months=14),
    )

    assert full.pair_component_map()["BTC/USDT"].components.activity_score == 1.0
    sparse = sparse_months.pair_component_map()["BTC/USDT"].components.activity_score
    expected_progress = 0.55 + 0.45 * ((14 / 35) / 0.75)
    assert sparse == pytest.approx(2.0 * expected_progress - 1.0)


def test_activity_targets_are_timeframe_specific_and_sparse_activity_is_negative():
    scenarios_15m = [
        replace(scenario, trade_count=17 * 35, active_months=27)
        for scenario in _scenarios("15m")
    ]
    scenarios_1h = [
        replace(scenario, trade_count=10 * 35, active_months=27)
        for scenario in _scenarios("1h")
    ]
    full_15m = score_raw_multipair(_panel("15m"), scenarios_15m)
    full_1h = score_raw_multipair(_panel("1h"), scenarios_1h)
    sparse = score_raw_multipair(
        _panel("15m"),
        _replace_pair(
            scenarios_15m,
            "BTC/USDT",
            trade_count=9,
            active_months=5,
            max_consecutive_losses=1,
        ),
    )

    assert full_15m.pair_component_map()["BTC/USDT"].components.activity_score == 1.0
    assert full_1h.pair_component_map()["BTC/USDT"].components.activity_score == 1.0
    assert sparse.pair_component_map()["BTC/USDT"].components.activity_score < 0.0


def test_holding_reward_declines_after_both_daytrading_targets():
    at_target = score_raw_multipair(_panel(), _scenarios())
    too_long = score_raw_multipair(
        _panel(),
        _replace_pair(
            _scenarios(),
            "BTC/USDT",
            median_holding_hours=48.0,
            p90_holding_hours=144.0,
        ),
    )

    assert at_target.pair_component_map()["BTC/USDT"].components.holding_score == 1.0
    assert too_long.pair_component_map()["BTC/USDT"].components.holding_score == 0.5


def test_overtrading_penalty_starts_strictly_above_sixty_per_month():
    at_limit = score_raw_multipair(
        _panel(),
        _replace_pair(
            _scenarios(),
            "BTC/USDT",
            trade_count=60 * 35,
            max_consecutive_losses=10,
        ),
    )
    double_limit = score_raw_multipair(
        _panel(),
        _replace_pair(
            _scenarios(),
            "BTC/USDT",
            trade_count=120 * 35,
            max_consecutive_losses=10,
        ),
    )

    assert at_limit.pair_component_map()["BTC/USDT"].components.overtrading_risk == 0.0
    assert double_limit.pair_component_map()[
        "BTC/USDT"
    ].components.overtrading_risk == pytest.approx(math.tanh(1.0))


def test_duplicate_or_incomplete_panel_is_invalid():
    duplicate = _scenarios()[:-1] + [_scenarios()[0]]
    duplicate_result = score_raw_multipair(_panel(), duplicate)
    incomplete_result = score_raw_multipair(_panel(), _scenarios()[:-1])

    assert duplicate_result.reason_code in {"DUPLICATE_PAIR", "PAIR_SET_MISMATCH"}
    assert incomplete_result.reason_code == "PAIR_SET_MISMATCH"
    assert duplicate_result.score is None and incomplete_result.score is None


def test_material_improvement_uses_versioned_absolute_or_relative_delta():
    assert material_improvement_threshold(0.0) == 0.25
    assert material_improvement_threshold(-20.0) == 0.25
    assert material_improvement_threshold(100.0) == 0.5
    assert is_material_improvement(None, -40.0)
    assert is_material_improvement(-20.0, -19.75)
    assert not is_material_improvement(-20.0, -19.751)
    assert is_material_improvement(100.0, 100.5)
    assert not is_material_improvement(100.0, 100.499)
    with pytest.raises(ValueError, match="finite"):
        is_material_improvement(1.0, math.nan)


def test_score_output_is_json_serializable_and_exposes_all_components():
    result = score_raw_multipair(_panel(), _scenarios())
    payload = result.to_dict()
    assert tuple(payload["aggregate_components"]) == COMPONENT_NAMES
    assert set(payload["pair_components"]) == set(ALL_PAIRS)
    assert payload["status"] == "VALID"
    json.dumps(payload, sort_keys=True)
