"""Pair-group and role-scoped V3 qualification tests."""

from __future__ import annotations

from datetime import date

import pytest

from genetic_algorithm.orchestration.promotion_policy_v3 import (
    EvaluationScenarioV3,
    PairGroupGateV3,
    QualificationPolicyV3,
    evaluate_candidate_v3,
    qualification_policy_v3_from_config,
)
from genetic_algorithm.config.schema import (
    config_contract_shape_v2,
    deep_merge,
    validate_config_or_raise,
)
from genetic_algorithm.orchestration.result_contract import BacktestRecordV2
from genetic_algorithm.tests.v2_metric_fixtures import expectancy_metrics


PAIRS = {
    "development": ["BTC/USDT", "SOL/USDT", "XRP/USDT"],
    "validation": ["BNB/USDT", "ETH/USDT", "PEPE/USDT"],
}


def _record(scenario: EvaluationScenarioV3, *, net_return=0.1, annual_lcb=0.04):
    trades = [
        {
            "pair": scenario.pair,
            "profit_ratio": 0.01,
            "trade_duration": 12 * 60,
        }
        for _ in range(60)
    ]
    return BacktestRecordV2(
        attempt_id="attempt-v3",
        wave_id="wave-v3",
        experiment_id="experiment-v3",
        candidate_id="candidate-v3",
        config_hash="c" * 64,
        phenotype_hash="p" * 64,
        code_version="v3-test",
        data_manifest_hash="d" * 64,
        fitness_policy_version="qualification-v3-test",
        seed=42,
        worker_count=1,
        fee_rate=0.001 * scenario.cost_multiplier,
        slippage_rate=0.0005 * scenario.cost_multiplier,
        spread_rate=0,
        funding_rate=0,
        equity_method="MARK_TO_MARKET",
        metrics={
            "scenario_id": scenario.scenario_id,
            "pair": scenario.pair,
            "timeframe": scenario.timeframe,
            "role": scenario.role,
            "period_start": scenario.period_start,
            "period_end": scenario.inclusive_period_end,
            "cost_multiplier": scenario.cost_multiplier,
            "status": "VALID",
            "success": True,
            "no_trades": False,
            "net_return": net_return,
            "annualized_net_return": 0.1,
            "annualized_net_return_lcb": annual_lcb,
            "max_drawdown": 0.1,
            "max_drawdown_ucb": 0.15,
            "daily_expected_shortfall_5": 0.01,
            "daily_expected_shortfall_5_ucb": 0.02,
            "net_expectancy": 0.003,
            "net_expectancy_lcb": 0.001,
            "profit_factor": 1.4,
            "profit_factor_censored": False,
            "profit_factor_contract_version": "right-censored-profit-factor-v1",
            "win_rate": 0.6,
            "trade_count": len(trades),
            "effective_sample_size": 40,
            **expectancy_metrics(
                trade_count=len(trades),
                mean_return=0.003,
                lower_confidence_bound=0.001,
                effective_sample_size=40,
            ),
            "active_months": 12 if scenario.role != "FINAL_TEST" else 4,
            "max_consecutive_losses": 3,
            "max_drawdown_duration_days": 30,
        },
        daily_net_returns=[0.01, -0.005, 0.004],
        equity_curve=[100, 101, 100.5, 101.2],
        trades=trades,
    )


def _policy() -> QualificationPolicyV3:
    scenarios = []
    for group_id, pairs in PAIRS.items():
        for pair in pairs:
            token = pair.split("/")[0].lower()
            scenarios.extend(
                [
                    EvaluationScenarioV3(
                        scenario_id=f"{group_id}-{token}",
                        pair=pair,
                        pair_group=group_id,
                        timeframe="1h",
                        role="PAIR_VALIDATION" if group_id == "validation" else "TRAIN",
                        period_start=date(2024, 1, 1),
                        period_end_exclusive=date(2025, 1, 1),
                        cost_multiplier=1.0,
                    ),
                    EvaluationScenarioV3(
                        scenario_id=f"final-{group_id}-{token}",
                        pair=pair,
                        pair_group=group_id,
                        timeframe="1h",
                        role="FINAL_TEST",
                        period_start=date(2026, 3, 28),
                        period_end_exclusive=date(2026, 8, 1),
                        cost_multiplier=1.5,
                    ),
                ]
            )
    scenarios.sort(key=lambda item: item.scenario_id)
    return QualificationPolicyV3(
        policy_version="qualification-v3-test",
        required_scenarios=scenarios,
        pair_groups=[
            PairGroupGateV3(group_id="development", pairs=sorted(PAIRS["development"]), min_profitable_pairs=2),
            PairGroupGateV3(group_id="validation", pairs=sorted(PAIRS["validation"]), min_profitable_pairs=2),
        ],
    )


def test_v3_allows_one_weak_pair_but_rejects_a_large_pair_loss():
    policy = _policy()
    records = [_record(item) for item in policy.required_scenarios]
    weak = next(item for item in policy.required_scenarios if item.pair == "PEPE/USDT" and item.cost_multiplier == 1.0)
    records = [
        _record(item, net_return=(-0.03 if item.scenario_id == weak.scenario_id else 0.1))
        for item in policy.required_scenarios
    ]
    candidate = evaluate_candidate_v3(records, policy)
    assert candidate.would_pass

    records = [
        _record(item, net_return=(-0.30 if item.scenario_id == weak.scenario_id else 0.1))
        for item in policy.required_scenarios
    ]
    candidate = evaluate_candidate_v3(records, policy)
    failed = {gate.gate_id for gate in candidate.gates if gate.passed is False}
    assert "VALIDATION_WORST_PAIR_RETURN" in failed
    assert not candidate.would_pass


def test_v3_applies_shorter_evidence_gate_to_final_test():
    policy = _policy()
    candidate = evaluate_candidate_v3(
        [_record(item) for item in policy.required_scenarios],
        policy,
    )

    assert candidate.would_pass
    assert next(g for g in candidate.gates if g.gate_id == "DEVELOPMENT_ACTIVE_MONTHS").observed == 12
    assert next(g for g in candidate.gates if g.gate_id == "FINAL_ACTIVE_MONTHS").observed == 4
    assert candidate.diagnostics.independent_pair_trades_per_day > 0
    assert candidate.diagnostics.median_hold_hours == pytest.approx(12)


def test_v3_final_test_must_be_positive_independently_of_development():
    policy = _policy()
    records = [
        _record(
            item,
            net_return=(-0.01 if item.role.value == "FINAL_TEST" else 0.20),
            annual_lcb=(-0.01 if item.role.value == "FINAL_TEST" else 0.10),
        )
        for item in policy.required_scenarios
    ]

    candidate = evaluate_candidate_v3(records, policy)

    failed = {gate.gate_id for gate in candidate.gates if gate.passed is False}
    assert "FINAL_DEVELOPMENT_PROFITABLE_PAIRS" in failed
    assert "FINAL_VALIDATION_GROUP_ANNUAL_RETURN_LCB" in failed
    assert not candidate.would_pass


def test_v3_half_open_window_must_match_record_matrix():
    policy = _policy()
    records = [_record(item) for item in policy.required_scenarios]
    payload = records[0].model_dump(mode="python")
    payload["metrics"]["period_end"] = date(2024, 12, 30)
    records[0] = BacktestRecordV2.model_validate(payload)

    candidate = evaluate_candidate_v3(records, policy)

    assert candidate.status.value == "INCONCLUSIVE"
    assert candidate.gates[0].reason_code == "INCOMPLETE_V3_SCENARIO_MATRIX"


def test_v3_policy_is_accepted_by_strict_config_contract():
    policy = _policy()
    config = {
        "config_schema_version": 2,
        "safety_profile": {"name": "schema-test", "enforce": False},
        "qualification_v3": {
            "enabled": True,
            **policy.model_dump(mode="json"),
        },
    }

    resolved = deep_merge(config_contract_shape_v2(), config)
    validate_config_or_raise(resolved)
    assert qualification_policy_v3_from_config(resolved) == policy
