"""Fail-closed candidate aggregation and shadow promotion policy for GA V2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from statistics import median
from typing import Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.result_contract import (
    BacktestRecordV2,
    CandidateEvaluationV2,
    EvaluationStatus,
    GateResultV2,
    PromotionDecisionV2,
    ScenarioRole,
    StrictV2Model,
)


class ScenarioRequirementV2(StrictV2Model):
    """Exactly one required pair x time x cost cell in an evaluation panel."""

    scenario_id: str = Field(min_length=1)
    pair: str = Field(min_length=3)
    timeframe: str = Field(min_length=1)
    role: ScenarioRole
    period_start: date
    period_end: date
    cost_multiplier: float = Field(gt=0)

    @model_validator(mode="after")
    def _valid_period(self) -> ScenarioRequirementV2:
        if self.period_end <= self.period_start:
            raise ValueError("period_end must be after period_start")
        return self


class ShadowGatePolicyV2(StrictV2Model):
    """Versioned, non-compensable gates fixed before candidate evaluation."""

    policy_version: str = Field(min_length=1)
    required_scenarios: list[ScenarioRequirementV2] = Field(min_length=1)
    min_expectancy_lcb: float = 0.0
    min_median_annual_return_lcb: float = 0.0
    min_worst_scenario_return: float = -0.05
    max_drawdown_ucb: float = Field(default=0.25, ge=0)
    max_daily_es5_ucb: float = Field(default=0.05, ge=0)
    min_effective_sample_size: float = Field(default=30.0, ge=1)
    min_active_months: int = Field(default=6, ge=1)
    min_trades_per_active_month: float = Field(default=5.0, ge=0)
    min_profitable_scenario_ratio: float = Field(default=0.6, ge=0, le=1)
    max_consecutive_losses: int = Field(default=10, ge=0)
    max_drawdown_duration_days: float = Field(default=90.0, ge=0)
    require_final_test: bool = True

    @model_validator(mode="after")
    def _coherent_panel(self) -> ShadowGatePolicyV2:
        scenario_ids = [item.scenario_id for item in self.required_scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("required scenario_id values must be unique")
        if self.require_final_test and not any(
            item.role == ScenarioRole.FINAL_TEST for item in self.required_scenarios
        ):
            raise ValueError("require_final_test needs a declared FINAL_TEST scenario")
        return self


def shadow_gate_policy_from_config(config: Mapping[str, object]) -> ShadowGatePolicyV2:
    """Load only an explicitly enabled and fully declared promotion policy."""

    raw = config.get("promotion_v2")
    if not isinstance(raw, Mapping) or not raw.get("enabled", False):
        raise ValueError("promotion_v2 must be explicitly enabled")
    payload = {key: value for key, value in raw.items() if key != "enabled"}
    return ShadowGatePolicyV2.model_validate(payload)


def _requirement_key(item: ScenarioRequirementV2) -> tuple[object, ...]:
    return (
        item.scenario_id,
        item.pair,
        item.timeframe,
        item.role,
        str(item.period_start),
        str(item.period_end),
        item.cost_multiplier,
    )


def _record_key(record: BacktestRecordV2) -> tuple[object, ...]:
    metrics = record.metrics
    return (
        metrics.scenario_id,
        metrics.pair,
        metrics.timeframe,
        metrics.role,
        str(metrics.period_start),
        str(metrics.period_end),
        metrics.cost_multiplier,
    )


def _gate(
    gate_id: str,
    passed: bool,
    observed: float,
    threshold: float,
    operator: Literal[">", ">=", "<", "<=", "=="],
    failed_reason: str,
) -> GateResultV2:
    return GateResultV2(
        gate_id=gate_id,
        passed=passed,
        status=EvaluationStatus.VALID,
        observed=observed,
        threshold=threshold,
        operator=operator,
        reason_code="PASS" if passed else failed_reason,
    )


def _inconclusive_gate(gate_id: str, reason: str, detail: str) -> GateResultV2:
    return GateResultV2(
        gate_id=gate_id,
        passed=False,
        status=EvaluationStatus.INCONCLUSIVE,
        reason_code=reason,
        detail=detail,
    )


def evaluate_candidate_shadow(
    records: Iterable[BacktestRecordV2],
    policy: ShadowGatePolicyV2,
) -> CandidateEvaluationV2:
    """Aggregate one frozen phenotype without imputing missing panel cells."""

    scenarios = list(records)
    if not scenarios:
        raise ValueError("at least one scenario record is required")

    first = scenarios[0]
    identity = (
        first.attempt_id,
        first.candidate_id,
        first.phenotype_hash,
        first.config_hash,
        first.code_version,
        first.data_manifest_hash,
        first.fitness_policy_version,
        first.seed,
        first.worker_count,
        first.wave_id,
        first.experiment_id,
    )
    if any(
        (
            row.attempt_id,
            row.candidate_id,
            row.phenotype_hash,
            row.config_hash,
            row.code_version,
            row.data_manifest_hash,
            row.fitness_policy_version,
            row.seed,
            row.worker_count,
            row.wave_id,
            row.experiment_id,
        )
        != identity
        for row in scenarios[1:]
    ):
        raise ValueError("scenario records do not describe one frozen candidate")

    actual_keys = [_record_key(record) for record in scenarios]
    required_keys = [_requirement_key(item) for item in policy.required_scenarios]
    duplicate_actual = len(actual_keys) != len(set(actual_keys))
    duplicate_required = len(required_keys) != len(set(required_keys))
    missing = sorted(set(required_keys) - set(actual_keys), key=str)
    unexpected = sorted(set(actual_keys) - set(required_keys), key=str)
    if duplicate_actual or duplicate_required or missing or unexpected:
        detail = json.dumps(
            {
                "duplicate_actual": duplicate_actual,
                "duplicate_required": duplicate_required,
                "missing": [list(item) for item in missing],
                "unexpected": [list(item) for item in unexpected],
            },
            sort_keys=True,
            default=str,
        )
        return CandidateEvaluationV2(
            candidate_id=first.candidate_id,
            phenotype_hash=first.phenotype_hash,
            fitness_policy_version=first.fitness_policy_version,
            status=EvaluationStatus.INCONCLUSIVE,
            scenarios=scenarios,
            gates=[_inconclusive_gate("SCENARIO_MATRIX", "INCOMPLETE_SCENARIO_MATRIX", detail)],
        )

    non_valid = [record for record in scenarios if record.metrics.status != EvaluationStatus.VALID]
    if non_valid:
        statuses = {record.metrics.status for record in non_valid}
        if EvaluationStatus.FAIL in statuses:
            candidate_status = EvaluationStatus.FAIL
        elif EvaluationStatus.INVALID in statuses:
            candidate_status = EvaluationStatus.INVALID
        else:
            candidate_status = EvaluationStatus.INCONCLUSIVE
        detail = ", ".join(
            f"{record.metrics.scenario_id}:{record.metrics.status.value}" for record in non_valid
        )
        return CandidateEvaluationV2(
            candidate_id=first.candidate_id,
            phenotype_hash=first.phenotype_hash,
            fitness_policy_version=first.fitness_policy_version,
            status=candidate_status,
            scenarios=scenarios,
            gates=[_inconclusive_gate("SCENARIO_VALIDITY", "NON_VALID_SCENARIO", detail)],
        )

    metrics = [record.metrics for record in scenarios]
    # A strategy must have positive edge both per completed trade and per
    # committed unit of margin/stake. One cannot compensate for the other.
    expectancy_lcb = min(
        min(
            float(item.net_expectancy_lcb),
            float(item.net_expectancy_on_committed_capital_lcb),
        )
        for item in metrics
    )
    annual_lcb = median(float(item.annualized_net_return_lcb) for item in metrics)
    worst_return = min(float(item.net_return) for item in metrics)
    drawdown_ucb = max(float(item.max_drawdown_ucb) for item in metrics)
    es_ucb = max(float(item.daily_expected_shortfall_5_ucb) for item in metrics)
    n_eff = min(float(item.effective_sample_size) for item in metrics)
    active_months = min(item.active_months for item in metrics)
    trades_per_month = min(
        item.trade_count / item.active_months if item.active_months else 0.0 for item in metrics
    )
    profitable_ratio = sum(float(item.net_return) > 0 for item in metrics) / len(metrics)
    loss_streak = max(int(item.max_consecutive_losses) for item in metrics)
    dd_duration = max(float(item.max_drawdown_duration_days) for item in metrics)
    final_count = sum(item.role == ScenarioRole.FINAL_TEST for item in metrics)

    gates = [
        _gate(
            "EXPECTANCY_LCB",
            expectancy_lcb > policy.min_expectancy_lcb,
            expectancy_lcb,
            policy.min_expectancy_lcb,
            ">",
            "EXPECTANCY_LCB_TOO_LOW",
        ),
        _gate(
            "MEDIAN_ANNUAL_RETURN_LCB",
            annual_lcb >= policy.min_median_annual_return_lcb,
            annual_lcb,
            policy.min_median_annual_return_lcb,
            ">=",
            "ANNUAL_RETURN_LCB_TOO_LOW",
        ),
        _gate(
            "WORST_SCENARIO_RETURN",
            worst_return >= policy.min_worst_scenario_return,
            worst_return,
            policy.min_worst_scenario_return,
            ">=",
            "WORST_SCENARIO_RETURN_TOO_LOW",
        ),
        _gate(
            "MAX_DRAWDOWN_UCB",
            drawdown_ucb <= policy.max_drawdown_ucb,
            drawdown_ucb,
            policy.max_drawdown_ucb,
            "<=",
            "MAX_DRAWDOWN_UCB_TOO_HIGH",
        ),
        _gate(
            "DAILY_ES5_UCB",
            es_ucb <= policy.max_daily_es5_ucb,
            es_ucb,
            policy.max_daily_es5_ucb,
            "<=",
            "DAILY_ES5_UCB_TOO_HIGH",
        ),
        _gate(
            "EFFECTIVE_SAMPLE_SIZE",
            n_eff >= policy.min_effective_sample_size,
            n_eff,
            policy.min_effective_sample_size,
            ">=",
            "EFFECTIVE_SAMPLE_SIZE_TOO_LOW",
        ),
        _gate(
            "ACTIVE_MONTHS",
            float(active_months) >= policy.min_active_months,
            float(active_months),
            float(policy.min_active_months),
            ">=",
            "ACTIVE_MONTHS_TOO_LOW",
        ),
        _gate(
            "TRADES_PER_ACTIVE_MONTH",
            trades_per_month >= policy.min_trades_per_active_month,
            trades_per_month,
            policy.min_trades_per_active_month,
            ">=",
            "TRADE_RATE_TOO_LOW",
        ),
        _gate(
            "PROFITABLE_SCENARIO_RATIO",
            profitable_ratio >= policy.min_profitable_scenario_ratio,
            profitable_ratio,
            policy.min_profitable_scenario_ratio,
            ">=",
            "PROFITABLE_SCENARIO_RATIO_TOO_LOW",
        ),
        _gate(
            "MAX_CONSECUTIVE_LOSSES",
            loss_streak <= policy.max_consecutive_losses,
            float(loss_streak),
            float(policy.max_consecutive_losses),
            "<=",
            "LOSS_STREAK_TOO_HIGH",
        ),
        _gate(
            "MAX_DRAWDOWN_DURATION",
            dd_duration <= policy.max_drawdown_duration_days,
            dd_duration,
            policy.max_drawdown_duration_days,
            "<=",
            "DRAWDOWN_DURATION_TOO_HIGH",
        ),
    ]
    if policy.require_final_test:
        gates.append(
            _gate(
                "FINAL_TEST_PRESENT",
                final_count >= 1,
                float(final_count),
                1.0,
                ">=",
                "FINAL_TEST_MISSING",
            )
        )

    # Diagnostic ranking only. It cannot compensate a failed gate.
    robust_score = annual_lcb - drawdown_ucb - es_ucb
    return CandidateEvaluationV2(
        candidate_id=first.candidate_id,
        phenotype_hash=first.phenotype_hash,
        fitness_policy_version=first.fitness_policy_version,
        status=EvaluationStatus.VALID,
        scenarios=scenarios,
        gates=gates,
        robust_score=robust_score,
    )


def make_shadow_promotion_decision(
    candidate: CandidateEvaluationV2,
    *,
    wave_id: str,
    policy_version: str,
    created_at: datetime,
) -> PromotionDecisionV2:
    """Create a deterministic read-only decision from already evaluated gates."""

    if candidate.status == EvaluationStatus.VALID:
        failed = [gate for gate in candidate.gates if gate.passed is not True]
        outcome = "FAIL" if failed else "WOULD_PASS"
        reasons = [gate.reason_code for gate in failed] or ["ALL_GATES_PASS"]
    elif candidate.status == EvaluationStatus.INVALID:
        outcome = "INVALID"
        reasons = ["CANDIDATE_INVALID"]
    elif candidate.status == EvaluationStatus.FAIL:
        outcome = "INVALID"
        reasons = ["CANDIDATE_EVALUATION_FAILED"]
    else:
        outcome = "INCONCLUSIVE"
        reasons = ["CANDIDATE_INCONCLUSIVE"]

    payload = {
        "wave_id": wave_id,
        "candidate_id": candidate.candidate_id,
        "policy_version": policy_version,
        "created_at": created_at.isoformat(),
        "gates": [gate.model_dump(mode="json") for gate in candidate.gates],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return PromotionDecisionV2(
        decision_id=f"shadow-{digest}",
        wave_id=wave_id,
        candidate_id=candidate.candidate_id,
        policy_version=policy_version,
        created_at=created_at,
        shadow_mode=True,
        outcome=outcome,
        promotion_authorized=False,
        gate_results=candidate.gates,
        reason_codes=reasons,
    )
