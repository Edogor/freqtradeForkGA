"""Role-scoped, pair-group qualification policy for GA V3 candidates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from statistics import median
from typing import Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.result_contract import (
    BacktestRecordV2,
    EvaluationStatus,
    GateResultV2,
    ScenarioRole,
    StrictV2Model,
)


class EvaluationScenarioV3(StrictV2Model):
    """One predeclared pair × half-open time window × cost cell."""

    scenario_id: str = Field(min_length=1)
    pair: str = Field(min_length=3)
    pair_group: str = Field(min_length=1)
    timeframe: Literal["1h", "15m"]
    role: ScenarioRole
    period_start: date
    period_end_exclusive: date
    cost_multiplier: float = Field(gt=0)

    @model_validator(mode="after")
    def _valid_window(self) -> "EvaluationScenarioV3":
        if self.period_end_exclusive <= self.period_start:
            raise ValueError("period_end_exclusive must be after period_start")
        if self.role == ScenarioRole.INNER_VALIDATION:
            raise ValueError("V3 forbids ambiguous INNER_VALIDATION role")
        return self

    @property
    def inclusive_period_end(self) -> date:
        return self.period_end_exclusive - timedelta(days=1)

    @property
    def record_key(self) -> tuple[object, ...]:
        return (
            self.scenario_id,
            self.pair,
            self.timeframe,
            self.role,
            str(self.period_start),
            str(self.inclusive_period_end),
            self.cost_multiplier,
        )


class PairGroupGateV3(StrictV2Model):
    group_id: str = Field(min_length=1)
    pairs: list[str] = Field(min_length=1)
    min_profitable_pairs: int = Field(ge=1)
    min_group_expectancy_lcb: float = 0.0
    min_group_annual_return_lcb: float = 0.0
    min_worst_pair_scenario_return: float = -0.05

    @model_validator(mode="after")
    def _coherent_group(self) -> "PairGroupGateV3":
        if self.pairs != sorted(set(self.pairs)):
            raise ValueError("pair group pairs must be sorted and unique")
        if self.min_profitable_pairs > len(self.pairs):
            raise ValueError("min_profitable_pairs exceeds pair count")
        return self


class EvidenceGateV3(StrictV2Model):
    min_effective_sample_size: float = Field(default=30.0, ge=1)
    min_active_months: int = Field(default=12, ge=1)


class PortfolioSoftTargetsV3(StrictV2Model):
    target_trades_per_day_min: float = Field(default=0.5, gt=0)
    target_trades_per_day_max: float = Field(default=1.5, gt=0)
    median_hold_hours_target: float = Field(default=24.0, gt=0)
    p90_hold_hours_target: float = Field(default=72.0, gt=0)

    @model_validator(mode="after")
    def _ordered_targets(self) -> "PortfolioSoftTargetsV3":
        if self.target_trades_per_day_max < self.target_trades_per_day_min:
            raise ValueError("trade-rate target range is reversed")
        if self.p90_hold_hours_target < self.median_hold_hours_target:
            raise ValueError("p90 holding target is below median target")
        return self


class QualificationPolicyV3(StrictV2Model):
    schema_version: Literal["3.0"] = "3.0"
    policy_version: str = Field(min_length=1)
    required_scenarios: list[EvaluationScenarioV3] = Field(min_length=1)
    pair_groups: list[PairGroupGateV3] = Field(min_length=1)
    development_evidence: EvidenceGateV3 = Field(default_factory=EvidenceGateV3)
    final_test_evidence: EvidenceGateV3 = Field(
        default_factory=lambda: EvidenceGateV3(
            min_effective_sample_size=20.0,
            min_active_months=1,
        )
    )
    max_drawdown_ucb: float = Field(default=0.20, ge=0)
    max_daily_es5_ucb: float = Field(default=0.05, ge=0)
    max_consecutive_losses: int = Field(default=12, ge=0)
    max_drawdown_duration_days: float = Field(default=120.0, ge=0)
    require_final_test: bool = True
    soft_targets: PortfolioSoftTargetsV3 = Field(default_factory=PortfolioSoftTargetsV3)

    @model_validator(mode="after")
    def _coherent_panel(self) -> "QualificationPolicyV3":
        ids = [item.scenario_id for item in self.required_scenarios]
        if ids != sorted(set(ids)):
            raise ValueError("required V3 scenario IDs must be sorted and unique")
        group_ids = [item.group_id for item in self.pair_groups]
        if group_ids != sorted(set(group_ids)):
            raise ValueError("V3 pair groups must be sorted and unique")
        groups = {item.group_id: set(item.pairs) for item in self.pair_groups}
        for scenario in self.required_scenarios:
            if scenario.pair_group not in groups:
                raise ValueError(f"scenario references unknown pair group: {scenario.pair_group}")
            if scenario.pair not in groups[scenario.pair_group]:
                raise ValueError("scenario pair is outside its declared group")
        for group_id, pairs in groups.items():
            base_pairs = {
                item.pair
                for item in self.required_scenarios
                if item.pair_group == group_id and math.isclose(item.cost_multiplier, 1.0)
            }
            if base_pairs != pairs:
                raise ValueError(f"pair group {group_id} lacks a base-cost scenario per pair")
        if self.require_final_test and not any(
            item.role == ScenarioRole.FINAL_TEST for item in self.required_scenarios
        ):
            raise ValueError("require_final_test needs a declared FINAL_TEST scenario")
        return self


class PairGroupResultV3(StrictV2Model):
    group_id: str
    profitable_pairs: int = Field(ge=0)
    pair_count: int = Field(ge=1)
    profitable_pair_ratio: float = Field(ge=0, le=1)
    group_expectancy_lcb: float
    group_annual_return_lcb: float
    worst_pair_scenario_return: float


class CandidateQualityDiagnosticsV3(StrictV2Model):
    independent_pair_trades_per_day: float = Field(ge=0)
    activity_adequacy_score: float = Field(ge=0, le=1)
    median_hold_hours: float | None = Field(default=None, ge=0)
    p90_hold_hours: float | None = Field(default=None, ge=0)
    holding_target_score: float | None = Field(default=None, ge=0, le=1)


class CandidateEvaluationV3(StrictV2Model):
    schema_version: Literal["3.0"] = "3.0"
    candidate_id: str = Field(min_length=1)
    phenotype_hash: str = Field(min_length=8)
    fitness_policy_version: str = Field(min_length=1)
    status: EvaluationStatus
    scenarios: list[BacktestRecordV2] = Field(min_length=1)
    group_results: list[PairGroupResultV3] = Field(default_factory=list)
    gates: list[GateResultV2] = Field(min_length=1)
    diagnostics: CandidateQualityDiagnosticsV3 | None = None
    robust_score: float | None = None

    @model_validator(mode="after")
    def _valid_evidence(self) -> "CandidateEvaluationV3":
        if self.status == EvaluationStatus.VALID:
            if not self.group_results or self.diagnostics is None or self.robust_score is None:
                raise ValueError("VALID V3 candidate lacks aggregate evidence")
            if any(item.status != EvaluationStatus.VALID for item in self.gates):
                raise ValueError("VALID V3 candidate has non-valid gates")
        return self

    @property
    def would_pass(self) -> bool:
        return self.status == EvaluationStatus.VALID and all(
            gate.passed is True for gate in self.gates
        )


def qualification_policy_v3_from_config(
    config: Mapping[str, object],
) -> QualificationPolicyV3:
    raw = config.get("qualification_v3")
    if not isinstance(raw, Mapping) or not raw.get("enabled", False):
        raise ValueError("qualification_v3 must be explicitly enabled")
    return QualificationPolicyV3.model_validate(
        {key: value for key, value in raw.items() if key != "enabled"}
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
    reason_code: str,
) -> GateResultV2:
    return GateResultV2(
        gate_id=gate_id,
        passed=passed,
        status=EvaluationStatus.VALID,
        observed=observed,
        threshold=threshold,
        operator=operator,
        reason_code="PASS" if passed else reason_code,
    )


def _inconclusive(reason: str, detail: str) -> GateResultV2:
    return GateResultV2(
        gate_id="V3_SCENARIO_MATRIX",
        passed=False,
        status=EvaluationStatus.INCONCLUSIVE,
        reason_code=reason,
        detail=detail,
    )


def _identity(record: BacktestRecordV2) -> tuple[object, ...]:
    return (
        record.attempt_id,
        record.candidate_id,
        record.phenotype_hash,
        record.config_hash,
        record.code_version,
        record.data_manifest_hash,
        record.fitness_policy_version,
        record.seed,
        record.worker_count,
        record.wave_id,
        record.experiment_id,
    )


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _quality_diagnostics(
    records: list[BacktestRecordV2],
    targets: PortfolioSoftTargetsV3,
) -> CandidateQualityDiagnosticsV3:
    base = [item for item in records if math.isclose(item.metrics.cost_multiplier, 1.0)]
    period_trade_counts: dict[tuple[date, date], int] = defaultdict(int)
    hold_hours: list[float] = []
    for record in base:
        metrics = record.metrics
        period_trade_counts[(metrics.period_start, metrics.period_end)] += metrics.trade_count
        for trade in record.trades:
            duration = trade.get("trade_duration")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                resolved = float(duration) / 60.0
                if math.isfinite(resolved) and resolved >= 0:
                    hold_hours.append(resolved)
    rates = [
        count / max(1, (end - start).days + 1)
        for (start, end), count in period_trade_counts.items()
    ]
    trade_rate = median(rates) if rates else 0.0
    activity_score = min(1.0, trade_rate / targets.target_trades_per_day_min)
    median_hold = _quantile(hold_hours, 0.5)
    p90_hold = _quantile(hold_hours, 0.9)
    holding_score = None
    if median_hold is not None and p90_hold is not None:
        holding_score = min(
            1.0,
            targets.median_hold_hours_target / max(median_hold, 1e-12),
            targets.p90_hold_hours_target / max(p90_hold, 1e-12),
        )
    return CandidateQualityDiagnosticsV3(
        independent_pair_trades_per_day=trade_rate,
        activity_adequacy_score=activity_score,
        median_hold_hours=median_hold,
        p90_hold_hours=p90_hold,
        holding_target_score=holding_score,
    )


def _pair_group_result(
    group_id: str,
    records: list[BacktestRecordV2],
    rules: PairGroupGateV3,
) -> PairGroupResultV3:
    """Aggregate one explicitly scoped slice without mixing evaluation roles."""
    by_pair: dict[str, list[BacktestRecordV2]] = defaultdict(list)
    for record in records:
        by_pair[record.metrics.pair].append(record)
    pair_returns = {
        pair: median(float(item.metrics.net_return) for item in items)
        for pair, items in by_pair.items()
    }
    pair_expectancies = {
        pair: min(
            min(
                float(item.metrics.net_expectancy_lcb),
                float(item.metrics.net_expectancy_on_committed_capital_lcb),
            )
            for item in items
        )
        for pair, items in by_pair.items()
    }
    pair_annual = {
        pair: median(float(item.metrics.annualized_net_return_lcb) for item in items)
        for pair, items in by_pair.items()
    }
    profitable_pairs = sum(value > 0 for value in pair_returns.values())
    return PairGroupResultV3(
        group_id=group_id,
        profitable_pairs=profitable_pairs,
        pair_count=len(rules.pairs),
        profitable_pair_ratio=profitable_pairs / len(rules.pairs),
        group_expectancy_lcb=median(pair_expectancies.values()),
        group_annual_return_lcb=median(pair_annual.values()),
        worst_pair_scenario_return=min(
            float(item.metrics.net_return) for item in records
        ),
    )


def _pair_group_gates(
    result: PairGroupResultV3,
    rules: PairGroupGateV3,
    prefix: str,
) -> list[GateResultV2]:
    return [
        _gate(
            f"{prefix}_GROUP_EXPECTANCY_LCB",
            result.group_expectancy_lcb > rules.min_group_expectancy_lcb,
            result.group_expectancy_lcb,
            rules.min_group_expectancy_lcb,
            ">",
            f"{prefix}_GROUP_EXPECTANCY_LCB_TOO_LOW",
        ),
        _gate(
            f"{prefix}_GROUP_ANNUAL_RETURN_LCB",
            result.group_annual_return_lcb > rules.min_group_annual_return_lcb,
            result.group_annual_return_lcb,
            rules.min_group_annual_return_lcb,
            ">",
            f"{prefix}_GROUP_ANNUAL_RETURN_LCB_TOO_LOW",
        ),
        _gate(
            f"{prefix}_PROFITABLE_PAIRS",
            result.profitable_pairs >= rules.min_profitable_pairs,
            float(result.profitable_pairs),
            float(rules.min_profitable_pairs),
            ">=",
            f"{prefix}_TOO_FEW_PROFITABLE_PAIRS",
        ),
        _gate(
            f"{prefix}_WORST_PAIR_RETURN",
            result.worst_pair_scenario_return >= rules.min_worst_pair_scenario_return,
            result.worst_pair_scenario_return,
            rules.min_worst_pair_scenario_return,
            ">=",
            f"{prefix}_PAIR_LOSS_TOO_LARGE",
        ),
    ]


def evaluate_candidate_v3(
    records: Iterable[BacktestRecordV2],
    policy: QualificationPolicyV3,
) -> CandidateEvaluationV3:
    scenarios = list(records)
    if not scenarios:
        raise ValueError("at least one V3 scenario record is required")
    first = scenarios[0]
    if any(_identity(item) != _identity(first) for item in scenarios[1:]):
        raise ValueError("V3 scenario records do not describe one frozen candidate")

    actual = [_record_key(item) for item in scenarios]
    required = [item.record_key for item in policy.required_scenarios]
    if len(actual) != len(set(actual)) or set(actual) != set(required):
        detail = json.dumps(
            {
                "duplicate_actual": len(actual) != len(set(actual)),
                "missing": [list(item) for item in sorted(set(required) - set(actual), key=str)],
                "unexpected": [list(item) for item in sorted(set(actual) - set(required), key=str)],
            },
            sort_keys=True,
            default=str,
        )
        return CandidateEvaluationV3(
            candidate_id=first.candidate_id,
            phenotype_hash=first.phenotype_hash,
            fitness_policy_version=first.fitness_policy_version,
            status=EvaluationStatus.INCONCLUSIVE,
            scenarios=scenarios,
            gates=[_inconclusive("INCOMPLETE_V3_SCENARIO_MATRIX", detail)],
        )
    non_valid = [item for item in scenarios if item.metrics.status != EvaluationStatus.VALID]
    if non_valid:
        detail = ", ".join(
            f"{item.metrics.scenario_id}:{item.metrics.status.value}" for item in non_valid
        )
        return CandidateEvaluationV3(
            candidate_id=first.candidate_id,
            phenotype_hash=first.phenotype_hash,
            fitness_policy_version=first.fitness_policy_version,
            status=EvaluationStatus.INCONCLUSIVE,
            scenarios=scenarios,
            gates=[_inconclusive("NON_VALID_V3_SCENARIO", detail)],
        )

    requirement_by_id = {item.scenario_id: item for item in policy.required_scenarios}
    records_by_group: dict[str, list[BacktestRecordV2]] = defaultdict(list)
    for record in scenarios:
        records_by_group[requirement_by_id[record.metrics.scenario_id].pair_group].append(record)

    gates: list[GateResultV2] = []
    group_results: list[PairGroupResultV3] = []
    group_policy = {item.group_id: item for item in policy.pair_groups}
    for group_id in sorted(records_by_group):
        rules = group_policy[group_id]
        base = [
            item
            for item in records_by_group[group_id]
            if math.isclose(item.metrics.cost_multiplier, 1.0)
        ]
        result = _pair_group_result(group_id, base, rules)
        group_results.append(result)
        prefix = group_id.upper().replace("-", "_")
        gates.extend(_pair_group_gates(result, rules, prefix))

        # The sealed window is an independent proof. Development performance may
        # never average away a weak final result.
        final_group = [
            item
            for item in records_by_group[group_id]
            if item.metrics.role == ScenarioRole.FINAL_TEST
        ]
        if final_group:
            final_result = _pair_group_result(group_id, final_group, rules)
            gates.extend(_pair_group_gates(final_result, rules, f"FINAL_{prefix}"))

    metrics = [item.metrics for item in scenarios]
    non_final = [item for item in metrics if item.role != ScenarioRole.FINAL_TEST]
    final = [item for item in metrics if item.role == ScenarioRole.FINAL_TEST]
    gates.extend(
        [
            _gate(
                "MAX_DRAWDOWN_UCB",
                max(float(item.max_drawdown_ucb) for item in metrics)
                <= policy.max_drawdown_ucb,
                max(float(item.max_drawdown_ucb) for item in metrics),
                policy.max_drawdown_ucb,
                "<=",
                "MAX_DRAWDOWN_UCB_TOO_HIGH",
            ),
            _gate(
                "DAILY_ES5_UCB",
                max(float(item.daily_expected_shortfall_5_ucb) for item in metrics)
                <= policy.max_daily_es5_ucb,
                max(float(item.daily_expected_shortfall_5_ucb) for item in metrics),
                policy.max_daily_es5_ucb,
                "<=",
                "DAILY_ES5_UCB_TOO_HIGH",
            ),
            _gate(
                "MAX_CONSECUTIVE_LOSSES",
                max(int(item.max_consecutive_losses) for item in metrics)
                <= policy.max_consecutive_losses,
                float(max(int(item.max_consecutive_losses) for item in metrics)),
                float(policy.max_consecutive_losses),
                "<=",
                "LOSS_STREAK_TOO_HIGH",
            ),
            _gate(
                "MAX_DRAWDOWN_DURATION",
                max(float(item.max_drawdown_duration_days) for item in metrics)
                <= policy.max_drawdown_duration_days,
                max(float(item.max_drawdown_duration_days) for item in metrics),
                policy.max_drawdown_duration_days,
                "<=",
                "DRAWDOWN_DURATION_TOO_HIGH",
            ),
        ]
    )
    if non_final:
        gates.extend(
            [
                _gate(
                    "DEVELOPMENT_EFFECTIVE_SAMPLE_SIZE",
                    min(float(item.effective_sample_size) for item in non_final)
                    >= policy.development_evidence.min_effective_sample_size,
                    min(float(item.effective_sample_size) for item in non_final),
                    policy.development_evidence.min_effective_sample_size,
                    ">=",
                    "DEVELOPMENT_EFFECTIVE_SAMPLE_SIZE_TOO_LOW",
                ),
                _gate(
                    "DEVELOPMENT_ACTIVE_MONTHS",
                    min(item.active_months for item in non_final)
                    >= policy.development_evidence.min_active_months,
                    float(min(item.active_months for item in non_final)),
                    float(policy.development_evidence.min_active_months),
                    ">=",
                    "DEVELOPMENT_ACTIVE_MONTHS_TOO_LOW",
                ),
            ]
        )
    if policy.require_final_test:
        gates.extend(
            [
                _gate(
                    "FINAL_TEST_PRESENT",
                    bool(final),
                    float(len(final)),
                    1.0,
                    ">=",
                    "FINAL_TEST_MISSING",
                ),
                _gate(
                    "FINAL_EFFECTIVE_SAMPLE_SIZE",
                    bool(final)
                    and min(float(item.effective_sample_size) for item in final)
                    >= policy.final_test_evidence.min_effective_sample_size,
                    min((float(item.effective_sample_size) for item in final), default=0.0),
                    policy.final_test_evidence.min_effective_sample_size,
                    ">=",
                    "FINAL_EFFECTIVE_SAMPLE_SIZE_TOO_LOW",
                ),
                _gate(
                    "FINAL_ACTIVE_MONTHS",
                    bool(final)
                    and min(item.active_months for item in final)
                    >= policy.final_test_evidence.min_active_months,
                    float(min((item.active_months for item in final), default=0)),
                    float(policy.final_test_evidence.min_active_months),
                    ">=",
                    "FINAL_ACTIVE_MONTHS_TOO_LOW",
                ),
            ]
        )

    diagnostics = _quality_diagnostics(scenarios, policy.soft_targets)
    robust_score = (
        median(item.group_annual_return_lcb for item in group_results)
        - max(float(item.max_drawdown_ucb) for item in metrics)
        - max(float(item.daily_expected_shortfall_5_ucb) for item in metrics)
    )
    return CandidateEvaluationV3(
        candidate_id=first.candidate_id,
        phenotype_hash=first.phenotype_hash,
        fitness_policy_version=first.fitness_policy_version,
        status=EvaluationStatus.VALID,
        scenarios=scenarios,
        group_results=group_results,
        gates=gates,
        diagnostics=diagnostics,
        robust_score=robust_score,
    )
