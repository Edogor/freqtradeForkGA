"""Strict, versioned result contracts for the GA V2 execution path.

These models are intentionally separate from legacy HOF/registry dictionaries.
They fail closed on unknown fields, non-finite numbers, and successful records
that omit required risk or return measurements.  Legacy data must be adapted
and replayed before it can become a valid V2 decision input.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from genetic_algorithm.evaluation.confidence_metrics_v2 import (
    EXPECTANCY_CONTRACT_VERSION,
)
from genetic_algorithm.evaluation.equity_metrics_v2 import (
    CALENDAR_DAYS_PER_YEAR,
    RISK_METRIC_CONTRACT_VERSION,
)


METRIC_SCHEMA_VERSION = "2.0"
RESULT_SCHEMA_VERSION = "2.0"


class StrictV2Model(BaseModel):
    """Base settings shared by every persisted V2 contract."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
    )


class EvaluationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAIL = "FAIL"


class ScenarioRole(str, Enum):
    TRAIN = "TRAIN"
    PAIR_VALIDATION = "PAIR_VALIDATION"
    TEMPORAL_VALIDATION = "TEMPORAL_VALIDATION"
    # Legacy ambiguous role. Canonical V2 split validation rejects it and
    # requires one of the two explicit validation axes above.
    INNER_VALIDATION = "INNER_VALIDATION"
    FINAL_TEST = "FINAL_TEST"


class AttemptStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    INVALID_RESULT = "INVALID_RESULT"


class GateResultV2(StrictV2Model):
    """One non-compensable promotion rule and its measured evidence."""

    gate_id: str = Field(min_length=1)
    passed: Optional[bool] = None
    status: EvaluationStatus
    observed: Optional[float] = None
    threshold: Optional[float] = None
    operator: Optional[Literal[">", ">=", "<", "<=", "=="]] = None
    reason_code: str = Field(min_length=1)
    detail: Optional[str] = None

    @model_validator(mode="after")
    def _consistent_pass_state(self) -> "GateResultV2":
        if self.status == EvaluationStatus.VALID and self.passed is None:
            raise ValueError("a VALID gate must set passed")
        if self.status != EvaluationStatus.VALID and self.passed is True:
            raise ValueError("an invalid/inconclusive gate cannot pass")
        return self


class ScenarioMetricsV2(StrictV2Model):
    """Metrics for exactly one pair × time block × cost scenario."""

    scenario_id: str = Field(min_length=1)
    pair: str = Field(min_length=3)
    timeframe: str = Field(min_length=1)
    role: ScenarioRole
    period_start: date
    period_end: date
    cost_multiplier: float = Field(gt=0)

    status: EvaluationStatus
    success: bool
    no_trades: bool = False
    error_code: Optional[str] = None
    error_detail: Optional[str] = None

    # Decimal returns/drawdowns, never percentage points.
    net_return: Optional[float] = None
    annualized_net_return: Optional[float] = None
    annualized_net_return_lcb: Optional[float] = None
    max_drawdown: Optional[float] = Field(default=None, ge=0)
    max_drawdown_ucb: Optional[float] = Field(default=None, ge=0)
    daily_expected_shortfall_5: Optional[float] = Field(default=None, ge=0)
    daily_expected_shortfall_5_ucb: Optional[float] = Field(default=None, ge=0)
    daily_sharpe_ratio: Optional[float] = None
    daily_sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    net_expectancy: Optional[float] = None
    net_expectancy_lcb: Optional[float] = None
    expectancy_contract_version: Optional[
        Literal["net-expectancy-clustered-v1"]
    ] = None
    expectancy_quote_currency: Optional[str] = None
    net_expectancy_on_committed_capital: Optional[float] = None
    net_expectancy_on_committed_capital_lcb: Optional[float] = None
    mean_net_profit_abs_per_trade: Optional[float] = None
    total_net_profit_abs: Optional[float] = None
    total_committed_capital: Optional[float] = Field(default=None, gt=0)
    expectancy_temporal_clusters: Optional[int] = Field(default=None, ge=1)
    expectancy_cluster_days: Optional[int] = Field(default=None, ge=1)
    expectancy_pair_count: Optional[int] = Field(default=None, ge=1)
    expectancy_effective_pair_count: Optional[float] = Field(default=None, ge=1)
    expectancy_pair_capital_hhi: Optional[float] = Field(default=None, gt=0, le=1)
    expectancy_max_trade_capital_share: Optional[float] = Field(
        default=None, gt=0, le=1
    )
    expectancy_max_cluster_capital_share: Optional[float] = Field(
        default=None, gt=0, le=1
    )
    expectancy_serial_effective_sample_size: Optional[float] = Field(
        default=None, ge=1
    )
    expectancy_capital_effective_sample_size: Optional[float] = Field(
        default=None, ge=1
    )
    expectancy_temporal_effective_sample_size: Optional[float] = Field(
        default=None, ge=1
    )
    profit_factor: Optional[float] = Field(default=None, ge=0)
    win_rate: Optional[float] = Field(default=None, ge=0, le=1)

    trade_count: int = Field(ge=0)
    effective_sample_size: Optional[float] = Field(default=None, ge=0)
    active_months: int = Field(ge=0)
    max_consecutive_losses: Optional[int] = Field(default=None, ge=0)
    max_drawdown_duration_days: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _fail_closed_validity(self) -> "ScenarioMetricsV2":
        if self.period_end <= self.period_start:
            raise ValueError("period_end must be after period_start")

        expectancy_required = (
            "net_expectancy",
            "net_expectancy_lcb",
            "expectancy_quote_currency",
            "net_expectancy_on_committed_capital",
            "net_expectancy_on_committed_capital_lcb",
            "mean_net_profit_abs_per_trade",
            "total_net_profit_abs",
            "total_committed_capital",
            "expectancy_temporal_clusters",
            "expectancy_cluster_days",
            "expectancy_pair_count",
            "expectancy_effective_pair_count",
            "expectancy_pair_capital_hhi",
            "expectancy_max_trade_capital_share",
            "expectancy_max_cluster_capital_share",
            "expectancy_serial_effective_sample_size",
            "expectancy_capital_effective_sample_size",
            "expectancy_temporal_effective_sample_size",
            "effective_sample_size",
        )
        required_when_valid = (
            "net_return",
            "annualized_net_return",
            "annualized_net_return_lcb",
            "max_drawdown",
            "max_drawdown_ucb",
            "daily_expected_shortfall_5",
            "daily_expected_shortfall_5_ucb",
            "expectancy_contract_version",
            "profit_factor",
            "win_rate",
            "max_consecutive_losses",
            "max_drawdown_duration_days",
            *expectancy_required,
        )
        if self.status == EvaluationStatus.VALID:
            if not self.success or self.no_trades or self.trade_count == 0:
                raise ValueError("VALID scenario requires a successful backtest with trades")
            missing = [name for name in required_when_valid if getattr(self, name) is None]
            if missing:
                raise ValueError(f"VALID scenario is missing metrics: {', '.join(missing)}")
        elif not self.error_code:
            raise ValueError("non-VALID scenario requires a machine-readable error_code")
        if (
            self.expectancy_contract_version is not None
            and self.expectancy_contract_version != EXPECTANCY_CONTRACT_VERSION
        ):
            raise ValueError("unsupported expectancy contract version")
        if self.expectancy_contract_version is not None:
            missing_expectancy = [
                name for name in expectancy_required if getattr(self, name) is None
            ]
            if missing_expectancy:
                raise ValueError(
                    "incomplete expectancy evidence: "
                    + ", ".join(missing_expectancy)
                )
            if not self.expectancy_quote_currency:
                raise ValueError("expectancy quote currency must be non-empty")
            if self.trade_count <= 0:
                raise ValueError("expectancy evidence requires trades")
            if int(self.expectancy_temporal_clusters) > self.trade_count:
                raise ValueError("temporal clusters exceed measured trades")
            if int(self.expectancy_pair_count) > self.trade_count:
                raise ValueError("pair count exceeds measured trades")
            expected_capital_return = (
                float(self.total_net_profit_abs) / float(self.total_committed_capital)
            )
            if not math.isclose(
                float(self.net_expectancy_on_committed_capital),
                expected_capital_return,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError("committed-capital expectancy does not reconcile")
            expected_mean_profit = float(self.total_net_profit_abs) / self.trade_count
            if not math.isclose(
                float(self.mean_net_profit_abs_per_trade),
                expected_mean_profit,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError("mean absolute trade profit does not reconcile")
            component_n_eff = min(
                float(self.expectancy_serial_effective_sample_size),
                float(self.expectancy_capital_effective_sample_size),
                float(self.expectancy_temporal_effective_sample_size),
            )
            if not math.isclose(
                float(self.effective_sample_size),
                component_n_eff,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError("effective sample size does not match cluster components")
            if float(self.effective_sample_size) > min(
                self.trade_count,
                int(self.expectancy_temporal_clusters),
            ):
                raise ValueError("effective sample size exceeds measured observations")
            if float(self.expectancy_serial_effective_sample_size) > int(
                self.expectancy_temporal_clusters
            ) + 1e-10:
                raise ValueError("serial effective sample size exceeds temporal clusters")
            if float(self.expectancy_capital_effective_sample_size) > (
                self.trade_count + 1e-10
            ):
                raise ValueError("capital effective sample size exceeds measured trades")
            if float(self.expectancy_temporal_effective_sample_size) > int(
                self.expectancy_temporal_clusters
            ) + 1e-10:
                raise ValueError("temporal effective sample size exceeds temporal clusters")
            inverse_pair_hhi = 1.0 / float(self.expectancy_pair_capital_hhi)
            if not math.isclose(
                float(self.expectancy_effective_pair_count),
                inverse_pair_hhi,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError("effective pair count does not match pair concentration")
            if float(self.expectancy_effective_pair_count) > int(
                self.expectancy_pair_count
            ) + 1e-10:
                raise ValueError("effective pair count exceeds measured pairs")
            if float(self.expectancy_max_trade_capital_share) + 1e-12 < (
                1.0 / self.trade_count
            ):
                raise ValueError("max trade capital share is below its feasible minimum")
            if float(self.expectancy_max_cluster_capital_share) + 1e-12 < (
                1.0 / int(self.expectancy_temporal_clusters)
            ):
                raise ValueError("max cluster capital share is below its feasible minimum")
        return self


class BacktestRecordV2(StrictV2Model):
    """Canonical measured output of one scenario backtest."""

    metric_schema_version: Literal["2.0"] = METRIC_SCHEMA_VERSION
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

    fee_rate: float = Field(ge=0)
    slippage_rate: float = Field(ge=0)
    spread_rate: float = Field(ge=0)
    funding_rate: float = Field(ge=0)
    risk_metric_contract_version: Literal["calendar-effective-v1"] = (
        RISK_METRIC_CONTRACT_VERSION
    )
    risk_periods_per_year: Literal[365] = CALENDAR_DAYS_PER_YEAR
    annual_risk_free_rate: float = Field(default=0.0, gt=-1.0)
    periodic_risk_free_rate: Optional[float] = 0.0
    equity_method: Optional[Literal["REALIZED_CLOSE", "MARK_TO_MARKET"]] = None
    metrics: ScenarioMetricsV2

    # Calendar-aligned, after-cost portfolio observations.
    daily_net_returns: List[float] = Field(default_factory=list)
    equity_curve: List[float] = Field(default_factory=list)
    trades: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid_records_need_raw_evidence(self) -> "BacktestRecordV2":
        expected_periodic_rate = math.expm1(
            math.log1p(self.annual_risk_free_rate) / self.risk_periods_per_year
        )
        if self.periodic_risk_free_rate is not None and not math.isclose(
            self.periodic_risk_free_rate,
            expected_periodic_rate,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("periodic risk-free rate does not match annual convention")
        if self.metrics.status == EvaluationStatus.VALID:
            if self.equity_method != "MARK_TO_MARKET":
                raise ValueError("VALID backtest requires MARK_TO_MARKET equity")
            if self.periodic_risk_free_rate is None:
                raise ValueError("VALID backtest requires risk-metric convention evidence")
            if not self.daily_net_returns or not self.equity_curve:
                raise ValueError("VALID backtest requires daily returns and an equity curve")
            if len(self.trades) != self.metrics.trade_count:
                raise ValueError("trade payload count differs from metrics.trade_count")
        return self


class CandidateEvaluationV2(StrictV2Model):
    """Comparable replay result and gate evidence for one frozen phenotype."""

    candidate_id: str = Field(min_length=1)
    phenotype_hash: str = Field(min_length=8)
    fitness_policy_version: str = Field(min_length=1)
    status: EvaluationStatus
    scenarios: List[BacktestRecordV2] = Field(min_length=1)
    gates: List[GateResultV2] = Field(default_factory=list)
    robust_score: Optional[float] = None
    pareto_rank: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _status_matches_scenarios(self) -> "CandidateEvaluationV2":
        scenario_statuses = {record.metrics.status for record in self.scenarios}
        if self.status == EvaluationStatus.VALID:
            if scenario_statuses != {EvaluationStatus.VALID}:
                raise ValueError("VALID candidate requires every scenario to be VALID")
            if not self.gates:
                raise ValueError("VALID candidate requires evaluated promotion gates")
            if any(gate.status != EvaluationStatus.VALID for gate in self.gates):
                raise ValueError("VALID candidate requires every gate to be evaluated")
            if self.robust_score is None:
                raise ValueError("VALID candidate requires robust_score")
        return self


class PromotionDecisionV2(StrictV2Model):
    """Persisted decision; shadow decisions can never authorize execution."""

    decision_id: str = Field(min_length=1)
    wave_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    created_at: datetime
    shadow_mode: bool = True
    outcome: Literal["WOULD_PASS", "PASS", "FAIL", "INCONCLUSIVE", "INVALID"]
    promotion_authorized: bool = False
    gate_results: List[GateResultV2] = Field(min_length=1)
    reason_codes: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _shadow_cannot_authorize(self) -> "PromotionDecisionV2":
        if self.shadow_mode and self.promotion_authorized:
            raise ValueError("shadow decisions cannot authorize promotion")
        if self.promotion_authorized and self.outcome != "PASS":
            raise ValueError("promotion can only be authorized for PASS")
        return self


class AttemptManifestV2(StrictV2Model):
    """Immutable provenance captured before an attempt is executed."""

    schema_version: Literal["2.0"] = RESULT_SCHEMA_VERSION
    attempt_id: str = Field(min_length=1)
    wave_id: str = Field(min_length=1)
    parent_wave_id: Optional[str] = None
    experiment_id: str = Field(min_length=1)
    created_at: datetime
    config_hash: str = Field(min_length=8)
    code_version: str = Field(min_length=1)
    dirty_patch_hash: Optional[str] = None
    data_manifest_hash: str = Field(min_length=8)
    split_manifest_hash: Optional[str] = None
    fitness_policy_version: str = Field(min_length=1)
    seeds: List[int] = Field(min_length=1)
    worker_count: int = Field(ge=1)
    resolved_config_path: str = Field(min_length=1)
    artifact_root: str = Field(min_length=1)


class AttemptResultV2(StrictV2Model):
    """Terminal attempt result consumed by reconciliation and wave planning."""

    schema_version: Literal["2.0"] = RESULT_SCHEMA_VERSION
    attempt_id: str = Field(min_length=1)
    status: AttemptStatus
    started_at: datetime
    finished_at: datetime
    manifest: AttemptManifestV2
    candidate_evaluations: List[CandidateEvaluationV2] = Field(default_factory=list)
    artifact_hashes: Dict[str, str] = Field(default_factory=dict)
    error_code: Optional[str] = None
    error_detail: Optional[str] = None

    @model_validator(mode="after")
    def _terminal_consistency(self) -> "AttemptResultV2":
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        if self.attempt_id != self.manifest.attempt_id:
            raise ValueError("result attempt_id differs from manifest")
        if self.status == AttemptStatus.SUCCEEDED:
            if self.error_code:
                raise ValueError("SUCCEEDED result cannot contain error_code")
            if not self.candidate_evaluations:
                raise ValueError("SUCCEEDED result requires candidate evaluations")
            if not self.artifact_hashes:
                raise ValueError("SUCCEEDED result requires artifact hashes")
        elif not self.error_code:
            raise ValueError("non-success result requires error_code")
        return self
