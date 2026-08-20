"""Deterministic, fail-closed analysis of a reconciled GA V2 wave.

The analyzer is deliberately read-only with respect to execution.  It consumes
one immutable parent snapshot, re-verifies every referenced result, aggregates
evidence by experiment and executable phenotype, and produces a hash-chained
``ANALYSIS`` decision.  It neither plans nor queues a child wave.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from statistics import mean, median
from typing import Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptResultV2,
    AttemptStatus,
    CandidateEvaluationV2,
    EvaluationStatus,
    StrictV2Model,
)
from genetic_algorithm.orchestration.wave_state_v2 import (
    AttemptEvidenceType,
    ExperimentArmType,
    ExperimentSpecV2,
    WaveAttemptSnapshotV2,
    WaveDecisionType,
    WaveDecisionV2,
    WaveLifecycleStatus,
    WaveResultSnapshotV2,
    WaveStateError,
    WaveStateStoreV2,
    WaveStateV2,
)


class WaveAnalysisError(ValueError):
    """Raised when analysis input is incomplete, inconsistent, or mutable."""


class AnalysisHealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    BLOCKED = "BLOCKED"


class CandidateEligibilityStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"


def _aware(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


class WaveAnalyzerPolicyV2(StrictV2Model):
    """Versioned health and evidence requirements fixed before analysis."""

    schema_version: Literal["2.0"] = "2.0"
    analysis_policy_version: str = Field(min_length=1)
    required_result_policy_version: str = Field(min_length=1)
    require_control_arm: bool = True
    max_abort_fraction: float = Field(default=0.0, ge=0, le=1)
    max_non_success_fraction: float = Field(default=0.0, ge=0, le=1)
    min_candidate_seed_count: int = Field(default=2, ge=1)
    require_all_candidate_gates: bool = True
    require_eligible_candidate_for_planning: bool = False

    @model_validator(mode="after")
    def _coherent_failure_limits(self) -> WaveAnalyzerPolicyV2:
        if self.max_abort_fraction > self.max_non_success_fraction:
            raise ValueError("max_abort_fraction cannot exceed max_non_success_fraction")
        return self

    @property
    def policy_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


class ExperimentAnalysisV2(StrictV2Model):
    experiment_id: str = Field(min_length=1)
    arm_type: ExperimentArmType
    expected_attempt_count: int = Field(ge=1)
    verified_result_count: int = Field(ge=0)
    successful_attempt_count: int = Field(ge=0)
    documented_abort_count: int = Field(ge=0)
    candidate_observation_count: int = Field(ge=0)
    abort_fraction: float = Field(ge=0, le=1)
    non_success_fraction: float = Field(ge=0, le=1)
    health_status: AnalysisHealthStatus
    reason_codes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_counts(self) -> ExperimentAnalysisV2:
        if self.verified_result_count + self.documented_abort_count != self.expected_attempt_count:
            raise ValueError("experiment evidence count differs from expected attempts")
        if self.successful_attempt_count > self.verified_result_count:
            raise ValueError("successful attempts exceed verified results")
        if self.health_status == AnalysisHealthStatus.HEALTHY:
            if self.reason_codes != ["EXPERIMENT_HEALTHY"]:
                raise ValueError("healthy experiment needs canonical reason code")
        elif self.reason_codes == ["EXPERIMENT_HEALTHY"]:
            raise ValueError("blocked experiment cannot be marked healthy")
        return self


class CandidateObservationRefV2(StrictV2Model):
    attempt_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    seed: int
    result_path: str = Field(min_length=1)
    result_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _absolute_result_path(self) -> CandidateObservationRefV2:
        if not Path(self.result_path).is_absolute():
            raise ValueError("candidate observation result_path must be absolute")
        return self


class CandidateAnalysisV2(StrictV2Model):
    experiment_id: str = Field(min_length=1)
    phenotype_hash: str = Field(min_length=8)
    candidate_ids: list[str] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    observation_refs: list[CandidateObservationRefV2] = Field(min_length=1)
    observation_count: int = Field(ge=1)
    valid_observation_count: int = Field(ge=0)
    eligibility_status: CandidateEligibilityStatus
    reason_codes: list[str] = Field(min_length=1)
    failed_gate_reason_codes: list[str] = Field(default_factory=list)
    comparison_panel_hash: str | None = Field(default=None, min_length=64, max_length=64)
    median_robust_score: float | None = None
    worst_annualized_return_lcb: float | None = None
    median_annualized_return_lcb: float | None = None
    worst_max_drawdown_ucb: float | None = Field(default=None, ge=0)
    worst_daily_es5_ucb: float | None = Field(default=None, ge=0)
    worst_net_expectancy_lcb: float | None = None
    median_profit_factor: float | None = Field(default=None, ge=0)
    median_win_rate: float | None = Field(default=None, ge=0, le=1)
    min_effective_sample_size: float | None = Field(default=None, ge=0)
    min_trades_per_active_month: float | None = Field(default=None, ge=0)
    min_scenario_net_return: float | None = None
    profitable_scenario_ratio: float | None = Field(default=None, ge=0, le=1)
    max_drawdown_duration_days: float | None = Field(default=None, ge=0)
    median_gate_alignment_score: float | None = Field(default=None, ge=0, le=1)
    failed_gate_count: int = Field(default=0, ge=0)
    scenario_trade_count_sum: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _canonical_candidate(self) -> CandidateAnalysisV2:
        self._validate_identity_refs()
        self._validate_candidate_evidence()
        return self

    def _validate_identity_refs(self) -> None:
        if self.candidate_ids != sorted(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique and sorted")
        if self.seeds != sorted(set(self.seeds)):
            raise ValueError("candidate seeds must be unique and sorted")
        if self.observation_count != len(self.seeds):
            raise ValueError("candidate observation count must equal unique seed count")
        ordered_refs = sorted(
            self.observation_refs,
            key=lambda item: (item.seed, item.attempt_id, item.candidate_id),
        )
        if self.observation_refs != ordered_refs:
            raise ValueError("candidate observation refs must be sorted")
        if len(self.observation_refs) != self.observation_count:
            raise ValueError("candidate observation refs differ from observation count")
        if [item.seed for item in self.observation_refs] != self.seeds:
            raise ValueError("candidate observation refs differ from candidate seeds")
        if sorted({item.candidate_id for item in self.observation_refs}) != self.candidate_ids:
            raise ValueError("candidate observation refs differ from candidate IDs")

    def _validate_candidate_evidence(self) -> None:
        if self.valid_observation_count > self.observation_count:
            raise ValueError("valid observations exceed total observations")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("candidate reason_codes must be unique and sorted")
        if self.failed_gate_reason_codes != sorted(set(self.failed_gate_reason_codes)):
            raise ValueError("failed gate reasons must be unique and sorted")
        if self.eligibility_status == CandidateEligibilityStatus.ELIGIBLE:
            if self.reason_codes != ["CANDIDATE_ELIGIBLE"]:
                raise ValueError("eligible candidate needs canonical reason code")
            if self.valid_observation_count != self.observation_count:
                raise ValueError("eligible candidate requires every observation to be valid")
            if self.comparison_panel_hash is None:
                raise ValueError("eligible candidate lacks common comparison panel")
            required_metrics = (
                self.median_robust_score,
                self.worst_annualized_return_lcb,
                self.median_annualized_return_lcb,
                self.worst_max_drawdown_ucb,
                self.worst_daily_es5_ucb,
                self.worst_net_expectancy_lcb,
                self.median_profit_factor,
                self.median_win_rate,
                self.min_effective_sample_size,
                self.min_trades_per_active_month,
            )
            if any(value is None for value in required_metrics):
                raise ValueError("eligible candidate lacks aggregate metrics")


class WaveAnalysisV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    wave_id: str = Field(min_length=1)
    created_at: datetime
    snapshot_hash: str = Field(min_length=64, max_length=64)
    result_policy_version: str = Field(min_length=1)
    analyzer_policy: WaveAnalyzerPolicyV2
    analyzer_policy_hash: str = Field(min_length=64, max_length=64)
    experiments: list[ExperimentAnalysisV2] = Field(min_length=1)
    candidates: list[CandidateAnalysisV2] = Field(default_factory=list)
    planning_allowed: bool
    has_eligible_candidates: bool
    reason_codes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_analysis(self) -> WaveAnalysisV2:
        _aware(self.created_at, "created_at")
        if self.analyzer_policy.policy_hash != self.analyzer_policy_hash:
            raise ValueError("analyzer policy hash differs from content")
        if self.result_policy_version != self.analyzer_policy.required_result_policy_version:
            raise ValueError("result policy differs from analyzer requirement")
        ordered_experiments = sorted(self.experiments, key=lambda item: item.experiment_id)
        if self.experiments != ordered_experiments:
            raise ValueError("experiment analyses must be sorted")
        ordered_candidates = sorted(
            self.candidates,
            key=lambda item: (item.experiment_id, item.phenotype_hash),
        )
        if self.candidates != ordered_candidates:
            raise ValueError("candidate analyses must be sorted")
        experiment_ids = {item.experiment_id for item in self.experiments}
        if any(item.experiment_id not in experiment_ids for item in self.candidates):
            raise ValueError("candidate analysis references unknown experiment")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("analysis reason_codes must be unique and sorted")
        actual_eligible = any(
            item.eligibility_status == CandidateEligibilityStatus.ELIGIBLE
            for item in self.candidates
        )
        if actual_eligible != self.has_eligible_candidates:
            raise ValueError("has_eligible_candidates differs from candidate analyses")
        if self.planning_allowed and any(
            item.health_status != AnalysisHealthStatus.HEALTHY
            for item in self.experiments
        ):
            raise ValueError("planning cannot be allowed with blocked experiment")
        return self

    @property
    def analysis_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))

    def to_decision(self) -> WaveDecisionV2:
        return WaveDecisionV2(
            decision_id=f"analysis-{self.analysis_hash[:24]}",
            wave_id=self.wave_id,
            decision_type=WaveDecisionType.ANALYSIS,
            created_at=self.created_at,
            actor="wave-analyzer-v2",
            reason_codes=self.reason_codes,
            input_hash=self.snapshot_hash,
            payload={
                "analysis": self.model_dump(mode="json"),
                "analysis_hash": self.analysis_hash,
                "analyzer_policy_hash": self.analyzer_policy_hash,
                "planning_allowed": self.planning_allowed,
                "has_eligible_candidates": self.has_eligible_candidates,
            },
        )


class _CandidateObservation:
    def __init__(
        self,
        *,
        attempt_id: str,
        seed: int,
        result_path: str,
        result_sha256: str,
        candidate: CandidateEvaluationV2,
    ) -> None:
        self.attempt_id = attempt_id
        self.seed = seed
        self.result_path = result_path
        self.result_sha256 = result_sha256
        self.candidate = candidate


def _verified_result(
    snapshot: WaveResultSnapshotV2,
    attempt: WaveAttemptSnapshotV2,
) -> AttemptResultV2:
    if attempt.result_path is None or attempt.result_sha256 is None:
        raise WaveAnalysisError("verified snapshot attempt lacks result provenance")
    path = Path(attempt.result_path)
    if not path.is_absolute() or path.name != V2ArtifactStore.RESULT_NAME:
        raise WaveAnalysisError("snapshot result path is not canonical")
    if not path.is_file():
        raise ArtifactIntegrityError("snapshot result path is missing")
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_hash != attempt.result_sha256:
        raise ArtifactIntegrityError("snapshot result hash differs from file")
    result = V2ArtifactStore(path.parent).read_verified_result()
    _validate_result_against_snapshot(snapshot, attempt, result)
    return result


def _validate_result_against_snapshot(
    snapshot: WaveResultSnapshotV2,
    attempt: WaveAttemptSnapshotV2,
    result: AttemptResultV2,
) -> None:
    if result.attempt_id != attempt.attempt_id:
        raise ArtifactIntegrityError("snapshot result belongs to another attempt")
    if result.status != attempt.result_status:
        raise ArtifactIntegrityError("snapshot result status differs from evidence")
    if result.finished_at != attempt.result_finished_at:
        raise ArtifactIntegrityError("snapshot result timestamp differs from evidence")
    manifest_hash = canonical_config_hash(result.manifest.model_dump(mode="json"))
    if manifest_hash != attempt.manifest_hash:
        raise ArtifactIntegrityError("snapshot result manifest hash differs from evidence")
    if result.manifest.wave_id != snapshot.wave_id:
        raise ArtifactIntegrityError("snapshot result belongs to another wave")
    if result.manifest.experiment_id != attempt.experiment_id:
        raise ArtifactIntegrityError("snapshot result belongs to another experiment")
    if result.manifest.seeds != [attempt.seed]:
        raise ArtifactIntegrityError("snapshot result seed differs from evidence")
    if result.manifest.fitness_policy_version != snapshot.policy_version:
        raise ArtifactIntegrityError("snapshot result policy differs from wave")
    if result.status != AttemptStatus.SUCCEEDED and result.candidate_evaluations:
        raise WaveAnalysisError("non-success result contains candidate evaluations")


def _validate_experiment_inputs(
    snapshot: WaveResultSnapshotV2,
    experiments: list[ExperimentSpecV2],
) -> dict[str, ExperimentSpecV2]:
    if not experiments:
        raise WaveAnalysisError("analysis requires experiment specifications")
    experiment_map = {item.experiment_id: item for item in experiments}
    if len(experiment_map) != len(experiments):
        raise WaveAnalysisError("duplicate experiment specification")
    if any(item.wave_id != snapshot.wave_id for item in experiments):
        raise WaveAnalysisError("experiment specification belongs to another wave")
    snapshot_ids = {item.experiment_id for item in snapshot.attempt_results}
    if snapshot_ids != set(experiment_map):
        missing = sorted(set(experiment_map) - snapshot_ids)
        unexpected = sorted(snapshot_ids - set(experiment_map))
        raise WaveAnalysisError(
            f"snapshot experiment set differs from specs; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for experiment in experiments:
        evidence = [
            item
            for item in snapshot.attempt_results
            if item.experiment_id == experiment.experiment_id
        ]
        actual = {(item.attempt_id, item.seed, item.ordinal) for item in evidence}
        expected = {
            (item.attempt_id, item.seed, item.ordinal)
            for item in experiment.expected_attempts
        }
        if actual != expected:
            raise WaveAnalysisError(
                f"snapshot attempts differ from experiment spec: {experiment.experiment_id}"
            )
    return experiment_map


def _candidate_evidence_status(
    candidate: CandidateEvaluationV2,
    policy: WaveAnalyzerPolicyV2,
) -> tuple[bool, list[str]]:
    reasons: set[str] = set()
    if candidate.status != EvaluationStatus.VALID:
        reasons.add(f"CANDIDATE_STATUS_{candidate.status.value}")
    if candidate.robust_score is None:
        reasons.add("ROBUST_SCORE_MISSING")
    if not candidate.gates:
        reasons.add("CANDIDATE_GATES_MISSING")
    for gate in candidate.gates:
        if gate.status != EvaluationStatus.VALID:
            reasons.add(f"GATE_{gate.gate_id}_{gate.status.value}")
    measurement_valid = not reasons
    if policy.require_all_candidate_gates:
        for gate in candidate.gates:
            if gate.status == EvaluationStatus.VALID and gate.passed is not True:
                reasons.add(gate.reason_code)
    return measurement_valid, sorted(reasons)


def _comparison_panel_hash(candidate: CandidateEvaluationV2) -> str:
    cells = [
        {
            "scenario_id": record.metrics.scenario_id,
            "pair": record.metrics.pair,
            "timeframe": record.metrics.timeframe,
            "role": record.metrics.role.value,
            "period_start": record.metrics.period_start.isoformat(),
            "period_end": record.metrics.period_end.isoformat(),
            "cost_multiplier": record.metrics.cost_multiplier,
            "fee_rate": record.fee_rate,
            "slippage_rate": record.slippage_rate,
            "spread_rate": record.spread_rate,
            "funding_rate": record.funding_rate,
            "equity_method": record.equity_method,
            "data_manifest_hash": record.data_manifest_hash,
            "code_version": record.code_version,
        }
        for record in candidate.scenarios
    ]
    cells.sort(
        key=lambda item: (
            str(item["scenario_id"]),
            str(item["pair"]),
            str(item["timeframe"]),
            str(item["role"]),
            str(item["period_start"]),
            float(item["cost_multiplier"]),
        )
    )
    return canonical_config_hash({"cells": cells})


def _gate_alignment_score(candidate: CandidateEvaluationV2) -> float:
    """Return a shadow-only 0..1 proximity score for declared numeric gates."""

    components: list[float] = []
    for gate in candidate.gates:
        if gate.status != EvaluationStatus.VALID:
            continue
        if gate.passed is True:
            components.append(1.0)
            continue
        if gate.observed is None or gate.threshold is None or gate.operator is None:
            components.append(0.0)
            continue
        observed = float(gate.observed)
        threshold = float(gate.threshold)
        if gate.operator in {">", ">="} and threshold > 0:
            components.append(max(0.0, min(1.0, observed / threshold)))
        elif gate.operator in {"<", "<="} and observed > 0 and threshold >= 0:
            components.append(max(0.0, min(1.0, threshold / observed)))
        else:
            # Zero/negative thresholds have no stable ratio interpretation.
            components.append(0.0)
    return mean(components) if components else 0.0


def _candidate_analysis(
    experiment_id: str,
    phenotype_hash: str,
    observations: list[_CandidateObservation],
    policy: WaveAnalyzerPolicyV2,
) -> CandidateAnalysisV2:
    by_seed: dict[int, _CandidateObservation] = {}
    for observation in observations:
        if observation.seed in by_seed:
            raise WaveAnalysisError(
                "duplicate phenotype observation for experiment and seed: "
                f"{experiment_id}/{phenotype_hash}/{observation.seed}"
            )
        by_seed[observation.seed] = observation

    valid: list[_CandidateObservation] = []
    reasons: set[str] = set()
    failed_gates: set[str] = set()
    for observation in by_seed.values():
        measurement_valid, observation_reasons = _candidate_evidence_status(
            observation.candidate, policy
        )
        if measurement_valid:
            valid.append(observation)
        else:
            reasons.add("NON_VALID_CANDIDATE_OBSERVATION")
        reasons.update(observation_reasons)
        failed_gates.update(
            reason
            for reason in observation_reasons
            if reason.startswith("GATE_")
            or any(gate.reason_code == reason for gate in observation.candidate.gates)
        )
    if len(by_seed) < policy.min_candidate_seed_count:
        reasons.add("INSUFFICIENT_CANDIDATE_SEEDS")
    panel_hashes = {_comparison_panel_hash(item.candidate) for item in by_seed.values()}
    comparison_panel_hash = next(iter(panel_hashes)) if len(panel_hashes) == 1 else None
    if comparison_panel_hash is None:
        reasons.add("INCONSISTENT_CANDIDATE_PANELS")
    eligible = not reasons
    metric_candidates = [item.candidate for item in valid]
    scenarios = [record for item in metric_candidates for record in item.scenarios]
    robust_scores = [float(item.robust_score) for item in metric_candidates]
    annual_lcbs = [
        float(record.metrics.annualized_net_return_lcb) for record in scenarios
    ]
    drawdown_ucbs = [float(record.metrics.max_drawdown_ucb) for record in scenarios]
    es_ucbs = [float(record.metrics.daily_expected_shortfall_5_ucb) for record in scenarios]
    expectancy_lcbs = [
        min(
            float(record.metrics.net_expectancy_lcb),
            float(record.metrics.net_expectancy_on_committed_capital_lcb),
        )
        for record in scenarios
    ]
    profit_factors = [float(record.metrics.profit_factor) for record in scenarios]
    win_rates = [float(record.metrics.win_rate) for record in scenarios]
    effective_samples = [float(record.metrics.effective_sample_size) for record in scenarios]
    scenario_returns = [float(record.metrics.net_return) for record in scenarios]
    trades_per_month = [
        record.metrics.trade_count / record.metrics.active_months
        if record.metrics.active_months
        else 0.0
        for record in scenarios
    ]
    drawdown_durations = [
        float(record.metrics.max_drawdown_duration_days) for record in scenarios
    ]
    gate_alignment_scores = [
        _gate_alignment_score(item.candidate) for item in valid
    ]
    return CandidateAnalysisV2(
        experiment_id=experiment_id,
        phenotype_hash=phenotype_hash,
        candidate_ids=sorted({item.candidate.candidate_id for item in by_seed.values()}),
        seeds=sorted(by_seed),
        observation_refs=[
            CandidateObservationRefV2(
                attempt_id=item.attempt_id,
                candidate_id=item.candidate.candidate_id,
                seed=item.seed,
                result_path=item.result_path,
                result_sha256=item.result_sha256,
            )
            for item in sorted(
                by_seed.values(),
                key=lambda item: (item.seed, item.attempt_id, item.candidate.candidate_id),
            )
        ],
        observation_count=len(by_seed),
        valid_observation_count=len(valid),
        eligibility_status=(
            CandidateEligibilityStatus.ELIGIBLE
            if eligible
            else CandidateEligibilityStatus.INELIGIBLE
        ),
        reason_codes=["CANDIDATE_ELIGIBLE"] if eligible else sorted(reasons),
        failed_gate_reason_codes=sorted(failed_gates),
        comparison_panel_hash=comparison_panel_hash,
        median_robust_score=median(robust_scores) if robust_scores else None,
        worst_annualized_return_lcb=min(annual_lcbs) if annual_lcbs else None,
        median_annualized_return_lcb=median(annual_lcbs) if annual_lcbs else None,
        worst_max_drawdown_ucb=max(drawdown_ucbs) if drawdown_ucbs else None,
        worst_daily_es5_ucb=max(es_ucbs) if es_ucbs else None,
        worst_net_expectancy_lcb=min(expectancy_lcbs) if expectancy_lcbs else None,
        median_profit_factor=median(profit_factors) if profit_factors else None,
        median_win_rate=median(win_rates) if win_rates else None,
        min_effective_sample_size=min(effective_samples) if effective_samples else None,
        min_trades_per_active_month=min(trades_per_month) if trades_per_month else None,
        min_scenario_net_return=min(scenario_returns) if scenario_returns else None,
        profitable_scenario_ratio=(
            sum(value > 0 for value in scenario_returns) / len(scenario_returns)
            if scenario_returns
            else None
        ),
        max_drawdown_duration_days=(
            max(drawdown_durations) if drawdown_durations else None
        ),
        median_gate_alignment_score=(
            median(gate_alignment_scores) if gate_alignment_scores else None
        ),
        failed_gate_count=len(failed_gates),
        scenario_trade_count_sum=sum(record.metrics.trade_count for record in scenarios),
    )


def _load_results_and_observations(
    snapshot: WaveResultSnapshotV2,
    experiment_map: dict[str, ExperimentSpecV2],
) -> tuple[
    dict[str, AttemptResultV2],
    dict[tuple[str, str], list[_CandidateObservation]],
]:
    results: dict[str, AttemptResultV2] = {}
    observations: dict[tuple[str, str], list[_CandidateObservation]] = defaultdict(list)
    for attempt in snapshot.attempt_results:
        if attempt.evidence == AttemptEvidenceType.DOCUMENTED_ABORT:
            continue
        result = _verified_result(snapshot, attempt)
        experiment_config_hash = experiment_map[
            attempt.experiment_id
        ].resolved_config_hash
        if result.manifest.config_hash != experiment_config_hash:
            raise WaveAnalysisError("result config differs from experiment spec")
        results[attempt.attempt_id] = result
        if result.status != AttemptStatus.SUCCEEDED:
            continue
        candidate_ids: set[str] = set()
        for candidate in result.candidate_evaluations:
            if candidate.candidate_id in candidate_ids:
                raise WaveAnalysisError("duplicate candidate_id in terminal result")
            candidate_ids.add(candidate.candidate_id)
            if candidate.fitness_policy_version != snapshot.policy_version:
                raise WaveAnalysisError("candidate policy differs from wave snapshot")
            observations[(attempt.experiment_id, candidate.phenotype_hash)].append(
                _CandidateObservation(
                    attempt_id=attempt.attempt_id,
                    seed=attempt.seed,
                    result_path=str(attempt.result_path),
                    result_sha256=str(attempt.result_sha256),
                    candidate=candidate,
                )
            )
    return results, observations


def _analyze_experiments(
    snapshot: WaveResultSnapshotV2,
    experiment_map: dict[str, ExperimentSpecV2],
    result_by_attempt: dict[str, AttemptResultV2],
    policy: WaveAnalyzerPolicyV2,
) -> list[ExperimentAnalysisV2]:
    analyses: list[ExperimentAnalysisV2] = []
    for experiment_id, experiment in experiment_map.items():
        evidence = [
            item
            for item in snapshot.attempt_results
            if item.experiment_id == experiment_id
        ]
        verified = [
            item for item in evidence if item.evidence == AttemptEvidenceType.VERIFIED_RESULT
        ]
        abort_count = len(evidence) - len(verified)
        success_count = sum(
            result_by_attempt[item.attempt_id].status == AttemptStatus.SUCCEEDED
            for item in verified
        )
        abort_fraction = abort_count / len(evidence)
        non_success_fraction = (len(evidence) - success_count) / len(evidence)
        reasons: set[str] = set()
        if success_count == 0:
            reasons.add("NO_SUCCESSFUL_ATTEMPTS")
        if abort_fraction > policy.max_abort_fraction:
            reasons.add("ABORT_FRACTION_EXCEEDED")
        if non_success_fraction > policy.max_non_success_fraction:
            reasons.add("NON_SUCCESS_FRACTION_EXCEEDED")
        candidate_count = sum(
            len(result_by_attempt[item.attempt_id].candidate_evaluations)
            for item in verified
            if result_by_attempt[item.attempt_id].status == AttemptStatus.SUCCEEDED
        )
        analyses.append(
            ExperimentAnalysisV2(
                experiment_id=experiment_id,
                arm_type=experiment.arm_type,
                expected_attempt_count=len(evidence),
                verified_result_count=len(verified),
                successful_attempt_count=success_count,
                documented_abort_count=abort_count,
                candidate_observation_count=candidate_count,
                abort_fraction=abort_fraction,
                non_success_fraction=non_success_fraction,
                health_status=(
                    AnalysisHealthStatus.HEALTHY
                    if not reasons
                    else AnalysisHealthStatus.BLOCKED
                ),
                reason_codes=["EXPERIMENT_HEALTHY"] if not reasons else sorted(reasons),
            )
        )
    return analyses


def _wave_outcome(
    experiments: list[ExperimentAnalysisV2],
    candidates: list[CandidateAnalysisV2],
    policy: WaveAnalyzerPolicyV2,
) -> tuple[bool, bool, list[str]]:
    has_eligible = any(
        item.eligibility_status == CandidateEligibilityStatus.ELIGIBLE
        for item in candidates
    )
    wave_reasons: set[str] = set()
    if policy.require_control_arm and not any(
        item.arm_type == ExperimentArmType.CONTROL for item in experiments
    ):
        wave_reasons.add("CONTROL_ARM_MISSING")
    if any(item.health_status == AnalysisHealthStatus.BLOCKED for item in experiments):
        wave_reasons.add("EXPERIMENT_HEALTH_BLOCKED")
    if not has_eligible:
        wave_reasons.add("NO_ELIGIBLE_CANDIDATES")
    planning_blockers = set(wave_reasons) - {"NO_ELIGIBLE_CANDIDATES"}
    if policy.require_eligible_candidate_for_planning and not has_eligible:
        planning_blockers.add("NO_ELIGIBLE_CANDIDATES")
    if not wave_reasons:
        wave_reasons.add("ANALYSIS_COMPLETE")
    return not planning_blockers, has_eligible, sorted(wave_reasons)


def analyze_wave_snapshot(
    snapshot: WaveResultSnapshotV2,
    experiments: Iterable[ExperimentSpecV2],
    policy: WaveAnalyzerPolicyV2,
) -> WaveAnalysisV2:
    """Pure analysis function; equal immutable inputs yield equal output bytes."""

    if snapshot.policy_version != policy.required_result_policy_version:
        raise WaveAnalysisError("snapshot result policy differs from analyzer policy")
    experiment_list = sorted(experiments, key=lambda item: item.experiment_id)
    experiment_map = _validate_experiment_inputs(snapshot, experiment_list)
    result_by_attempt, observations = _load_results_and_observations(
        snapshot, experiment_map
    )
    experiment_analyses = _analyze_experiments(
        snapshot, experiment_map, result_by_attempt, policy
    )

    candidate_analyses = [
        _candidate_analysis(experiment_id, phenotype_hash, items, policy)
        for (experiment_id, phenotype_hash), items in sorted(observations.items())
    ]
    planning_allowed, has_eligible, wave_reasons = _wave_outcome(
        experiment_analyses, candidate_analyses, policy
    )

    return WaveAnalysisV2(
        wave_id=snapshot.wave_id,
        created_at=snapshot.created_at,
        snapshot_hash=snapshot.snapshot_hash,
        result_policy_version=snapshot.policy_version,
        analyzer_policy=policy,
        analyzer_policy_hash=policy.policy_hash,
        experiments=sorted(experiment_analyses, key=lambda item: item.experiment_id),
        candidates=candidate_analyses,
        planning_allowed=planning_allowed,
        has_eligible_candidates=has_eligible,
        reason_codes=wave_reasons,
    )


class WaveAnalyzerV2:
    """Store adapter around the pure snapshot analyzer."""

    def __init__(self, store: WaveStateStoreV2, policy: WaveAnalyzerPolicyV2) -> None:
        self.store = store
        self.policy = policy

    def build(self, wave_id: str) -> WaveAnalysisV2:
        wave = self.store.get_wave(wave_id)
        if wave.status not in {WaveLifecycleStatus.RECONCILED, WaveLifecycleStatus.ANALYZED}:
            raise WaveStateError("wave analysis requires RECONCILED wave")
        if wave.result_snapshot is None:
            raise WaveStateError("wave analysis requires immutable result snapshot")
        return analyze_wave_snapshot(
            wave.result_snapshot,
            self.store.experiments(wave_id),
            self.policy,
        )

    def analyze_and_record(self, wave_id: str) -> tuple[WaveAnalysisV2, WaveStateV2]:
        analysis = self.build(wave_id)
        state = self.store.apply_decision(analysis.to_decision())
        return analysis, state
