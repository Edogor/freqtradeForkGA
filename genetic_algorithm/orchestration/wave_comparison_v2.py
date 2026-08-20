"""Verified, read-only comparison view for analyzed GA V2 waves."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.result_contract import StrictV2Model
from genetic_algorithm.orchestration.wave_analyzer_v2 import (
    WaveAnalysisV2,
    analyze_wave_snapshot,
)
from genetic_algorithm.orchestration.wave_state_v2 import (
    WaveDecisionType,
    WaveLifecycleStatus,
    WaveStateError,
    WaveStateStoreV2,
)


class WaveComparisonError(ValueError):
    """Raised when a wave cannot prove one coherent comparison."""


class VerifiedWaveComparisonV2(StrictV2Model):
    """A report envelope bound to one immutable snapshot and analysis decision."""

    schema_version: Literal["2.0"] = "2.0"
    wave_id: str = Field(min_length=1)
    lifecycle_status: WaveLifecycleStatus
    wave_state_version: int = Field(ge=0)
    snapshot_hash: str = Field(min_length=64, max_length=64)
    analysis_decision_id: str = Field(min_length=1)
    analysis_decision_hash: str = Field(min_length=64, max_length=64)
    analysis: WaveAnalysisV2

    @model_validator(mode="after")
    def _same_evidence_chain(self) -> VerifiedWaveComparisonV2:
        if self.analysis.wave_id != self.wave_id:
            raise ValueError("comparison analysis belongs to another wave")
        if self.analysis.snapshot_hash != self.snapshot_hash:
            raise ValueError("comparison analysis differs from wave snapshot")
        return self

    @property
    def comparison_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


_ANALYSIS_PAYLOAD_KEYS = {
    "analysis",
    "analysis_hash",
    "analyzer_policy_hash",
    "planning_allowed",
    "has_eligible_candidates",
}
_POST_ANALYSIS_STATES = {
    WaveLifecycleStatus.ANALYZED,
    WaveLifecycleStatus.PROPOSED,
    WaveLifecycleStatus.APPROVED,
    WaveLifecycleStatus.QUEUED,
    WaveLifecycleStatus.BLOCKED,
    WaveLifecycleStatus.REJECTED,
}


def build_verified_wave_comparison(
    store: WaveStateStoreV2,
    wave_id: str,
) -> VerifiedWaveComparisonV2:
    """Rebuild recorded analysis from verified artifacts and require byte semantics."""

    wave = store.get_wave(wave_id)
    if wave.result_snapshot is None or wave.result_snapshot_hash is None:
        raise WaveComparisonError("wave comparison requires a reconciled result snapshot")

    analysis_decisions = [
        decision
        for decision in store.decisions(wave_id)
        if decision.decision_type == WaveDecisionType.ANALYSIS
    ]
    if len(analysis_decisions) != 1:
        raise WaveComparisonError("wave comparison requires exactly one recorded ANALYSIS decision")
    if wave.status not in _POST_ANALYSIS_STATES:
        raise WaveComparisonError("wave comparison requires a post-ANALYSIS lifecycle state")
    decision = analysis_decisions[0]
    if decision.actor != "wave-analyzer-v2":
        raise WaveComparisonError("analysis decision actor is not canonical")
    if decision.input_hash != wave.result_snapshot_hash:
        raise WaveComparisonError("analysis decision is not bound to the current snapshot")
    if set(decision.payload) != _ANALYSIS_PAYLOAD_KEYS:
        raise WaveComparisonError("analysis decision payload shape is not canonical")

    try:
        recorded = WaveAnalysisV2.model_validate(decision.payload["analysis"])
    except Exception as exc:
        raise WaveComparisonError("recorded wave analysis is invalid") from exc
    if decision != recorded.to_decision():
        raise WaveComparisonError(
            "recorded ANALYSIS decision differs from the canonical analysis decision"
        )
    if decision.created_at != recorded.created_at:
        raise WaveComparisonError("analysis decision timestamp differs from analysis")
    if decision.reason_codes != recorded.reason_codes:
        raise WaveComparisonError("analysis decision reasons differ from analysis")
    if decision.payload["analysis_hash"] != recorded.analysis_hash:
        raise WaveComparisonError("recorded analysis hash differs from analysis")
    if decision.payload["analyzer_policy_hash"] != recorded.analyzer_policy_hash:
        raise WaveComparisonError("recorded analyzer policy hash differs from analysis")
    if decision.payload["planning_allowed"] is not recorded.planning_allowed:
        raise WaveComparisonError("recorded planning outcome differs from analysis")
    if decision.payload["has_eligible_candidates"] is not recorded.has_eligible_candidates:
        raise WaveComparisonError("recorded candidate outcome differs from analysis")

    # This reloads every snapshot result through V2ArtifactStore, including
    # result/artifact/manifest hashes and candidate/scenario provenance.
    rebuilt = analyze_wave_snapshot(
        wave.result_snapshot,
        store.experiments(wave_id),
        recorded.analyzer_policy,
    )
    if rebuilt != recorded:
        raise WaveComparisonError(
            "recorded analysis differs from current verified structured evidence"
        )

    try:
        return VerifiedWaveComparisonV2(
            wave_id=wave.wave_id,
            lifecycle_status=wave.status,
            wave_state_version=wave.version,
            snapshot_hash=wave.result_snapshot_hash,
            analysis_decision_id=decision.decision_id,
            analysis_decision_hash=decision.decision_hash,
            analysis=recorded,
        )
    except ValueError as exc:
        raise WaveStateError("verified comparison envelope is inconsistent") from exc
