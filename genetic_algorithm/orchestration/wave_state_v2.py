"""Fail-closed Wave, Experiment and Decision state for GA V2 orchestration.

The wave store intentionally shares the SQLite database and transaction boundary
with :mod:`attempt_state_v2`.  A parent snapshot therefore cannot observe a
half-written attempt transition or silently lose an expected attempt.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    AttemptStateStoreV2,
    AttemptStateV2,
    WorkerBindingV2,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptStatus,
    StrictV2Model,
)


class WaveStateError(ValueError):
    """Raised when a wave contract, transition, or evidence check fails."""


class WaveLifecycleStatus(StrEnum):
    DRAFT = "DRAFT"
    COLLECTING = "COLLECTING"
    RECONCILED = "RECONCILED"
    ANALYZED = "ANALYZED"
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    QUEUED = "QUEUED"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"


class ExperimentArmType(StrEnum):
    CONTROL = "CONTROL"
    REPLICATION = "REPLICATION"
    EXPLOIT = "EXPLOIT"
    EXPLORE = "EXPLORE"
    VALIDATION = "VALIDATION"


class WaveDecisionType(StrEnum):
    ATTEMPT_ABORT = "ATTEMPT_ABORT"
    ANALYSIS = "ANALYSIS"
    PROPOSAL = "PROPOSAL"
    APPROVAL = "APPROVAL"
    REJECTION = "REJECTION"
    BLOCK = "BLOCK"


class AttemptEvidenceType(StrEnum):
    VERIFIED_RESULT = "VERIFIED_RESULT"
    DOCUMENTED_ABORT = "DOCUMENTED_ABORT"


_TERMINAL_ATTEMPTS = {
    AttemptLifecycleStatus.SUCCEEDED,
    AttemptLifecycleStatus.FAILED,
    AttemptLifecycleStatus.INTERRUPTED,
    AttemptLifecycleStatus.INVALID_RESULT,
}
_FINAL_WAVES = {
    WaveLifecycleStatus.QUEUED,
    WaveLifecycleStatus.BLOCKED,
    WaveLifecycleStatus.REJECTED,
}


def _aware(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _iso(value: datetime) -> str:
    if value.utcoffset() is None:
        raise WaveStateError("wave timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class WaveBudgetV2(StrictV2Model):
    max_attempts: int = Field(ge=1)
    max_parallel: int = Field(ge=1)
    max_wallclock_seconds: int = Field(ge=1)
    max_retries: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _parallel_fits_attempt_budget(self) -> WaveBudgetV2:
        if self.max_parallel > self.max_attempts:
            raise ValueError("max_parallel cannot exceed max_attempts")
        return self


class WaveSpecV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    wave_id: str = Field(min_length=1)
    parent_wave_id: str | None = None
    policy_version: str = Field(min_length=1)
    search_space_version: str = Field(min_length=1)
    budget: WaveBudgetV2
    created_at: datetime

    @model_validator(mode="after")
    def _valid_time_and_parent(self) -> WaveSpecV2:
        _aware(self.created_at, "created_at")
        if self.parent_wave_id == self.wave_id:
            raise ValueError("wave cannot be its own parent")
        return self

    @property
    def spec_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


class AttemptExpectationV2(StrictV2Model):
    attempt_id: str = Field(min_length=1)
    manifest_hash: str = Field(min_length=64, max_length=64)
    seed: int
    ordinal: int = Field(ge=0)


class ExperimentSpecV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    experiment_id: str = Field(min_length=1)
    wave_id: str = Field(min_length=1)
    arm_type: ExperimentArmType
    hypothesis: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    factor_delta: dict[str, Any] = Field(default_factory=dict)
    seeds: list[int] = Field(min_length=1)
    resolved_config_hash: str = Field(min_length=64, max_length=64)
    expected_attempts: list[AttemptExpectationV2] = Field(min_length=1)
    created_at: datetime

    @model_validator(mode="after")
    def _canonical_experiment(self) -> ExperimentSpecV2:
        _aware(self.created_at, "created_at")
        if self.seeds != sorted(set(self.seeds)):
            raise ValueError("experiment seeds must be unique and sorted")
        ordered = sorted(self.expected_attempts, key=lambda item: item.ordinal)
        if self.expected_attempts != ordered:
            raise ValueError("expected_attempts must be sorted by ordinal")
        if [item.ordinal for item in ordered] != list(range(len(ordered))):
            raise ValueError("expected attempt ordinals must be contiguous from zero")
        attempt_ids = [item.attempt_id for item in ordered]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("expected attempt IDs must be unique")
        expected_seeds = [item.seed for item in ordered]
        if sorted(expected_seeds) != self.seeds:
            raise ValueError("expected attempt seeds must match experiment seeds")
        return self

    @property
    def spec_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


class WaveDecisionV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    decision_id: str = Field(min_length=1)
    wave_id: str = Field(min_length=1)
    decision_type: WaveDecisionType
    created_at: datetime
    actor: str = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)
    attempt_id: str | None = None
    input_hash: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _decision_shape(self) -> WaveDecisionV2:
        _aware(self.created_at, "created_at")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("decision reason_codes must be unique")
        if any(not reason for reason in self.reason_codes):
            raise ValueError("decision reason_codes cannot be empty")
        if self.decision_type == WaveDecisionType.ATTEMPT_ABORT:
            if not self.attempt_id:
                raise ValueError("ATTEMPT_ABORT requires attempt_id")
            if self.input_hash is not None:
                raise ValueError("ATTEMPT_ABORT cannot use input_hash")
        elif self.attempt_id is not None:
            raise ValueError("only ATTEMPT_ABORT may reference attempt_id")
        if self.input_hash is not None and len(self.input_hash) != 64:
            raise ValueError("input_hash must be SHA-256")
        return self

    @property
    def decision_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


class WaveAttemptSnapshotV2(StrictV2Model):
    attempt_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    manifest_hash: str = Field(min_length=64, max_length=64)
    seed: int
    ordinal: int = Field(ge=0)
    lifecycle_status: AttemptLifecycleStatus
    evidence: AttemptEvidenceType
    result_status: AttemptStatus | None = None
    result_path: str | None = None
    result_sha256: str | None = None
    result_finished_at: datetime | None = None
    abort_decision_hash: str | None = None
    attempt_state_version: int = Field(ge=0)

    @model_validator(mode="after")
    def _exclusive_evidence(self) -> WaveAttemptSnapshotV2:
        result_fields = (
            self.result_status,
            self.result_path,
            self.result_sha256,
            self.result_finished_at,
        )
        if self.evidence == AttemptEvidenceType.VERIFIED_RESULT:
            if any(value is None for value in result_fields):
                raise ValueError("verified evidence requires complete result provenance")
            if self.abort_decision_hash is not None:
                raise ValueError("verified evidence cannot also use an abort decision")
        else:
            if any(value is not None for value in result_fields):
                raise ValueError("documented abort cannot contain result provenance")
            if self.abort_decision_hash is None:
                raise ValueError("documented abort requires decision hash")
        return self


class WaveResultSnapshotV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    wave_id: str = Field(min_length=1)
    created_at: datetime
    policy_version: str = Field(min_length=1)
    search_space_version: str = Field(min_length=1)
    attempt_results: list[WaveAttemptSnapshotV2] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_snapshot(self) -> WaveResultSnapshotV2:
        _aware(self.created_at, "created_at")
        ordered = sorted(
            self.attempt_results,
            key=lambda item: (item.experiment_id, item.ordinal, item.attempt_id),
        )
        if ordered != self.attempt_results:
            raise ValueError("snapshot attempts must use canonical experiment/ordinal order")
        ids = [item.attempt_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("snapshot attempt IDs must be unique")
        return self

    @property
    def snapshot_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


class WaveStateV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    wave_id: str = Field(min_length=1)
    parent_wave_id: str | None = None
    policy_version: str = Field(min_length=1)
    search_space_version: str = Field(min_length=1)
    budget: WaveBudgetV2
    spec_hash: str = Field(min_length=64, max_length=64)
    status: WaveLifecycleStatus
    created_at: datetime
    updated_at: datetime
    result_snapshot: WaveResultSnapshotV2 | None = None
    result_snapshot_hash: str | None = None
    version: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent_state(self) -> WaveStateV2:
        _aware(self.created_at, "created_at")
        _aware(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at precedes created_at")
        if (self.result_snapshot is None) != (self.result_snapshot_hash is None):
            raise ValueError("wave result snapshot and hash must be set together")
        if self.result_snapshot is not None:
            if self.result_snapshot.wave_id != self.wave_id:
                raise ValueError("wave snapshot belongs to another wave")
            if self.result_snapshot.snapshot_hash != self.result_snapshot_hash:
                raise ValueError("wave result snapshot hash differs from content")
        if self.status in {
            WaveLifecycleStatus.RECONCILED,
            WaveLifecycleStatus.ANALYZED,
            WaveLifecycleStatus.PROPOSED,
            WaveLifecycleStatus.APPROVED,
            WaveLifecycleStatus.QUEUED,
            WaveLifecycleStatus.REJECTED,
        } and self.result_snapshot is None:
            raise ValueError("post-reconciliation wave lacks immutable result snapshot")
        return self


class WaveEventV2(StrictV2Model):
    event_id: int = Field(ge=1)
    wave_id: str = Field(min_length=1)
    from_status: WaveLifecycleStatus | None = None
    to_status: WaveLifecycleStatus
    occurred_at: datetime
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    wave_version: int = Field(ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)


class WaveStateStoreV2(AttemptStateStoreV2):
    """Wave state backed by the exact transaction domain used for attempts."""

    @staticmethod
    def _wave_event(
        connection: sqlite3.Connection,
        *,
        wave_id: str,
        from_status: WaveLifecycleStatus | None,
        to_status: WaveLifecycleStatus,
        occurred_at: datetime,
        actor: str,
        reason: str,
        wave_version: int,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO wave_events (
                wave_id, from_status, to_status, occurred_at, actor,
                reason, wave_version, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                wave_id,
                from_status.value if from_status else None,
                to_status.value,
                _iso(occurred_at),
                actor,
                reason,
                wave_version,
                _canonical_json(dict(payload or {})),
            ),
        )

    def register_wave(
        self,
        spec: WaveSpecV2,
        *,
        actor: str = "controller",
    ) -> WaveStateV2:
        budget_json = _canonical_json(spec.budget.model_dump(mode="json"))
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM waves WHERE wave_id = ?", (spec.wave_id,)
            ).fetchone()
            if row is not None:
                state = self._wave_state(row)
                if state.spec_hash != spec.spec_hash:
                    raise WaveStateError("wave_id is already registered with a different spec")
                return state
            created_at = _iso(spec.created_at)
            connection.execute(
                """
                INSERT INTO waves (
                    wave_id, parent_wave_id, policy_version, search_space_version,
                    budget_json, spec_hash, status, created_at, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, 'DRAFT', ?, ?, 0)
                """,
                (
                    spec.wave_id,
                    spec.parent_wave_id,
                    spec.policy_version,
                    spec.search_space_version,
                    budget_json,
                    spec.spec_hash,
                    created_at,
                    created_at,
                ),
            )
            self._wave_event(
                connection,
                wave_id=spec.wave_id,
                from_status=None,
                to_status=WaveLifecycleStatus.DRAFT,
                occurred_at=spec.created_at,
                actor=actor,
                reason="REGISTERED",
                wave_version=0,
                payload={"spec_hash": spec.spec_hash},
            )
            return self._read_wave(connection, spec.wave_id)

    def register_experiment(self, spec: ExperimentSpecV2) -> ExperimentSpecV2:
        payload = _canonical_json(spec.model_dump(mode="json"))
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM experiment_specs WHERE experiment_id = ?",
                (spec.experiment_id,),
            ).fetchone()
            if existing is not None:
                persisted = ExperimentSpecV2.model_validate_json(existing["spec_json"])
                if existing["spec_hash"] != spec.spec_hash or persisted != spec:
                    raise WaveStateError(
                        "experiment_id is already registered with a different spec"
                    )
                return persisted
            wave = self._read_wave(connection, spec.wave_id)
            if wave.status != WaveLifecycleStatus.DRAFT:
                raise WaveStateError("experiments can only be added while wave is DRAFT")
            if spec.created_at < wave.created_at:
                raise WaveStateError("experiment creation precedes its wave")
            current_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM wave_attempts WHERE wave_id = ?", (spec.wave_id,)
                ).fetchone()[0]
            )
            if current_count + len(spec.expected_attempts) > wave.budget.max_attempts:
                raise WaveStateError("experiment attempts exceed the wave attempt budget")
            try:
                connection.execute(
                    """
                    INSERT INTO experiment_specs (
                        experiment_id, wave_id, arm_type, spec_json, spec_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.experiment_id,
                        spec.wave_id,
                        spec.arm_type.value,
                        payload,
                        spec.spec_hash,
                        _iso(spec.created_at),
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO wave_attempts (
                        attempt_id, wave_id, experiment_id, manifest_hash, seed, ordinal
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.attempt_id,
                            spec.wave_id,
                            spec.experiment_id,
                            item.manifest_hash,
                            item.seed,
                            item.ordinal,
                        )
                        for item in spec.expected_attempts
                    ],
                )
            except sqlite3.IntegrityError as exc:
                raise WaveStateError(
                    "experiment contains an attempt already expected by another arm"
                ) from exc
            return spec

    def begin_collection(
        self,
        wave_id: str,
        *,
        started_at: datetime,
        actor: str = "controller",
    ) -> WaveStateV2:
        with self._transaction() as connection:
            wave = self._read_wave(connection, wave_id)
            if wave.status == WaveLifecycleStatus.COLLECTING:
                return wave
            if wave.status != WaveLifecycleStatus.DRAFT:
                raise WaveStateError("collection can only begin from DRAFT")
            self._require_wave_time(wave, started_at)
            expected = self._expected_rows(connection, wave_id)
            if not expected:
                raise WaveStateError("wave has no registered experiment attempts")
            actual = connection.execute(
                "SELECT * FROM attempts WHERE wave_id = ? ORDER BY attempt_id", (wave_id,)
            ).fetchall()
            self._require_exact_attempt_set(expected, actual)
            actual_by_id = {row["attempt_id"]: row for row in actual}
            for item in expected:
                row = actual_by_id[item["attempt_id"]]
                self._validate_expected_attempt(item, row)
                if row["status"] != AttemptLifecycleStatus.QUEUED.value:
                    raise WaveStateError(
                        f"expected attempt is not QUEUED: {item['attempt_id']} ({row['status']})"
                    )
            return self._transition_wave(
                connection,
                wave,
                WaveLifecycleStatus.COLLECTING,
                occurred_at=started_at,
                actor=actor,
                reason="COLLECTION_STARTED",
                payload={"expected_attempt_count": len(expected)},
            )

    def apply_decision(self, decision: WaveDecisionV2) -> WaveStateV2:
        """Persist one immutable decision and apply its gated wave transition."""

        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM wave_decisions WHERE decision_id = ?",
                (decision.decision_id,),
            ).fetchone()
            if existing is not None:
                persisted = self._decision(existing)
                if persisted != decision or existing["decision_hash"] != decision.decision_hash:
                    raise WaveStateError(
                        "decision_id is already registered with different content"
                    )
                return self._read_wave(connection, decision.wave_id)

            wave = self._read_wave(connection, decision.wave_id)
            self._require_wave_time(wave, decision.created_at)
            target = self._validate_decision(connection, wave, decision)
            self._insert_decision(connection, decision)
            if target is None:
                return wave
            return self._transition_wave(
                connection,
                wave,
                target,
                occurred_at=decision.created_at,
                actor=decision.actor,
                reason=decision.decision_type.value,
                payload={
                    "decision_id": decision.decision_id,
                    "decision_hash": decision.decision_hash,
                    "reason_codes": decision.reason_codes,
                },
            )

    def reconcile_wave(
        self,
        wave_id: str,
        *,
        reconciled_at: datetime,
        actor: str = "reconciler",
    ) -> WaveStateV2:
        """Atomically freeze the complete set of terminal parent evidence."""

        with self._transaction() as connection:
            wave = self._read_wave(connection, wave_id)
            if wave.status in {
                WaveLifecycleStatus.RECONCILED,
                WaveLifecycleStatus.ANALYZED,
                WaveLifecycleStatus.PROPOSED,
                WaveLifecycleStatus.APPROVED,
                WaveLifecycleStatus.QUEUED,
                WaveLifecycleStatus.REJECTED,
            }:
                return wave
            if wave.status != WaveLifecycleStatus.COLLECTING:
                raise WaveStateError("reconciliation requires COLLECTING wave")
            self._require_wave_time(wave, reconciled_at)
            expected = self._expected_rows(connection, wave_id)
            actual = connection.execute(
                "SELECT * FROM attempts WHERE wave_id = ? ORDER BY attempt_id", (wave_id,)
            ).fetchall()
            self._require_exact_attempt_set(expected, actual)
            actual_by_id = {row["attempt_id"]: row for row in actual}
            evidence: list[WaveAttemptSnapshotV2] = []
            for item in expected:
                row = actual_by_id[item["attempt_id"]]
                self._validate_expected_attempt(item, row)
                status = AttemptLifecycleStatus(row["status"])
                if status not in _TERMINAL_ATTEMPTS:
                    raise WaveStateError(
                        f"expected attempt is not terminal: {item['attempt_id']} ({status.value})"
                    )
                evidence.append(self._attempt_evidence(connection, item, row))

            evidence.sort(key=lambda item: (item.experiment_id, item.ordinal, item.attempt_id))
            snapshot = WaveResultSnapshotV2(
                wave_id=wave_id,
                created_at=reconciled_at,
                policy_version=wave.policy_version,
                search_space_version=wave.search_space_version,
                attempt_results=evidence,
            )
            snapshot_json = _canonical_json(snapshot.model_dump(mode="json"))
            version = wave.version + 1
            connection.execute(
                """
                UPDATE waves
                SET status = 'RECONCILED', updated_at = ?, result_snapshot_json = ?,
                    result_snapshot_hash = ?, version = ?
                WHERE wave_id = ? AND status = 'COLLECTING' AND version = ?
                """,
                (
                    _iso(reconciled_at),
                    snapshot_json,
                    snapshot.snapshot_hash,
                    version,
                    wave_id,
                    wave.version,
                ),
            )
            self._wave_event(
                connection,
                wave_id=wave_id,
                from_status=wave.status,
                to_status=WaveLifecycleStatus.RECONCILED,
                occurred_at=reconciled_at,
                actor=actor,
                reason="COMPLETE_EVIDENCE_RECONCILED",
                wave_version=version,
                payload={
                    "attempt_count": len(evidence),
                    "snapshot_hash": snapshot.snapshot_hash,
                },
            )
            return self._read_wave(connection, wave_id)

    def mark_queued(
        self,
        wave_id: str,
        *,
        queued_at: datetime,
        actor: str = "controller",
    ) -> WaveStateV2:
        """Reject the old state-only queue marker.

        A parent may become ``QUEUED`` only in the same transaction that
        registers the exact child wave, attempts, and immutable workers.
        """

        del wave_id, queued_at, actor
        raise WaveStateError(
            "direct queue marking is disabled; use queue_materialized_child_wave"
        )

    def queue_materialized_child_wave(
        self,
        *,
        parent_wave_id: str,
        child_wave: WaveSpecV2,
        experiments: list[ExperimentSpecV2],
        attempts: list[tuple[AttemptManifestV2, WorkerBindingV2]],
        plan_hash: str,
        materialization_hash: str,
        queued_at: datetime,
        actor: str = "materialized-wave-controller-v2",
    ) -> WaveStateV2:
        """Atomically register and queue one fully approved child wave."""

        self._validate_materialized_child_inputs(
            parent_wave_id=parent_wave_id,
            child_wave=child_wave,
            experiments=experiments,
            attempts=attempts,
            plan_hash=plan_hash,
            materialization_hash=materialization_hash,
            queued_at=queued_at,
        )
        with self._transaction() as connection:
            parent = self._read_wave(connection, parent_wave_id)
            if parent.status == WaveLifecycleStatus.QUEUED:
                self._verify_existing_materialized_handoff(
                    connection,
                    parent_wave_id=parent_wave_id,
                    child_wave=child_wave,
                    experiments=experiments,
                    attempts=attempts,
                    plan_hash=plan_hash,
                    materialization_hash=materialization_hash,
                )
                return parent
            if parent.status != WaveLifecycleStatus.APPROVED:
                raise WaveStateError("materialized queue handoff requires APPROVED parent wave")
            self._require_wave_time(parent, queued_at)
            approval = self._latest_decision(
                connection, parent_wave_id, WaveDecisionType.APPROVAL
            )
            if (
                approval.payload.get("approved_plan_hash") != plan_hash
                or approval.payload.get("approved_materialization_hash")
                != materialization_hash
            ):
                raise WaveStateError("approval differs from materialized child inputs")
            occupied = connection.execute(
                """
                SELECT wave_id FROM waves WHERE wave_id = ?
                UNION ALL SELECT wave_id FROM experiment_specs WHERE wave_id = ?
                UNION ALL SELECT wave_id FROM attempts WHERE wave_id = ?
                LIMIT 1
                """,
                (child_wave.wave_id, child_wave.wave_id, child_wave.wave_id),
            ).fetchone()
            if occupied is not None:
                raise WaveStateError("child wave ID already has persisted state")

            try:
                self._insert_materialized_child_wave(
                    connection,
                    child_wave=child_wave,
                    experiments=experiments,
                    attempts=attempts,
                    queued_at=queued_at,
                    actor=actor,
                    plan_hash=plan_hash,
                    materialization_hash=materialization_hash,
                )
            except sqlite3.IntegrityError as exc:
                raise WaveStateError(
                    "materialized child conflicts with existing immutable state"
                ) from exc
            return self._transition_wave(
                connection,
                parent,
                WaveLifecycleStatus.QUEUED,
                occurred_at=queued_at,
                actor=actor,
                reason="APPROVED_PLAN_QUEUED",
                payload={
                    "child_wave_id": child_wave.wave_id,
                    "plan_hash": plan_hash,
                    "materialization_hash": materialization_hash,
                    "attempt_count": len(attempts),
                },
            )

    @staticmethod
    def _validate_materialized_child_inputs(
        *,
        parent_wave_id: str,
        child_wave: WaveSpecV2,
        experiments: list[ExperimentSpecV2],
        attempts: list[tuple[AttemptManifestV2, WorkerBindingV2]],
        plan_hash: str,
        materialization_hash: str,
        queued_at: datetime,
    ) -> None:
        _aware(queued_at, "queued_at")
        if len(plan_hash) != 64 or len(materialization_hash) != 64:
            raise WaveStateError("plan and materialization hashes must be SHA-256")
        if child_wave.parent_wave_id != parent_wave_id:
            raise WaveStateError("child wave parent differs from approved parent")
        if child_wave.created_at > queued_at:
            raise WaveStateError("child wave creation follows queue handoff")
        WaveStateStoreV2._validate_materialized_collections(
            child_wave=child_wave,
            experiments=experiments,
            attempts=attempts,
        )

        expected = {
            item.attempt_id: (experiment, item)
            for experiment in experiments
            for item in experiment.expected_attempts
        }
        if len(expected) != sum(len(item.expected_attempts) for item in experiments):
            raise WaveStateError("materialized attempt expectations are duplicated")
        actual_ids = [manifest.attempt_id for manifest, _ in attempts]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected):
            raise WaveStateError("materialized manifests differ from experiment expectations")
        for manifest, binding in attempts:
            experiment, expectation = expected[manifest.attempt_id]
            WaveStateStoreV2._validate_materialized_attempt_input(
                child_wave=child_wave,
                experiment=experiment,
                expectation=expectation,
                manifest=manifest,
                binding=binding,
                queued_at=queued_at,
            )

    @staticmethod
    def _validate_materialized_collections(
        *,
        child_wave: WaveSpecV2,
        experiments: list[ExperimentSpecV2],
        attempts: list[tuple[AttemptManifestV2, WorkerBindingV2]],
    ) -> None:
        if experiments != sorted(experiments, key=lambda item: item.experiment_id):
            raise WaveStateError("materialized experiments must be canonically sorted")
        if attempts != sorted(attempts, key=lambda item: item[0].attempt_id):
            raise WaveStateError("materialized attempts must be canonically sorted")
        if not experiments or not attempts:
            raise WaveStateError("materialized child wave cannot be empty")
        experiment_ids = [item.experiment_id for item in experiments]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise WaveStateError("materialized experiment IDs must be unique")
        if any(item.wave_id != child_wave.wave_id for item in experiments):
            raise WaveStateError("materialized experiment belongs to another child wave")
        if len(attempts) > child_wave.budget.max_attempts:
            raise WaveStateError("materialized attempts exceed child wave budget")

    @staticmethod
    def _validate_materialized_attempt_input(
        *,
        child_wave: WaveSpecV2,
        experiment: ExperimentSpecV2,
        expectation: AttemptExpectationV2,
        manifest: AttemptManifestV2,
        binding: WorkerBindingV2,
        queued_at: datetime,
    ) -> None:
        provenance = (
            manifest.wave_id == child_wave.wave_id,
            manifest.experiment_id == experiment.experiment_id,
            manifest.config_hash == experiment.resolved_config_hash,
            manifest.fitness_policy_version == child_wave.policy_version,
            manifest.seeds == [expectation.seed],
            canonical_config_hash(manifest.model_dump(mode="json"))
            == expectation.manifest_hash,
        )
        if not all(provenance):
            raise WaveStateError("materialized manifest provenance differs from child spec")
        if manifest.created_at > queued_at:
            raise WaveStateError("materialized attempt creation follows queue handoff")
        draft = AttemptStateV2(
            attempt_id=manifest.attempt_id,
            wave_id=manifest.wave_id,
            experiment_id=manifest.experiment_id,
            manifest_hash=expectation.manifest_hash,
            artifact_root=manifest.artifact_root,
            status=AttemptLifecycleStatus.DRAFT,
            created_at=manifest.created_at,
            updated_at=manifest.created_at,
            version=0,
        )
        AttemptStateStoreV2._validate_worker_binding(draft, binding)

    @classmethod
    def _insert_materialized_child_wave(
        cls,
        connection: sqlite3.Connection,
        *,
        child_wave: WaveSpecV2,
        experiments: list[ExperimentSpecV2],
        attempts: list[tuple[AttemptManifestV2, WorkerBindingV2]],
        queued_at: datetime,
        actor: str,
        plan_hash: str,
        materialization_hash: str,
    ) -> None:
        created_at = _iso(child_wave.created_at)
        connection.execute(
            """
            INSERT INTO waves (
                wave_id, parent_wave_id, policy_version, search_space_version,
                budget_json, spec_hash, status, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, 'DRAFT', ?, ?, 0)
            """,
            (
                child_wave.wave_id,
                child_wave.parent_wave_id,
                child_wave.policy_version,
                child_wave.search_space_version,
                _canonical_json(child_wave.budget.model_dump(mode="json")),
                child_wave.spec_hash,
                created_at,
                created_at,
            ),
        )
        cls._wave_event(
            connection,
            wave_id=child_wave.wave_id,
            from_status=None,
            to_status=WaveLifecycleStatus.DRAFT,
            occurred_at=child_wave.created_at,
            actor=actor,
            reason="MATERIALIZED_CHILD_REGISTERED",
            wave_version=0,
            payload={
                "plan_hash": plan_hash,
                "materialization_hash": materialization_hash,
            },
        )
        for experiment in experiments:
            connection.execute(
                """
                INSERT INTO experiment_specs (
                    experiment_id, wave_id, arm_type, spec_json, spec_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.wave_id,
                    experiment.arm_type.value,
                    _canonical_json(experiment.model_dump(mode="json")),
                    experiment.spec_hash,
                    _iso(experiment.created_at),
                ),
            )
            connection.executemany(
                """
                INSERT INTO wave_attempts (
                    attempt_id, wave_id, experiment_id, manifest_hash, seed, ordinal
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.attempt_id,
                        experiment.wave_id,
                        experiment.experiment_id,
                        item.manifest_hash,
                        item.seed,
                        item.ordinal,
                    )
                    for item in experiment.expected_attempts
                ],
            )

        for manifest, binding in attempts:
            manifest_hash = canonical_config_hash(manifest.model_dump(mode="json"))
            manifest_payload = _canonical_json(manifest.model_dump(mode="json"))
            manifest_created_at = _iso(manifest.created_at)
            connection.execute(
                """
                INSERT INTO attempts (
                    attempt_id, wave_id, experiment_id, manifest_hash, manifest_json,
                    artifact_root, status, priority, retry_number, created_at,
                    updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, 'DRAFT', 0, 0, ?, ?, 0)
                """,
                (
                    manifest.attempt_id,
                    manifest.wave_id,
                    manifest.experiment_id,
                    manifest_hash,
                    manifest_payload,
                    manifest.artifact_root,
                    manifest_created_at,
                    manifest_created_at,
                ),
            )
            AttemptStateStoreV2._event(
                connection,
                attempt_id=manifest.attempt_id,
                from_status=None,
                to_status=AttemptLifecycleStatus.DRAFT,
                occurred_at=manifest.created_at,
                actor=actor,
                reason="REGISTERED_FROM_MATERIALIZATION",
                attempt_version=0,
                payload={"manifest_hash": manifest_hash},
            )
            binding_payload = _canonical_json(binding.model_dump(mode="json"))
            connection.execute(
                """
                UPDATE attempts
                SET worker_kind = ?, worker_binding_json = ?, worker_binding_hash = ?,
                    updated_at = ?, version = 1
                WHERE attempt_id = ?
                """,
                (
                    binding.worker_kind.value,
                    binding_payload,
                    binding.binding_hash,
                    _iso(queued_at),
                    manifest.attempt_id,
                ),
            )
            AttemptStateStoreV2._event(
                connection,
                attempt_id=manifest.attempt_id,
                from_status=AttemptLifecycleStatus.DRAFT,
                to_status=AttemptLifecycleStatus.DRAFT,
                occurred_at=queued_at,
                actor=actor,
                reason="WORKER_BOUND_FROM_MATERIALIZATION",
                attempt_version=1,
                payload={"worker_binding_hash": binding.binding_hash},
            )
            for version, status, reason in (
                (2, AttemptLifecycleStatus.VALIDATED, "MATERIALIZED_INPUTS_VALID"),
                (3, AttemptLifecycleStatus.QUEUED, "QUEUED_FROM_APPROVED_MATERIALIZATION"),
            ):
                previous = (
                    AttemptLifecycleStatus.DRAFT
                    if status == AttemptLifecycleStatus.VALIDATED
                    else AttemptLifecycleStatus.VALIDATED
                )
                connection.execute(
                    """
                    UPDATE attempts SET status = ?, updated_at = ?, version = ?
                    WHERE attempt_id = ?
                    """,
                    (status.value, _iso(queued_at), version, manifest.attempt_id),
                )
                AttemptStateStoreV2._event(
                    connection,
                    attempt_id=manifest.attempt_id,
                    from_status=previous,
                    to_status=status,
                    occurred_at=queued_at,
                    actor=actor,
                    reason=reason,
                    attempt_version=version,
                )

    @staticmethod
    def _verify_existing_materialized_handoff(
        connection: sqlite3.Connection,
        *,
        parent_wave_id: str,
        child_wave: WaveSpecV2,
        experiments: list[ExperimentSpecV2],
        attempts: list[tuple[AttemptManifestV2, WorkerBindingV2]],
        plan_hash: str,
        materialization_hash: str,
    ) -> None:
        event = connection.execute(
            """
            SELECT payload_json FROM wave_events
            WHERE wave_id = ? AND reason = 'APPROVED_PLAN_QUEUED'
            ORDER BY event_id DESC LIMIT 1
            """,
            (parent_wave_id,),
        ).fetchone()
        expected_payload = {
            "child_wave_id": child_wave.wave_id,
            "plan_hash": plan_hash,
            "materialization_hash": materialization_hash,
            "attempt_count": len(attempts),
        }
        if event is None or json.loads(event["payload_json"]) != expected_payload:
            raise WaveStateError("queued parent differs from requested materialized handoff")
        wave_row = connection.execute(
            "SELECT spec_hash FROM waves WHERE wave_id = ?", (child_wave.wave_id,)
        ).fetchone()
        if wave_row is None or wave_row["spec_hash"] != child_wave.spec_hash:
            raise WaveStateError("persisted child wave differs from materialization")
        persisted_experiments = connection.execute(
            """
            SELECT spec_hash FROM experiment_specs
            WHERE wave_id = ? ORDER BY experiment_id
            """,
            (child_wave.wave_id,),
        ).fetchall()
        if [row["spec_hash"] for row in persisted_experiments] != [
            item.spec_hash for item in experiments
        ]:
            raise WaveStateError("persisted child experiments differ from materialization")
        persisted_attempts = connection.execute(
            """
            SELECT manifest_hash, worker_binding_hash, status FROM attempts
            WHERE wave_id = ? ORDER BY attempt_id
            """,
            (child_wave.wave_id,),
        ).fetchall()
        expected_attempts = [
            (
                canonical_config_hash(manifest.model_dump(mode="json")),
                binding.binding_hash,
                AttemptLifecycleStatus.QUEUED.value,
            )
            for manifest, binding in attempts
        ]
        if [tuple(row) for row in persisted_attempts] != expected_attempts:
            raise WaveStateError("persisted child attempts differ from materialization")

    def get_wave(self, wave_id: str) -> WaveStateV2:
        connection = self._connect()
        try:
            return self._read_wave(connection, wave_id)
        finally:
            connection.close()

    def list_waves(self) -> list[WaveStateV2]:
        """Return every canonical wave in deterministic creation order."""

        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM waves ORDER BY created_at, wave_id"
            ).fetchall()
            return [self._wave_state(row) for row in rows]
        finally:
            connection.close()

    def experiments(self, wave_id: str) -> list[ExperimentSpecV2]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM experiment_specs
                WHERE wave_id = ? ORDER BY experiment_id
                """,
                (wave_id,),
            ).fetchall()
            experiments = [
                ExperimentSpecV2.model_validate_json(row["spec_json"]) for row in rows
            ]
            for row, experiment in zip(rows, experiments, strict=True):
                if experiment.spec_hash != row["spec_hash"]:
                    raise WaveStateError("persisted experiment hash differs from content")
            return experiments
        finally:
            connection.close()

    def decisions(self, wave_id: str) -> list[WaveDecisionV2]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM wave_decisions
                WHERE wave_id = ? ORDER BY created_at, decision_id
                """,
                (wave_id,),
            ).fetchall()
            return [self._decision(row) for row in rows]
        finally:
            connection.close()

    def wave_events(self, wave_id: str) -> list[WaveEventV2]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM wave_events WHERE wave_id = ? ORDER BY event_id",
                (wave_id,),
            ).fetchall()
            return [
                WaveEventV2(
                    event_id=row["event_id"],
                    wave_id=row["wave_id"],
                    from_status=row["from_status"],
                    to_status=row["to_status"],
                    occurred_at=datetime.fromisoformat(row["occurred_at"]),
                    actor=row["actor"],
                    reason=row["reason"],
                    wave_version=row["wave_version"],
                    payload=json.loads(row["payload_json"]),
                )
                for row in rows
            ]
        finally:
            connection.close()

    def _validate_decision(
        self,
        connection: sqlite3.Connection,
        wave: WaveStateV2,
        decision: WaveDecisionV2,
    ) -> WaveLifecycleStatus | None:
        if decision.decision_type == WaveDecisionType.ATTEMPT_ABORT:
            self._validate_abort_decision(connection, wave, decision)
            return None

        transitions = {
            WaveDecisionType.ANALYSIS: (
                WaveLifecycleStatus.RECONCILED,
                WaveLifecycleStatus.ANALYZED,
            ),
            WaveDecisionType.PROPOSAL: (
                WaveLifecycleStatus.ANALYZED,
                WaveLifecycleStatus.PROPOSED,
            ),
            WaveDecisionType.APPROVAL: (
                WaveLifecycleStatus.PROPOSED,
                WaveLifecycleStatus.APPROVED,
            ),
            WaveDecisionType.REJECTION: (
                WaveLifecycleStatus.PROPOSED,
                WaveLifecycleStatus.REJECTED,
            ),
        }
        if decision.decision_type == WaveDecisionType.BLOCK:
            if wave.status in _FINAL_WAVES:
                raise WaveStateError("final wave cannot be blocked again")
            expected_input = self._current_wave_input(connection, wave)
            if decision.input_hash != expected_input:
                raise WaveStateError("BLOCK input_hash differs from immutable input")
            return WaveLifecycleStatus.BLOCKED
        required, target = transitions[decision.decision_type]
        if wave.status != required:
            raise WaveStateError(
                f"{decision.decision_type.value} decision requires {required.value} wave"
            )
        expected_input = self._expected_decision_input(connection, wave, decision.decision_type)
        if decision.input_hash != expected_input:
            raise WaveStateError(
                f"{decision.decision_type.value} input_hash differs from immutable input"
            )
        if decision.decision_type == WaveDecisionType.APPROVAL:
            self._validate_materialized_approval(connection, wave, decision)
        elif decision.decision_type == WaveDecisionType.PROPOSAL:
            self._validate_materialized_proposal(decision)
        return target

    @classmethod
    def _validate_materialized_approval(
        cls,
        connection: sqlite3.Connection,
        wave: WaveStateV2,
        decision: WaveDecisionV2,
    ) -> None:
        proposal = cls._latest_decision(
            connection, wave.wave_id, WaveDecisionType.PROPOSAL
        )
        if proposal.payload.get("planning_allowed") is not True:
            raise WaveStateError("blocked or incomplete proposal cannot be approved")
        plan_hash = proposal.payload.get("plan_hash")
        materialization_hash = proposal.payload.get("materialization_hash")
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (plan_hash, materialization_hash)
        ):
            raise WaveStateError("proposal lacks materialized plan hashes")
        if (
            decision.payload.get("approved_plan_hash") != plan_hash
            or decision.payload.get("approved_materialization_hash")
            != materialization_hash
        ):
            raise WaveStateError("approval does not bind the exact materialized plan")

    @staticmethod
    def _validate_materialized_proposal(decision: WaveDecisionV2) -> None:
        if decision.payload.get("planning_allowed") is not True:
            return
        hashes = (
            decision.payload.get("plan_hash"),
            decision.payload.get("materialization_hash"),
        )
        if not all(isinstance(value, str) and len(value) == 64 for value in hashes):
            raise WaveStateError("executable proposal lacks materialized plan hashes")

    def _validate_abort_decision(
        self,
        connection: sqlite3.Connection,
        wave: WaveStateV2,
        decision: WaveDecisionV2,
    ) -> None:
        if wave.status != WaveLifecycleStatus.COLLECTING:
            raise WaveStateError("attempt abort can only be documented while COLLECTING")
        expectation = next(
            (
                item
                for item in self._expected_rows(connection, wave.wave_id)
                if item["attempt_id"] == decision.attempt_id
            ),
            None,
        )
        if expectation is None:
            raise WaveStateError("abort decision references an unexpected attempt")
        attempt = connection.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (decision.attempt_id,)
        ).fetchone()
        if attempt is None:
            raise WaveStateError("abort decision references an unregistered attempt")
        self._validate_expected_attempt(expectation, attempt)
        status = AttemptLifecycleStatus(attempt["status"])
        if status not in _TERMINAL_ATTEMPTS - {AttemptLifecycleStatus.SUCCEEDED}:
            raise WaveStateError("abort decision requires a non-success terminal attempt")
        if attempt["result_path"] is not None or attempt["result_sha256"] is not None:
            raise WaveStateError("attempt with result evidence cannot be documented as abort")

    @classmethod
    def _latest_decision(
        cls,
        connection: sqlite3.Connection,
        wave_id: str,
        decision_type: WaveDecisionType,
    ) -> WaveDecisionV2:
        row = connection.execute(
            """
            SELECT * FROM wave_decisions
            WHERE wave_id = ? AND decision_type = ?
            ORDER BY created_at DESC, decision_id DESC LIMIT 1
            """,
            (wave_id, decision_type.value),
        ).fetchone()
        if row is None:
            raise WaveStateError(f"missing predecessor decision: {decision_type.value}")
        return cls._decision(row)

    @staticmethod
    def _expected_decision_input(
        connection: sqlite3.Connection,
        wave: WaveStateV2,
        decision_type: WaveDecisionType,
    ) -> str:
        if decision_type == WaveDecisionType.ANALYSIS:
            if wave.result_snapshot_hash is None:
                raise WaveStateError("analysis requires reconciled snapshot")
            return wave.result_snapshot_hash
        predecessor = {
            WaveDecisionType.PROPOSAL: WaveDecisionType.ANALYSIS,
            WaveDecisionType.APPROVAL: WaveDecisionType.PROPOSAL,
            WaveDecisionType.REJECTION: WaveDecisionType.PROPOSAL,
        }[decision_type]
        row = connection.execute(
            """
            SELECT decision_hash FROM wave_decisions
            WHERE wave_id = ? AND decision_type = ?
            ORDER BY created_at DESC, decision_id DESC LIMIT 1
            """,
            (wave.wave_id, predecessor.value),
        ).fetchone()
        if row is None:
            raise WaveStateError(f"missing predecessor decision: {predecessor.value}")
        return str(row["decision_hash"])

    @staticmethod
    def _current_wave_input(
        connection: sqlite3.Connection,
        wave: WaveStateV2,
    ) -> str:
        predecessor = {
            WaveLifecycleStatus.ANALYZED: WaveDecisionType.ANALYSIS,
            WaveLifecycleStatus.PROPOSED: WaveDecisionType.PROPOSAL,
            WaveLifecycleStatus.APPROVED: WaveDecisionType.APPROVAL,
        }.get(wave.status)
        if predecessor is not None:
            row = connection.execute(
                """
                SELECT decision_hash FROM wave_decisions
                WHERE wave_id = ? AND decision_type = ?
                ORDER BY created_at DESC, decision_id DESC LIMIT 1
                """,
                (wave.wave_id, predecessor.value),
            ).fetchone()
            if row is None:
                raise WaveStateError(
                    f"missing current-state decision: {predecessor.value}"
                )
            return str(row["decision_hash"])
        if wave.status == WaveLifecycleStatus.RECONCILED:
            if wave.result_snapshot_hash is None:
                raise WaveStateError("reconciled wave lacks snapshot hash")
            return wave.result_snapshot_hash
        return wave.spec_hash

    @staticmethod
    def _insert_decision(
        connection: sqlite3.Connection,
        decision: WaveDecisionV2,
    ) -> None:
        try:
            connection.execute(
                """
                INSERT INTO wave_decisions (
                    decision_id, wave_id, decision_type, attempt_id, input_hash,
                    decision_json, decision_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.wave_id,
                    decision.decision_type.value,
                    decision.attempt_id,
                    decision.input_hash,
                    _canonical_json(decision.model_dump(mode="json")),
                    decision.decision_hash,
                    _iso(decision.created_at),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise WaveStateError("decision conflicts with immutable wave history") from exc

    def _attempt_evidence(
        self,
        connection: sqlite3.Connection,
        expected: sqlite3.Row,
        attempt: sqlite3.Row,
    ) -> WaveAttemptSnapshotV2:
        state = self._state(attempt)
        if state.result_path is not None:
            canonical_path = (
                Path(state.artifact_root).resolve() / V2ArtifactStore.RESULT_NAME
            ).resolve()
            if Path(state.result_path).resolve() != canonical_path:
                raise ArtifactIntegrityError("attempt result path is not canonical")
            if not canonical_path.is_file():
                raise ArtifactIntegrityError("attempt result path is missing")
            actual_sha = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
            if actual_sha != state.result_sha256:
                raise ArtifactIntegrityError("attempt result hash differs from state")
            result = V2ArtifactStore(state.artifact_root).read_verified_result()
            if result.attempt_id != state.attempt_id:
                raise ArtifactIntegrityError("terminal result belongs to another attempt")
            manifest_hash = canonical_config_hash(result.manifest.model_dump(mode="json"))
            if manifest_hash != state.manifest_hash:
                raise ArtifactIntegrityError("terminal result manifest differs from attempt state")
            if result.status.value != state.status.value:
                raise ArtifactIntegrityError("terminal result status differs from attempt state")
            return WaveAttemptSnapshotV2(
                attempt_id=state.attempt_id,
                experiment_id=state.experiment_id,
                manifest_hash=state.manifest_hash,
                seed=expected["seed"],
                ordinal=expected["ordinal"],
                lifecycle_status=state.status,
                evidence=AttemptEvidenceType.VERIFIED_RESULT,
                result_status=result.status,
                result_path=str(canonical_path),
                result_sha256=actual_sha,
                result_finished_at=result.finished_at,
                attempt_state_version=state.version,
            )

        if state.result_sha256 is not None or state.result_status is not None:
            raise ArtifactIntegrityError("attempt contains incomplete result provenance")
        abort = connection.execute(
            """
            SELECT * FROM wave_decisions
            WHERE wave_id = ? AND attempt_id = ? AND decision_type = 'ATTEMPT_ABORT'
            """,
            (expected["wave_id"], state.attempt_id),
        ).fetchone()
        if abort is None:
            raise WaveStateError(
                f"terminal attempt lacks verified result or abort decision: {state.attempt_id}"
            )
        decision = self._decision(abort)
        if decision.decision_hash != abort["decision_hash"]:
            raise WaveStateError("persisted abort decision hash differs from content")
        return WaveAttemptSnapshotV2(
            attempt_id=state.attempt_id,
            experiment_id=state.experiment_id,
            manifest_hash=state.manifest_hash,
            seed=expected["seed"],
            ordinal=expected["ordinal"],
            lifecycle_status=state.status,
            evidence=AttemptEvidenceType.DOCUMENTED_ABORT,
            abort_decision_hash=decision.decision_hash,
            attempt_state_version=state.version,
        )

    @staticmethod
    def _validate_expected_attempt(expected: sqlite3.Row, attempt: sqlite3.Row) -> None:
        mismatches = {
            field: (expected[field], attempt[field])
            for field in ("wave_id", "experiment_id", "manifest_hash")
            if expected[field] != attempt[field]
        }
        if mismatches:
            raise WaveStateError(f"attempt differs from experiment expectation: {mismatches}")
        try:
            manifest = AttemptManifestV2.model_validate_json(attempt["manifest_json"])
        except Exception as exc:
            raise WaveStateError("registered attempt manifest is invalid") from exc
        actual_manifest_hash = canonical_config_hash(manifest.model_dump(mode="json"))
        if actual_manifest_hash != attempt["manifest_hash"]:
            raise WaveStateError("registered attempt manifest hash differs from content")
        try:
            experiment = ExperimentSpecV2.model_validate_json(
                expected["experiment_spec_json"]
            )
        except Exception as exc:
            raise WaveStateError("persisted experiment spec is invalid") from exc
        if experiment.spec_hash != expected["experiment_spec_hash"]:
            raise WaveStateError("persisted experiment hash differs from content")
        if manifest.config_hash != experiment.resolved_config_hash:
            raise WaveStateError("attempt config hash differs from experiment spec")
        if manifest.fitness_policy_version != expected["expected_policy_version"]:
            raise WaveStateError("attempt policy version differs from wave spec")
        if manifest.parent_wave_id != expected["expected_parent_wave_id"]:
            raise WaveStateError("attempt parent differs from wave spec")
        if manifest.seeds != [expected["seed"]]:
            raise WaveStateError("attempt manifest must contain exactly its expected seed")

    @staticmethod
    def _require_exact_attempt_set(
        expected: list[sqlite3.Row],
        actual: list[sqlite3.Row],
    ) -> None:
        expected_ids = {row["attempt_id"] for row in expected}
        actual_ids = {row["attempt_id"] for row in actual}
        if expected_ids != actual_ids:
            missing = sorted(expected_ids - actual_ids)
            unexpected = sorted(actual_ids - expected_ids)
            raise WaveStateError(
                f"wave attempt set differs from spec; missing={missing}, unexpected={unexpected}"
            )

    @staticmethod
    def _expected_rows(
        connection: sqlite3.Connection,
        wave_id: str,
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """
            SELECT
                wa.*,
                es.spec_json AS experiment_spec_json,
                es.spec_hash AS experiment_spec_hash,
                w.parent_wave_id AS expected_parent_wave_id,
                w.policy_version AS expected_policy_version
            FROM wave_attempts AS wa
            JOIN experiment_specs AS es ON es.experiment_id = wa.experiment_id
            JOIN waves AS w ON w.wave_id = wa.wave_id
            WHERE wa.wave_id = ?
            ORDER BY wa.experiment_id, wa.ordinal, wa.attempt_id
            """,
            (wave_id,),
        ).fetchall()

    @staticmethod
    def _require_wave_time(wave: WaveStateV2, occurred_at: datetime) -> None:
        _iso(occurred_at)
        if occurred_at < wave.updated_at:
            raise WaveStateError("wave transition timestamp precedes last update")

    def _transition_wave(
        self,
        connection: sqlite3.Connection,
        wave: WaveStateV2,
        target: WaveLifecycleStatus,
        *,
        occurred_at: datetime,
        actor: str,
        reason: str,
        payload: Mapping[str, Any] | None = None,
    ) -> WaveStateV2:
        version = wave.version + 1
        cursor = connection.execute(
            """
            UPDATE waves SET status = ?, updated_at = ?, version = ?
            WHERE wave_id = ? AND status = ? AND version = ?
            """,
            (
                target.value,
                _iso(occurred_at),
                version,
                wave.wave_id,
                wave.status.value,
                wave.version,
            ),
        )
        if cursor.rowcount != 1:
            raise WaveStateError("wave changed during transition")
        self._wave_event(
            connection,
            wave_id=wave.wave_id,
            from_status=wave.status,
            to_status=target,
            occurred_at=occurred_at,
            actor=actor,
            reason=reason,
            wave_version=version,
            payload=payload,
        )
        return self._read_wave(connection, wave.wave_id)

    @staticmethod
    def _decision(row: sqlite3.Row) -> WaveDecisionV2:
        decision = WaveDecisionV2.model_validate_json(row["decision_json"])
        if decision.decision_hash != row["decision_hash"]:
            raise WaveStateError("persisted decision hash differs from content")
        return decision

    @staticmethod
    def _wave_state(row: sqlite3.Row) -> WaveStateV2:
        budget = WaveBudgetV2.model_validate_json(row["budget_json"])
        spec = WaveSpecV2(
            wave_id=row["wave_id"],
            parent_wave_id=row["parent_wave_id"],
            policy_version=row["policy_version"],
            search_space_version=row["search_space_version"],
            budget=budget,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        if spec.spec_hash != row["spec_hash"]:
            raise WaveStateError("persisted wave spec hash differs from content")
        snapshot = (
            WaveResultSnapshotV2.model_validate_json(row["result_snapshot_json"])
            if row["result_snapshot_json"]
            else None
        )
        return WaveStateV2(
            wave_id=row["wave_id"],
            parent_wave_id=row["parent_wave_id"],
            policy_version=row["policy_version"],
            search_space_version=row["search_space_version"],
            budget=budget,
            spec_hash=row["spec_hash"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            result_snapshot=snapshot,
            result_snapshot_hash=row["result_snapshot_hash"],
            version=row["version"],
        )

    @classmethod
    def _read_wave(cls, connection: sqlite3.Connection, wave_id: str) -> WaveStateV2:
        row = connection.execute(
            "SELECT * FROM waves WHERE wave_id = ?", (wave_id,)
        ).fetchone()
        if row is None:
            raise WaveStateError(f"unknown wave: {wave_id}")
        return cls._wave_state(row)
