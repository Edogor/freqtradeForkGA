"""Transactional SQLite state machine for canonical GA V2 attempts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptStatus,
    StrictV2Model,
)


class AttemptStateError(ValueError):
    """Raised when an attempt transition or ownership check fails."""


class AttemptClaimConflict(AttemptStateError):
    """Raised when a stale or foreign worker tries to mutate an attempt."""


class AttemptLifecycleStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    INVALID_RESULT = "INVALID_RESULT"


class WorkerKind(StrEnum):
    SHADOW_REPLAY = "SHADOW_REPLAY"
    STANDARD_EVOLUTION = "STANDARD_EVOLUTION"
    GENERIC_ISLAND_EVOLUTION = "GENERIC_ISLAND_EVOLUTION"


CLAIM_RECOVERY_CONTRACT_V1 = "GUARDED_SPAWN_RECOVERY_V1"


def execution_command_hash(argv: list[str], working_directory: str | Path) -> str:
    return canonical_config_hash(
        {
            "argv": argv,
            "working_directory": str(Path(working_directory).resolve()),
        }
    )


class WorkerBindingV2(StrictV2Model):
    worker_kind: WorkerKind
    argv: list[str] = Field(min_length=1)
    working_directory: str = Field(min_length=1)
    spec_path: str = Field(min_length=1)
    spec_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _canonical_binding(self) -> WorkerBindingV2:
        if any(not item or "\x00" in item for item in self.argv):
            raise ValueError("worker argv contains an empty or NUL-bearing item")
        for field_name in ("working_directory", "spec_path"):
            if not Path(getattr(self, field_name)).is_absolute():
                raise ValueError(f"{field_name} must be absolute")
        return self

    @property
    def command_hash(self) -> str:
        return execution_command_hash(self.argv, self.working_directory)

    @property
    def binding_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


class SpawnGuardBindingV1(StrictV2Model):
    """Immutable one-time handshake paths for a guarded process spawn."""

    protocol: Literal["PARENT_DEATH_GUARD_V1"] = "PARENT_DEATH_GUARD_V1"
    ready_path: str = Field(min_length=1)
    release_path: str = Field(min_length=1)
    nonce_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _canonical_paths(self) -> SpawnGuardBindingV1:
        ready = Path(self.ready_path)
        release = Path(self.release_path)
        if not ready.is_absolute() or not release.is_absolute():
            raise ValueError("spawn guard paths must be absolute")
        if ready == release:
            raise ValueError("spawn guard paths must be distinct")
        return self

    @property
    def binding_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


_TERMINAL = {
    AttemptLifecycleStatus.SUCCEEDED,
    AttemptLifecycleStatus.FAILED,
    AttemptLifecycleStatus.INTERRUPTED,
    AttemptLifecycleStatus.INVALID_RESULT,
}
_ALLOWED_TRANSITIONS = {
    AttemptLifecycleStatus.DRAFT: {AttemptLifecycleStatus.VALIDATED},
    AttemptLifecycleStatus.VALIDATED: {AttemptLifecycleStatus.QUEUED},
    AttemptLifecycleStatus.QUEUED: {AttemptLifecycleStatus.CLAIMED},
    AttemptLifecycleStatus.CLAIMED: {
        AttemptLifecycleStatus.RUNNING,
        AttemptLifecycleStatus.FAILED,
        AttemptLifecycleStatus.INTERRUPTED,
        AttemptLifecycleStatus.INVALID_RESULT,
    },
    AttemptLifecycleStatus.RUNNING: _TERMINAL,
}


class AttemptStateV2(StrictV2Model):
    schema_version: str = "2.0"
    attempt_id: str = Field(min_length=1)
    wave_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    manifest_hash: str = Field(min_length=64, max_length=64)
    artifact_root: str = Field(min_length=1)
    status: AttemptLifecycleStatus
    priority: int = 0
    retry_number: int = Field(default=0, ge=0)
    worker_binding: WorkerBindingV2 | None = None
    worker_binding_hash: str | None = None
    claim_recovery_contract: str | None = None
    spawn_guard_binding: SpawnGuardBindingV1 | None = None
    spawn_guard_binding_hash: str | None = None
    created_at: datetime
    updated_at: datetime
    claimed_by: str | None = None
    claim_token: str | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    pid: int | None = Field(default=None, ge=1)
    process_start_token: str | None = None
    command_hash: str | None = None
    working_directory: str | None = None
    log_path: str | None = None
    log_sha256: str | None = None
    result_status: AttemptStatus | None = None
    result_path: str | None = None
    result_sha256: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    version: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent_lifecycle(self) -> AttemptStateV2:
        self._validate_timestamps()
        self._validate_claim_fields()
        self._validate_result_fields()
        return self

    def _validate_timestamps(self) -> None:
        for value in (
            self.created_at,
            self.updated_at,
            self.lease_expires_at,
            self.heartbeat_at,
        ):
            if value is not None and value.utcoffset() is None:
                raise ValueError("attempt-state timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at precedes created_at")

    def _validate_claim_fields(self) -> None:
        ownership = (self.claimed_by, self.claim_token, self.heartbeat_at)
        execution = (self.command_hash, self.working_directory, self.log_path)
        if self.status in {
            AttemptLifecycleStatus.DRAFT,
            AttemptLifecycleStatus.VALIDATED,
            AttemptLifecycleStatus.QUEUED,
        }:
            if any(value is not None for value in ownership + (self.lease_expires_at,)):
                raise ValueError("unclaimed attempt contains claim fields")
            if self.pid is not None or self.process_start_token is not None:
                raise ValueError("unclaimed attempt contains process identity")
            if any(value is not None for value in execution + (self.log_sha256,)):
                raise ValueError("unclaimed attempt contains execution metadata")
            if self.claim_recovery_contract is not None:
                raise ValueError("unclaimed attempt contains a recovery contract")
            if self.spawn_guard_binding is not None:
                raise ValueError("unclaimed attempt contains a spawn guard")
        elif self.status in {AttemptLifecycleStatus.CLAIMED, AttemptLifecycleStatus.RUNNING}:
            self._validate_active_claim(ownership, execution)
        else:
            if self.status not in _TERMINAL:
                raise ValueError("unknown terminal attempt state")
            if self.lease_expires_at is not None:
                raise ValueError("terminal attempt cannot retain an active lease")
        self._validate_execution_metadata(execution)
        if (self.worker_binding is None) != (self.worker_binding_hash is None):
            raise ValueError("worker binding and hash must be set together")
        if (
            self.worker_binding is not None
            and self.worker_binding.binding_hash != self.worker_binding_hash
        ):
            raise ValueError("worker binding hash differs from content")
        if (self.spawn_guard_binding is None) != (self.spawn_guard_binding_hash is None):
            raise ValueError("spawn guard binding and hash must be set together")
        if (
            self.spawn_guard_binding is not None
            and self.spawn_guard_binding.binding_hash != self.spawn_guard_binding_hash
        ):
            raise ValueError("spawn guard binding hash differs from content")
        if self.spawn_guard_binding is not None and any(value is None for value in execution):
            raise ValueError("spawn guard requires prepared execution metadata")
        if self.claim_recovery_contract not in {
            None,
            CLAIM_RECOVERY_CONTRACT_V1,
        }:
            raise ValueError("unknown claim recovery contract")

    def _validate_execution_metadata(
        self,
        execution: tuple[str | None, str | None, str | None],
    ) -> None:
        if any(value is not None for value in execution) and any(
            value is None for value in execution
        ):
            raise ValueError("execution metadata must be complete")
        if self.command_hash is not None and len(self.command_hash) != 64:
            raise ValueError("command_hash must be SHA-256")
        if self.log_sha256 is not None:
            if len(self.log_sha256) != 64:
                raise ValueError("log_sha256 must be SHA-256")
            if self.log_path is None:
                raise ValueError("log_sha256 requires log_path")

    def _validate_active_claim(
        self,
        ownership: tuple[str | None, str | None, datetime | None],
        execution: tuple[str | None, str | None, str | None],
    ) -> None:
        if any(value is None for value in ownership + (self.lease_expires_at,)):
            raise ValueError("active attempt lacks claim/lease fields")
        if self.status == AttemptLifecycleStatus.CLAIMED:
            if self.pid is not None or self.process_start_token is not None:
                raise ValueError("CLAIMED attempt cannot have process identity")
            return
        if self.pid is None or not self.process_start_token:
            raise ValueError("RUNNING attempt lacks process identity")
        if any(value is None for value in execution):
            raise ValueError("RUNNING attempt lacks execution metadata")

    def _validate_result_fields(self) -> None:
        if (self.result_path is None) != (self.result_sha256 is None):
            raise ValueError("result_path and result_sha256 must be set together")
        if self.result_sha256 is not None and len(self.result_sha256) != 64:
            raise ValueError("result_sha256 must be SHA-256")
        if self.result_status is not None and self.status.value != self.result_status.value:
            raise ValueError("result status differs from lifecycle status")
        if self.status == AttemptLifecycleStatus.SUCCEEDED:
            if self.result_status is None or self.result_path is None:
                raise ValueError("SUCCEEDED attempt requires verified result provenance")


class AttemptEventV2(StrictV2Model):
    event_id: int = Field(ge=1)
    attempt_id: str = Field(min_length=1)
    from_status: AttemptLifecycleStatus | None = None
    to_status: AttemptLifecycleStatus
    occurred_at: datetime
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    attempt_version: int = Field(ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)


def _iso(value: datetime) -> str:
    if value.utcoffset() is None:
        raise AttemptStateError("state timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _manifest_hash(manifest: AttemptManifestV2) -> str:
    return canonical_config_hash(manifest.model_dump(mode="json"))


def _runtime_log_hash(state: AttemptStateV2) -> str | None:
    if state.log_path is None:
        return None
    path = Path(state.log_path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AttemptStateStoreV2:
    """WAL-backed attempt state with transactional claims and fenced workers."""

    SCHEMA_VERSION = 6

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 30_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = busy_timeout_ms
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, 1, 2, 3, 4, 5, self.SCHEMA_VERSION}:
                raise AttemptStateError(f"unsupported attempt-state schema version: {version}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    wave_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    manifest_hash TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    artifact_root TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    retry_number INTEGER NOT NULL,
                    worker_kind TEXT,
                    worker_binding_json TEXT,
                    worker_binding_hash TEXT,
                    claim_recovery_contract TEXT,
                    spawn_guard_json TEXT,
                    spawn_guard_hash TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    claimed_by TEXT,
                    claim_token TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    pid INTEGER,
                    process_start_token TEXT,
                    command_hash TEXT,
                    working_directory TEXT,
                    log_path TEXT,
                    log_sha256 TEXT,
                    result_status TEXT,
                    result_path TEXT,
                    result_sha256 TEXT,
                    error_code TEXT,
                    error_detail TEXT,
                    version INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_attempt_claim_token
                    ON attempts(claim_token) WHERE claim_token IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_attempt_queue
                    ON attempts(status, priority DESC, created_at, attempt_id);
                CREATE TABLE IF NOT EXISTS attempt_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    attempt_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attempt_events_attempt
                    ON attempt_events(attempt_id, event_id);
                CREATE TRIGGER IF NOT EXISTS trg_attempt_events_no_update
                BEFORE UPDATE ON attempt_events
                BEGIN
                    SELECT RAISE(ABORT, 'attempt events are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_attempt_events_no_delete
                BEFORE DELETE ON attempt_events
                BEGIN
                    SELECT RAISE(ABORT, 'attempt events are immutable');
                END;

                CREATE TABLE IF NOT EXISTS waves (
                    wave_id TEXT PRIMARY KEY,
                    parent_wave_id TEXT,
                    policy_version TEXT NOT NULL,
                    search_space_version TEXT NOT NULL,
                    budget_json TEXT NOT NULL,
                    spec_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result_snapshot_json TEXT,
                    result_snapshot_hash TEXT,
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiment_specs (
                    experiment_id TEXT PRIMARY KEY,
                    wave_id TEXT NOT NULL REFERENCES waves(wave_id),
                    arm_type TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    spec_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_experiment_wave
                    ON experiment_specs(wave_id, experiment_id);
                CREATE TABLE IF NOT EXISTS wave_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    wave_id TEXT NOT NULL REFERENCES waves(wave_id),
                    experiment_id TEXT NOT NULL REFERENCES experiment_specs(experiment_id),
                    manifest_hash TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    UNIQUE(wave_id, experiment_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_wave_attempts_wave
                    ON wave_attempts(wave_id, experiment_id, ordinal);
                CREATE TABLE IF NOT EXISTS wave_decisions (
                    decision_id TEXT PRIMARY KEY,
                    wave_id TEXT NOT NULL REFERENCES waves(wave_id),
                    decision_type TEXT NOT NULL,
                    attempt_id TEXT,
                    input_hash TEXT,
                    decision_json TEXT NOT NULL,
                    decision_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_wave_decisions_wave
                    ON wave_decisions(wave_id, decision_type, created_at, decision_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_wave_attempt_abort
                    ON wave_decisions(wave_id, attempt_id)
                    WHERE decision_type = 'ATTEMPT_ABORT';
                CREATE TABLE IF NOT EXISTS wave_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wave_id TEXT NOT NULL REFERENCES waves(wave_id),
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    wave_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_wave_events_wave
                    ON wave_events(wave_id, event_id);
                CREATE TABLE IF NOT EXISTS legacy_registry_imports (
                    import_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_sha256 TEXT NOT NULL UNIQUE,
                    source_path TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    experiment_count INTEGER NOT NULL,
                    envelope_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS legacy_experiment_versions (
                    import_id INTEGER NOT NULL
                        REFERENCES legacy_registry_imports(import_id),
                    experiment_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(import_id, experiment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_legacy_experiment_latest
                    ON legacy_experiment_versions(experiment_id, import_id DESC);
                CREATE TRIGGER IF NOT EXISTS trg_waves_spec_immutable
                BEFORE UPDATE ON waves
                WHEN NEW.wave_id != OLD.wave_id
                    OR NEW.parent_wave_id IS NOT OLD.parent_wave_id
                    OR NEW.policy_version != OLD.policy_version
                    OR NEW.search_space_version != OLD.search_space_version
                    OR NEW.budget_json != OLD.budget_json
                    OR NEW.spec_hash != OLD.spec_hash
                    OR NEW.created_at != OLD.created_at
                BEGIN
                    SELECT RAISE(ABORT, 'wave specs are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_experiment_specs_no_update
                BEFORE UPDATE ON experiment_specs
                BEGIN
                    SELECT RAISE(ABORT, 'experiment specs are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_experiment_specs_no_delete
                BEFORE DELETE ON experiment_specs
                BEGIN
                    SELECT RAISE(ABORT, 'experiment specs are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_wave_attempts_no_update
                BEFORE UPDATE ON wave_attempts
                BEGIN
                    SELECT RAISE(ABORT, 'wave attempt expectations are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_wave_attempts_no_delete
                BEFORE DELETE ON wave_attempts
                BEGIN
                    SELECT RAISE(ABORT, 'wave attempt expectations are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_wave_decisions_no_update
                BEFORE UPDATE ON wave_decisions
                BEGIN
                    SELECT RAISE(ABORT, 'wave decisions are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_wave_decisions_no_delete
                BEFORE DELETE ON wave_decisions
                BEGIN
                    SELECT RAISE(ABORT, 'wave decisions are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_wave_events_no_update
                BEFORE UPDATE ON wave_events
                BEGIN
                    SELECT RAISE(ABORT, 'wave events are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_wave_events_no_delete
                BEFORE DELETE ON wave_events
                BEGIN
                    SELECT RAISE(ABORT, 'wave events are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_legacy_registry_imports_no_update
                BEFORE UPDATE ON legacy_registry_imports
                BEGIN
                    SELECT RAISE(ABORT, 'legacy registry imports are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_legacy_registry_imports_no_delete
                BEFORE DELETE ON legacy_registry_imports
                BEGIN
                    SELECT RAISE(ABORT, 'legacy registry imports are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_legacy_experiment_versions_no_update
                BEFORE UPDATE ON legacy_experiment_versions
                BEGIN
                    SELECT RAISE(ABORT, 'legacy experiment versions are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_legacy_experiment_versions_no_delete
                BEFORE DELETE ON legacy_experiment_versions
                BEGIN
                    SELECT RAISE(ABORT, 'legacy experiment versions are immutable');
                END;
                """
            )
            if version == 1:
                active_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM attempts WHERE status IN ('CLAIMED', 'RUNNING')"
                    ).fetchone()[0]
                )
                if active_count:
                    raise AttemptStateError(
                        "cannot migrate attempt-state v1 with active claims; "
                        "reconcile them before upgrading"
                    )
                connection.execute("ALTER TABLE attempts ADD COLUMN command_hash TEXT")
                connection.execute("ALTER TABLE attempts ADD COLUMN working_directory TEXT")
                connection.execute("ALTER TABLE attempts ADD COLUMN log_path TEXT")
                connection.execute("ALTER TABLE attempts ADD COLUMN log_sha256 TEXT")
            if version in {1, 2}:
                connection.execute("ALTER TABLE attempts ADD COLUMN worker_kind TEXT")
                connection.execute("ALTER TABLE attempts ADD COLUMN worker_binding_json TEXT")
                connection.execute("ALTER TABLE attempts ADD COLUMN worker_binding_hash TEXT")
            attempt_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(attempts)").fetchall()
            }
            for column in (
                "claim_recovery_contract",
                "spawn_guard_json",
                "spawn_guard_hash",
            ):
                if column not in attempt_columns:
                    connection.execute(f"ALTER TABLE attempts ADD COLUMN {column} TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempt_worker_queue "
                "ON attempts(worker_kind, status, priority DESC, created_at, attempt_id)"
            )
            connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        began = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            began = True
            yield connection
            connection.execute("COMMIT")
        except Exception:
            if began:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        *,
        attempt_id: str,
        from_status: AttemptLifecycleStatus | None,
        to_status: AttemptLifecycleStatus,
        occurred_at: datetime,
        actor: str,
        reason: str,
        attempt_version: int,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO attempt_events (
                attempt_id, from_status, to_status, occurred_at, actor,
                reason, attempt_version, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                from_status.value if from_status else None,
                to_status.value,
                _iso(occurred_at),
                actor,
                reason,
                attempt_version,
                json.dumps(dict(payload or {}), sort_keys=True, separators=(",", ":")),
            ),
        )

    def register(
        self,
        manifest: AttemptManifestV2,
        *,
        priority: int = 0,
        retry_number: int = 0,
        actor: str = "controller",
    ) -> AttemptStateV2:
        manifest_hash = _manifest_hash(manifest)
        manifest_payload = json.dumps(
            manifest.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        created_at_iso = _iso(manifest.created_at)
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?",
                (manifest.attempt_id,),
            ).fetchone()
            if existing is not None:
                state = self._state(existing)
                if (
                    state.manifest_hash != manifest_hash
                    or state.priority != priority
                    or state.retry_number != retry_number
                ):
                    raise AttemptStateError(
                        "attempt_id is already registered with different inputs"
                    )
                return state
            connection.execute(
                """
                INSERT INTO attempts (
                    attempt_id, wave_id, experiment_id, manifest_hash, manifest_json,
                    artifact_root, status, priority, retry_number, created_at,
                    updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, 'DRAFT', ?, ?, ?, ?, 0)
                """,
                (
                    manifest.attempt_id,
                    manifest.wave_id,
                    manifest.experiment_id,
                    manifest_hash,
                    manifest_payload,
                    manifest.artifact_root,
                    priority,
                    retry_number,
                    created_at_iso,
                    created_at_iso,
                ),
            )
            self._event(
                connection,
                attempt_id=manifest.attempt_id,
                from_status=None,
                to_status=AttemptLifecycleStatus.DRAFT,
                occurred_at=manifest.created_at,
                actor=actor,
                reason="REGISTERED",
                attempt_version=0,
                payload={"manifest_hash": manifest_hash},
            )
            return self._read_state(connection, manifest.attempt_id)

    def bind_worker(
        self,
        attempt_id: str,
        binding: WorkerBindingV2,
        *,
        bound_at: datetime,
        actor: str = "controller",
    ) -> AttemptStateV2:
        """Bind exact immutable worker inputs before validation and queueing."""

        binding_payload = json.dumps(
            binding.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._transaction() as connection:
            current = self._read_state(connection, attempt_id)
            if current.worker_binding is not None:
                if current.worker_binding == binding:
                    self._validate_worker_binding(current, binding)
                    return current
                raise AttemptStateError("attempt already has a different worker binding")
            if current.status != AttemptLifecycleStatus.DRAFT:
                raise AttemptStateError("worker binding must precede VALIDATED")
            self._require_monotonic_time(current, bound_at)
            self._validate_worker_binding(current, binding)
            version = current.version + 1
            connection.execute(
                """
                UPDATE attempts
                SET worker_kind = ?, worker_binding_json = ?, worker_binding_hash = ?,
                    updated_at = ?, version = ?
                WHERE attempt_id = ?
                """,
                (
                    binding.worker_kind.value,
                    binding_payload,
                    binding.binding_hash,
                    _iso(bound_at),
                    version,
                    attempt_id,
                ),
            )
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=current.status,
                to_status=current.status,
                occurred_at=bound_at,
                actor=actor,
                reason="WORKER_BOUND",
                attempt_version=version,
                payload={
                    "worker_kind": binding.worker_kind.value,
                    "worker_binding_hash": binding.binding_hash,
                    "command_hash": binding.command_hash,
                },
            )
            return self._read_state(connection, attempt_id)

    @staticmethod
    def _validate_worker_binding(
        current: AttemptStateV2,
        binding: WorkerBindingV2,
    ) -> None:
        spec_path = Path(binding.spec_path).resolve()
        expected_spec = Path(current.artifact_root).resolve() / V2ArtifactStore.WORKER_SPEC_NAME
        if spec_path != expected_spec or not spec_path.is_file():
            raise AttemptStateError("worker spec path is missing or not canonical")
        if hashlib.sha256(spec_path.read_bytes()).hexdigest() != binding.spec_sha256:
            raise AttemptStateError("worker spec hash differs before queueing")
        workdir = Path(binding.working_directory)
        if not workdir.is_dir():
            raise AttemptStateError("worker working directory does not exist")
        try:
            spec_index = binding.argv.index("--spec")
            hash_index = binding.argv.index("--spec-sha256")
        except ValueError as exc:
            raise AttemptStateError("worker argv lacks spec fencing arguments") from exc
        if spec_index + 1 >= len(binding.argv) or binding.argv[spec_index + 1] != str(spec_path):
            raise AttemptStateError("worker argv spec path differs from binding")
        if (
            hash_index + 1 >= len(binding.argv)
            or binding.argv[hash_index + 1] != binding.spec_sha256
        ):
            raise AttemptStateError("worker argv spec hash differs from binding")

    def transition(
        self,
        attempt_id: str,
        to_status: AttemptLifecycleStatus,
        *,
        occurred_at: datetime,
        actor: str,
        reason: str,
    ) -> AttemptStateV2:
        if (
            to_status
            in {AttemptLifecycleStatus.CLAIMED, AttemptLifecycleStatus.RUNNING} | _TERMINAL
        ):
            raise AttemptStateError(
                "claim/runtime/terminal transitions require specialized methods"
            )
        with self._transaction() as connection:
            current = self._read_state(connection, attempt_id)
            self._require_transition(current.status, to_status)
            self._require_monotonic_time(current, occurred_at)
            version = current.version + 1
            connection.execute(
                "UPDATE attempts SET status = ?, updated_at = ?, version = ? WHERE attempt_id = ?",
                (to_status.value, _iso(occurred_at), version, attempt_id),
            )
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=current.status,
                to_status=to_status,
                occurred_at=occurred_at,
                actor=actor,
                reason=reason,
                attempt_version=version,
            )
            return self._read_state(connection, attempt_id)

    def claim_next(
        self,
        *,
        worker_id: str,
        claimed_at: datetime,
        lease_seconds: int,
        worker_kind: WorkerKind | None = None,
    ) -> AttemptStateV2 | None:
        if not worker_id:
            raise AttemptStateError("worker_id cannot be empty")
        if lease_seconds <= 0:
            raise AttemptStateError("lease_seconds must be positive")
        claimed_at_iso = _iso(claimed_at)
        lease_expires = claimed_at + timedelta(seconds=lease_seconds)
        claim_token = uuid.uuid4().hex
        with self._transaction() as connection:
            if worker_kind is None:
                row = connection.execute(
                    """
                    SELECT * FROM attempts
                    WHERE status = 'QUEUED'
                    ORDER BY priority DESC, created_at, attempt_id
                    LIMIT 1
                    """
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT * FROM attempts
                    WHERE status = 'QUEUED' AND worker_kind = ?
                    ORDER BY priority DESC, created_at, attempt_id
                    LIMIT 1
                    """,
                    (worker_kind.value,),
                ).fetchone()
            if row is None:
                return None
            current = self._state(row)
            self._require_transition(current.status, AttemptLifecycleStatus.CLAIMED)
            if claimed_at < current.updated_at:
                raise AttemptStateError("claimed_at precedes last state update")
            version = current.version + 1
            connection.execute(
                """
                UPDATE attempts
                SET status = 'CLAIMED', updated_at = ?, claimed_by = ?, claim_token = ?,
                    lease_expires_at = ?, heartbeat_at = ?,
                    claim_recovery_contract = ?, version = ?
                WHERE attempt_id = ? AND status = 'QUEUED' AND version = ?
                """,
                (
                    claimed_at_iso,
                    worker_id,
                    claim_token,
                    _iso(lease_expires),
                    claimed_at_iso,
                    CLAIM_RECOVERY_CONTRACT_V1,
                    version,
                    current.attempt_id,
                    current.version,
                ),
            )
            self._event(
                connection,
                attempt_id=current.attempt_id,
                from_status=current.status,
                to_status=AttemptLifecycleStatus.CLAIMED,
                occurred_at=claimed_at,
                actor=worker_id,
                reason="CLAIMED_FROM_QUEUE",
                attempt_version=version,
                payload={
                    "claim_token": claim_token,
                    "lease_seconds": lease_seconds,
                    "claim_recovery_contract": CLAIM_RECOVERY_CONTRACT_V1,
                },
            )
            return self._read_state(connection, current.attempt_id)

    def claim_attempt(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        claimed_at: datetime,
        lease_seconds: int,
    ) -> AttemptStateV2:
        """Atomically claim one exact queued attempt.

        Interactive/manual execution must never call ``claim_next`` after
        queueing its own attempt: another, older queue entry could otherwise
        be executed under the caller's identity.  Schedulers use
        :meth:`claim_next`; direct runners use this exact-ID variant.
        """

        if not attempt_id:
            raise AttemptStateError("attempt_id cannot be empty")
        if not worker_id:
            raise AttemptStateError("worker_id cannot be empty")
        if lease_seconds <= 0:
            raise AttemptStateError("lease_seconds must be positive")
        claimed_at_iso = _iso(claimed_at)
        lease_expires = claimed_at + timedelta(seconds=lease_seconds)
        claim_token = uuid.uuid4().hex
        with self._transaction() as connection:
            current = self._read_state(connection, attempt_id)
            if current.status != AttemptLifecycleStatus.QUEUED:
                raise AttemptClaimConflict(f"attempt is not claimable: {current.status.value}")
            self._require_transition(current.status, AttemptLifecycleStatus.CLAIMED)
            if claimed_at < current.updated_at:
                raise AttemptStateError("claimed_at precedes last state update")
            version = current.version + 1
            cursor = connection.execute(
                """
                UPDATE attempts
                SET status = 'CLAIMED', updated_at = ?, claimed_by = ?, claim_token = ?,
                    lease_expires_at = ?, heartbeat_at = ?,
                    claim_recovery_contract = ?, version = ?
                WHERE attempt_id = ? AND status = 'QUEUED' AND version = ?
                """,
                (
                    claimed_at_iso,
                    worker_id,
                    claim_token,
                    _iso(lease_expires),
                    claimed_at_iso,
                    CLAIM_RECOVERY_CONTRACT_V1,
                    version,
                    attempt_id,
                    current.version,
                ),
            )
            if cursor.rowcount != 1:
                raise AttemptClaimConflict("attempt was claimed concurrently")
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=current.status,
                to_status=AttemptLifecycleStatus.CLAIMED,
                occurred_at=claimed_at,
                actor=worker_id,
                reason="CLAIMED_BY_ID",
                attempt_version=version,
                payload={
                    "claim_token": claim_token,
                    "lease_seconds": lease_seconds,
                    "claim_recovery_contract": CLAIM_RECOVERY_CONTRACT_V1,
                },
            )
            return self._read_state(connection, attempt_id)

    def prepare_execution(
        self,
        attempt_id: str,
        *,
        claim_token: str,
        command_hash: str,
        working_directory: str | Path,
        log_path: str | Path,
        prepared_at: datetime,
        spawn_guard_binding: SpawnGuardBindingV1 | None = None,
    ) -> AttemptStateV2:
        """Persist immutable execution inputs before a child process is spawned."""

        if len(command_hash) != 64:
            raise AttemptStateError("command_hash must be SHA-256")
        resolved_workdir = Path(working_directory).resolve()
        resolved_log = Path(log_path).resolve()
        with self._transaction() as connection:
            current = self._read_state(connection, attempt_id)
            self._require_owner(current, claim_token)
            self._require_live_lease(current, prepared_at)
            if current.status != AttemptLifecycleStatus.CLAIMED:
                raise AttemptStateError("execution preparation requires CLAIMED attempt")
            self._require_monotonic_time(current, prepared_at)
            expected_log = Path(current.artifact_root).resolve() / "runtime" / "attempt.log"
            if resolved_log != expected_log:
                raise AttemptStateError("log_path must be <artifact_root>/runtime/attempt.log")
            if current.worker_binding is not None:
                if current.worker_binding.command_hash != command_hash:
                    raise AttemptClaimConflict("runtime command differs from queued worker binding")
                if Path(current.worker_binding.working_directory) != resolved_workdir:
                    raise AttemptClaimConflict(
                        "runtime working directory differs from queued worker binding"
                    )
            if spawn_guard_binding is not None:
                runtime_root = Path(current.artifact_root).resolve() / "runtime"
                expected_ready = runtime_root / "spawn.ready.json"
                expected_release = runtime_root / "spawn.release.json"
                if Path(spawn_guard_binding.ready_path) != expected_ready:
                    raise AttemptStateError("spawn ready path is not canonical")
                if Path(spawn_guard_binding.release_path) != expected_release:
                    raise AttemptStateError("spawn release path is not canonical")
                if current.claim_recovery_contract != CLAIM_RECOVERY_CONTRACT_V1:
                    raise AttemptStateError(
                        "spawn guard requires the canonical claim recovery contract"
                    )
            if any(
                value is not None
                for value in (
                    current.command_hash,
                    current.working_directory,
                    current.log_path,
                )
            ):
                if (
                    current.command_hash == command_hash
                    and current.working_directory == str(resolved_workdir)
                    and current.log_path == str(resolved_log)
                    and current.spawn_guard_binding == spawn_guard_binding
                ):
                    return current
                raise AttemptStateError("execution inputs are already prepared and immutable")
            guard_payload = (
                json.dumps(
                    spawn_guard_binding.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if spawn_guard_binding is not None
                else None
            )
            guard_hash = (
                spawn_guard_binding.binding_hash if spawn_guard_binding is not None else None
            )
            version = current.version + 1
            connection.execute(
                """
                UPDATE attempts
                SET updated_at = ?, command_hash = ?, working_directory = ?,
                    log_path = ?, spawn_guard_json = ?, spawn_guard_hash = ?,
                    version = ?
                WHERE attempt_id = ?
                """,
                (
                    _iso(prepared_at),
                    command_hash,
                    str(resolved_workdir),
                    str(resolved_log),
                    guard_payload,
                    guard_hash,
                    version,
                    attempt_id,
                ),
            )
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=current.status,
                to_status=current.status,
                occurred_at=prepared_at,
                actor=str(current.claimed_by),
                reason="EXECUTION_PREPARED",
                attempt_version=version,
                payload={
                    "command_hash": command_hash,
                    "working_directory": str(resolved_workdir),
                    "log_path": str(resolved_log),
                    "spawn_guard_binding_hash": guard_hash,
                },
            )
            return self._read_state(connection, attempt_id)

    def mark_running(
        self,
        attempt_id: str,
        *,
        claim_token: str,
        pid: int,
        process_start_token: str,
        started_at: datetime,
        lease_seconds: int,
    ) -> AttemptStateV2:
        if pid < 1 or not process_start_token:
            raise AttemptStateError("running process identity is incomplete")
        if lease_seconds <= 0:
            raise AttemptStateError("lease_seconds must be positive")
        with self._transaction() as connection:
            current = self._read_state(connection, attempt_id)
            self._require_owner(current, claim_token)
            self._require_live_lease(current, started_at)
            self._require_transition(current.status, AttemptLifecycleStatus.RUNNING)
            self._require_monotonic_time(current, started_at)
            if any(
                value is None
                for value in (
                    current.command_hash,
                    current.working_directory,
                    current.log_path,
                )
            ):
                raise AttemptStateError("execution must be prepared before process start")
            version = current.version + 1
            lease_expires = started_at + timedelta(seconds=lease_seconds)
            connection.execute(
                """
                UPDATE attempts
                SET status = 'RUNNING', updated_at = ?, lease_expires_at = ?,
                    heartbeat_at = ?, pid = ?, process_start_token = ?, version = ?
                WHERE attempt_id = ?
                """,
                (
                    _iso(started_at),
                    _iso(lease_expires),
                    _iso(started_at),
                    pid,
                    process_start_token,
                    version,
                    attempt_id,
                ),
            )
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=current.status,
                to_status=AttemptLifecycleStatus.RUNNING,
                occurred_at=started_at,
                actor=str(current.claimed_by),
                reason="PROCESS_STARTED",
                attempt_version=version,
                payload={"pid": pid, "process_start_token": process_start_token},
            )
            return self._read_state(connection, attempt_id)

    def heartbeat(
        self,
        attempt_id: str,
        *,
        claim_token: str,
        process_start_token: str | None,
        heartbeat_at: datetime,
        lease_seconds: int,
    ) -> AttemptStateV2:
        if lease_seconds <= 0:
            raise AttemptStateError("lease_seconds must be positive")
        with self._transaction() as connection:
            current = self._read_state(connection, attempt_id)
            self._require_owner(current, claim_token)
            self._require_live_lease(current, heartbeat_at)
            if current.status not in {
                AttemptLifecycleStatus.CLAIMED,
                AttemptLifecycleStatus.RUNNING,
            }:
                raise AttemptStateError("heartbeat requires CLAIMED or RUNNING attempt")
            if current.status == AttemptLifecycleStatus.RUNNING:
                if process_start_token != current.process_start_token:
                    raise AttemptClaimConflict("process start token differs from running attempt")
            elif process_start_token is not None:
                raise AttemptClaimConflict("CLAIMED heartbeat cannot provide a process token")
            self._require_monotonic_time(current, heartbeat_at)
            version = current.version + 1
            lease_expires = heartbeat_at + timedelta(seconds=lease_seconds)
            connection.execute(
                """
                UPDATE attempts
                SET updated_at = ?, heartbeat_at = ?, lease_expires_at = ?, version = ?
                WHERE attempt_id = ?
                """,
                (
                    _iso(heartbeat_at),
                    _iso(heartbeat_at),
                    _iso(lease_expires),
                    version,
                    attempt_id,
                ),
            )
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=current.status,
                to_status=current.status,
                occurred_at=heartbeat_at,
                actor=str(current.claimed_by),
                reason="HEARTBEAT",
                attempt_version=version,
                payload={"lease_seconds": lease_seconds},
            )
            return self._read_state(connection, attempt_id)

    def finalize_from_artifacts(
        self,
        attempt_id: str,
        *,
        claim_token: str,
        process_start_token: str,
        finalized_at: datetime,
    ) -> AttemptStateV2:
        """Transition from RUNNING only after verified terminal result readback."""

        current = self.get(attempt_id)
        self._require_owner(current, claim_token)
        self._require_live_lease(current, finalized_at)
        if current.status != AttemptLifecycleStatus.RUNNING:
            raise AttemptStateError("verified completion requires RUNNING attempt")
        if process_start_token != current.process_start_token:
            raise AttemptClaimConflict("process start token differs from running attempt")
        try:
            result = V2ArtifactStore(current.artifact_root).read_verified_result()
        except ArtifactIntegrityError:
            raise
        if result.attempt_id != attempt_id:
            raise ArtifactIntegrityError("terminal result belongs to another attempt")
        if _manifest_hash(result.manifest) != current.manifest_hash:
            raise ArtifactIntegrityError("terminal result manifest differs from registered attempt")
        if finalized_at < result.finished_at:
            raise AttemptStateError("finalized_at precedes terminal result timestamp")
        result_path = Path(current.artifact_root) / V2ArtifactStore.RESULT_NAME
        result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
        log_sha256 = _runtime_log_hash(current)
        target = AttemptLifecycleStatus(result.status.value)

        with self._transaction() as connection:
            latest = self._read_state(connection, attempt_id)
            self._require_owner(latest, claim_token)
            self._require_live_lease(latest, finalized_at)
            if latest.process_start_token != process_start_token:
                raise AttemptClaimConflict("process identity changed during result verification")
            self._require_transition(latest.status, target)
            self._require_monotonic_time(latest, finalized_at)
            version = latest.version + 1
            connection.execute(
                """
                UPDATE attempts
                SET status = ?, updated_at = ?, lease_expires_at = NULL,
                    result_status = ?, result_path = ?, result_sha256 = ?,
                    log_sha256 = ?, error_code = ?, error_detail = ?, version = ?
                WHERE attempt_id = ?
                """,
                (
                    target.value,
                    _iso(finalized_at),
                    result.status.value,
                    str(result_path.resolve()),
                    result_sha256,
                    log_sha256,
                    result.error_code,
                    result.error_detail,
                    version,
                    attempt_id,
                ),
            )
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=latest.status,
                to_status=target,
                occurred_at=finalized_at,
                actor=str(latest.claimed_by),
                reason="VERIFIED_TERMINAL_RESULT",
                attempt_version=version,
                payload={"result_sha256": result_sha256},
            )
            return self._read_state(connection, attempt_id)

    def mark_runtime_failure(
        self,
        attempt_id: str,
        *,
        claim_token: str,
        process_start_token: str | None,
        status: AttemptLifecycleStatus,
        failed_at: datetime,
        error_code: str,
        error_detail: str | None = None,
    ) -> AttemptStateV2:
        if status not in {
            AttemptLifecycleStatus.FAILED,
            AttemptLifecycleStatus.INTERRUPTED,
            AttemptLifecycleStatus.INVALID_RESULT,
        }:
            raise AttemptStateError("runtime failure must use a non-success terminal state")
        if not error_code:
            raise AttemptStateError("runtime failure requires error_code")
        with self._transaction() as connection:
            current = self._read_state(connection, attempt_id)
            self._require_owner(current, claim_token)
            self._require_live_lease(current, failed_at)
            if current.status == AttemptLifecycleStatus.RUNNING:
                if process_start_token != current.process_start_token:
                    raise AttemptClaimConflict("process start token differs from running attempt")
            elif current.status == AttemptLifecycleStatus.CLAIMED:
                if process_start_token is not None:
                    raise AttemptClaimConflict("CLAIMED failure cannot provide process token")
            self._require_transition(current.status, status)
            self._require_monotonic_time(current, failed_at)
            log_sha256 = _runtime_log_hash(current)
            version = current.version + 1
            connection.execute(
                """
                UPDATE attempts
                SET status = ?, updated_at = ?, lease_expires_at = NULL,
                    log_sha256 = ?, error_code = ?, error_detail = ?, version = ?
                WHERE attempt_id = ?
                """,
                (
                    status.value,
                    _iso(failed_at),
                    log_sha256,
                    error_code,
                    error_detail,
                    version,
                    attempt_id,
                ),
            )
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=current.status,
                to_status=status,
                occurred_at=failed_at,
                actor=str(current.claimed_by),
                reason=error_code,
                attempt_version=version,
                payload={"error_detail": error_detail},
            )
            return self._read_state(connection, attempt_id)

    def reconcile_expired_from_artifacts(
        self,
        attempt_id: str,
        *,
        expected_version: int,
        expected_process_start_token: str,
        reconciled_at: datetime,
        actor: str,
        process_evidence: str,
    ) -> AttemptStateV2:
        """Finalize an expired RUNNING attempt after its process is proven gone.

        Process liveness is deliberately established outside this persistence layer.
        Version and process-token fencing make the evidence apply to exactly the
        state that the reconciler inspected.
        """

        current = self.get(attempt_id)
        self._require_reconcilable_expired_runtime(
            current,
            expected_version=expected_version,
            expected_process_start_token=expected_process_start_token,
            reconciled_at=reconciled_at,
        )
        result = V2ArtifactStore(current.artifact_root).read_verified_result()
        if result.attempt_id != attempt_id:
            raise ArtifactIntegrityError("terminal result belongs to another attempt")
        if _manifest_hash(result.manifest) != current.manifest_hash:
            raise ArtifactIntegrityError("terminal result manifest differs from registered attempt")
        if reconciled_at < result.finished_at:
            raise AttemptStateError("reconciled_at precedes terminal result timestamp")
        result_path = Path(current.artifact_root) / V2ArtifactStore.RESULT_NAME
        result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
        log_sha256 = _runtime_log_hash(current)
        target = AttemptLifecycleStatus(result.status.value)

        with self._transaction() as connection:
            latest = self._read_state(connection, attempt_id)
            self._require_reconcilable_expired_runtime(
                latest,
                expected_version=expected_version,
                expected_process_start_token=expected_process_start_token,
                reconciled_at=reconciled_at,
            )
            self._require_transition(latest.status, target)
            version = latest.version + 1
            connection.execute(
                """
                UPDATE attempts
                SET status = ?, updated_at = ?, lease_expires_at = NULL,
                    result_status = ?, result_path = ?, result_sha256 = ?,
                    log_sha256 = ?, error_code = ?, error_detail = ?, version = ?
                WHERE attempt_id = ? AND version = ?
                """,
                (
                    target.value,
                    _iso(reconciled_at),
                    result.status.value,
                    str(result_path.resolve()),
                    result_sha256,
                    log_sha256,
                    result.error_code,
                    result.error_detail,
                    version,
                    attempt_id,
                    expected_version,
                ),
            )
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=latest.status,
                to_status=target,
                occurred_at=reconciled_at,
                actor=actor,
                reason="RECOVERED_VERIFIED_TERMINAL_RESULT",
                attempt_version=version,
                payload={
                    "process_evidence": process_evidence,
                    "result_sha256": result_sha256,
                },
            )
            return self._read_state(connection, attempt_id)

    def reconcile_expired_claimed_failure(
        self,
        attempt_id: str,
        *,
        expected_version: int,
        reconciled_at: datetime,
        actor: str,
        error_code: str,
        error_detail: str | None = None,
    ) -> AttemptStateV2:
        """Close a canonical expired CLAIMED state that could not run a child.

        A new claim carries ``GUARDED_SPAWN_RECOVERY_V1``. Before preparation
        no spawn is permitted. After preparation the parent-death guard cannot
        start the target without a release receipt, and that receipt is written
        only after the transaction changing the state to ``RUNNING`` commits.
        """

        if not error_code:
            raise AttemptStateError("claimed reconciliation requires error_code")
        with self._transaction() as connection:
            current = self._read_state(connection, attempt_id)
            if current.status != AttemptLifecycleStatus.CLAIMED:
                raise AttemptStateError("claimed reconciliation requires CLAIMED attempt")
            if current.version != expected_version:
                raise AttemptClaimConflict(
                    "attempt changed after claimed recovery evidence was collected"
                )
            if current.lease_expires_at is None or reconciled_at < current.lease_expires_at:
                raise AttemptStateError("claim lease has not expired")
            if current.claim_recovery_contract != CLAIM_RECOVERY_CONTRACT_V1:
                raise AttemptStateError("claim has no canonical recovery contract")
            if current.command_hash is not None and current.spawn_guard_binding is None:
                raise AttemptStateError(
                    "prepared claim has no guarded-spawn proof and is not recoverable"
                )
            if (
                current.spawn_guard_binding is not None
                and Path(current.spawn_guard_binding.release_path).exists()
            ):
                raise AttemptStateError(
                    "CLAIMED state has a release receipt and requires manual recovery"
                )
            self._require_transition(
                current.status,
                AttemptLifecycleStatus.INTERRUPTED,
            )
            self._require_monotonic_time(current, reconciled_at)
            log_sha256 = _runtime_log_hash(current)
            version = current.version + 1
            cursor = connection.execute(
                """
                UPDATE attempts
                SET status = 'INTERRUPTED', updated_at = ?,
                    lease_expires_at = NULL, log_sha256 = ?,
                    error_code = ?, error_detail = ?, version = ?
                WHERE attempt_id = ? AND status = 'CLAIMED' AND version = ?
                """,
                (
                    _iso(reconciled_at),
                    log_sha256,
                    error_code,
                    error_detail,
                    version,
                    attempt_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise AttemptClaimConflict("claimed attempt changed during reconciliation")
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=current.status,
                to_status=AttemptLifecycleStatus.INTERRUPTED,
                occurred_at=reconciled_at,
                actor=actor,
                reason=error_code,
                attempt_version=version,
                payload={
                    "claim_recovery_contract": current.claim_recovery_contract,
                    "spawn_guard_binding_hash": current.spawn_guard_binding_hash,
                    "error_detail": error_detail,
                },
            )
            return self._read_state(connection, attempt_id)

    def reconcile_expired_failure(
        self,
        attempt_id: str,
        *,
        expected_version: int,
        expected_process_start_token: str,
        reconciled_at: datetime,
        actor: str,
        process_evidence: str,
        status: AttemptLifecycleStatus,
        error_code: str,
        error_detail: str | None = None,
    ) -> AttemptStateV2:
        """Close an expired RUNNING attempt after its process is proven gone."""

        if status not in {
            AttemptLifecycleStatus.INTERRUPTED,
            AttemptLifecycleStatus.INVALID_RESULT,
        }:
            raise AttemptStateError(
                "expired runtime reconciliation permits INTERRUPTED or INVALID_RESULT"
            )
        if not error_code:
            raise AttemptStateError("expired runtime reconciliation requires error_code")
        with self._transaction() as connection:
            current = self._read_state(connection, attempt_id)
            self._require_reconcilable_expired_runtime(
                current,
                expected_version=expected_version,
                expected_process_start_token=expected_process_start_token,
                reconciled_at=reconciled_at,
            )
            self._require_transition(current.status, status)
            log_sha256 = _runtime_log_hash(current)
            version = current.version + 1
            connection.execute(
                """
                UPDATE attempts
                SET status = ?, updated_at = ?, lease_expires_at = NULL,
                    log_sha256 = ?, error_code = ?, error_detail = ?, version = ?
                WHERE attempt_id = ? AND version = ?
                """,
                (
                    status.value,
                    _iso(reconciled_at),
                    log_sha256,
                    error_code,
                    error_detail,
                    version,
                    attempt_id,
                    expected_version,
                ),
            )
            self._event(
                connection,
                attempt_id=attempt_id,
                from_status=current.status,
                to_status=status,
                occurred_at=reconciled_at,
                actor=actor,
                reason=error_code,
                attempt_version=version,
                payload={
                    "process_evidence": process_evidence,
                    "error_detail": error_detail,
                },
            )
            return self._read_state(connection, attempt_id)

    def expired_leases(self, *, as_of: datetime) -> list[AttemptStateV2]:
        as_of_iso = _iso(as_of)
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM attempts
                WHERE status IN ('CLAIMED', 'RUNNING') AND lease_expires_at <= ?
                ORDER BY lease_expires_at, attempt_id
                """,
                (as_of_iso,),
            ).fetchall()
            return [self._state(row) for row in rows]
        finally:
            connection.close()

    def list_attempts(
        self,
        *,
        statuses: set[AttemptLifecycleStatus] | None = None,
        worker_kind: WorkerKind | None = None,
    ) -> list[AttemptStateV2]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM attempts ORDER BY created_at, attempt_id"
            ).fetchall()
            states = [self._state(row) for row in rows]
            if statuses:
                states = [state for state in states if state.status in statuses]
            if worker_kind is not None:
                states = [
                    state
                    for state in states
                    if state.worker_binding is not None
                    and state.worker_binding.worker_kind == worker_kind
                ]
            return states
        finally:
            connection.close()

    def count_attempts(
        self,
        *,
        statuses: set[AttemptLifecycleStatus] | None = None,
        worker_kind: WorkerKind | None = None,
    ) -> int:
        return len(self.list_attempts(statuses=statuses, worker_kind=worker_kind))

    @staticmethod
    def _require_reconcilable_expired_runtime(
        current: AttemptStateV2,
        *,
        expected_version: int,
        expected_process_start_token: str,
        reconciled_at: datetime,
    ) -> None:
        _iso(reconciled_at)
        if current.status != AttemptLifecycleStatus.RUNNING:
            raise AttemptStateError("reconciliation requires RUNNING attempt")
        if current.version != expected_version:
            raise AttemptClaimConflict("attempt changed after process evidence was collected")
        if current.process_start_token != expected_process_start_token:
            raise AttemptClaimConflict("process identity changed after evidence was collected")
        if current.lease_expires_at is None or reconciled_at < current.lease_expires_at:
            raise AttemptStateError("runtime lease has not expired")

    def get(self, attempt_id: str) -> AttemptStateV2:
        connection = self._connect()
        try:
            return self._read_state(connection, attempt_id)
        finally:
            connection.close()

    def events(self, attempt_id: str) -> list[AttemptEventV2]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM attempt_events
                WHERE attempt_id = ? ORDER BY event_id
                """,
                (attempt_id,),
            ).fetchall()
            return [
                AttemptEventV2(
                    event_id=row["event_id"],
                    attempt_id=row["attempt_id"],
                    from_status=row["from_status"],
                    to_status=row["to_status"],
                    occurred_at=datetime.fromisoformat(row["occurred_at"]),
                    actor=row["actor"],
                    reason=row["reason"],
                    attempt_version=row["attempt_version"],
                    payload=json.loads(row["payload_json"]),
                )
                for row in rows
            ]
        finally:
            connection.close()

    @staticmethod
    def _require_transition(
        from_status: AttemptLifecycleStatus,
        to_status: AttemptLifecycleStatus,
    ) -> None:
        if to_status not in _ALLOWED_TRANSITIONS.get(from_status, set()):
            raise AttemptStateError(
                f"invalid attempt transition: {from_status.value} -> {to_status.value}"
            )

    @staticmethod
    def _require_owner(current: AttemptStateV2, claim_token: str) -> None:
        if not claim_token or current.claim_token != claim_token:
            raise AttemptClaimConflict("claim token does not own attempt")

    @staticmethod
    def _require_live_lease(current: AttemptStateV2, at: datetime) -> None:
        _iso(at)
        if current.lease_expires_at is None or at >= current.lease_expires_at:
            raise AttemptClaimConflict("attempt lease has expired")

    @staticmethod
    def _require_monotonic_time(current: AttemptStateV2, at: datetime) -> None:
        _iso(at)
        if at < current.updated_at:
            raise AttemptStateError("transition timestamp precedes last state update")

    @staticmethod
    def _state(row: sqlite3.Row) -> AttemptStateV2:
        def timestamp(name: str) -> datetime | None:
            value = row[name]
            return datetime.fromisoformat(value) if value else None

        raw_binding = row["worker_binding_json"]
        binding = WorkerBindingV2.model_validate_json(raw_binding) if raw_binding else None
        raw_spawn_guard = row["spawn_guard_json"]
        spawn_guard = (
            SpawnGuardBindingV1.model_validate_json(raw_spawn_guard) if raw_spawn_guard else None
        )
        return AttemptStateV2(
            attempt_id=row["attempt_id"],
            wave_id=row["wave_id"],
            experiment_id=row["experiment_id"],
            manifest_hash=row["manifest_hash"],
            artifact_root=row["artifact_root"],
            status=row["status"],
            priority=row["priority"],
            retry_number=row["retry_number"],
            worker_binding=binding,
            worker_binding_hash=row["worker_binding_hash"],
            claim_recovery_contract=row["claim_recovery_contract"],
            spawn_guard_binding=spawn_guard,
            spawn_guard_binding_hash=row["spawn_guard_hash"],
            created_at=timestamp("created_at"),
            updated_at=timestamp("updated_at"),
            claimed_by=row["claimed_by"],
            claim_token=row["claim_token"],
            lease_expires_at=timestamp("lease_expires_at"),
            heartbeat_at=timestamp("heartbeat_at"),
            pid=row["pid"],
            process_start_token=row["process_start_token"],
            command_hash=row["command_hash"],
            working_directory=row["working_directory"],
            log_path=row["log_path"],
            log_sha256=row["log_sha256"],
            result_status=row["result_status"],
            result_path=row["result_path"],
            result_sha256=row["result_sha256"],
            error_code=row["error_code"],
            error_detail=row["error_detail"],
            version=row["version"],
        )

    @classmethod
    def _read_state(cls, connection: sqlite3.Connection, attempt_id: str) -> AttemptStateV2:
        row = connection.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise AttemptStateError(f"unknown attempt: {attempt_id}")
        return cls._state(row)
