"""Read-only experiment catalog, legacy import, and JSON export for GA V2.

SQLite remains the sole mutable source of truth.  Imported JSON rows are
immutable historical observations and never become executable attempts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
)
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    AttemptStateStoreV2,
    AttemptStateV2,
    WorkerKind,
)
from genetic_algorithm.orchestration.result_contract import StrictV2Model


class ExperimentCatalogError(ValueError):
    """Raised when catalog input/output cannot be proven safe."""


class ExperimentRecordV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    experiment_id: str = Field(min_length=1)
    source: Literal["CANONICAL_V2", "LEGACY_IMPORT"]
    status: Literal[
        "queued",
        "running",
        "completed",
        "failed",
        "cancelled",
        "unknown",
    ]
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt_ids: list[str] = Field(default_factory=list)
    wave_ids: list[str] = Field(default_factory=list)
    artifact_roots: list[str] = Field(default_factory=list)
    worker_kinds: list[WorkerKind] = Field(default_factory=list)
    pid: int | None = Field(default=None, ge=1)
    log_path: str | None = None
    best_score: float | None = None
    best_net_return: float | None = None
    best_candidate_id: str | None = None
    best_attempt_id: str | None = None
    best_net_return_basis: (
        Literal[
            "WORST_SCENARIO_OF_BEST_SCORE",
            "LEGACY_PERCENT_BEST_PROFIT",
        ]
        | None
    ) = None
    error_code: str | None = None
    error_detail: str | None = None
    tags: list[str] = Field(default_factory=list)
    config_path: str | None = None
    legacy_payload: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _canonical_collections(self) -> ExperimentRecordV2:
        for field_name in (
            "attempt_ids",
            "wave_ids",
            "artifact_roots",
            "worker_kinds",
            "tags",
        ):
            values = getattr(self, field_name)
            if values != sorted(set(values)):
                raise ValueError(f"{field_name} must be sorted and unique")
        if self.source == "CANONICAL_V2":
            if not self.attempt_ids or self.legacy_payload is not None:
                raise ValueError("canonical catalog record lacks attempts or has legacy data")
            if (self.best_candidate_id is None) != (self.best_attempt_id is None):
                raise ValueError("canonical best candidate/attempt provenance is incomplete")
            if self.best_attempt_id is not None and self.best_attempt_id not in self.attempt_ids:
                raise ValueError("best_attempt_id is not part of the experiment")
            if self.best_net_return_basis == "LEGACY_PERCENT_BEST_PROFIT":
                raise ValueError("canonical record has a legacy return basis")
        elif self.legacy_payload is None or self.attempt_ids:
            raise ValueError("legacy catalog record has invalid provenance")
        elif self.best_candidate_id is not None or self.best_attempt_id is not None:
            raise ValueError("legacy record cannot claim canonical candidate provenance")
        elif self.best_net_return_basis == "WORST_SCENARIO_OF_BEST_SCORE":
            raise ValueError("legacy record has a canonical return basis")
        return self

    def compatibility_dict(self) -> dict[str, Any]:
        """Return the old CLI/monitor shape without hiding record provenance."""

        payload = self.model_dump(mode="json")
        payload.update(
            {
                "best_fitness": self.best_score,
                "best_profit": (
                    self.best_net_return * 100.0 if self.best_net_return is not None else None
                ),
                "generation": None,
                "generations_total": None,
                "ga_type": (
                    "generic_island"
                    if WorkerKind.GENERIC_ISLAND_EVOLUTION in self.worker_kinds
                    else "standard"
                    if WorkerKind.STANDARD_EVOLUTION in self.worker_kinds
                    else "replay"
                    if self.worker_kinds
                    else str((self.legacy_payload or {}).get("ga_type", "legacy"))
                ),
                "data_dir": self.artifact_roots[-1] if self.artifact_roots else None,
            }
        )
        return payload


class LegacyRegistryImportV2(StrictV2Model):
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    import_id: int = Field(ge=1)
    imported_at: datetime
    experiment_count: int = Field(ge=0)
    already_imported: bool = False


class CatalogExportV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    exported_at: datetime
    state_schema_version: int = Field(ge=1)
    attempts: list[AttemptStateV2]
    experiments: list[ExperimentRecordV2]
    legacy_imports: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _canonical_order(self) -> CatalogExportV2:
        if self.exported_at.utcoffset() is None:
            raise ValueError("exported_at must be timezone-aware")
        if self.attempts != sorted(self.attempts, key=lambda item: item.attempt_id):
            raise ValueError("attempt export is not sorted")
        if self.experiments != sorted(
            self.experiments, key=lambda item: (item.source, item.experiment_id)
        ):
            raise ValueError("experiment export is not sorted")
        return self


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ExperimentCatalogError(f"legacy registry contains non-finite JSON value: {value}")


def _aware_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _legacy_status(value: Any) -> str:
    return (
        str(value)
        if value in {"queued", "running", "completed", "failed", "cancelled"}
        else "unknown"
    )


def _legacy_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _legacy_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _legacy_pid(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _legacy_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({item for item in value if isinstance(item, str) and item})


def _legacy_record(
    experiment_id: str,
    payload: Mapping[str, Any],
    imported_at: str | datetime,
) -> ExperimentRecordV2:
    best_profit = _legacy_number(payload.get("best_profit"))
    imported = imported_at.isoformat() if isinstance(imported_at, datetime) else imported_at
    return ExperimentRecordV2(
        experiment_id=experiment_id,
        source="LEGACY_IMPORT",
        status=_legacy_status(payload.get("status")),
        created_at=_aware_datetime(payload.get("created_at")),
        updated_at=_aware_datetime(payload.get("updated_at") or imported),
        started_at=_aware_datetime(payload.get("started_at")),
        finished_at=_aware_datetime(payload.get("finished_at")),
        pid=_legacy_pid(payload.get("pid")),
        log_path=_legacy_string(payload.get("log_path")),
        best_score=_legacy_number(payload.get("best_fitness")),
        best_net_return=best_profit / 100.0 if best_profit is not None else None,
        best_net_return_basis=("LEGACY_PERCENT_BEST_PROFIT" if best_profit is not None else None),
        error_code="LEGACY_IMPORT_ONLY" if payload.get("error") else None,
        error_detail=(str(payload.get("error"))[:2000] if payload.get("error") else None),
        tags=_legacy_tags(payload.get("tags")),
        config_path=_legacy_string(payload.get("config_path")),
        legacy_payload=dict(payload),
    )


def _select_catalog_winner(
    observations: Sequence[tuple[str, str, float | None, list[float]]],
) -> tuple[str, str, float | None, float | None] | None:
    """Select one display winner without combining metrics across candidates.

    Robust score is the primary diagnostic rank.  The displayed return is the
    worst scenario return of that exact candidate, never a maximum cherry-picked
    from another pair, period, cost scenario, candidate, or attempt.
    """

    normalized = [
        (
            attempt_id,
            candidate_id,
            robust_score,
            min(net_returns) if net_returns else None,
        )
        for attempt_id, candidate_id, robust_score, net_returns in observations
        if robust_score is not None or net_returns
    ]
    if not normalized:
        return None
    return max(
        normalized,
        key=lambda item: (
            item[2] is not None,
            item[2] if item[2] is not None else float("-inf"),
            item[3] is not None,
            item[3] if item[3] is not None else float("-inf"),
            item[1],
            item[0],
        ),
    )


class ExperimentCatalogV2:
    """Query canonical attempts and immutable legacy history from one database."""

    def __init__(self, state_store: AttemptStateStoreV2) -> None:
        self.state_store = state_store

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.state_store.path,
            timeout=self.state_store.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self.state_store.busy_timeout_ms}")
        return connection

    def import_legacy_registry(
        self,
        source_path: str | Path,
        *,
        imported_at: datetime | None = None,
    ) -> LegacyRegistryImportV2:
        """Import one complete JSON snapshot idempotently and immutably."""

        source = Path(source_path).resolve()
        if not source.is_file():
            raise ExperimentCatalogError(f"legacy registry not found: {source}")
        raw = source.read_bytes()
        digest = _sha256(raw)
        try:
            envelope = json.loads(raw, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise ExperimentCatalogError("legacy registry is not valid JSON") from exc
        if (
            not isinstance(envelope, Mapping)
            or envelope.get("version") != 1
            or not isinstance(envelope.get("experiments"), Mapping)
        ):
            raise ExperimentCatalogError("legacy registry envelope is unsupported")
        experiments = dict(envelope["experiments"])
        canonical_entries: dict[str, dict[str, Any]] = {}
        for experiment_id, payload in experiments.items():
            if not isinstance(experiment_id, str) or not experiment_id:
                raise ExperimentCatalogError("legacy experiment ID is invalid")
            if not isinstance(payload, Mapping):
                raise ExperimentCatalogError(
                    f"legacy experiment payload is invalid: {experiment_id}"
                )
            entry = dict(payload)
            embedded_id = entry.get("experiment_id", experiment_id)
            if embedded_id != experiment_id:
                raise ExperimentCatalogError(
                    f"legacy experiment key/payload ID differs: {experiment_id}"
                )
            entry["experiment_id"] = experiment_id
            canonical_entries[experiment_id] = entry
        timestamp = imported_at or datetime.now(UTC)
        if timestamp.utcoffset() is None:
            raise ExperimentCatalogError("imported_at must be timezone-aware")
        # Materialize every row before opening the write transaction.  A
        # malformed historical row must reject the complete snapshot instead
        # of leaving a partially queryable import behind.
        for experiment_id, payload in canonical_entries.items():
            _legacy_record(experiment_id, payload, timestamp)
        envelope_json = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT import_id, imported_at, experiment_count
                FROM legacy_registry_imports
                WHERE source_sha256 = ?
                """,
                (digest,),
            ).fetchone()
            if existing is not None:
                connection.execute("COMMIT")
                return LegacyRegistryImportV2(
                    source_path=str(source),
                    source_sha256=digest,
                    import_id=int(existing["import_id"]),
                    imported_at=datetime.fromisoformat(existing["imported_at"]),
                    experiment_count=int(existing["experiment_count"]),
                    already_imported=True,
                )
            cursor = connection.execute(
                """
                INSERT INTO legacy_registry_imports (
                    source_sha256, source_path, imported_at,
                    experiment_count, envelope_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    digest,
                    str(source),
                    timestamp.astimezone(UTC).isoformat(),
                    len(canonical_entries),
                    envelope_json,
                ),
            )
            import_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO legacy_experiment_versions (
                    import_id, experiment_id, payload_json
                ) VALUES (?, ?, ?)
                """,
                [
                    (
                        import_id,
                        experiment_id,
                        json.dumps(
                            payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    )
                    for experiment_id, payload in sorted(canonical_entries.items())
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return LegacyRegistryImportV2(
            source_path=str(source),
            source_sha256=digest,
            import_id=import_id,
            imported_at=timestamp,
            experiment_count=len(canonical_entries),
        )

    def list_records(
        self,
        *,
        statuses: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        include_legacy: bool = True,
        limit: int | None = None,
    ) -> list[ExperimentRecordV2]:
        status_filter = set(statuses or [])
        tag_filter = set(tags or [])
        records = self._canonical_records()
        if include_legacy:
            canonical_ids = {record.experiment_id for record in records}
            records.extend(
                record
                for record in self._legacy_records()
                if record.experiment_id not in canonical_ids
            )
        if status_filter:
            records = [record for record in records if record.status in status_filter]
        if tag_filter:
            records = [record for record in records if tag_filter.issubset(set(record.tags))]
        records.sort(
            key=lambda item: (
                item.updated_at or item.created_at or datetime.min.replace(tzinfo=UTC),
                item.experiment_id,
            ),
            reverse=True,
        )
        return records[:limit] if limit is not None else records

    def get_record(
        self,
        identifier: str,
        *,
        include_legacy: bool = True,
    ) -> ExperimentRecordV2 | None:
        for record in self.list_records(include_legacy=include_legacy):
            if record.experiment_id == identifier or identifier in record.attempt_ids:
                return record
        return None

    def summary(self, *, include_legacy: bool = True) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.list_records(include_legacy=include_legacy):
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def export_json(
        self,
        destination: str | Path,
        *,
        exported_at: datetime | None = None,
        include_legacy: bool = True,
    ) -> CatalogExportV2:
        timestamp = exported_at or datetime.now(UTC)
        attempts = sorted(self.state_store.list_attempts(), key=lambda item: item.attempt_id)
        experiments = sorted(
            self.list_records(include_legacy=include_legacy),
            key=lambda item: (item.source, item.experiment_id),
        )
        connection = self._connect()
        try:
            imports = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT import_id, source_sha256, source_path,
                           imported_at, experiment_count
                    FROM legacy_registry_imports
                    ORDER BY import_id
                    """
                ).fetchall()
            ]
        finally:
            connection.close()
        export = CatalogExportV2(
            exported_at=timestamp,
            state_schema_version=AttemptStateStoreV2.SCHEMA_VERSION,
            attempts=attempts,
            experiments=experiments,
            legacy_imports=imports if include_legacy else [],
        )
        path = Path(destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (export.model_dump_json(indent=2) + "\n").encode()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return export

    def _canonical_records(self) -> list[ExperimentRecordV2]:
        grouped: dict[str, list[AttemptStateV2]] = {}
        for attempt in self.state_store.list_attempts():
            grouped.setdefault(attempt.experiment_id, []).append(attempt)
        return [
            self._canonical_record(experiment_id, attempts)
            for experiment_id, attempts in sorted(grouped.items())
        ]

    def _canonical_record(
        self,
        experiment_id: str,
        attempts: list[AttemptStateV2],
    ) -> ExperimentRecordV2:
        attempts = sorted(attempts, key=lambda item: (item.created_at, item.attempt_id))
        active = [
            item
            for item in attempts
            if item.status in {AttemptLifecycleStatus.CLAIMED, AttemptLifecycleStatus.RUNNING}
        ]
        pending = [
            item
            for item in attempts
            if item.status
            in {
                AttemptLifecycleStatus.DRAFT,
                AttemptLifecycleStatus.VALIDATED,
                AttemptLifecycleStatus.QUEUED,
            }
        ]
        if active:
            status = "running"
        elif pending:
            status = "queued"
        elif all(item.status == AttemptLifecycleStatus.SUCCEEDED for item in attempts):
            status = "completed"
        else:
            status = "failed"
        observations: list[tuple[str, str, float | None, list[float]]] = []
        integrity_error: str | None = None
        for item in attempts:
            if item.status != AttemptLifecycleStatus.SUCCEEDED:
                continue
            try:
                result = V2ArtifactStore(item.artifact_root).read_verified_result()
            except ArtifactIntegrityError as exc:
                integrity_error = f"{type(exc).__name__}: {exc}"[:2000]
                status = "failed"
                continue
            for evaluation in result.candidate_evaluations:
                observations.append(
                    (
                        item.attempt_id,
                        evaluation.candidate_id,
                        evaluation.robust_score,
                        [
                            scenario.metrics.net_return
                            for scenario in evaluation.scenarios
                            if scenario.metrics.net_return is not None
                        ],
                    )
                )
        winner = _select_catalog_winner(observations)
        lifecycle_owner = max(
            active or attempts,
            key=lambda item: (item.updated_at, item.attempt_id),
        )
        connection = self._connect()
        try:
            started_raw = connection.execute(
                """
                SELECT MIN(occurred_at)
                FROM attempt_events
                WHERE attempt_id IN (
                    SELECT attempt_id FROM attempts WHERE experiment_id = ?
                ) AND to_status = 'RUNNING'
                """,
                (experiment_id,),
            ).fetchone()[0]
        finally:
            connection.close()
        terminal = {
            AttemptLifecycleStatus.SUCCEEDED,
            AttemptLifecycleStatus.FAILED,
            AttemptLifecycleStatus.INTERRUPTED,
            AttemptLifecycleStatus.INVALID_RESULT,
        }
        finished = (
            max(item.updated_at for item in attempts)
            if all(item.status in terminal for item in attempts)
            else None
        )
        bindings = [
            item.worker_binding.worker_kind for item in attempts if item.worker_binding is not None
        ]
        errors = [item for item in reversed(attempts) if item.error_code]
        return ExperimentRecordV2(
            experiment_id=experiment_id,
            source="CANONICAL_V2",
            status=status,
            created_at=min(item.created_at for item in attempts),
            updated_at=max(item.updated_at for item in attempts),
            started_at=datetime.fromisoformat(started_raw) if started_raw else None,
            finished_at=finished,
            attempt_ids=sorted(item.attempt_id for item in attempts),
            wave_ids=sorted({item.wave_id for item in attempts}),
            artifact_roots=sorted({item.artifact_root for item in attempts}),
            worker_kinds=sorted(set(bindings)),
            pid=lifecycle_owner.pid,
            log_path=lifecycle_owner.log_path,
            best_score=winner[2] if winner else None,
            best_net_return=winner[3] if winner else None,
            best_attempt_id=winner[0] if winner else None,
            best_candidate_id=winner[1] if winner else None,
            best_net_return_basis=(
                "WORST_SCENARIO_OF_BEST_SCORE"
                if winner is not None and winner[3] is not None
                else None
            ),
            error_code=(
                "ARTIFACT_INTEGRITY_FAILED"
                if integrity_error
                else errors[0].error_code
                if errors
                else None
            ),
            error_detail=integrity_error or (errors[0].error_detail if errors else None),
            config_path=self._manifest_config_path(lifecycle_owner.attempt_id),
        )

    def _manifest_config_path(self, attempt_id: str) -> str:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT manifest_json FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ExperimentCatalogError(f"attempt disappeared during query: {attempt_id}")
        manifest = json.loads(row["manifest_json"])
        return str(manifest["resolved_config_path"])

    def _legacy_records(self) -> list[ExperimentRecordV2]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT versions.experiment_id, versions.payload_json,
                       imports.imported_at
                FROM legacy_experiment_versions AS versions
                JOIN legacy_registry_imports AS imports
                  ON imports.import_id = versions.import_id
                WHERE versions.import_id = (
                    SELECT MAX(newer.import_id)
                    FROM legacy_experiment_versions AS newer
                    WHERE newer.experiment_id = versions.experiment_id
                )
                ORDER BY versions.experiment_id
                """
            ).fetchall()
        finally:
            connection.close()
        records: list[ExperimentRecordV2] = []
        for row in rows:
            payload = json.loads(
                row["payload_json"],
                parse_constant=_reject_json_constant,
            )
            records.append(
                _legacy_record(
                    row["experiment_id"],
                    payload,
                    row["imported_at"],
                )
            )
        return records
