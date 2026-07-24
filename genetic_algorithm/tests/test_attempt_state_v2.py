"""Tests for transactional attempt lifecycle, claims, leases, and fencing."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from genetic_algorithm.orchestration.artifact_store_v2 import ArtifactIntegrityError
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptClaimConflict,
    AttemptLifecycleStatus,
    AttemptStateError,
    AttemptStateStoreV2,
)
from genetic_algorithm.orchestration.result_contract import AttemptManifestV2


NOW = datetime(2026, 7, 21, 18, 0, tzinfo=UTC)


def _manifest(tmp_path: Path, attempt_id: str) -> AttemptManifestV2:
    root = tmp_path / "attempts" / attempt_id
    return AttemptManifestV2(
        attempt_id=attempt_id,
        wave_id="wave-state-test",
        experiment_id="experiment-state-test",
        created_at=NOW,
        config_hash="c" * 64,
        code_version="commit-state-test",
        data_manifest_hash="d" * 64,
        split_manifest_hash="s" * 64,
        fitness_policy_version="fitness-state-test",
        seeds=[42],
        worker_count=1,
        resolved_config_path=str(root / "resolved_config.yaml"),
        artifact_root=str(root),
    )


def _queue(
    store: AttemptStateStoreV2,
    manifest: AttemptManifestV2,
    *,
    priority: int = 0,
) -> None:
    store.register(manifest, priority=priority)
    store.transition(
        manifest.attempt_id,
        AttemptLifecycleStatus.VALIDATED,
        occurred_at=NOW + timedelta(seconds=1),
        actor="validator",
        reason="MANIFEST_VALID",
    )
    store.transition(
        manifest.attempt_id,
        AttemptLifecycleStatus.QUEUED,
        occurred_at=NOW + timedelta(seconds=2),
        actor="controller",
        reason="QUEUED_FOR_EXECUTION",
    )


def _prepare_execution(
    store: AttemptStateStoreV2,
    manifest: AttemptManifestV2,
    claim_token: str,
) -> None:
    log_path = Path(manifest.artifact_root) / "runtime" / "attempt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"state test log\n")
    store.prepare_execution(
        manifest.attempt_id,
        claim_token=claim_token,
        command_hash="e" * 64,
        working_directory=Path(manifest.artifact_root).parent,
        log_path=log_path,
        prepared_at=NOW + timedelta(milliseconds=3500),
    )


def test_schema_v4_migrates_to_recovery_schema_v6(tmp_path: Path):
    path = tmp_path / "attempts.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA user_version=4")
    finally:
        connection.close()

    AttemptStateStoreV2(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        columns = {row[1] for row in connection.execute("PRAGMA table_info(attempts)")}
    finally:
        connection.close()
    assert "legacy_registry_imports" in tables
    assert "legacy_experiment_versions" in tables
    assert {
        "claim_recovery_contract",
        "spawn_guard_json",
        "spawn_guard_hash",
    }.issubset(columns)


def test_schema_v5_migration_preserves_existing_attempts(tmp_path: Path):
    path = tmp_path / "attempts.sqlite3"
    store = AttemptStateStoreV2(path)
    manifest = _manifest(tmp_path, "pre-v6-attempt")
    before = store.register(manifest)
    connection = sqlite3.connect(path)
    try:
        connection.execute("ALTER TABLE attempts DROP COLUMN spawn_guard_hash")
        connection.execute("ALTER TABLE attempts DROP COLUMN spawn_guard_json")
        connection.execute("ALTER TABLE attempts DROP COLUMN claim_recovery_contract")
        connection.execute("PRAGMA user_version=5")
    finally:
        connection.close()

    migrated = AttemptStateStoreV2(path)

    assert migrated.get(manifest.attempt_id) == before
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
    finally:
        connection.close()


def test_register_is_idempotent_but_attempt_id_inputs_are_immutable(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "attempts.sqlite3")
    manifest = _manifest(tmp_path, "attempt-001")

    first = store.register(manifest, priority=7)
    second = store.register(manifest, priority=7)

    assert first == second
    assert first.status == AttemptLifecycleStatus.DRAFT
    assert first.version == 0
    with pytest.raises(AttemptStateError, match="different inputs"):
        store.register(manifest.model_copy(update={"config_hash": "x" * 64}), priority=7)


def test_state_machine_rejects_skipped_and_generic_runtime_transitions(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "attempts.sqlite3")
    manifest = _manifest(tmp_path, "attempt-001")
    store.register(manifest)

    with pytest.raises(AttemptStateError, match="invalid attempt transition"):
        store.transition(
            manifest.attempt_id,
            AttemptLifecycleStatus.QUEUED,
            occurred_at=NOW + timedelta(seconds=1),
            actor="controller",
            reason="SKIP_VALIDATION",
        )
    with pytest.raises(AttemptStateError, match="specialized methods"):
        store.transition(
            manifest.attempt_id,
            AttemptLifecycleStatus.CLAIMED,
            occurred_at=NOW + timedelta(seconds=1),
            actor="worker",
            reason="UNSAFE_CLAIM",
        )


def test_claim_running_heartbeat_and_process_fencing(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "attempts.sqlite3")
    manifest = _manifest(tmp_path, "attempt-001")
    _queue(store, manifest)
    claimed = store.claim_next(
        worker_id="worker-a",
        claimed_at=NOW + timedelta(seconds=3),
        lease_seconds=30,
    )

    assert claimed is not None
    assert claimed.status == AttemptLifecycleStatus.CLAIMED
    _prepare_execution(store, manifest, str(claimed.claim_token))
    running = store.mark_running(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        pid=1234,
        process_start_token="linux-proc-start-9876",
        started_at=NOW + timedelta(seconds=4),
        lease_seconds=30,
    )
    assert running.status == AttemptLifecycleStatus.RUNNING

    with pytest.raises(AttemptClaimConflict, match="claim token"):
        store.heartbeat(
            manifest.attempt_id,
            claim_token="stale-worker-token",
            process_start_token="linux-proc-start-9876",
            heartbeat_at=NOW + timedelta(seconds=5),
            lease_seconds=30,
        )
    with pytest.raises(AttemptClaimConflict, match="process start token"):
        store.heartbeat(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            process_start_token="pid-reused-with-other-start",
            heartbeat_at=NOW + timedelta(seconds=5),
            lease_seconds=30,
        )

    heartbeat = store.heartbeat(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        process_start_token="linux-proc-start-9876",
        heartbeat_at=NOW + timedelta(seconds=5),
        lease_seconds=30,
    )
    assert heartbeat.version == running.version + 1
    assert heartbeat.lease_expires_at == NOW + timedelta(seconds=35)


def test_direct_runner_claims_exact_attempt_instead_of_queue_head(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "attempts.sqlite3")
    older = _manifest(tmp_path, "attempt-older")
    requested = _manifest(tmp_path, "attempt-requested").model_copy(
        update={"created_at": NOW + timedelta(microseconds=1)}
    )
    _queue(store, older, priority=100)
    store.register(requested, priority=0)
    store.transition(
        requested.attempt_id,
        AttemptLifecycleStatus.VALIDATED,
        occurred_at=NOW + timedelta(seconds=1),
        actor="validator",
        reason="MANIFEST_VALID",
    )
    store.transition(
        requested.attempt_id,
        AttemptLifecycleStatus.QUEUED,
        occurred_at=NOW + timedelta(seconds=2),
        actor="controller",
        reason="QUEUED_FOR_EXECUTION",
    )

    claimed = store.claim_attempt(
        requested.attempt_id,
        worker_id="manual-runner",
        claimed_at=NOW + timedelta(seconds=3),
        lease_seconds=30,
    )

    assert claimed.attempt_id == requested.attempt_id
    assert store.get(older.attempt_id).status == AttemptLifecycleStatus.QUEUED
    with pytest.raises(AttemptClaimConflict, match="not claimable"):
        store.claim_attempt(
            requested.attempt_id,
            worker_id="second-runner",
            claimed_at=NOW + timedelta(seconds=4),
            lease_seconds=30,
        )


def test_expired_lease_is_reported_and_cannot_be_revived_by_stale_worker(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "attempts.sqlite3")
    manifest = _manifest(tmp_path, "attempt-001")
    _queue(store, manifest)
    claimed = store.claim_next(
        worker_id="worker-a",
        claimed_at=NOW + timedelta(seconds=3),
        lease_seconds=5,
    )
    assert claimed is not None

    expired_at = NOW + timedelta(seconds=8)
    assert [item.attempt_id for item in store.expired_leases(as_of=expired_at)] == [
        manifest.attempt_id
    ]
    with pytest.raises(AttemptClaimConflict, match="lease has expired"):
        store.heartbeat(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            process_start_token=None,
            heartbeat_at=expired_at,
            lease_seconds=10,
        )


def test_missing_terminal_artifact_never_becomes_success(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "attempts.sqlite3")
    manifest = _manifest(tmp_path, "attempt-001")
    _queue(store, manifest)
    claimed = store.claim_next(
        worker_id="worker-a",
        claimed_at=NOW + timedelta(seconds=3),
        lease_seconds=60,
    )
    assert claimed is not None
    _prepare_execution(store, manifest, str(claimed.claim_token))
    store.mark_running(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        pid=1234,
        process_start_token="proc-start",
        started_at=NOW + timedelta(seconds=4),
        lease_seconds=60,
    )

    with pytest.raises(ArtifactIntegrityError, match="complete terminal result"):
        store.finalize_from_artifacts(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            process_start_token="proc-start",
            finalized_at=NOW + timedelta(seconds=5),
        )
    assert store.get(manifest.attempt_id).status == AttemptLifecycleStatus.RUNNING


def test_runtime_failure_is_terminal_and_audited(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "attempts.sqlite3")
    manifest = _manifest(tmp_path, "attempt-001")
    _queue(store, manifest)
    claimed = store.claim_next(
        worker_id="worker-a",
        claimed_at=NOW + timedelta(seconds=3),
        lease_seconds=60,
    )
    assert claimed is not None
    failed = store.mark_runtime_failure(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        process_start_token=None,
        status=AttemptLifecycleStatus.FAILED,
        failed_at=NOW + timedelta(seconds=4),
        error_code="SPAWN_FAILED",
        error_detail="worker could not create child process",
    )

    assert failed.status == AttemptLifecycleStatus.FAILED
    assert failed.lease_expires_at is None
    assert [event.to_status for event in store.events(manifest.attempt_id)] == [
        AttemptLifecycleStatus.DRAFT,
        AttemptLifecycleStatus.VALIDATED,
        AttemptLifecycleStatus.QUEUED,
        AttemptLifecycleStatus.CLAIMED,
        AttemptLifecycleStatus.FAILED,
    ]
    connection = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="events are immutable"):
            connection.execute("DELETE FROM attempt_events")
    finally:
        connection.close()


def test_parallel_workers_can_claim_each_attempt_exactly_once(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "attempts.sqlite3")
    for index in range(60):
        _queue(
            store,
            _manifest(tmp_path, f"attempt-{index:02d}"),
            priority=index % 3,
        )

    def claim(index: int) -> str | None:
        state = store.claim_next(
            worker_id=f"worker-{index:02d}",
            claimed_at=NOW + timedelta(seconds=3),
            lease_seconds=60,
        )
        return state.attempt_id if state else None

    with ThreadPoolExecutor(max_workers=24) as pool:
        claimed_ids = list(pool.map(claim, range(72)))

    actual = [attempt_id for attempt_id in claimed_ids if attempt_id is not None]
    assert len(actual) == 60
    assert len(set(actual)) == 60
    assert claimed_ids.count(None) == 12


def test_parallel_heartbeats_do_not_lose_versions_or_events(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "attempts.sqlite3")
    manifest = _manifest(tmp_path, "attempt-001")
    _queue(store, manifest)
    claimed = store.claim_next(
        worker_id="worker-a",
        claimed_at=NOW + timedelta(seconds=3),
        lease_seconds=120,
    )
    assert claimed is not None
    _prepare_execution(store, manifest, str(claimed.claim_token))
    running = store.mark_running(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        pid=1234,
        process_start_token="proc-start",
        started_at=NOW + timedelta(seconds=4),
        lease_seconds=120,
    )
    heartbeat_at = NOW + timedelta(seconds=5)

    def heartbeat(_: int) -> None:
        store.heartbeat(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            process_start_token="proc-start",
            heartbeat_at=heartbeat_at,
            lease_seconds=120,
        )

    with ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(heartbeat, range(60)))

    final = store.get(manifest.attempt_id)
    assert final.version == running.version + 60
    assert len(store.events(manifest.attempt_id)) == final.version + 1
