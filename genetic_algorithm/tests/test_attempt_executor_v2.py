"""Process, heartbeat, terminal-readback, and restart tests for the V2 executor."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from genetic_algorithm.orchestration.artifact_store_v2 import (
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.attempt_executor_v2 import (
    AttemptCommandV2,
    AttemptExecutorV2,
    AttemptReconcilerV2,
    LinuxProcessInspector,
    ProcessProbeStatus,
    ProcessProbeV2,
    ReconciliationAction,
)
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    AttemptStateStoreV2,
    SpawnGuardBindingV1,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptResultV2,
    AttemptStatus,
)


class _ProbeInspector:
    def __init__(self, status: ProcessProbeStatus) -> None:
        self.status = status

    def identity_token(self, pid: int) -> str | None:
        return f"stub:{pid}"

    def probe(self, pid: int, expected_start_token: str) -> ProcessProbeV2:
        return ProcessProbeV2(status=self.status)


def _registered_queue(
    tmp_path: Path,
    *,
    attempt_id: str = "attempt-executor-test",
    terminal_result: AttemptStatus | None = None,
) -> tuple[AttemptStateStoreV2, AttemptManifestV2]:
    created_at = datetime.now(UTC) - timedelta(minutes=1)
    artifact_root = tmp_path / "attempts" / attempt_id
    config: dict[str, object] = {"test": "executor-v2"}
    manifest = AttemptManifestV2(
        attempt_id=attempt_id,
        wave_id="wave-executor-test",
        experiment_id="experiment-executor-test",
        created_at=created_at,
        config_hash=canonical_config_hash(config),
        code_version="executor-test-commit",
        data_manifest_hash="d" * 64,
        split_manifest_hash="s" * 64,
        fitness_policy_version="executor-test-policy",
        seeds=[41],
        worker_count=1,
        resolved_config_path=str(artifact_root / V2ArtifactStore.CONFIG_NAME),
        artifact_root=str(artifact_root),
    )
    artifacts = V2ArtifactStore(artifact_root)
    artifacts.write_manifest(manifest, config)
    if terminal_result is not None:
        artifacts.finalize(
            AttemptResultV2(
                attempt_id=attempt_id,
                status=terminal_result,
                started_at=created_at + timedelta(seconds=3),
                finished_at=created_at + timedelta(seconds=4),
                manifest=manifest,
                error_code="CHILD_REPORTED_FAILURE",
                error_detail="synthetic terminal result",
            )
        )
    state_store = AttemptStateStoreV2(tmp_path / "attempt-state.sqlite3")
    state_store.register(manifest)
    state_store.transition(
        attempt_id,
        AttemptLifecycleStatus.VALIDATED,
        occurred_at=created_at + timedelta(seconds=1),
        actor="validator",
        reason="VALIDATED",
    )
    state_store.transition(
        attempt_id,
        AttemptLifecycleStatus.QUEUED,
        occurred_at=created_at + timedelta(seconds=2),
        actor="controller",
        reason="QUEUED",
    )
    return state_store, manifest


def _python_command(tmp_path: Path, source: str) -> AttemptCommandV2:
    return AttemptCommandV2(
        argv=[sys.executable, "-c", source],
        working_directory=str(tmp_path),
    )


def test_nonzero_child_is_failed_and_runtime_log_is_hashed(tmp_path: Path):
    state_store, manifest = _registered_queue(tmp_path)
    executor = AttemptExecutorV2(
        state_store,
        worker_id="worker-a",
        lease_seconds=2,
        heartbeat_interval_seconds=0.05,
    )

    terminal = executor.claim_and_execute_next(
        lambda _: _python_command(
            tmp_path,
            "print('child failed deliberately', flush=True); raise SystemExit(7)",
        )
    )

    assert terminal is not None
    assert terminal.status == AttemptLifecycleStatus.FAILED
    assert terminal.error_code == "PROCESS_EXIT_NONZERO"
    assert terminal.command_hash is not None
    assert terminal.log_path == str(
        (Path(manifest.artifact_root) / "runtime" / "attempt.log").resolve()
    )
    log_payload = Path(str(terminal.log_path)).read_bytes()
    assert b"child failed deliberately" in log_payload
    assert terminal.log_sha256 == hashlib.sha256(log_payload).hexdigest()
    assert terminal.spawn_guard_binding is not None
    assert Path(terminal.spawn_guard_binding.ready_path).is_file()
    assert Path(terminal.spawn_guard_binding.release_path).is_file()


def test_command_build_failure_does_not_leave_an_unowned_claim(tmp_path: Path):
    state_store, _ = _registered_queue(tmp_path)
    executor = AttemptExecutorV2(
        state_store,
        worker_id="worker-a",
        lease_seconds=2,
        heartbeat_interval_seconds=0.05,
    )

    def fail_to_build(_: object) -> AttemptCommandV2:
        raise ValueError("resolved command is unavailable")

    terminal = executor.claim_and_execute_next(fail_to_build)

    assert terminal is not None
    assert terminal.status == AttemptLifecycleStatus.FAILED
    assert terminal.error_code == "COMMAND_BUILD_FAILED"
    assert terminal.lease_expires_at is None


def test_exit_zero_without_verified_result_is_invalid_result(tmp_path: Path):
    state_store, _ = _registered_queue(tmp_path)
    executor = AttemptExecutorV2(
        state_store,
        worker_id="worker-a",
        lease_seconds=2,
        heartbeat_interval_seconds=0.05,
    )

    terminal = executor.claim_and_execute_next(
        lambda _: _python_command(tmp_path, "print('no result produced')")
    )

    assert terminal is not None
    assert terminal.status == AttemptLifecycleStatus.INVALID_RESULT
    assert terminal.error_code == "INVALID_TERMINAL_ARTIFACTS"
    assert terminal.result_sha256 is None


def test_exit_zero_uses_verified_terminal_result_not_log_text(tmp_path: Path):
    state_store, _ = _registered_queue(tmp_path, terminal_result=AttemptStatus.FAILED)
    executor = AttemptExecutorV2(
        state_store,
        worker_id="worker-a",
        lease_seconds=2,
        heartbeat_interval_seconds=0.05,
    )

    terminal = executor.claim_and_execute_next(
        lambda _: _python_command(
            tmp_path,
            "print('EVOLUTION COMPLETE and everything looks successful')",
        )
    )

    assert terminal is not None
    assert terminal.status == AttemptLifecycleStatus.FAILED
    assert terminal.result_status == AttemptStatus.FAILED
    assert terminal.result_sha256 is not None
    assert state_store.events(terminal.attempt_id)[-1].reason == "VERIFIED_TERMINAL_RESULT"


def test_longer_child_receives_periodic_heartbeats(tmp_path: Path):
    state_store, _ = _registered_queue(tmp_path)
    executor = AttemptExecutorV2(
        state_store,
        worker_id="worker-a",
        lease_seconds=2,
        heartbeat_interval_seconds=0.03,
    )

    terminal = executor.claim_and_execute_next(
        lambda _: _python_command(tmp_path, "import time; time.sleep(0.16)")
    )

    assert terminal is not None
    assert terminal.status == AttemptLifecycleStatus.INVALID_RESULT
    reasons = [event.reason for event in state_store.events(terminal.attempt_id)]
    assert reasons.count("HEARTBEAT") >= 3


def test_execution_wallclock_timeout_terminates_process_group(tmp_path: Path):
    state_store, _ = _registered_queue(tmp_path)
    executor = AttemptExecutorV2(
        state_store,
        worker_id="worker-timeout",
        lease_seconds=2,
        heartbeat_interval_seconds=0.03,
        max_execution_seconds=0.12,
    )
    started = time.monotonic()

    terminal = executor.claim_and_execute_next(
        lambda _: _python_command(tmp_path, "import time; time.sleep(60)")
    )

    assert time.monotonic() - started < 2
    assert terminal is not None
    assert terminal.status == AttemptLifecycleStatus.FAILED
    assert terminal.error_code == "PROCESS_EXECUTION_TIMEOUT"
    assert terminal.lease_expires_at is None
    assert terminal.log_sha256 is not None


def test_reconciler_closes_canonical_claim_that_expired_before_spawn(tmp_path: Path):
    state_store, _ = _registered_queue(tmp_path)
    claimed_at = datetime.now(UTC)
    claimed = state_store.claim_next(
        worker_id="worker-a",
        claimed_at=claimed_at,
        lease_seconds=1,
    )
    assert claimed is not None

    records = AttemptReconcilerV2(
        state_store,
        process_inspector=_ProbeInspector(ProcessProbeStatus.NOT_FOUND),
    ).reconcile_expired(as_of=claimed_at + timedelta(seconds=2))

    assert records[0].action == ReconciliationAction.INTERRUPTED_CLAIM_NO_SPAWN
    terminal = state_store.get(claimed.attempt_id)
    assert terminal.status == AttemptLifecycleStatus.INTERRUPTED
    assert terminal.error_code == "RECOVERY_EXPIRED_BEFORE_SPAWN"


def test_reconciler_keeps_migrated_claim_without_recovery_contract(tmp_path: Path):
    state_store, _ = _registered_queue(tmp_path)
    claimed_at = datetime.now(UTC)
    claimed = state_store.claim_next(
        worker_id="legacy-worker",
        claimed_at=claimed_at,
        lease_seconds=1,
    )
    assert claimed is not None
    connection = state_store._connect()
    try:
        connection.execute(
            "UPDATE attempts SET claim_recovery_contract = NULL WHERE attempt_id = ?",
            (claimed.attempt_id,),
        )
    finally:
        connection.close()

    records = AttemptReconcilerV2(
        state_store,
        process_inspector=_ProbeInspector(ProcessProbeStatus.NOT_FOUND),
    ).reconcile_expired(as_of=claimed_at + timedelta(seconds=2))

    assert records[0].action == ReconciliationAction.NO_ACTION_CLAIM_UNPROVEN
    assert state_store.get(claimed.attempt_id).status == AttemptLifecycleStatus.CLAIMED


@pytest.mark.parametrize(
    ("probe_status", "expected_action", "expected_lifecycle"),
    [
        (
            ProcessProbeStatus.ALIVE_MATCH,
            ReconciliationAction.NO_ACTION_GUARD_ALIVE,
            AttemptLifecycleStatus.CLAIMED,
        ),
        (
            ProcessProbeStatus.NOT_FOUND,
            ReconciliationAction.INTERRUPTED_GUARD_GONE,
            AttemptLifecycleStatus.INTERRUPTED,
        ),
        (
            ProcessProbeStatus.PID_REUSED,
            ReconciliationAction.INTERRUPTED_GUARD_GONE,
            AttemptLifecycleStatus.INTERRUPTED,
        ),
    ],
)
def test_claimed_guard_reconciliation_is_fenced_by_ready_process_identity(
    tmp_path: Path,
    probe_status: ProcessProbeStatus,
    expected_action: ReconciliationAction,
    expected_lifecycle: AttemptLifecycleStatus,
):
    state_store, manifest = _registered_queue(tmp_path)
    claimed_at = datetime.now(UTC)
    claimed = state_store.claim_next(
        worker_id="worker-before-running",
        claimed_at=claimed_at,
        lease_seconds=1,
    )
    assert claimed is not None
    runtime = Path(manifest.artifact_root) / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    nonce = "guard-nonce"
    binding = SpawnGuardBindingV1(
        ready_path=str((runtime / "spawn.ready.json").resolve()),
        release_path=str((runtime / "spawn.release.json").resolve()),
        nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest(),
    )
    state_store.prepare_execution(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        command_hash="e" * 64,
        working_directory=tmp_path,
        log_path=runtime / "attempt.log",
        prepared_at=claimed_at + timedelta(milliseconds=100),
        spawn_guard_binding=binding,
    )
    Path(binding.ready_path).write_text(
        json.dumps(
            {
                "nonce": nonce,
                "pid": 424242,
                "process_start_token": "guard-process-token",
                "protocol": "PARENT_DEATH_GUARD_V1",
            }
        )
    )

    records = AttemptReconcilerV2(
        state_store,
        process_inspector=_ProbeInspector(probe_status),
    ).reconcile_expired(as_of=claimed_at + timedelta(seconds=2))

    assert records[0].action == expected_action
    assert state_store.get(manifest.attempt_id).status == expected_lifecycle


def test_controller_kill_before_running_never_starts_target_and_is_recoverable(
    tmp_path: Path,
):
    state_store, manifest = _registered_queue(tmp_path)
    controller_ready = tmp_path / "controller-observed-guard-ready"
    target_marker = tmp_path / "target-started"
    controller_source = """
import sys
import time
from pathlib import Path
from datetime import UTC, datetime
from genetic_algorithm.orchestration.attempt_executor_v2 import AttemptCommandV2, AttemptExecutorV2
from genetic_algorithm.orchestration.attempt_state_v2 import AttemptStateStoreV2

class PausingExecutor(AttemptExecutorV2):
    def _await_spawn_guard_ready(self, process, *, spawn_guard, nonce):
        token = super()._await_spawn_guard_ready(
            process,
            spawn_guard=spawn_guard,
            nonce=nonce,
        )
        Path(sys.argv[3]).write_text("ready")
        time.sleep(30)
        return token

store = AttemptStateStoreV2(sys.argv[1])
claimed = store.claim_next(
    worker_id="crash-window-controller",
    claimed_at=datetime.now(UTC),
    lease_seconds=2,
)
command = AttemptCommandV2(
    argv=[
        sys.executable,
        "-c",
        "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('started')",
        sys.argv[4],
    ],
    working_directory=sys.argv[2],
)
PausingExecutor(
    store,
    worker_id="crash-window-controller",
    lease_seconds=2,
    heartbeat_interval_seconds=0.1,
).execute_attempt(claimed, command)
"""
    controller = subprocess.Popen(
        [
            sys.executable,
            "-c",
            controller_source,
            str(state_store.path),
            str(tmp_path),
            str(controller_ready),
            str(target_marker),
        ],
        cwd=Path.cwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 8
    while not controller_ready.exists() and time.monotonic() < deadline:
        if controller.poll() is not None:
            pytest.fail(f"controller exited before crash point: {controller.returncode}")
        time.sleep(0.02)
    if not controller_ready.exists():
        controller.kill()
        controller.wait(timeout=5)
        pytest.fail("controller did not reach guarded-spawn crash point")

    claimed = state_store.get(manifest.attempt_id)
    assert claimed.status == AttemptLifecycleStatus.CLAIMED
    assert claimed.spawn_guard_binding is not None
    ready = json.loads(Path(claimed.spawn_guard_binding.ready_path).read_text())
    guard_pid = int(ready["pid"])
    guard_token = str(ready["process_start_token"])

    controller.kill()
    controller.wait(timeout=5)
    inspector = LinuxProcessInspector()
    deadline = time.monotonic() + 5
    while (
        inspector.probe(guard_pid, guard_token).status == ProcessProbeStatus.ALIVE_MATCH
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)

    assert inspector.probe(guard_pid, guard_token).status in {
        ProcessProbeStatus.NOT_FOUND,
        ProcessProbeStatus.PID_REUSED,
    }
    assert not target_marker.exists()
    recovered = AttemptReconcilerV2(
        state_store,
        process_inspector=inspector,
    ).reconcile_expired(as_of=claimed.lease_expires_at + timedelta(seconds=1))

    assert recovered[0].action == ReconciliationAction.INTERRUPTED_GUARD_GONE
    terminal = state_store.get(manifest.attempt_id)
    assert terminal.status == AttemptLifecycleStatus.INTERRUPTED
    assert terminal.error_code == "RECOVERY_GUARD_GONE_BEFORE_RELEASE"


def test_restart_reconciliation_keeps_matching_process_then_closes_it_after_kill(
    tmp_path: Path,
):
    state_store, manifest = _registered_queue(tmp_path)
    inspector = LinuxProcessInspector()
    claimed_at = datetime.now(UTC)
    claimed = state_store.claim_next(
        worker_id="worker-before-restart",
        claimed_at=claimed_at,
        lease_seconds=5,
    )
    assert claimed is not None
    log_path = Path(manifest.artifact_root) / "runtime" / "attempt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state_store.prepare_execution(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        command_hash="e" * 64,
        working_directory=tmp_path,
        log_path=log_path,
        prepared_at=claimed_at + timedelta(milliseconds=100),
    )
    with log_path.open("xb", buffering=0) as log_handle:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        token = inspector.identity_token(process.pid)
        if token is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            pytest.skip("Linux process start token is unavailable")
        running = state_store.mark_running(
            manifest.attempt_id,
            claim_token=str(claimed.claim_token),
            pid=process.pid,
            process_start_token=token,
            started_at=claimed_at + timedelta(milliseconds=200),
            lease_seconds=1,
        )
        as_of = claimed_at + timedelta(seconds=2)
        reconciler = AttemptReconcilerV2(state_store, process_inspector=inspector)

        alive = reconciler.reconcile_expired(as_of=as_of)
        assert alive[0].action == ReconciliationAction.NO_ACTION_PROCESS_ALIVE
        assert state_store.get(manifest.attempt_id).version == running.version

        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)

    recovered = reconciler.reconcile_expired(as_of=as_of)

    assert recovered[0].action == ReconciliationAction.INTERRUPTED_PROCESS_GONE
    terminal = state_store.get(manifest.attempt_id)
    assert terminal.status == AttemptLifecycleStatus.INTERRUPTED
    assert terminal.error_code == "RECOVERY_PROCESS_GONE_NO_RESULT"


def test_restart_reconciliation_accepts_verified_result_only_after_process_is_gone(
    tmp_path: Path,
):
    state_store, manifest = _registered_queue(
        tmp_path,
        terminal_result=AttemptStatus.FAILED,
    )
    claimed_at = datetime.now(UTC)
    claimed = state_store.claim_next(
        worker_id="worker-before-restart",
        claimed_at=claimed_at,
        lease_seconds=5,
    )
    assert claimed is not None
    log_path = Path(manifest.artifact_root) / "runtime" / "attempt.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"recovered child log\n")
    state_store.prepare_execution(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        command_hash="e" * 64,
        working_directory=tmp_path,
        log_path=log_path,
        prepared_at=claimed_at + timedelta(milliseconds=100),
    )
    state_store.mark_running(
        manifest.attempt_id,
        claim_token=str(claimed.claim_token),
        pid=424242,
        process_start_token="missing-process-token",
        started_at=claimed_at + timedelta(milliseconds=200),
        lease_seconds=1,
    )

    recovered = AttemptReconcilerV2(
        state_store,
        process_inspector=_ProbeInspector(ProcessProbeStatus.NOT_FOUND),
    ).reconcile_expired(as_of=claimed_at + timedelta(seconds=2))

    assert recovered[0].action == ReconciliationAction.FINALIZED_FROM_ARTIFACTS
    terminal = state_store.get(manifest.attempt_id)
    assert terminal.status == AttemptLifecycleStatus.FAILED
    assert terminal.result_sha256 is not None
    assert state_store.events(manifest.attempt_id)[-1].reason == (
        "RECOVERED_VERIFIED_TERMINAL_RESULT"
    )
