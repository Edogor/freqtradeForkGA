"""Canonical subprocess execution and conservative recovery for GA V2 attempts."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
)
from genetic_algorithm.orchestration.attempt_state_v2 import (
    CLAIM_RECOVERY_CONTRACT_V1,
    AttemptClaimConflict,
    AttemptLifecycleStatus,
    AttemptStateError,
    AttemptStateStoreV2,
    AttemptStateV2,
    SpawnGuardBindingV1,
    execution_command_hash,
)
from genetic_algorithm.orchestration.result_contract import StrictV2Model


class AttemptExecutionError(RuntimeError):
    """Raised when a child cannot be executed under the V2 contract."""


class AttemptExecutionTimeout(AttemptExecutionError):
    """Raised when a child exceeds its controller-owned wallclock budget."""


class AttemptCommandV2(StrictV2Model):
    """Exact argv and working directory of one already-resolved attempt."""

    argv: list[str] = Field(min_length=1)
    working_directory: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_command(self) -> AttemptCommandV2:
        if any(not item or "\x00" in item for item in self.argv):
            raise ValueError("command argv contains an empty or NUL-bearing item")
        return self

    @property
    def resolved_working_directory(self) -> Path:
        return Path(self.working_directory).resolve()

    @property
    def command_hash(self) -> str:
        return execution_command_hash(self.argv, self.resolved_working_directory)


class ProcessProbeStatus(StrEnum):
    ALIVE_MATCH = "ALIVE_MATCH"
    NOT_FOUND = "NOT_FOUND"
    PID_REUSED = "PID_REUSED"
    UNKNOWN = "UNKNOWN"


class ProcessProbeV2(StrictV2Model):
    status: ProcessProbeStatus
    observed_start_token: str | None = None
    detail: str | None = None


class ProcessInspector(Protocol):
    def identity_token(self, pid: int) -> str | None: ...

    def probe(self, pid: int, expected_start_token: str) -> ProcessProbeV2: ...


class LinuxProcessInspector:
    """Fence a PID with Linux boot ID and `/proc/<pid>/stat` start ticks."""

    def __init__(self, *, proc_root: str | Path = "/proc") -> None:
        self.proc_root = Path(proc_root)

    def identity_token(self, pid: int) -> str | None:
        try:
            boot_id = (self.proc_root / "sys/kernel/random/boot_id").read_text().strip()
            stat_payload = (self.proc_root / str(pid) / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return None
        except (OSError, UnicodeError):
            return None
        closing_parenthesis = stat_payload.rfind(")")
        if closing_parenthesis < 0:
            return None
        # Tokens after the command begin with field 3 (state); starttime is field 22.
        fields = stat_payload[closing_parenthesis + 1 :].split()
        if len(fields) <= 19 or not boot_id:
            return None
        return f"linux-proc-v1:{boot_id}:{fields[19]}"

    def probe(self, pid: int, expected_start_token: str) -> ProcessProbeV2:
        stat_path = self.proc_root / str(pid) / "stat"
        try:
            exists = stat_path.exists()
        except OSError as exc:
            return ProcessProbeV2(status=ProcessProbeStatus.UNKNOWN, detail=str(exc))
        if not exists:
            return ProcessProbeV2(status=ProcessProbeStatus.NOT_FOUND)
        observed = self.identity_token(pid)
        if observed is None:
            return ProcessProbeV2(
                status=ProcessProbeStatus.UNKNOWN,
                detail="process exists but its start identity cannot be read",
            )
        if observed != expected_start_token:
            return ProcessProbeV2(
                status=ProcessProbeStatus.PID_REUSED,
                observed_start_token=observed,
            )
        return ProcessProbeV2(
            status=ProcessProbeStatus.ALIVE_MATCH,
            observed_start_token=observed,
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AttemptExecutorV2:
    """Execute claimed attempts with fenced ownership and periodic heartbeats."""

    def __init__(
        self,
        state_store: AttemptStateStoreV2,
        *,
        worker_id: str,
        lease_seconds: int = 30,
        heartbeat_interval_seconds: float = 5.0,
        process_inspector: ProcessInspector | None = None,
        clock: Callable[[], datetime] = _utc_now,
        spawn_ready_timeout_seconds: float | None = None,
        max_execution_seconds: float | None = None,
    ) -> None:
        if not worker_id:
            raise AttemptExecutionError("worker_id cannot be empty")
        if lease_seconds <= 0:
            raise AttemptExecutionError("lease_seconds must be positive")
        if not 0 < heartbeat_interval_seconds < lease_seconds:
            raise AttemptExecutionError(
                "heartbeat interval must be positive and shorter than the lease"
            )
        ready_timeout = (
            min(1.0, lease_seconds / 2)
            if spawn_ready_timeout_seconds is None
            else spawn_ready_timeout_seconds
        )
        if not 0 < ready_timeout < lease_seconds:
            raise AttemptExecutionError(
                "spawn-ready timeout must be positive and shorter than the lease"
            )
        if max_execution_seconds is not None and max_execution_seconds <= 0:
            raise AttemptExecutionError("max_execution_seconds must be positive or null")
        self.state_store = state_store
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.process_inspector = process_inspector or LinuxProcessInspector()
        self.clock = clock
        self.spawn_ready_timeout_seconds = ready_timeout
        self.max_execution_seconds = max_execution_seconds

    def claim_and_execute_next(
        self,
        command_factory: Callable[[AttemptStateV2], AttemptCommandV2],
    ) -> AttemptStateV2 | None:
        claimed = self.state_store.claim_next(
            worker_id=self.worker_id,
            claimed_at=self.clock(),
            lease_seconds=self.lease_seconds,
        )
        if claimed is None:
            return None
        try:
            command = command_factory(claimed)
        except Exception as exc:
            return self.state_store.mark_runtime_failure(
                claimed.attempt_id,
                claim_token=str(claimed.claim_token),
                process_start_token=None,
                status=AttemptLifecycleStatus.FAILED,
                failed_at=self.clock(),
                error_code="COMMAND_BUILD_FAILED",
                error_detail=f"{type(exc).__name__}: {exc}",
            )
        return self.execute_attempt(claimed, command)

    def execute_attempt(
        self,
        claimed: AttemptStateV2,
        command: AttemptCommandV2,
    ) -> AttemptStateV2:
        current = self.state_store.get(claimed.attempt_id)
        if current.status != AttemptLifecycleStatus.CLAIMED:
            raise AttemptExecutionError("execute_attempt requires a CLAIMED attempt")
        if current.claim_token != claimed.claim_token or current.claimed_by != self.worker_id:
            raise AttemptExecutionError("claimed attempt is not owned by this worker")
        claim_token = str(current.claim_token)
        workdir = command.resolved_working_directory
        if not workdir.is_dir():
            return self.state_store.mark_runtime_failure(
                current.attempt_id,
                claim_token=claim_token,
                process_start_token=None,
                status=AttemptLifecycleStatus.FAILED,
                failed_at=self.clock(),
                error_code="WORKING_DIRECTORY_MISSING",
                error_detail=str(workdir),
            )

        log_path = Path(current.artifact_root).resolve() / "runtime" / "attempt.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        nonce = secrets.token_hex(32)
        spawn_guard = SpawnGuardBindingV1(
            ready_path=str(log_path.parent / "spawn.ready.json"),
            release_path=str(log_path.parent / "spawn.release.json"),
            nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest(),
        )
        current = self.state_store.prepare_execution(
            current.attempt_id,
            claim_token=claim_token,
            command_hash=command.command_hash,
            working_directory=workdir,
            log_path=log_path,
            prepared_at=self.clock(),
            spawn_guard_binding=spawn_guard,
        )
        occupied_paths = [
            path
            for path in (
                log_path,
                Path(spawn_guard.ready_path),
                Path(spawn_guard.release_path),
            )
            if path.exists()
        ]
        if occupied_paths:
            return self.state_store.mark_runtime_failure(
                current.attempt_id,
                claim_token=claim_token,
                process_start_token=None,
                status=AttemptLifecycleStatus.INVALID_RESULT,
                failed_at=self.clock(),
                error_code="RUNTIME_HANDSHAKE_PATH_ALREADY_EXISTS",
                error_detail=", ".join(str(path) for path in occupied_paths),
            )

        process: subprocess.Popen[bytes] | None = None
        process_start_token: str | None = None
        exit_code: int | None = None
        interrupted = False
        try:
            with log_path.open("xb", buffering=0) as log_handle:
                try:
                    process = subprocess.Popen(
                        self._guard_command(
                            command,
                            spawn_guard=spawn_guard,
                            nonce=nonce,
                        ),
                        cwd=workdir,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        start_new_session=True,
                    )
                except OSError as exc:
                    return self.state_store.mark_runtime_failure(
                        current.attempt_id,
                        claim_token=claim_token,
                        process_start_token=None,
                        status=AttemptLifecycleStatus.FAILED,
                        failed_at=self.clock(),
                        error_code="PROCESS_SPAWN_FAILED",
                        error_detail=f"{type(exc).__name__}: {exc}",
                    )
                try:
                    ready_token = self._await_spawn_guard_ready(
                        process,
                        spawn_guard=spawn_guard,
                        nonce=nonce,
                    )
                except KeyboardInterrupt:
                    self._terminate(process)
                    raise
                except Exception as exc:
                    self._terminate(process)
                    return self.state_store.mark_runtime_failure(
                        current.attempt_id,
                        claim_token=claim_token,
                        process_start_token=None,
                        status=AttemptLifecycleStatus.FAILED,
                        failed_at=self.clock(),
                        error_code="SPAWN_GUARD_NOT_READY",
                        error_detail=f"{type(exc).__name__}: {exc}",
                    )
                try:
                    process_start_token = self.process_inspector.identity_token(process.pid)
                except KeyboardInterrupt:
                    self._terminate(process)
                    raise
                except Exception as exc:
                    self._terminate(process)
                    return self.state_store.mark_runtime_failure(
                        current.attempt_id,
                        claim_token=claim_token,
                        process_start_token=None,
                        status=AttemptLifecycleStatus.FAILED,
                        failed_at=self.clock(),
                        error_code="PROCESS_IDENTITY_INSPECTION_FAILED",
                        error_detail=f"{type(exc).__name__}: {exc}",
                    )
                if process_start_token is None:
                    self._terminate(process)
                    return self.state_store.mark_runtime_failure(
                        current.attempt_id,
                        claim_token=claim_token,
                        process_start_token=None,
                        status=AttemptLifecycleStatus.FAILED,
                        failed_at=self.clock(),
                        error_code="PROCESS_IDENTITY_UNAVAILABLE",
                        error_detail=f"pid={process.pid}",
                    )
                if ready_token != process_start_token:
                    self._terminate(process)
                    return self.state_store.mark_runtime_failure(
                        current.attempt_id,
                        claim_token=claim_token,
                        process_start_token=None,
                        status=AttemptLifecycleStatus.FAILED,
                        failed_at=self.clock(),
                        error_code="SPAWN_GUARD_IDENTITY_MISMATCH",
                        error_detail=(f"ready={ready_token}; inspected={process_start_token}"),
                    )
                try:
                    self.state_store.mark_running(
                        current.attempt_id,
                        claim_token=claim_token,
                        pid=process.pid,
                        process_start_token=process_start_token,
                        started_at=self.clock(),
                        lease_seconds=self.lease_seconds,
                    )
                    try:
                        self._release_spawn_guard(
                            spawn_guard,
                            nonce=nonce,
                        )
                    except Exception as exc:
                        self._terminate(process)
                        return self.state_store.mark_runtime_failure(
                            current.attempt_id,
                            claim_token=claim_token,
                            process_start_token=process_start_token,
                            status=AttemptLifecycleStatus.FAILED,
                            failed_at=self.clock(),
                            error_code="SPAWN_GUARD_RELEASE_FAILED",
                            error_detail=f"{type(exc).__name__}: {exc}",
                        )
                    try:
                        exit_code = self._wait_with_heartbeats(
                            process,
                            attempt_id=current.attempt_id,
                            claim_token=claim_token,
                            process_start_token=process_start_token,
                        )
                    except AttemptExecutionTimeout as exc:
                        self._terminate(process)
                        return self.state_store.mark_runtime_failure(
                            current.attempt_id,
                            claim_token=claim_token,
                            process_start_token=process_start_token,
                            status=AttemptLifecycleStatus.FAILED,
                            failed_at=self.clock(),
                            error_code="PROCESS_EXECUTION_TIMEOUT",
                            error_detail=str(exc),
                        )
                except KeyboardInterrupt:
                    interrupted = True
                    self._terminate(process)
                except Exception:
                    self._terminate(process)
                    raise
                finally:
                    log_handle.flush()
                    os.fsync(log_handle.fileno())
        except FileExistsError:
            return self.state_store.mark_runtime_failure(
                current.attempt_id,
                claim_token=claim_token,
                process_start_token=None,
                status=AttemptLifecycleStatus.INVALID_RESULT,
                failed_at=self.clock(),
                error_code="RUNTIME_LOG_ALREADY_EXISTS",
                error_detail=str(log_path),
            )

        return self._finalize_child_exit(
            current,
            claim_token=claim_token,
            process_start_token=process_start_token,
            exit_code=exit_code,
            interrupted=interrupted,
        )

    def _guard_command(
        self,
        command: AttemptCommandV2,
        *,
        spawn_guard: SpawnGuardBindingV1,
        nonce: str,
    ) -> list[str]:
        guard_script = Path(__file__).with_name("spawn_guard_v2.py").resolve()
        if not guard_script.is_file():
            raise AttemptExecutionError(f"spawn guard is missing: {guard_script}")
        return [
            sys.executable,
            str(guard_script),
            "--parent-pid",
            str(os.getpid()),
            "--ready-path",
            spawn_guard.ready_path,
            "--release-path",
            spawn_guard.release_path,
            "--nonce",
            nonce,
            "--release-timeout-seconds",
            str(self.lease_seconds),
            "--",
            *command.argv,
        ]

    def _await_spawn_guard_ready(
        self,
        process: subprocess.Popen[bytes],
        *,
        spawn_guard: SpawnGuardBindingV1,
        nonce: str,
    ) -> str:
        ready_path = Path(spawn_guard.ready_path)
        deadline = time.monotonic() + self.spawn_ready_timeout_seconds
        while time.monotonic() < deadline:
            if ready_path.is_file():
                try:
                    payload = json.loads(ready_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise AttemptExecutionError("spawn guard ready receipt is unreadable") from exc
                expected = {
                    "nonce": nonce,
                    "pid": process.pid,
                    "protocol": "PARENT_DEATH_GUARD_V1",
                }
                if any(payload.get(key) != value for key, value in expected.items()):
                    raise AttemptExecutionError("spawn guard ready receipt differs from binding")
                token = payload.get("process_start_token")
                if not isinstance(token, str) or not token:
                    raise AttemptExecutionError("spawn guard ready receipt lacks process identity")
                if hashlib.sha256(nonce.encode()).hexdigest() != (spawn_guard.nonce_sha256):
                    raise AttemptExecutionError("spawn nonce differs from binding")
                return token
            return_code = process.poll()
            if return_code is not None:
                raise AttemptExecutionError(
                    f"spawn guard exited before ready: exit_code={return_code}"
                )
            time.sleep(0.01)
        raise AttemptExecutionError("spawn guard did not become ready before timeout")

    @staticmethod
    def _release_spawn_guard(
        spawn_guard: SpawnGuardBindingV1,
        *,
        nonce: str,
    ) -> None:
        path = Path(spawn_guard.release_path)
        payload = (
            json.dumps(
                {
                    "nonce": nonce,
                    "protocol": "PARENT_DEATH_GUARD_V1",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _finalize_child_exit(
        self,
        current: AttemptStateV2,
        *,
        claim_token: str,
        process_start_token: str | None,
        exit_code: int | None,
        interrupted: bool,
    ) -> AttemptStateV2:
        if interrupted:
            terminal = self.state_store.mark_runtime_failure(
                current.attempt_id,
                claim_token=claim_token,
                process_start_token=process_start_token,
                status=AttemptLifecycleStatus.INTERRUPTED,
                failed_at=self.clock(),
                error_code="CONTROLLER_INTERRUPTED",
            )
            raise KeyboardInterrupt(
                f"attempt {terminal.attempt_id} was interrupted"
            ) from AttemptExecutionError(f"attempt {terminal.attempt_id} was interrupted")
        if exit_code != 0:
            return self.state_store.mark_runtime_failure(
                current.attempt_id,
                claim_token=claim_token,
                process_start_token=process_start_token,
                status=AttemptLifecycleStatus.FAILED,
                failed_at=self.clock(),
                error_code="PROCESS_EXIT_NONZERO",
                error_detail=f"exit_code={exit_code}",
            )
        try:
            return self.state_store.finalize_from_artifacts(
                current.attempt_id,
                claim_token=claim_token,
                process_start_token=str(process_start_token),
                finalized_at=self.clock(),
            )
        except ArtifactIntegrityError as exc:
            return self.state_store.mark_runtime_failure(
                current.attempt_id,
                claim_token=claim_token,
                process_start_token=process_start_token,
                status=AttemptLifecycleStatus.INVALID_RESULT,
                failed_at=self.clock(),
                error_code="INVALID_TERMINAL_ARTIFACTS",
                error_detail=f"{type(exc).__name__}: {exc}",
            )

    def _wait_with_heartbeats(
        self,
        process: subprocess.Popen[bytes],
        *,
        attempt_id: str,
        claim_token: str,
        process_start_token: str,
    ) -> int:
        deadline = (
            None
            if self.max_execution_seconds is None
            else time.monotonic() + self.max_execution_seconds
        )
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AttemptExecutionTimeout(
                        f"worker exceeded {self.max_execution_seconds:.3f}s"
                    )
                wait_seconds = min(self.heartbeat_interval_seconds, remaining)
            else:
                wait_seconds = self.heartbeat_interval_seconds
            try:
                return process.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    raise AttemptExecutionTimeout(
                        f"worker exceeded {self.max_execution_seconds:.3f}s"
                    )
                self.state_store.heartbeat(
                    attempt_id,
                    claim_token=claim_token,
                    process_start_token=process_start_token,
                    heartbeat_at=self.clock(),
                    lease_seconds=self.lease_seconds,
                )

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes], timeout_seconds: float = 5.0) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=timeout_seconds)


class ReconciliationAction(StrEnum):
    FINALIZED_FROM_ARTIFACTS = "FINALIZED_FROM_ARTIFACTS"
    INTERRUPTED_PROCESS_GONE = "INTERRUPTED_PROCESS_GONE"
    INVALID_RESULT_PROCESS_GONE = "INVALID_RESULT_PROCESS_GONE"
    INTERRUPTED_CLAIM_NO_SPAWN = "INTERRUPTED_CLAIM_NO_SPAWN"
    INTERRUPTED_GUARD_GONE = "INTERRUPTED_GUARD_GONE"
    NO_ACTION_PROCESS_ALIVE = "NO_ACTION_PROCESS_ALIVE"
    NO_ACTION_GUARD_ALIVE = "NO_ACTION_GUARD_ALIVE"
    NO_ACTION_PROCESS_UNKNOWN = "NO_ACTION_PROCESS_UNKNOWN"
    NO_ACTION_CLAIM_UNPROVEN = "NO_ACTION_CLAIM_UNPROVEN"
    RACE_LOST = "RACE_LOST"


class ReconciliationRecordV2(StrictV2Model):
    attempt_id: str = Field(min_length=1)
    action: ReconciliationAction
    process_probe: ProcessProbeStatus | None = None
    terminal_status: AttemptLifecycleStatus | None = None
    detail: str | None = None


class AttemptReconcilerV2:
    """Recover expired attempts only when process identity provides safe evidence."""

    def __init__(
        self,
        state_store: AttemptStateStoreV2,
        *,
        process_inspector: ProcessInspector | None = None,
        actor: str = "attempt-reconciler-v2",
    ) -> None:
        self.state_store = state_store
        self.process_inspector = process_inspector or LinuxProcessInspector()
        self.actor = actor

    def reconcile_expired(self, *, as_of: datetime) -> list[ReconciliationRecordV2]:
        records: list[ReconciliationRecordV2] = []
        for state in self.state_store.expired_leases(as_of=as_of):
            records.append(self._reconcile_one(state, as_of=as_of))
        return records

    def _reconcile_one(
        self,
        state: AttemptStateV2,
        *,
        as_of: datetime,
    ) -> ReconciliationRecordV2:
        if state.status == AttemptLifecycleStatus.CLAIMED:
            return self._reconcile_claimed(state, as_of=as_of)
        if state.pid is None or state.process_start_token is None:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.NO_ACTION_PROCESS_UNKNOWN,
                process_probe=ProcessProbeStatus.UNKNOWN,
                detail="RUNNING state has no complete process identity",
            )
        probe = self.process_inspector.probe(state.pid, state.process_start_token)
        if probe.status == ProcessProbeStatus.ALIVE_MATCH:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.NO_ACTION_PROCESS_ALIVE,
                process_probe=probe.status,
            )
        if probe.status == ProcessProbeStatus.UNKNOWN:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.NO_ACTION_PROCESS_UNKNOWN,
                process_probe=probe.status,
                detail=probe.detail,
            )

        evidence = probe.status.value
        try:
            terminal = self.state_store.reconcile_expired_from_artifacts(
                state.attempt_id,
                expected_version=state.version,
                expected_process_start_token=state.process_start_token,
                reconciled_at=as_of,
                actor=self.actor,
                process_evidence=evidence,
            )
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.FINALIZED_FROM_ARTIFACTS,
                process_probe=probe.status,
                terminal_status=terminal.status,
            )
        except ArtifactIntegrityError as exc:
            result_path = Path(state.artifact_root) / V2ArtifactStore.RESULT_NAME
            checksum_path = Path(state.artifact_root) / V2ArtifactStore.RESULT_CHECKSUM_NAME
            has_terminal_files = result_path.exists() or checksum_path.exists()
            target = (
                AttemptLifecycleStatus.INVALID_RESULT
                if has_terminal_files
                else AttemptLifecycleStatus.INTERRUPTED
            )
            error_code = (
                "RECOVERY_INVALID_TERMINAL_ARTIFACTS"
                if has_terminal_files
                else "RECOVERY_PROCESS_GONE_NO_RESULT"
            )
            try:
                terminal = self.state_store.reconcile_expired_failure(
                    state.attempt_id,
                    expected_version=state.version,
                    expected_process_start_token=state.process_start_token,
                    reconciled_at=as_of,
                    actor=self.actor,
                    process_evidence=evidence,
                    status=target,
                    error_code=error_code,
                    error_detail=f"{type(exc).__name__}: {exc}",
                )
            except (AttemptClaimConflict, AttemptStateError) as race:
                return ReconciliationRecordV2(
                    attempt_id=state.attempt_id,
                    action=ReconciliationAction.RACE_LOST,
                    process_probe=probe.status,
                    detail=str(race),
                )
            action = (
                ReconciliationAction.INVALID_RESULT_PROCESS_GONE
                if has_terminal_files
                else ReconciliationAction.INTERRUPTED_PROCESS_GONE
            )
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=action,
                process_probe=probe.status,
                terminal_status=terminal.status,
                detail=str(exc),
            )
        except (AttemptClaimConflict, AttemptStateError) as exc:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.RACE_LOST,
                process_probe=probe.status,
                detail=str(exc),
            )

    def _reconcile_claimed(
        self,
        state: AttemptStateV2,
        *,
        as_of: datetime,
    ) -> ReconciliationRecordV2:
        if state.claim_recovery_contract != CLAIM_RECOVERY_CONTRACT_V1:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.NO_ACTION_CLAIM_UNPROVEN,
                detail="claim predates the guarded-spawn recovery contract",
            )
        if state.command_hash is None:
            return self._close_claimed(
                state,
                as_of=as_of,
                action=ReconciliationAction.INTERRUPTED_CLAIM_NO_SPAWN,
                error_code="RECOVERY_EXPIRED_BEFORE_SPAWN",
                detail="claim expired before execution preparation",
            )
        binding = state.spawn_guard_binding
        if binding is None:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.NO_ACTION_CLAIM_UNPROVEN,
                detail="prepared claim has no parent-death guard binding",
            )
        if Path(binding.release_path).exists():
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.NO_ACTION_CLAIM_UNPROVEN,
                detail="CLAIMED state unexpectedly has a spawn release receipt",
            )
        ready_path = Path(binding.ready_path)
        if not ready_path.exists():
            return self._close_claimed(
                state,
                as_of=as_of,
                action=ReconciliationAction.INTERRUPTED_CLAIM_NO_SPAWN,
                error_code="RECOVERY_GUARD_NEVER_READY",
                detail="guard did not publish a ready receipt before claim expiry",
            )
        try:
            payload = json.loads(ready_path.read_text())
            nonce = payload["nonce"]
            pid = payload["pid"]
            process_start_token = payload["process_start_token"]
            if (
                payload.get("protocol") != "PARENT_DEATH_GUARD_V1"
                or not isinstance(nonce, str)
                or hashlib.sha256(nonce.encode()).hexdigest() != binding.nonce_sha256
                or not isinstance(pid, int)
                or pid < 1
                or not isinstance(process_start_token, str)
                or not process_start_token
            ):
                raise ValueError("ready receipt differs from spawn guard binding")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.NO_ACTION_CLAIM_UNPROVEN,
                detail=f"spawn ready receipt cannot prove identity: {exc}",
            )
        probe = self.process_inspector.probe(pid, process_start_token)
        if probe.status == ProcessProbeStatus.ALIVE_MATCH:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.NO_ACTION_GUARD_ALIVE,
                process_probe=probe.status,
                detail="guard is alive and still cannot start without a release receipt",
            )
        if probe.status == ProcessProbeStatus.UNKNOWN:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.NO_ACTION_PROCESS_UNKNOWN,
                process_probe=probe.status,
                detail=probe.detail,
            )
        return self._close_claimed(
            state,
            as_of=as_of,
            action=ReconciliationAction.INTERRUPTED_GUARD_GONE,
            error_code="RECOVERY_GUARD_GONE_BEFORE_RELEASE",
            detail=f"guard process evidence: {probe.status.value}",
            process_probe=probe.status,
        )

    def _close_claimed(
        self,
        state: AttemptStateV2,
        *,
        as_of: datetime,
        action: ReconciliationAction,
        error_code: str,
        detail: str,
        process_probe: ProcessProbeStatus | None = None,
    ) -> ReconciliationRecordV2:
        try:
            terminal = self.state_store.reconcile_expired_claimed_failure(
                state.attempt_id,
                expected_version=state.version,
                reconciled_at=as_of,
                actor=self.actor,
                error_code=error_code,
                error_detail=detail,
            )
        except (AttemptClaimConflict, AttemptStateError) as exc:
            return ReconciliationRecordV2(
                attempt_id=state.attempt_id,
                action=ReconciliationAction.RACE_LOST,
                process_probe=process_probe,
                detail=str(exc),
            )
        return ReconciliationRecordV2(
            attempt_id=state.attempt_id,
            action=action,
            process_probe=process_probe,
            terminal_status=terminal.status,
            detail=detail,
        )


def execute_attempt(
    state_store: AttemptStateStoreV2,
    claimed: AttemptStateV2,
    command: AttemptCommandV2,
    *,
    worker_id: str,
    lease_seconds: int = 30,
    heartbeat_interval_seconds: float = 5.0,
) -> AttemptStateV2:
    """Convenience entry point used by queue, CLI, and future controller adapters."""

    return AttemptExecutorV2(
        state_store,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    ).execute_attempt(claimed, command)
