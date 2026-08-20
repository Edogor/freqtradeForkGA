"""Linux parent-death guard for the canonical GA V2 subprocess handshake.

The guard becomes a new session leader, proves that it is armed with
``PR_SET_PDEATHSIG``, publishes a one-time ready receipt, and waits for a
release nonce.  It starts the real command only after the controller has
committed the guard PID/start-token as ``RUNNING`` in SQLite.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


_PR_SET_PDEATHSIG = 1


def _kill_process_group(signum: int, _frame: object) -> None:
    """Kill the complete guarded session when the controller disappears."""

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except ProcessLookupError:
        pass
    os._exit(128 + signum)


def _arm_parent_death(expected_parent_pid: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    # The parent may have died between fork/exec and prctl(). Checking after
    # arming closes that race: either it is still the expected controller or
    # this guard exits without ever starting the target.
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("controller disappeared before parent-death guard was armed")
    signal.signal(signal.SIGTERM, _kill_process_group)


def _linux_start_token(pid: int) -> str:
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    stat_payload = Path(f"/proc/{pid}/stat").read_text()
    closing_parenthesis = stat_payload.rfind(")")
    if closing_parenthesis < 0:
        raise RuntimeError("guard process stat is malformed")
    fields = stat_payload[closing_parenthesis + 1 :].split()
    if len(fields) <= 19 or not boot_id:
        raise RuntimeError("guard process start identity is unavailable")
    return f"linux-proc-v1:{boot_id}:{fields[19]}"


def _write_once(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
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
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wait_for_release(
    path: Path,
    *,
    nonce: str,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            release = json.loads(path.read_text())
        except FileNotFoundError:
            time.sleep(0.01)
            continue
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("spawn release receipt is unreadable") from exc
        if release != {"nonce": nonce, "protocol": "PARENT_DEATH_GUARD_V1"}:
            raise RuntimeError("spawn release receipt differs from guard binding")
        return
    raise TimeoutError("controller did not release guarded process before lease deadline")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--ready-path", required=True)
    parser.add_argument("--release-path", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--release-timeout-seconds", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if (
        args.parent_pid < 1
        or args.release_timeout_seconds <= 0
        or not args.nonce
        or not args.command
        or any(not item or "\x00" in item for item in args.command)
    ):
        parser.error("invalid guarded command contract")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ready_path = Path(args.ready_path).resolve()
    release_path = Path(args.release_path).resolve()
    if ready_path == release_path or ready_path.exists() or release_path.exists():
        raise RuntimeError("spawn handshake paths must be distinct and unused")

    _arm_parent_death(args.parent_pid)
    pid = os.getpid()
    _write_once(
        ready_path,
        {
            "nonce": args.nonce,
            "pid": pid,
            "process_start_token": _linux_start_token(pid),
            "protocol": "PARENT_DEATH_GUARD_V1",
        },
    )
    _wait_for_release(
        release_path,
        nonce=args.nonce,
        timeout_seconds=args.release_timeout_seconds,
    )

    target = subprocess.Popen(args.command, start_new_session=False)
    return_code = target.wait()
    return return_code if return_code >= 0 else 128 + abs(return_code)


if __name__ == "__main__":
    sys.exit(main())
