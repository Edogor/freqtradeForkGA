"""Operational facade for the canonical GA V2 attempt execution contract.

This module is the only supported bridge from a user-supplied standard-GA
configuration to the immutable worker queue.  It deliberately keeps
materialization/registration in the parent process and execution in a child
that knows only its hash-bound worker spec.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from genetic_algorithm.config.schema import load_config
from genetic_algorithm.orchestration.attempt_executor_v2 import (
    AttemptCommandV2,
    AttemptExecutorV2,
)
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    AttemptStateError,
    AttemptStateStoreV2,
    AttemptStateV2,
    WorkerKind,
)
from genetic_algorithm.orchestration.data_manifest_v2 import resolve_spot_data_root
from genetic_algorithm.orchestration.evolution_scheduler_v2 import (
    queue_prepared_evolution_attempt,
)
from genetic_algorithm.orchestration.evolution_worker_v2 import (
    PreparedEvolutionWorkerV2,
    prepare_evolution_worker,
)
from genetic_algorithm.orchestration.manifest_builder_v2 import (
    AttemptManifestBundleV2,
    build_attempt_manifest_bundle,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    shadow_gate_policy_from_config,
)


class CanonicalRunnerError(RuntimeError):
    """Raised when an operational start cannot satisfy the V2 contract."""


@dataclass(frozen=True)
class PreparedOperationalAttemptV2:
    bundle: AttemptManifestBundleV2
    worker: PreparedEvolutionWorkerV2
    state: AttemptStateV2

    @property
    def attempt_id(self) -> str:
        return self.bundle.manifest.attempt_id


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_v2_root(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root).resolve() if repo_root is not None else repository_root()
    return root / "genetic_algorithm" / "data" / "v2"


def default_state_path(repo_root: str | Path | None = None) -> Path:
    return default_v2_root(repo_root) / "orchestration.sqlite3"


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    if not normalized:
        raise CanonicalRunnerError("experiment name has no usable characters")
    return normalized[:80]


def _attempt_id(experiment_name: str, created_at: datetime) -> str:
    stamp = created_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{_slug(experiment_name)}-{stamp}"


def _validate_supported_config(config: Mapping[str, Any]) -> None:
    supported_safety_profiles = {
        "safe_v2",
        "automation_island_v2",
        "quality_experiment_v3",
    }
    if config.get("generic_island_model", {}).get("enabled", False):
        safety = config.get("safety_profile", {})
        if (
            safety.get("name") not in {
                "automation_island_v2",
                "quality_experiment_v3",
            }
            or not safety.get("enforce", False)
            or not safety.get("shadow_mode", False)
            or safety.get("automation_eligible", False)
        ):
            raise CanonicalRunnerError(
                "generic island evolution requires an enforced shadow-only safety profile"
            )
    if config.get("island_model", {}).get("enabled", False):
        raise CanonicalRunnerError("island evolution has no canonical V2 worker yet")
    safety = config.get("safety_profile", {})
    if safety.get("name") not in supported_safety_profiles or not safety.get(
        "enforce", False
    ):
        raise CanonicalRunnerError(
            "operational starts require an enforced V2 safety profile"
        )
    if config.get("genetic_algorithm", {}).get("mode") != "single_objective":
        raise CanonicalRunnerError(
            "the canonical evolution worker currently supports single_objective only"
        )


def prepare_and_queue_standard_attempt(
    config_path: str | Path,
    *,
    experiment_name: str | None = None,
    state_path: str | Path | None = None,
    artifact_base: str | Path | None = None,
    priority: int = 0,
    created_at: datetime | None = None,
    repo_root: str | Path | None = None,
    python_executable: str | Path = sys.executable,
) -> PreparedOperationalAttemptV2:
    """Resolve, freeze, register, and queue one standard evolution attempt."""

    resolved_repo = (
        Path(repo_root).resolve() if repo_root is not None else repository_root()
    )
    source = Path(config_path).resolve()
    if not source.is_file():
        raise CanonicalRunnerError(f"config not found: {source}")
    config = load_config(source)
    _validate_supported_config(config)
    now = created_at or datetime.now(UTC)
    if now.utcoffset() is None:
        raise CanonicalRunnerError("created_at must be timezone-aware")
    name = experiment_name or source.stem
    attempt_id = _attempt_id(name, now)
    v2_root = default_v2_root(resolved_repo)
    attempts_root = (
        Path(artifact_base).resolve()
        if artifact_base is not None
        else v2_root / "attempts"
    )
    artifact_root = attempts_root / attempt_id
    database = (
        Path(state_path).resolve()
        if state_path is not None
        else default_state_path(resolved_repo)
    )
    policy = shadow_gate_policy_from_config(config)
    pairs = sorted({item.pair for item in policy.required_scenarios})
    data_root = resolved_repo / resolve_spot_data_root(config, pairs)
    ga = config.get("genetic_algorithm", {})
    parallel = config.get("parallel_evaluation", {})
    seed = int(ga["random_seed"])
    worker_count = (
        int(parallel.get("num_workers", 1))
        if parallel.get("enabled", False)
        else 1
    )
    top_n = int(config.get("output", {}).get("top_n", 5))
    bundle = build_attempt_manifest_bundle(
        attempt_id=attempt_id,
        wave_id=f"manual-{attempt_id}",
        experiment_id=_slug(name),
        created_at=now,
        resolved_config=config,
        policy=policy,
        fitness_policy_version=policy.policy_version,
        seeds=[seed],
        worker_count=worker_count,
        artifact_root=artifact_root,
        repo_root=resolved_repo,
        data_root=data_root,
    )
    worker = prepare_evolution_worker(
        bundle=bundle,
        resolved_config=config,
        policy=policy,
        seeds=[],
        repo_root=resolved_repo,
        final_test_ledger_path=v2_root / "final_test_usage.sqlite3",
        created_at=now + timedelta(microseconds=1),
        top_n=top_n,
    )
    store = AttemptStateStoreV2(database)
    state = queue_prepared_evolution_attempt(
        store,
        manifest=bundle.manifest,
        prepared=worker,
        bound_at=now + timedelta(microseconds=2),
        validated_at=now + timedelta(microseconds=3),
        queued_at=now + timedelta(microseconds=4),
        priority=priority,
        actor="operational-runner-v2",
        python_executable=python_executable,
        working_directory=resolved_repo,
    )
    return PreparedOperationalAttemptV2(bundle=bundle, worker=worker, state=state)


def execute_queued_attempt(
    state_store: AttemptStateStoreV2,
    attempt_id: str,
    *,
    worker_id: str,
    lease_seconds: int = 30,
    heartbeat_interval_seconds: float = 5.0,
) -> AttemptStateV2:
    """Claim and execute exactly one queued attempt through ``execute_attempt``."""

    claimed = state_store.claim_attempt(
        attempt_id,
        worker_id=worker_id,
        claimed_at=datetime.now(UTC),
        lease_seconds=lease_seconds,
    )
    binding = claimed.worker_binding
    if binding is None or binding.worker_kind not in {
        WorkerKind.STANDARD_EVOLUTION,
        WorkerKind.GENERIC_ISLAND_EVOLUTION,
        WorkerKind.SHADOW_REPLAY,
    }:
        raise AttemptStateError("claimed attempt has no supported immutable worker")
    command = AttemptCommandV2(
        argv=binding.argv,
        working_directory=binding.working_directory,
    )
    return AttemptExecutorV2(
        state_store,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    ).execute_attempt(claimed, command)


def run_standard_attempt(
    config_path: str | Path,
    *,
    experiment_name: str | None = None,
    state_path: str | Path | None = None,
    artifact_base: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> AttemptStateV2:
    """Materialize, queue, claim, and execute a standard GA attempt."""

    prepared = prepare_and_queue_standard_attempt(
        config_path,
        experiment_name=experiment_name,
        state_path=state_path,
        artifact_base=artifact_base,
        repo_root=repo_root,
    )
    database = (
        Path(state_path).resolve()
        if state_path is not None
        else default_state_path(repo_root)
    )
    terminal = execute_queued_attempt(
        AttemptStateStoreV2(database),
        prepared.attempt_id,
        worker_id=f"manual-runner:{prepared.attempt_id}",
    )
    if terminal.status not in {
        AttemptLifecycleStatus.SUCCEEDED,
        AttemptLifecycleStatus.FAILED,
        AttemptLifecycleStatus.INTERRUPTED,
        AttemptLifecycleStatus.INVALID_RESULT,
    }:
        raise CanonicalRunnerError(
            f"attempt did not reach a terminal state: {terminal.status.value}"
        )
    return terminal
