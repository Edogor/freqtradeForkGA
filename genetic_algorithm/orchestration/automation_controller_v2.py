"""Guarded, restartable automation for canonical V2 evolution waves.

The controller automates *search only*.  Every materialized config must retain
the enforced ``automation_island_v2`` shadow contract, and every executable
proposal is hash-bound before it receives the controller's narrow approval.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from genetic_algorithm.config.schema import (
    load_config,
    validate_resolved_config_v2_or_raise,
)
from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.attempt_scheduler_v2 import AttemptSchedulerV2
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    WorkerBindingV2,
    WorkerKind,
)
from genetic_algorithm.orchestration.candidate_selector_v2 import (
    CandidateSelectionPolicyV2,
)
from genetic_algorithm.orchestration.data_manifest_v2 import resolve_spot_data_root
from genetic_algorithm.orchestration.data_manifest_v2 import build_data_manifest
from genetic_algorithm.orchestration.code_manifest_v2 import capture_code_manifest
from genetic_algorithm.orchestration.evolution_worker_v2 import (
    evolution_worker_argv,
    load_evolution_worker,
    prepare_evolution_worker,
)
from genetic_algorithm.orchestration.manifest_builder_v2 import (
    build_attempt_manifest_bundle,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    shadow_gate_policy_from_config,
)
from genetic_algorithm.orchestration.result_contract import StrictV2Model
from genetic_algorithm.orchestration.split_contract_v2 import (
    build_evaluation_split_plan,
)
from genetic_algorithm.orchestration.shadow_scheduler_v2 import (
    ShadowSchedulerConfigV2,
)
from genetic_algorithm.orchestration.wave_analyzer_v2 import (
    WaveAnalysisV2,
    WaveAnalyzerPolicyV2,
    WaveAnalyzerV2,
)
from genetic_algorithm.orchestration.wave_materializer_v2 import (
    MaterializedAttemptV2,
    PreparedChildWaveMaterializationV2,
    materialize_evolution_wave,
    queue_approved_materialization,
)
from genetic_algorithm.orchestration.wave_planner_v2 import (
    ChildWavePlanV2,
    PlannerSourceMode,
    WaveArmTemplateV2,
    WavePlanMode,
    WavePlannerPolicyV2,
    select_and_plan_child_wave,
)
from genetic_algorithm.orchestration.wave_state_v2 import (
    AttemptExpectationV2,
    ExperimentArmType,
    ExperimentSpecV2,
    WaveBudgetV2,
    WaveDecisionType,
    WaveDecisionV2,
    WaveLifecycleStatus,
    WaveSpecV2,
    WaveStateError,
    WaveStateStoreV2,
    WaveStateV2,
)


class AutomationControllerError(RuntimeError):
    """Raised when unattended execution cannot prove that it is safe."""


class AutomationPolicyV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    automation_policy_version: str = Field(min_length=1)
    root_seeds: list[int] = Field(min_length=1)
    max_waves: int = Field(ge=1)
    max_total_attempts: int = Field(ge=1)
    max_consecutive_failed_waves: int = Field(ge=1)
    max_concurrent: int = Field(ge=1)
    max_runtime_seconds: int = Field(ge=1)
    max_attempt_runtime_seconds: int = Field(ge=1)
    max_artifact_bytes: int = Field(ge=1)
    min_free_disk_bytes: int = Field(ge=1)
    min_available_memory_bytes: int = Field(ge=1)
    poll_interval_seconds: float = Field(default=5.0, gt=0)
    kill_switch_path: str = Field(min_length=1)
    allowed_factor_paths: list[str] = Field(default_factory=list)
    analyzer_policy: WaveAnalyzerPolicyV2
    selection_policy: CandidateSelectionPolicyV2
    planner_policy: WavePlannerPolicyV2

    @model_validator(mode="after")
    def _coherent_limits(self) -> AutomationPolicyV2:
        if self.root_seeds != sorted(set(self.root_seeds)):
            raise ValueError("root_seeds must be unique and sorted")
        if self.max_total_attempts < len(self.root_seeds):
            raise ValueError("max_total_attempts cannot be smaller than root seeds")
        if self.planner_policy.budget.max_parallel > self.max_concurrent:
            raise ValueError("wave parallel budget exceeds controller concurrency")
        if self.allowed_factor_paths != sorted(set(self.allowed_factor_paths)):
            raise ValueError("allowed_factor_paths must be sorted and unique")
        if not Path(self.kill_switch_path).is_absolute():
            raise ValueError("kill_switch_path must be absolute")
        if (
            not self.planner_policy.allow_control_fallback_when_no_candidates
            or not self.planner_policy.allow_control_recovery_after_technical_failure
        ):
            raise ValueError(
                "unattended policy requires healthy fallback and bounded recovery"
            )
        return self

    @property
    def policy_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


class AutomationBootstrapIntentV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    created_at: datetime
    config_hash: str = Field(min_length=64, max_length=64)
    automation_policy_hash: str = Field(min_length=64, max_length=64)
    seeds: list[int] = Field(min_length=1)
    wave_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)


class AutomationBootstrapReceiptV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    intent: AutomationBootstrapIntentV2
    wave_spec: WaveSpecV2
    experiment_spec: ExperimentSpecV2
    attempts: list[MaterializedAttemptV2] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_receipt(self) -> AutomationBootstrapReceiptV2:
        if self.wave_spec.wave_id != self.intent.wave_id:
            raise ValueError("bootstrap wave differs from intent")
        if self.experiment_spec.experiment_id != self.intent.experiment_id:
            raise ValueError("bootstrap experiment differs from intent")
        if self.experiment_spec.wave_id != self.wave_spec.wave_id:
            raise ValueError("bootstrap experiment belongs to another wave")
        manifests = {item.manifest.attempt_id: item for item in self.attempts}
        expected = {
            item.attempt_id: item for item in self.experiment_spec.expected_attempts
        }
        if set(manifests) != set(expected):
            raise ValueError("bootstrap attempts differ from expectations")
        for attempt_id, materialized in manifests.items():
            expectation = expected[attempt_id]
            if materialized.manifest_hash != expectation.manifest_hash:
                raise ValueError("bootstrap manifest hash differs from expectation")
            if materialized.manifest.seeds != [expectation.seed]:
                raise ValueError("bootstrap attempt seed differs from expectation")
        return self


class AutomationTickV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    observed_at: datetime
    root_wave_id: str
    active_wave_id: str
    active_wave_status: WaveLifecycleStatus
    outcome: Literal[
        "WAITING",
        "PROGRESSED",
        "STOPPED_KILL_SWITCH",
        "STOPPED_LIMIT",
        "BLOCKED",
    ]
    reason_codes: list[str] = Field(min_length=1)
    launched_attempt_ids: list[str] = Field(default_factory=list)


class AutomationPreflightV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    checked_at: datetime
    ready: bool
    config_hash: str = Field(min_length=64, max_length=64)
    code_manifest_hash: str = Field(min_length=64, max_length=64)
    data_manifest_hash: str = Field(min_length=64, max_length=64)
    split_hash: str = Field(min_length=64, max_length=64)
    automation_policy_hash: str = Field(min_length=64, max_length=64)
    python_executable: str = Field(min_length=1)
    virtualenv_active: bool
    disk_free_bytes: int = Field(ge=0)
    disk_reserve_bytes: int = Field(ge=0)
    memory_available_bytes: int = Field(ge=0)
    memory_reserve_bytes: int = Field(ge=0)
    evolution_pairs: list[str] = Field(min_length=1)
    validation_pairs: list[str] = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)


_TERMINAL_ATTEMPTS = {
    AttemptLifecycleStatus.SUCCEEDED,
    AttemptLifecycleStatus.FAILED,
    AttemptLifecycleStatus.INTERRUPTED,
    AttemptLifecycleStatus.INVALID_RESULT,
}


def _canonical_bytes(model: StrictV2Model) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json", exclude_none=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise AutomationControllerError(f"immutable automation file differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise AutomationControllerError(
                    f"concurrent automation file differs: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _available_memory_bytes() -> int:
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _validate_automation_config(config: dict[str, Any]) -> None:
    validate_resolved_config_v2_or_raise(config)
    safety = config.get("safety_profile", {})
    if safety != {
        "name": "automation_island_v2",
        "enforce": True,
        "shadow_mode": True,
        "automation_eligible": False,
    }:
        raise AutomationControllerError(
            "automation requires the exact enforced search-only safety profile"
        )
    if not config.get("generic_island_model", {}).get("enabled", False):
        raise AutomationControllerError("automation requires Generic Island evolution")
    if config.get("pair_validation", {}).get("validate_top_n_only", 0) != 0:
        raise AutomationControllerError("automation must pair-validate every candidate")


def default_automation_policy(
    config: dict[str, Any],
    *,
    automation_root: str | Path,
) -> AutomationPolicyV2:
    """Return the conservative one-week policy shipped with the V2 preset."""

    _validate_automation_config(config)
    result_policy = shadow_gate_policy_from_config(config).policy_version
    wave_budget = WaveBudgetV2(
        max_attempts=3,
        max_parallel=1,
        max_wallclock_seconds=48 * 60 * 60,
    )
    return AutomationPolicyV2(
        automation_policy_version="guarded-island-search-v2.0",
        root_seeds=[1001],
        max_waves=50,
        max_total_attempts=148,
        max_consecutive_failed_waves=3,
        max_concurrent=1,
        max_runtime_seconds=7 * 24 * 60 * 60,
        max_attempt_runtime_seconds=12 * 60 * 60,
        max_artifact_bytes=40 * 1024**3,
        min_free_disk_bytes=20 * 1024**3,
        min_available_memory_bytes=2 * 1024**3,
        poll_interval_seconds=5.0,
        kill_switch_path=str(Path(automation_root).resolve() / "STOP_AUTOMATION"),
        allowed_factor_paths=["genetic_algorithm.mutation_rate"],
        analyzer_policy=WaveAnalyzerPolicyV2(
            analysis_policy_version="guarded-island-analysis-v2.0",
            required_result_policy_version=result_policy,
            require_control_arm=True,
            max_abort_fraction=0.25,
            max_non_success_fraction=0.5,
            min_candidate_seed_count=1,
            require_all_candidate_gates=True,
            require_eligible_candidate_for_planning=False,
        ),
        selection_policy=CandidateSelectionPolicyV2(
            selection_policy_version="guarded-island-pareto-v2.0",
            max_selected=1,
            min_selected=1,
            max_per_experiment=1,
            max_pareto_rank=1,
            min_normalized_objective_distance=0,
        ),
        planner_policy=WavePlannerPolicyV2(
            planner_policy_version="guarded-island-next-wave-v2.0",
            child_result_policy_version=result_policy,
            child_search_space_version="automation-island-v2",
            plan_mode=WavePlanMode.EVOLUTION_EXPERIMENT,
            budget=wave_budget,
            allow_control_fallback_when_no_candidates=True,
            allow_control_recovery_after_technical_failure=True,
            arm_templates=[
                WaveArmTemplateV2(
                    arm_id="control",
                    arm_type=ExperimentArmType.CONTROL,
                    source_mode=PlannerSourceMode.BASELINE_CONTROL,
                    hypothesis="Fresh paired Generic-Island control.",
                    primary_metric="worst_annualized_return_lcb",
                    seeds=[1001],
                ),
                WaveArmTemplateV2(
                    arm_id="replication",
                    arm_type=ExperimentArmType.REPLICATION,
                    source_mode=PlannerSourceMode.SELECTED_CANDIDATES,
                    hypothesis="Replicate search from the selected robust genome.",
                    primary_metric="worst_annualized_return_lcb",
                    seeds=[1001],
                ),
                WaveArmTemplateV2(
                    arm_id="explore",
                    arm_type=ExperimentArmType.EXPLORE,
                    source_mode=PlannerSourceMode.SELECTED_CANDIDATES,
                    hypothesis="Increase mutation around the selected robust genome.",
                    primary_metric="worst_annualized_return_lcb",
                    factor_delta={"genetic_algorithm.mutation_rate": 0.30},
                    seeds=[1001],
                ),
            ],
        ),
    )


def build_automation_preflight(
    config_path: str | Path,
    *,
    automation_root: str | Path,
    repo_root: str | Path,
    checked_at: datetime | None = None,
) -> AutomationPreflightV2:
    """Read-only proof that the real automation inputs and resources are ready."""

    repository = Path(repo_root).resolve()
    root = Path(automation_root).resolve()
    config = load_config(config_path)
    _validate_automation_config(config)
    policy = default_automation_policy(config, automation_root=root)
    promotion = shadow_gate_policy_from_config(config)
    pairs = sorted({item.pair for item in promotion.required_scenarios})
    data_root = repository / resolve_spot_data_root(config, pairs)
    data_manifest = build_data_manifest(config, promotion, data_root=data_root)
    split = build_evaluation_split_plan(config, promotion)
    code_manifest = capture_code_manifest(repository)
    disk_probe = root if root.exists() else root.parent
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    disk_free = shutil.disk_usage(disk_probe).free
    memory_available = _available_memory_bytes()
    venv_active = sys.prefix != sys.base_prefix
    reasons: list[str] = []
    if not venv_active:
        reasons.append("VIRTUALENV_NOT_ACTIVE")
    if disk_free <= policy.min_free_disk_bytes:
        reasons.append("INSUFFICIENT_FREE_DISK")
    if memory_available and memory_available <= policy.min_available_memory_bytes:
        reasons.append("INSUFFICIENT_AVAILABLE_MEMORY")
    if not memory_available:
        reasons.append("MEMORY_AVAILABILITY_UNPROVEN")
    if not reasons:
        reasons.append("PREFLIGHT_READY")
    return AutomationPreflightV2(
        checked_at=checked_at or datetime.now(UTC),
        ready=reasons == ["PREFLIGHT_READY"],
        config_hash=canonical_config_hash(config),
        code_manifest_hash=canonical_config_hash(
            code_manifest.model_dump(mode="json")
        ),
        data_manifest_hash=data_manifest.manifest_hash,
        split_hash=split.split_hash,
        automation_policy_hash=policy.policy_hash,
        python_executable=str(Path(sys.executable).absolute()),
        virtualenv_active=venv_active,
        disk_free_bytes=disk_free,
        disk_reserve_bytes=policy.min_free_disk_bytes,
        memory_available_bytes=memory_available,
        memory_reserve_bytes=policy.min_available_memory_bytes,
        evolution_pairs=split.evolution_pairs,
        validation_pairs=sorted(config["pair_validation"]["validation_pairs"]),
        reason_codes=reasons,
    )


def render_systemd_user_unit(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    state_path: str | Path,
    automation_root: str | Path,
) -> str:
    """Render a restart-on-crash user service without installing it."""

    repository = Path(repo_root).resolve()
    executable = (repository / ".venv/bin/python").absolute()
    config = Path(config_path).resolve()
    state = Path(state_path).resolve()
    artifacts = Path(automation_root).resolve()
    for path in (repository, executable, config):
        if not path.exists():
            raise AutomationControllerError(f"systemd unit input does not exist: {path}")

    def quote(value: str | Path) -> str:
        text = str(value)
        if "\n" in text or "\r" in text or "\x00" in text:
            raise AutomationControllerError("systemd path contains control characters")
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    working_directory = str(repository)
    if any(character in working_directory for character in ("\n", "\r", "\x00", "%")):
        raise AutomationControllerError(
            "systemd working directory contains unsupported characters"
        )

    command = " ".join(
        [
            quote(executable),
            "-m",
            "genetic_algorithm",
            "automation",
            "start",
            quote(config),
            "--state-db",
            quote(state),
            "--automation-root",
            quote(artifacts),
        ]
    )
    return (
        "[Unit]\n"
        "Description=Guarded Freqtrade GA V2 automation\n"
        "After=default.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={working_directory}\n"
        f"ExecStart={command}\n"
        "Restart=on-failure\n"
        "RestartPreventExitStatus=2\n"
        "RestartSec=30s\n"
        "TimeoutStopSec=30s\n"
        "KillMode=control-group\n"
        "NoNewPrivileges=true\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def bootstrap_automation_wave(
    config_path: str | Path,
    *,
    store: WaveStateStoreV2,
    policy: AutomationPolicyV2,
    automation_root: str | Path,
    repo_root: str | Path,
    created_at: datetime | None = None,
    python_executable: str | Path = sys.executable,
) -> AutomationBootstrapReceiptV2:
    """Materialize and register the deterministic root CONTROL wave."""

    root = Path(automation_root).resolve()
    repository = Path(repo_root).resolve()
    # Preserve a virtualenv launcher symlink.  Resolving it to /usr/bin/python
    # silently drops the venv's site-packages in the spawned worker.
    executable = Path(python_executable).absolute()
    config = load_config(config_path)
    _validate_automation_config(config)
    config_hash = canonical_config_hash(config)
    identity_hash = canonical_config_hash(
        {
            "config_hash": config_hash,
            "automation_policy_hash": policy.policy_hash,
            "seeds": policy.root_seeds,
        }
    )
    intent_path = root / "bootstrap_intent.json"
    if intent_path.exists():
        intent = AutomationBootstrapIntentV2.model_validate_json(
            intent_path.read_bytes()
        )
        if (
            intent.config_hash != config_hash
            or intent.automation_policy_hash != policy.policy_hash
            or intent.seeds != policy.root_seeds
        ):
            raise AutomationControllerError(
                "existing bootstrap intent differs from config or automation policy"
            )
    else:
        now = created_at or datetime.now(UTC)
        if now.utcoffset() is None:
            raise AutomationControllerError("bootstrap timestamp must be timezone-aware")
        intent = AutomationBootstrapIntentV2(
            created_at=now,
            config_hash=config_hash,
            automation_policy_hash=policy.policy_hash,
            seeds=policy.root_seeds,
            wave_id=f"wave-root-{identity_hash[:20]}",
            experiment_id=f"experiment-root-{identity_hash[:20]}",
        )
        _write_immutable(intent_path, _canonical_bytes(intent))

    receipt_path = root / "bootstrap_receipt.json"
    if receipt_path.exists():
        receipt = AutomationBootstrapReceiptV2.model_validate_json(
            receipt_path.read_bytes()
        )
        if receipt.intent != intent:
            raise AutomationControllerError("bootstrap receipt differs from intent")
    else:
        promotion = shadow_gate_policy_from_config(config)
        pairs = sorted({item.pair for item in promotion.required_scenarios})
        data_root = repository / resolve_spot_data_root(config, pairs)
        parallel = config["parallel_evaluation"]
        worker_count = (
            int(parallel.get("num_workers", 1))
            if parallel.get("enabled", False)
            else 1
        )
        top_n = int(config.get("output", {}).get("top_n", 5))
        wave_spec = WaveSpecV2(
            wave_id=intent.wave_id,
            policy_version=promotion.policy_version,
            search_space_version=policy.planner_policy.child_search_space_version,
            budget=WaveBudgetV2(
                max_attempts=len(intent.seeds),
                max_parallel=min(policy.max_concurrent, len(intent.seeds)),
                max_wallclock_seconds=policy.planner_policy.budget.max_wallclock_seconds,
            ),
            created_at=intent.created_at,
        )
        expectations: list[AttemptExpectationV2] = []
        attempts: list[MaterializedAttemptV2] = []
        for ordinal, seed in enumerate(intent.seeds):
            attempt_hash = canonical_config_hash(
                {"experiment_id": intent.experiment_id, "seed": seed}
            )
            attempt_id = f"attempt-{attempt_hash[:24]}"
            attempt_root = root / intent.wave_id / "attempts" / attempt_id
            bundle = build_attempt_manifest_bundle(
                attempt_id=attempt_id,
                wave_id=intent.wave_id,
                experiment_id=intent.experiment_id,
                created_at=intent.created_at,
                resolved_config=config,
                policy=promotion,
                fitness_policy_version=promotion.policy_version,
                seeds=[seed],
                worker_count=worker_count,
                artifact_root=attempt_root,
                repo_root=repository,
                data_root=data_root,
            )
            prepared = prepare_evolution_worker(
                bundle=bundle,
                resolved_config=config,
                policy=promotion,
                seeds=[],
                repo_root=repository,
                final_test_ledger_path=root / "final_test_usage.sqlite3",
                created_at=intent.created_at,
                top_n=top_n,
            )
            binding = WorkerBindingV2(
                worker_kind=WorkerKind(prepared.spec.worker_kind),
                argv=evolution_worker_argv(
                    prepared, python_executable=executable
                ),
                working_directory=str(repository),
                spec_path=str(prepared.spec_path.resolve()),
                spec_sha256=prepared.spec_file_sha256,
            )
            manifest_hash = canonical_config_hash(
                bundle.manifest.model_dump(mode="json")
            )
            expectations.append(
                AttemptExpectationV2(
                    attempt_id=attempt_id,
                    manifest_hash=manifest_hash,
                    seed=seed,
                    ordinal=ordinal,
                )
            )
            attempts.append(
                MaterializedAttemptV2(
                    manifest=bundle.manifest,
                    manifest_hash=manifest_hash,
                    worker_binding=binding,
                    worker_binding_hash=binding.binding_hash,
                )
            )
        experiment = ExperimentSpecV2(
            experiment_id=intent.experiment_id,
            wave_id=intent.wave_id,
            arm_type=ExperimentArmType.CONTROL,
            hypothesis="Guarded Generic-Island baseline search.",
            primary_metric="worst_annualized_return_lcb",
            seeds=intent.seeds,
            resolved_config_hash=config_hash,
            expected_attempts=expectations,
            created_at=intent.created_at,
        )
        receipt = AutomationBootstrapReceiptV2(
            intent=intent,
            wave_spec=wave_spec,
            experiment_spec=experiment,
            attempts=sorted(attempts, key=lambda item: item.manifest.attempt_id),
        )
        _write_immutable(receipt_path, _canonical_bytes(receipt))

    store.register_wave(receipt.wave_spec, actor="automation-bootstrap-v2")
    for index, item in enumerate(receipt.attempts):
        loaded = load_evolution_worker(
            item.worker_binding.spec_path,
            expected_spec_sha256=item.worker_binding.spec_sha256,
            # Bootstrap is idempotent and also runs after service restarts.
            # At that point a RUNNING/terminal attempt may legitimately have
            # immutable output artifacts below its root.  Input hashes still
            # have to match; only the pre-execution emptiness check is skipped.
            require_pristine_artifact_root=False,
        )
        if loaded.manifest != item.manifest:
            raise AutomationControllerError("bootstrap worker differs from receipt")
        state = store.register(item.manifest, actor="automation-bootstrap-v2")
        state = store.bind_worker(
            item.manifest.attempt_id,
            item.worker_binding,
            bound_at=intent.created_at + timedelta(microseconds=1 + index * 3),
            actor="automation-bootstrap-v2",
        )
        if state.status == AttemptLifecycleStatus.DRAFT:
            state = store.transition(
                item.manifest.attempt_id,
                AttemptLifecycleStatus.VALIDATED,
                occurred_at=intent.created_at
                + timedelta(microseconds=2 + index * 3),
                actor="automation-bootstrap-v2",
                reason="BOOTSTRAP_WORKER_VALID",
            )
        if state.status == AttemptLifecycleStatus.VALIDATED:
            store.transition(
                item.manifest.attempt_id,
                AttemptLifecycleStatus.QUEUED,
                occurred_at=intent.created_at
                + timedelta(microseconds=3 + index * 3),
                actor="automation-bootstrap-v2",
                reason="BOOTSTRAP_QUEUED",
            )
    store.register_experiment(receipt.experiment_spec)
    store.begin_collection(
        intent.wave_id,
        started_at=intent.created_at
        + timedelta(microseconds=4 + len(receipt.attempts) * 3),
        actor="automation-bootstrap-v2",
    )
    return receipt


class AutomationControllerV2:
    """Advance one linear automation lineage and execute its queued workers."""

    def __init__(
        self,
        *,
        store: WaveStateStoreV2,
        root_wave_id: str,
        policy: AutomationPolicyV2,
        automation_root: str | Path,
        repo_root: str | Path,
        scheduler: AttemptSchedulerV2 | None = None,
    ) -> None:
        self.store = store
        self.root_wave_id = root_wave_id
        self.policy = policy
        self.automation_root = Path(automation_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        self._owns_scheduler = scheduler is None
        self.scheduler = scheduler or AttemptSchedulerV2(
            store,
            scheduler_id="automation-controller-v2",
            config=ShadowSchedulerConfigV2(
                max_concurrent=policy.max_concurrent,
                poll_interval_seconds=policy.poll_interval_seconds,
                max_execution_seconds=policy.max_attempt_runtime_seconds,
            ),
        )

    def close(self) -> None:
        if self._owns_scheduler:
            self.scheduler.close(wait=True)

    def __enter__(self) -> AutomationControllerV2:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _lineage(self) -> list[WaveStateV2]:
        by_parent: dict[str, list[WaveStateV2]] = {}
        waves = {item.wave_id: item for item in self.store.list_waves()}
        if self.root_wave_id not in waves:
            raise AutomationControllerError("automation root wave is not registered")
        for wave in waves.values():
            if wave.parent_wave_id is not None:
                by_parent.setdefault(wave.parent_wave_id, []).append(wave)
        lineage = [waves[self.root_wave_id]]
        while lineage[-1].wave_id in by_parent:
            children = by_parent[lineage[-1].wave_id]
            if len(children) != 1:
                raise AutomationControllerError(
                    "automation lineage branches and cannot be advanced safely"
                )
            lineage.append(children[0])
        return lineage

    def _attempts(self, wave_id: str):
        return [
            item
            for item in self.store.list_attempts()
            if item.wave_id == wave_id
        ]

    def _artifact_bytes(self) -> int:
        if not self.automation_root.exists():
            return 0
        return sum(
            path.stat().st_size
            for path in self.automation_root.rglob("*")
            if path.is_file()
        )

    def _kill_switch_exists(self) -> bool:
        return Path(self.policy.kill_switch_path).is_file()

    def _analysis(self, wave: WaveStateV2) -> WaveAnalysisV2:
        if wave.status in {
            WaveLifecycleStatus.RECONCILED,
            WaveLifecycleStatus.ANALYZED,
        }:
            return WaveAnalyzerV2(
                self.store, self.policy.analyzer_policy
            ).build(wave.wave_id)
        decisions = self.store.decisions(wave.wave_id)
        analysis_decisions = [
            item for item in decisions if item.decision_type == WaveDecisionType.ANALYSIS
        ]
        if not analysis_decisions:
            raise AutomationControllerError("post-analysis wave lacks ANALYSIS decision")
        payload = analysis_decisions[-1].payload.get("analysis")
        return WaveAnalysisV2.model_validate(payload)

    def _resolved_configs(
        self, experiments: list[ExperimentSpecV2]
    ) -> dict[str, dict[str, Any]]:
        configs: dict[str, dict[str, Any]] = {}
        for experiment in experiments:
            attempt = self.store.get(experiment.expected_attempts[0].attempt_id)
            path = Path(attempt.artifact_root) / "resolved_config.yaml"
            raw = yaml.safe_load(path.read_text())
            if not isinstance(raw, dict):
                raise AutomationControllerError("parent resolved config is not a mapping")
            validate_resolved_config_v2_or_raise(raw)
            if canonical_config_hash(raw) != experiment.resolved_config_hash:
                raise AutomationControllerError(
                    "parent resolved config differs from experiment"
                )
            configs[experiment.resolved_config_hash] = raw
        return configs

    def _plan(self, wave: WaveStateV2) -> tuple[WaveAnalysisV2, ChildWavePlanV2]:
        analysis = self._analysis(wave)
        experiments = self.store.experiments(wave.wave_id)
        _, plan = select_and_plan_child_wave(
            analysis,
            self.policy.selection_policy,
            experiments,
            self._resolved_configs(experiments),
            self.policy.planner_policy,
        )
        return analysis, plan

    def _guard_plan(
        self,
        lineage: list[WaveStateV2],
        plan: ChildWavePlanV2,
    ) -> list[str]:
        reasons: list[str] = []
        if len(lineage) >= self.policy.max_waves:
            reasons.append("MAX_WAVES_REACHED")
        current_attempts = sum(len(self._attempts(item.wave_id)) for item in lineage)
        planned_attempts = sum(
            len(experiment.attempts) for experiment in plan.experiments
        )
        if current_attempts + planned_attempts > self.policy.max_total_attempts:
            reasons.append("MAX_TOTAL_ATTEMPTS_REACHED")
        if self._artifact_bytes() >= self.policy.max_artifact_bytes:
            reasons.append("MAX_ARTIFACT_BYTES_REACHED")
        consecutive_failed = 0
        for wave in reversed(lineage):
            attempts = self._attempts(wave.wave_id)
            if attempts and not any(
                item.status == AttemptLifecycleStatus.SUCCEEDED
                for item in attempts
            ):
                consecutive_failed += 1
            else:
                break
        if consecutive_failed >= self.policy.max_consecutive_failed_waves:
            reasons.append("MAX_CONSECUTIVE_FAILED_WAVES_REACHED")
        allowed_paths = set(self.policy.allowed_factor_paths)
        used_paths = {
            path
            for experiment in plan.experiments
            for path in experiment.factor_delta
        }
        if not used_paths.issubset(allowed_paths):
            reasons.append("UNAPPROVED_FACTOR_PATH")
        for experiment in plan.experiments:
            try:
                _validate_automation_config(experiment.resolved_config)
            except (AutomationControllerError, ValueError):
                reasons.append("UNSAFE_CHILD_CONFIG")
                break
        return sorted(set(reasons))

    def _block(
        self,
        wave: WaveStateV2,
        *,
        analysis: WaveAnalysisV2,
        reasons: list[str],
    ) -> WaveStateV2:
        analysis_decision = next(
            item
            for item in self.store.decisions(wave.wave_id)
            if item.decision_type == WaveDecisionType.ANALYSIS
        )
        return self.store.apply_decision(
            WaveDecisionV2(
                decision_id=f"block-{canonical_config_hash({'wave': wave.wave_id, 'reasons': reasons})[:24]}",
                wave_id=wave.wave_id,
                decision_type=WaveDecisionType.BLOCK,
                created_at=analysis.created_at + timedelta(microseconds=1),
                actor="guarded-automation-controller-v2",
                reason_codes=reasons,
                input_hash=analysis_decision.decision_hash,
                payload={"automation_policy_hash": self.policy.policy_hash},
            )
        )

    def _materialize(
        self,
        analysis: WaveAnalysisV2,
        plan: ChildWavePlanV2,
    ) -> PreparedChildWaveMaterializationV2:
        prepared = materialize_evolution_wave(
            plan,
            materialization_root=self.automation_root / "waves",
            repo_root=self.repo_root,
            final_test_ledger_path=self.automation_root / "final_test_usage.sqlite3",
            materialized_at=analysis.created_at + timedelta(microseconds=1),
        )
        if any(
            item.worker_binding.worker_kind
            != WorkerKind.GENERIC_ISLAND_EVOLUTION
            for item in prepared.materialization.attempts
        ):
            raise AutomationControllerError(
                "automation materialized a non-Generic-Island worker"
            )
        return prepared

    def run_once(self, *, observed_at: datetime | None = None) -> AutomationTickV2:
        now = observed_at or datetime.now(UTC)
        if self._kill_switch_exists():
            tip = self._lineage()[-1]
            return AutomationTickV2(
                observed_at=now,
                root_wave_id=self.root_wave_id,
                active_wave_id=tip.wave_id,
                active_wave_status=tip.status,
                outcome="STOPPED_KILL_SWITCH",
                reason_codes=["KILL_SWITCH_PRESENT"],
            )

        lineage = self._lineage()
        if (now - lineage[0].created_at).total_seconds() >= self.policy.max_runtime_seconds:
            tip = lineage[-1]
            return AutomationTickV2(
                observed_at=now,
                root_wave_id=self.root_wave_id,
                active_wave_id=tip.wave_id,
                active_wave_status=tip.status,
                outcome="STOPPED_LIMIT",
                reason_codes=["MAX_RUNTIME_REACHED"],
            )
        if self._artifact_bytes() >= self.policy.max_artifact_bytes:
            tip = lineage[-1]
            return AutomationTickV2(
                observed_at=now,
                root_wave_id=self.root_wave_id,
                active_wave_id=tip.wave_id,
                active_wave_status=tip.status,
                outcome="STOPPED_LIMIT",
                reason_codes=["MAX_ARTIFACT_BYTES_REACHED"],
            )
        disk_probe = (
            self.automation_root
            if self.automation_root.exists()
            else self.automation_root.parent
        )
        if (
            shutil.disk_usage(disk_probe).free
            <= self.policy.min_free_disk_bytes
        ):
            tip = lineage[-1]
            return AutomationTickV2(
                observed_at=now,
                root_wave_id=self.root_wave_id,
                active_wave_id=tip.wave_id,
                active_wave_status=tip.status,
                outcome="STOPPED_LIMIT",
                reason_codes=["MIN_FREE_DISK_RESERVE_REACHED"],
            )
        available_memory = _available_memory_bytes()
        if (
            available_memory
            and available_memory <= self.policy.min_available_memory_bytes
        ):
            tip = lineage[-1]
            return AutomationTickV2(
                observed_at=now,
                root_wave_id=self.root_wave_id,
                active_wave_id=tip.wave_id,
                active_wave_status=tip.status,
                outcome="WAITING",
                reason_codes=["MEMORY_RESERVE_ACTIVE"],
            )

        scheduler_tick = self.scheduler.run_once(observed_at=now)
        progressed = bool(
            scheduler_tick.launched_attempt_ids
            or scheduler_tick.completed_attempt_ids
            or scheduler_tick.reconciliation
        )
        lineage = self._lineage()
        wave = lineage[-1]

        if wave.status == WaveLifecycleStatus.DRAFT:
            wave = self.store.begin_collection(
                wave.wave_id,
                started_at=max(now, wave.updated_at),
                actor="guarded-automation-controller-v2",
            )
            progressed = True

        if wave.status == WaveLifecycleStatus.COLLECTING:
            attempts = self._attempts(wave.wave_id)
            if all(item.status in _TERMINAL_ATTEMPTS for item in attempts):
                for item in attempts:
                    if (
                        item.status != AttemptLifecycleStatus.SUCCEEDED
                        and item.result_path is None
                    ):
                        decision = WaveDecisionV2(
                            decision_id=f"abort-{item.attempt_id}",
                            wave_id=wave.wave_id,
                            decision_type=WaveDecisionType.ATTEMPT_ABORT,
                            created_at=max(now, item.updated_at, wave.updated_at),
                            actor="guarded-automation-controller-v2",
                            reason_codes=["TERMINAL_ATTEMPT_WITHOUT_RESULT"],
                            attempt_id=item.attempt_id,
                        )
                        self.store.apply_decision(decision)
                wave = self.store.reconcile_wave(
                    wave.wave_id,
                    reconciled_at=max(now, wave.updated_at),
                    actor="guarded-automation-controller-v2",
                )
                progressed = True

        if wave.status == WaveLifecycleStatus.RECONCILED:
            _, wave = WaveAnalyzerV2(
                self.store, self.policy.analyzer_policy
            ).analyze_and_record(wave.wave_id)
            progressed = True

        if wave.status in {
            WaveLifecycleStatus.ANALYZED,
            WaveLifecycleStatus.PROPOSED,
            WaveLifecycleStatus.APPROVED,
        }:
            analysis, plan = self._plan(wave)
            if not plan.planning_allowed:
                blocked = self.store.apply_decision(
                    plan.to_decision(
                        analysis_decision_hash=next(
                            item.decision_hash
                            for item in self.store.decisions(wave.wave_id)
                            if item.decision_type == WaveDecisionType.ANALYSIS
                        )
                    )
                )
                return AutomationTickV2(
                    observed_at=now,
                    root_wave_id=self.root_wave_id,
                    active_wave_id=blocked.wave_id,
                    active_wave_status=blocked.status,
                    outcome="BLOCKED",
                    reason_codes=plan.reason_codes,
                    launched_attempt_ids=scheduler_tick.launched_attempt_ids,
                )
            guard_reasons = (
                self._guard_plan(lineage, plan)
                if wave.status == WaveLifecycleStatus.ANALYZED
                else []
            )
            if guard_reasons:
                blocked = self._block(
                    wave, analysis=analysis, reasons=guard_reasons
                )
                return AutomationTickV2(
                    observed_at=now,
                    root_wave_id=self.root_wave_id,
                    active_wave_id=blocked.wave_id,
                    active_wave_status=blocked.status,
                    outcome="STOPPED_LIMIT",
                    reason_codes=guard_reasons,
                    launched_attempt_ids=scheduler_tick.launched_attempt_ids,
                )
            prepared = self._materialize(analysis, plan)
            analysis_decision = next(
                item
                for item in self.store.decisions(wave.wave_id)
                if item.decision_type == WaveDecisionType.ANALYSIS
            )
            proposal = prepared.materialization.to_proposal_decision(
                analysis_decision_hash=analysis_decision.decision_hash
            )
            if wave.status == WaveLifecycleStatus.ANALYZED:
                wave = self.store.apply_decision(proposal)
                progressed = True
            approval = prepared.materialization.to_approval_decision(
                proposal=proposal,
                approved_at=analysis.created_at + timedelta(microseconds=2),
                actor="guarded-automation-controller-v2",
                reason_codes=["SEARCH_ONLY_POLICY_APPROVED"],
            )
            if wave.status == WaveLifecycleStatus.PROPOSED:
                wave = self.store.apply_decision(approval)
                progressed = True
            if wave.status == WaveLifecycleStatus.APPROVED:
                queue_approved_materialization(
                    self.store,
                    prepared,
                    queued_at=analysis.created_at + timedelta(microseconds=3),
                    actor="guarded-automation-controller-v2",
                )
                wave = self._lineage()[-1]
                progressed = True

        if wave.status in {
            WaveLifecycleStatus.BLOCKED,
            WaveLifecycleStatus.REJECTED,
        }:
            outcome = "BLOCKED"
            reasons = ["WAVE_TERMINAL_BLOCK"]
        else:
            outcome = "PROGRESSED" if progressed else "WAITING"
            reasons = ["STATE_ADVANCED" if progressed else "ATTEMPTS_PENDING"]
        return AutomationTickV2(
            observed_at=now,
            root_wave_id=self.root_wave_id,
            active_wave_id=wave.wave_id,
            active_wave_status=wave.status,
            outcome=outcome,
            reason_codes=reasons,
            launched_attempt_ids=scheduler_tick.launched_attempt_ids,
        )

    def run_forever(self) -> AutomationTickV2:
        """Run until a guard stops the lineage or the process is interrupted."""

        while True:
            tick = self.run_once()
            if tick.outcome in {
                "STOPPED_KILL_SWITCH",
                "STOPPED_LIMIT",
                "BLOCKED",
            }:
                return tick
            time.sleep(self.policy.poll_interval_seconds)
