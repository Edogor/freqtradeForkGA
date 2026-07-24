"""Immutable materialization and atomic handoff for approved replay waves."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from genetic_algorithm.config.schema import validate_config
from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.attempt_state_v2 import WorkerBindingV2, WorkerKind
from genetic_algorithm.orchestration.data_manifest_v2 import resolve_spot_data_root
from genetic_algorithm.orchestration.evolution_worker_v2 import (
    FrozenEvolutionSeedV2,
    evolution_worker_argv,
    load_evolution_worker,
    prepare_evolution_worker,
)
from genetic_algorithm.orchestration.manifest_builder_v2 import build_attempt_manifest_bundle
from genetic_algorithm.orchestration.promotion_policy_v2 import shadow_gate_policy_from_config
from genetic_algorithm.orchestration.replay_runner_v2 import FrozenCandidateV2
from genetic_algorithm.orchestration.replay_worker_v2 import (
    load_replay_worker,
    prepare_replay_worker,
    replay_worker_argv,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptStatus,
    EvaluationStatus,
    ScenarioRole,
    StrictV2Model,
)
from genetic_algorithm.orchestration.wave_planner_v2 import (
    ChildWavePlanV2,
    PlannedExperimentV2,
    PlannerSourceMode,
    WavePlanMode,
)
from genetic_algorithm.orchestration.wave_state_v2 import (
    AttemptExpectationV2,
    ExperimentArmType,
    ExperimentSpecV2,
    WaveDecisionType,
    WaveDecisionV2,
    WaveSpecV2,
    WaveStateStoreV2,
    WaveStateV2,
)


class WaveMaterializationError(ValueError):
    """Raised when a plan cannot be converted to immutable worker inputs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_model_bytes(model: StrictV2Model) -> bytes:
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
        if path.read_bytes() == payload:
            return
        raise WaveMaterializationError(f"immutable materialization already differs: {path}")
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
                raise WaveMaterializationError(
                    f"concurrent materialization write differs: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


class VerifiedCandidateSourceV2(StrictV2Model):
    parent_attempt_id: str = Field(min_length=1)
    parent_experiment_id: str = Field(min_length=1)
    parent_candidate_id: str = Field(min_length=1)
    parent_result_path: str = Field(min_length=1)
    parent_result_sha256: str = Field(min_length=64, max_length=64)
    parent_config_hash: str = Field(min_length=64, max_length=64)
    seed: int
    phenotype_hash: str = Field(min_length=64, max_length=64)
    frozen_candidate_path: str = Field(min_length=1)
    frozen_candidate_sha256: str = Field(min_length=64, max_length=64)
    frozen_candidate: FrozenCandidateV2
    evolution_seed_path: str | None = None
    evolution_seed_sha256: str | None = None
    evolution_seed: FrozenEvolutionSeedV2 | None = None

    @model_validator(mode="after")
    def _consistent_source(self) -> VerifiedCandidateSourceV2:
        for field_name in ("parent_result_path", "frozen_candidate_path"):
            if not Path(getattr(self, field_name)).is_absolute():
                raise ValueError(f"{field_name} must be absolute")
        if self.frozen_candidate.candidate_id != self.parent_candidate_id:
            raise ValueError("frozen candidate ID differs from parent candidate")
        if self.frozen_candidate.phenotype_hash != self.phenotype_hash:
            raise ValueError("frozen candidate phenotype differs from selected source")
        evolution_fields = (
            self.evolution_seed_path,
            self.evolution_seed_sha256,
            self.evolution_seed,
        )
        if any(value is not None for value in evolution_fields) and any(
            value is None for value in evolution_fields
        ):
            raise ValueError("evolution source fields must be set together")
        if self.evolution_seed is not None:
            if not Path(str(self.evolution_seed_path)).is_absolute():
                raise ValueError("evolution_seed_path must be absolute")
            if len(str(self.evolution_seed_sha256)) != 64:
                raise ValueError("evolution_seed_sha256 must be SHA-256")
            if (
                self.evolution_seed.candidate_id != self.parent_candidate_id
                or self.evolution_seed.phenotype_hash != self.phenotype_hash
                or self.evolution_seed.frozen_candidate != self.frozen_candidate
            ):
                raise ValueError("evolution seed differs from selected frozen candidate")
        return self


class MaterializedAttemptV2(StrictV2Model):
    manifest: AttemptManifestV2
    manifest_hash: str = Field(min_length=64, max_length=64)
    worker_binding: WorkerBindingV2
    worker_binding_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _consistent_attempt(self) -> MaterializedAttemptV2:
        if canonical_config_hash(self.manifest.model_dump(mode="json")) != self.manifest_hash:
            raise ValueError("materialized manifest hash differs from content")
        if self.worker_binding.binding_hash != self.worker_binding_hash:
            raise ValueError("materialized worker binding hash differs from content")
        if self.worker_binding.worker_kind not in {
            WorkerKind.SHADOW_REPLAY,
            WorkerKind.STANDARD_EVOLUTION,
            WorkerKind.GENERIC_ISLAND_EVOLUTION,
        }:
            raise ValueError("materialized attempt uses an unsupported worker")
        return self


class MaterializedExperimentV2(StrictV2Model):
    experiment_spec: ExperimentSpecV2
    verified_source: VerifiedCandidateSourceV2 | None = None
    config_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _canonical_warnings(self) -> MaterializedExperimentV2:
        if self.config_warnings != sorted(set(self.config_warnings)):
            raise ValueError("config warnings must be unique and sorted")
        return self


class ChildWaveMaterializationV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    plan: ChildWavePlanV2
    plan_hash: str = Field(min_length=64, max_length=64)
    materialized_at: datetime
    materialization_root: str = Field(min_length=1)
    child_wave_spec: WaveSpecV2
    experiments: list[MaterializedExperimentV2] = Field(min_length=1)
    attempts: list[MaterializedAttemptV2] = Field(min_length=1)
    reason_codes: list[str] = Field(
        default_factory=lambda: ["MATERIALIZATION_COMPLETE"]
    )

    @model_validator(mode="after")
    def _consistent_materialization(self) -> ChildWaveMaterializationV2:
        if self.materialized_at.utcoffset() is None:
            raise ValueError("materialized_at must be timezone-aware")
        if not Path(self.materialization_root).is_absolute():
            raise ValueError("materialization_root must be absolute")
        if self.plan.plan_hash != self.plan_hash:
            raise ValueError("materialized plan hash differs from content")
        if not self.plan.planning_allowed:
            raise ValueError("blocked plan cannot be materialized")
        if self.child_wave_spec.wave_id != self.plan.wave_id:
            raise ValueError("materialized child wave differs from plan")
        if self.child_wave_spec.parent_wave_id != self.plan.parent_wave_id:
            raise ValueError("materialized parent differs from plan")
        if (
            self.child_wave_spec.policy_version != self.plan.result_policy_version
            or self.child_wave_spec.search_space_version != self.plan.search_space_version
            or self.child_wave_spec.budget != self.plan.budget
        ):
            raise ValueError("materialized child wave policy or budget differs from plan")
        if self.reason_codes != ["MATERIALIZATION_COMPLETE"]:
            raise ValueError("completed materialization needs canonical reason code")
        ordered_experiments = sorted(
            self.experiments, key=lambda item: item.experiment_spec.experiment_id
        )
        if self.experiments != ordered_experiments:
            raise ValueError("materialized experiments must be sorted")
        ordered_attempts = sorted(self.attempts, key=lambda item: item.manifest.attempt_id)
        if self.attempts != ordered_attempts:
            raise ValueError("materialized attempts must be sorted")
        self._validate_plan_projection()
        return self

    def _validate_plan_projection(self) -> None:
        planned = {item.experiment_id: item for item in self.plan.experiments}
        materialized = {item.experiment_spec.experiment_id: item for item in self.experiments}
        if set(planned) != set(materialized):
            raise ValueError("materialized experiments differ from plan")
        attempt_map = {item.manifest.attempt_id: item for item in self.attempts}
        if len(attempt_map) != len(self.attempts):
            raise ValueError("materialized attempt IDs must be unique")
        expected_attempt_ids: set[str] = set()
        for experiment_id, planned_experiment in planned.items():
            materialized_experiment = materialized[experiment_id]
            self._validate_materialized_source(
                planned_experiment,
                materialized_experiment,
            )
            spec = materialized_experiment.experiment_spec
            if (
                spec.arm_type != planned_experiment.arm_type
                or spec.factor_delta != planned_experiment.factor_delta
                or spec.seeds != planned_experiment.seeds
                or spec.resolved_config_hash != planned_experiment.resolved_config_hash
            ):
                raise ValueError("materialized experiment differs from planned experiment")
            expected_attempt_ids.update(item.attempt_id for item in planned_experiment.attempts)
            for expectation in spec.expected_attempts:
                attempt = attempt_map.get(expectation.attempt_id)
                if attempt is None or attempt.manifest_hash != expectation.manifest_hash:
                    raise ValueError("materialized attempt differs from experiment expectation")
        if set(attempt_map) != expected_attempt_ids:
            raise ValueError("materialized attempts differ from plan")
        root = Path(self.materialization_root).resolve()
        for attempt_id, attempt in attempt_map.items():
            expected_root = root / "attempts" / attempt_id
            if Path(attempt.manifest.artifact_root).resolve() != expected_root:
                raise ValueError("materialized attempt root is not canonical")
            if Path(attempt.worker_binding.spec_path).resolve() != (
                expected_root / V2ArtifactStore.WORKER_SPEC_NAME
            ):
                raise ValueError("materialized worker spec path is not canonical")

    def _validate_materialized_source(
        self,
        planned: PlannedExperimentV2,
        materialized: MaterializedExperimentV2,
    ) -> None:
        baseline = planned.source.source_mode == PlannerSourceMode.BASELINE_CONTROL
        if baseline != (materialized.verified_source is None):
            raise ValueError("materialized source differs from planned source mode")
        if self.plan.planner_policy.plan_mode != WavePlanMode.EVOLUTION_EXPERIMENT:
            return
        if (
            materialized.verified_source is not None
            and materialized.verified_source.evolution_seed is None
        ):
            raise ValueError("selected evolution source lacks an immutable genome")

    @property
    def materialization_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))

    def to_proposal_decision(self, *, analysis_decision_hash: str) -> WaveDecisionV2:
        if len(analysis_decision_hash) != 64:
            raise WaveMaterializationError("analysis_decision_hash must be SHA-256")
        return WaveDecisionV2(
            decision_id=f"proposal-{self.materialization_hash[:24]}",
            wave_id=self.plan.parent_wave_id,
            decision_type=WaveDecisionType.PROPOSAL,
            created_at=self.materialized_at,
            actor="wave-materializer-v2",
            reason_codes=["MATERIALIZED_PLAN_READY"],
            input_hash=analysis_decision_hash,
            payload={
                "child_wave_id": self.plan.wave_id,
                "plan_hash": self.plan_hash,
                "materialization_hash": self.materialization_hash,
                "planning_allowed": True,
            },
        )

    def to_approval_decision(
        self,
        *,
        proposal: WaveDecisionV2,
        approved_at: datetime,
        actor: str,
        reason_codes: list[str],
    ) -> WaveDecisionV2:
        if (
            proposal.decision_type != WaveDecisionType.PROPOSAL
            or proposal.wave_id != self.plan.parent_wave_id
            or proposal.payload.get("plan_hash") != self.plan_hash
            or proposal.payload.get("materialization_hash") != self.materialization_hash
        ):
            raise WaveMaterializationError("proposal differs from this materialization")
        return WaveDecisionV2(
            decision_id=f"approval-{proposal.decision_hash[:24]}",
            wave_id=proposal.wave_id,
            decision_type=WaveDecisionType.APPROVAL,
            created_at=approved_at,
            actor=actor,
            reason_codes=reason_codes,
            input_hash=proposal.decision_hash,
            payload={
                "approved_plan_hash": self.plan_hash,
                "approved_materialization_hash": self.materialization_hash,
            },
        )


@dataclass(frozen=True)
class PreparedChildWaveMaterializationV2:
    materialization: ChildWaveMaterializationV2
    receipt_path: Path
    receipt_file_sha256: str


def _validate_replay_config(experiment: PlannedExperimentV2) -> list[str]:
    config = experiment.resolved_config
    errors, warnings = validate_config(config)
    if errors:
        raise WaveMaterializationError(
            f"config validation failed for {experiment.experiment_id}: {'; '.join(errors)}"
        )
    safety = config.get("safety_profile", {})
    evaluation = config.get("evaluation_v2", {})
    backtesting = config.get("backtesting", {})
    if not safety.get("shadow_mode", False) or safety.get("automation_eligible", False):
        raise WaveMaterializationError("replay materialization requires non-eligible shadow_mode")
    if not evaluation.get("enabled", False) or not evaluation.get("mark_to_market", False):
        raise WaveMaterializationError("replay materialization requires mark-to-market V2")
    if backtesting.get("fee_noise_std", 0.0) != 0.0:
        raise WaveMaterializationError("replay materialization forbids fee noise")
    if backtesting.get("dynamic_slippage", False):
        raise WaveMaterializationError("replay materialization forbids dynamic slippage")
    if float(backtesting.get("spread_pct", 0.0) or 0.0) != 0.0:
        raise WaveMaterializationError("replay materialization forbids untracked spread")
    if float(backtesting.get("funding_rate", 0.0) or 0.0) != 0.0:
        raise WaveMaterializationError("replay materialization forbids untracked funding")
    if config.get("short_selling", {}).get("enabled", False):
        raise WaveMaterializationError("replay materialization currently supports spot-long only")
    shadow_gate_policy_from_config(config)
    return sorted(set(warnings))


def _verify_candidate_source(experiment: PlannedExperimentV2) -> VerifiedCandidateSourceV2:
    source = experiment.source
    reference = source.observation_ref
    if source.source_mode != PlannerSourceMode.SELECTED_CANDIDATES or reference is None:
        raise WaveMaterializationError("replay validation lacks a selected candidate source")
    result_path = Path(reference.result_path).resolve()
    if result_path.name != V2ArtifactStore.RESULT_NAME or not result_path.is_file():
        raise WaveMaterializationError("selected source result path is missing or not canonical")
    if _sha256_file(result_path) != reference.result_sha256:
        raise ArtifactIntegrityError("selected source result SHA-256 differs from analysis")
    result = V2ArtifactStore(result_path.parent).read_verified_result()
    if (
        result.status != AttemptStatus.SUCCEEDED
        or result.manifest.attempt_id != reference.attempt_id
        or result.manifest.experiment_id != source.parent_experiment_id
        or result.manifest.config_hash != source.parent_config_hash
        or result.manifest.seeds != [reference.seed]
    ):
        raise WaveMaterializationError("selected source result provenance differs from plan")
    evaluations = [
        item for item in result.candidate_evaluations if item.candidate_id == reference.candidate_id
    ]
    if len(evaluations) != 1:
        raise WaveMaterializationError(
            "selected candidate is absent or duplicated in parent result"
        )
    evaluation = evaluations[0]
    if (
        evaluation.status != EvaluationStatus.VALID
        or evaluation.phenotype_hash != source.phenotype_hash
    ):
        raise WaveMaterializationError("selected candidate is not a valid matching phenotype")
    relative = f"candidates/{reference.candidate_id}/frozen_candidate.json"
    expected_frozen_hash = result.artifact_hashes.get(relative)
    frozen_path = (result_path.parent / relative).resolve()
    if expected_frozen_hash is None or not frozen_path.is_file():
        raise WaveMaterializationError("selected candidate lacks a frozen executable artifact")
    actual_frozen_hash = _sha256_file(frozen_path)
    if actual_frozen_hash != expected_frozen_hash:
        raise ArtifactIntegrityError("selected frozen candidate hash differs from parent result")
    try:
        frozen = FrozenCandidateV2.model_validate_json(frozen_path.read_bytes())
    except Exception as exc:
        raise WaveMaterializationError("selected frozen candidate violates its contract") from exc
    return VerifiedCandidateSourceV2(
        parent_attempt_id=reference.attempt_id,
        parent_experiment_id=source.parent_experiment_id,
        parent_candidate_id=reference.candidate_id,
        parent_result_path=str(result_path),
        parent_result_sha256=reference.result_sha256,
        parent_config_hash=source.parent_config_hash,
        seed=reference.seed,
        phenotype_hash=str(source.phenotype_hash),
        frozen_candidate_path=str(frozen_path),
        frozen_candidate_sha256=actual_frozen_hash,
        frozen_candidate=frozen,
    )


def _verify_evolution_source(experiment: PlannedExperimentV2) -> VerifiedCandidateSourceV2:
    source = _verify_candidate_source(experiment)
    result_root = Path(source.parent_result_path).parent
    relative = f"candidates/{source.parent_candidate_id}/evolution_seed.json"
    result = V2ArtifactStore(result_root).read_verified_result()
    expected_hash = result.artifact_hashes.get(relative)
    seed_path = (result_root / relative).resolve()
    if expected_hash is None or not seed_path.is_file():
        raise WaveMaterializationError(
            "selected candidate has no immutable evolution seed; "
            "legacy/replay-only candidates cannot warm-start an evolution arm"
        )
    actual_hash = _sha256_file(seed_path)
    if actual_hash != expected_hash:
        raise ArtifactIntegrityError("selected evolution seed hash differs from parent result")
    try:
        seed = FrozenEvolutionSeedV2.model_validate_json(seed_path.read_bytes())
    except Exception as exc:
        raise WaveMaterializationError("selected evolution seed violates its contract") from exc
    payload = source.model_dump(mode="python")
    payload.update(
        {
            "evolution_seed_path": str(seed_path),
            "evolution_seed_sha256": actual_hash,
            "evolution_seed": seed,
        }
    )
    return VerifiedCandidateSourceV2.model_validate(payload)


def _validate_evolution_config(experiment: PlannedExperimentV2) -> list[str]:
    warnings = _validate_replay_config(experiment)
    policy = shadow_gate_policy_from_config(experiment.resolved_config)
    if any(item.role == ScenarioRole.FINAL_TEST for item in policy.required_scenarios):
        raise WaveMaterializationError(
            "evolution waves cannot expose FINAL_TEST cells; schedule final replay separately"
        )
    parallel = experiment.resolved_config.get("parallel_evaluation", {})
    workers = parallel.get("num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise WaveMaterializationError(
            "evolution materialization requires explicit parallel_evaluation.num_workers >= 1"
        )
    top_n = experiment.resolved_config.get("output", {}).get("top_n")
    if (
        not isinstance(top_n, int)
        or isinstance(top_n, bool)
        or not 1 <= top_n <= 10
    ):
        raise WaveMaterializationError(
            "evolution materialization requires output.top_n between 1 and 10"
        )
    if experiment.resolved_config.get("genetic_algorithm", {}).get("mode") != "single_objective":
        raise WaveMaterializationError("canonical evolution currently supports single_objective")
    return warnings


def _reject_reused_final_test_cells(
    preflight: list[tuple[PlannedExperimentV2, list[str], VerifiedCandidateSourceV2]],
) -> None:
    seen: dict[tuple[str, str, str, str, str], str] = {}
    for experiment, _, _ in preflight:
        policy = shadow_gate_policy_from_config(experiment.resolved_config)
        exchange = str(experiment.resolved_config["backtesting"]["exchange"])
        final_cells = [
            (
                exchange,
                item.pair,
                item.timeframe,
                str(item.period_start),
                str(item.period_end),
            )
            for item in policy.required_scenarios
            if item.role == ScenarioRole.FINAL_TEST
        ]
        for attempt in experiment.attempts:
            for cell in final_cells:
                previous = seen.setdefault(cell, attempt.attempt_id)
                if previous != attempt.attempt_id:
                    raise WaveMaterializationError(
                        "FINAL_TEST cell would be exposed by multiple attempts; "
                        "group candidates into one replay attempt or declare a non-final panel"
                    )


def materialize_replay_validation_wave(
    plan: ChildWavePlanV2,
    *,
    materialization_root: str | Path,
    repo_root: str | Path,
    final_test_ledger_path: str | Path,
    materialized_at: datetime,
    python_executable: str | Path = sys.executable,
) -> PreparedChildWaveMaterializationV2:
    """Validate, freeze, and commit every input needed by a replay child wave."""

    if not plan.planning_allowed:
        raise WaveMaterializationError("blocked plan cannot be materialized")
    if plan.planner_policy.plan_mode != WavePlanMode.REPLAY_VALIDATION:
        raise WaveMaterializationError(
            "GA evolution worker is not implemented; only REPLAY_VALIDATION is materializable"
        )
    if materialized_at.utcoffset() is None or materialized_at < plan.created_at:
        raise WaveMaterializationError("materialized_at must be aware and follow plan creation")
    resolved_repo = Path(repo_root).resolve()
    # Keep virtualenv launcher symlinks intact for spawned workers.
    executable = Path(python_executable).absolute()
    ledger_path = Path(final_test_ledger_path).resolve()
    if not resolved_repo.is_dir() or not executable.is_file():
        raise WaveMaterializationError("repo root or Python executable does not exist")

    preflight: list[tuple[PlannedExperimentV2, list[str], VerifiedCandidateSourceV2]] = []
    for experiment in plan.experiments:
        if experiment.arm_type != ExperimentArmType.VALIDATION:
            raise WaveMaterializationError("replay materialization only supports VALIDATION arms")
        warnings = _validate_replay_config(experiment)
        source = _verify_candidate_source(experiment)
        preflight.append((experiment, warnings, source))
    _reject_reused_final_test_cells(preflight)

    wave_root = (Path(materialization_root).resolve() / plan.wave_id).resolve()
    child_wave = WaveSpecV2(
        wave_id=plan.wave_id,
        parent_wave_id=plan.parent_wave_id,
        policy_version=plan.result_policy_version,
        search_space_version=plan.search_space_version,
        budget=plan.budget,
        created_at=materialized_at,
    )
    materialized_attempts: list[MaterializedAttemptV2] = []
    materialized_experiments: list[MaterializedExperimentV2] = []

    for experiment, warnings, source in preflight:
        policy = shadow_gate_policy_from_config(experiment.resolved_config)
        pairs = sorted({item.pair for item in policy.required_scenarios})
        data_root = resolved_repo / resolve_spot_data_root(experiment.resolved_config, pairs)
        expectations: list[AttemptExpectationV2] = []
        for planned_attempt in experiment.attempts:
            attempt_root = wave_root / "attempts" / planned_attempt.attempt_id
            bundle = build_attempt_manifest_bundle(
                attempt_id=planned_attempt.attempt_id,
                wave_id=plan.wave_id,
                parent_wave_id=plan.parent_wave_id,
                experiment_id=experiment.experiment_id,
                created_at=materialized_at,
                resolved_config=experiment.resolved_config,
                policy=policy,
                fitness_policy_version=plan.result_policy_version,
                seeds=[planned_attempt.seed],
                worker_count=1,
                artifact_root=attempt_root,
                repo_root=resolved_repo,
                data_root=data_root,
            )
            prepared = prepare_replay_worker(
                bundle=bundle,
                resolved_config=experiment.resolved_config,
                policy=policy,
                candidates=[source.frozen_candidate],
                repo_root=resolved_repo,
                final_test_ledger_path=ledger_path,
                created_at=materialized_at,
            )
            binding = WorkerBindingV2(
                worker_kind=WorkerKind.SHADOW_REPLAY,
                argv=replay_worker_argv(prepared, python_executable=executable),
                working_directory=str(resolved_repo),
                spec_path=str(prepared.spec_path.resolve()),
                spec_sha256=prepared.spec_file_sha256,
            )
            manifest_hash = canonical_config_hash(bundle.manifest.model_dump(mode="json"))
            expectations.append(
                AttemptExpectationV2(
                    attempt_id=planned_attempt.attempt_id,
                    manifest_hash=manifest_hash,
                    seed=planned_attempt.seed,
                    ordinal=planned_attempt.ordinal,
                )
            )
            materialized_attempts.append(
                MaterializedAttemptV2(
                    manifest=bundle.manifest,
                    manifest_hash=manifest_hash,
                    worker_binding=binding,
                    worker_binding_hash=binding.binding_hash,
                )
            )
        spec = ExperimentSpecV2(
            experiment_id=experiment.experiment_id,
            wave_id=plan.wave_id,
            arm_type=experiment.arm_type,
            hypothesis=experiment.hypothesis,
            primary_metric=experiment.primary_metric,
            factor_delta=experiment.factor_delta,
            seeds=experiment.seeds,
            resolved_config_hash=experiment.resolved_config_hash,
            expected_attempts=expectations,
            created_at=materialized_at,
        )
        materialized_experiments.append(
            MaterializedExperimentV2(
                experiment_spec=spec,
                verified_source=source,
                config_warnings=warnings,
            )
        )

    materialization = ChildWaveMaterializationV2(
        plan=plan,
        plan_hash=plan.plan_hash,
        materialized_at=materialized_at,
        materialization_root=str(wave_root),
        child_wave_spec=child_wave,
        experiments=sorted(
            materialized_experiments,
            key=lambda item: item.experiment_spec.experiment_id,
        ),
        attempts=sorted(materialized_attempts, key=lambda item: item.manifest.attempt_id),
    )
    receipt_path = wave_root / "materialization.json"
    _write_immutable(receipt_path, _canonical_model_bytes(materialization))
    return PreparedChildWaveMaterializationV2(
        materialization=materialization,
        receipt_path=receipt_path,
        receipt_file_sha256=_sha256_file(receipt_path),
    )


def materialize_evolution_wave(
    plan: ChildWavePlanV2,
    *,
    materialization_root: str | Path,
    repo_root: str | Path,
    final_test_ledger_path: str | Path,
    materialized_at: datetime,
    python_executable: str | Path = sys.executable,
) -> PreparedChildWaveMaterializationV2:
    """Freeze Control/Replication/Exploit/Explore standard-evolution workers."""

    if not plan.planning_allowed:
        raise WaveMaterializationError("blocked plan cannot be materialized")
    if plan.planner_policy.plan_mode != WavePlanMode.EVOLUTION_EXPERIMENT:
        raise WaveMaterializationError("plan is not an EVOLUTION_EXPERIMENT")
    if materialized_at.utcoffset() is None or materialized_at < plan.created_at:
        raise WaveMaterializationError("materialized_at must be aware and follow plan creation")
    resolved_repo = Path(repo_root).resolve()
    # Keep virtualenv launcher symlinks intact for spawned workers.
    executable = Path(python_executable).absolute()
    ledger_path = Path(final_test_ledger_path).resolve()
    if not resolved_repo.is_dir() or not executable.is_file():
        raise WaveMaterializationError("repo root or Python executable does not exist")

    supported_arms = {
        ExperimentArmType.CONTROL,
        ExperimentArmType.REPLICATION,
        ExperimentArmType.EXPLOIT,
        ExperimentArmType.EXPLORE,
    }
    preflight: list[
        tuple[PlannedExperimentV2, list[str], VerifiedCandidateSourceV2 | None]
    ] = []
    for experiment in plan.experiments:
        if experiment.arm_type not in supported_arms:
            raise WaveMaterializationError(
                f"unsupported evolution arm: {experiment.arm_type.value}"
            )
        warnings = _validate_evolution_config(experiment)
        source = (
            None
            if experiment.source.source_mode == PlannerSourceMode.BASELINE_CONTROL
            else _verify_evolution_source(experiment)
        )
        preflight.append((experiment, warnings, source))

    wave_root = (Path(materialization_root).resolve() / plan.wave_id).resolve()
    child_wave = WaveSpecV2(
        wave_id=plan.wave_id,
        parent_wave_id=plan.parent_wave_id,
        policy_version=plan.result_policy_version,
        search_space_version=plan.search_space_version,
        budget=plan.budget,
        created_at=materialized_at,
    )
    materialized_attempts: list[MaterializedAttemptV2] = []
    materialized_experiments: list[MaterializedExperimentV2] = []

    for experiment, warnings, source in preflight:
        policy = shadow_gate_policy_from_config(experiment.resolved_config)
        pairs = sorted({item.pair for item in policy.required_scenarios})
        data_root = resolved_repo / resolve_spot_data_root(experiment.resolved_config, pairs)
        worker_count = int(experiment.resolved_config["parallel_evaluation"]["num_workers"])
        top_n = int(experiment.resolved_config.get("output", {}).get("top_n", 5))
        expectations: list[AttemptExpectationV2] = []
        for planned_attempt in experiment.attempts:
            attempt_root = wave_root / "attempts" / planned_attempt.attempt_id
            bundle = build_attempt_manifest_bundle(
                attempt_id=planned_attempt.attempt_id,
                wave_id=plan.wave_id,
                parent_wave_id=plan.parent_wave_id,
                experiment_id=experiment.experiment_id,
                created_at=materialized_at,
                resolved_config=experiment.resolved_config,
                policy=policy,
                fitness_policy_version=plan.result_policy_version,
                seeds=[planned_attempt.seed],
                worker_count=worker_count,
                artifact_root=attempt_root,
                repo_root=resolved_repo,
                data_root=data_root,
            )
            prepared = prepare_evolution_worker(
                bundle=bundle,
                resolved_config=experiment.resolved_config,
                policy=policy,
                seeds=[] if source is None else [source.evolution_seed],
                repo_root=resolved_repo,
                final_test_ledger_path=ledger_path,
                created_at=materialized_at,
                top_n=top_n,
            )
            binding = WorkerBindingV2(
                worker_kind=WorkerKind(prepared.spec.worker_kind),
                argv=evolution_worker_argv(prepared, python_executable=executable),
                working_directory=str(resolved_repo),
                spec_path=str(prepared.spec_path.resolve()),
                spec_sha256=prepared.spec_file_sha256,
            )
            manifest_hash = canonical_config_hash(bundle.manifest.model_dump(mode="json"))
            expectations.append(
                AttemptExpectationV2(
                    attempt_id=planned_attempt.attempt_id,
                    manifest_hash=manifest_hash,
                    seed=planned_attempt.seed,
                    ordinal=planned_attempt.ordinal,
                )
            )
            materialized_attempts.append(
                MaterializedAttemptV2(
                    manifest=bundle.manifest,
                    manifest_hash=manifest_hash,
                    worker_binding=binding,
                    worker_binding_hash=binding.binding_hash,
                )
            )
        spec = ExperimentSpecV2(
            experiment_id=experiment.experiment_id,
            wave_id=plan.wave_id,
            arm_type=experiment.arm_type,
            hypothesis=experiment.hypothesis,
            primary_metric=experiment.primary_metric,
            factor_delta=experiment.factor_delta,
            seeds=experiment.seeds,
            resolved_config_hash=experiment.resolved_config_hash,
            expected_attempts=expectations,
            created_at=materialized_at,
        )
        materialized_experiments.append(
            MaterializedExperimentV2(
                experiment_spec=spec,
                verified_source=source,
                config_warnings=warnings,
            )
        )

    materialization = ChildWaveMaterializationV2(
        plan=plan,
        plan_hash=plan.plan_hash,
        materialized_at=materialized_at,
        materialization_root=str(wave_root),
        child_wave_spec=child_wave,
        experiments=sorted(
            materialized_experiments,
            key=lambda item: item.experiment_spec.experiment_id,
        ),
        attempts=sorted(materialized_attempts, key=lambda item: item.manifest.attempt_id),
    )
    receipt_path = wave_root / "materialization.json"
    _write_immutable(receipt_path, _canonical_model_bytes(materialization))
    return PreparedChildWaveMaterializationV2(
        materialization=materialization,
        receipt_path=receipt_path,
        receipt_file_sha256=_sha256_file(receipt_path),
    )


def queue_approved_materialization(
    store: WaveStateStoreV2,
    prepared: PreparedChildWaveMaterializationV2,
    *,
    queued_at: datetime,
    actor: str = "materialized-wave-controller-v2",
) -> WaveStateV2:
    """Re-verify the receipt and workers, then perform one SQLite handoff."""

    expected_receipt = (
        Path(prepared.materialization.materialization_root).resolve()
        / "materialization.json"
    )
    if prepared.receipt_path.resolve() != expected_receipt or not expected_receipt.is_file():
        raise ArtifactIntegrityError("materialization receipt path is not canonical")
    if _sha256_file(prepared.receipt_path) != prepared.receipt_file_sha256:
        raise ArtifactIntegrityError("materialization receipt SHA-256 differs before queueing")
    try:
        persisted = ChildWaveMaterializationV2.model_validate_json(
            prepared.receipt_path.read_bytes()
        )
    except Exception as exc:
        raise WaveMaterializationError("materialization receipt violates its contract") from exc
    if persisted != prepared.materialization:
        raise ArtifactIntegrityError("materialization receipt differs from prepared content")
    persisted_receipt = (
        Path(persisted.materialization_root).resolve() / "materialization.json"
    )
    if prepared.receipt_path.resolve() != persisted_receipt:
        raise ArtifactIntegrityError("materialization receipt path is not canonical")
    for item in persisted.attempts:
        binding = item.worker_binding
        if binding.worker_kind == WorkerKind.SHADOW_REPLAY:
            loaded = load_replay_worker(
                binding.spec_path,
                expected_spec_sha256=binding.spec_sha256,
            )
        elif binding.worker_kind in {
            WorkerKind.STANDARD_EVOLUTION,
            WorkerKind.GENERIC_ISLAND_EVOLUTION,
        }:
            loaded = load_evolution_worker(
                binding.spec_path,
                expected_spec_sha256=binding.spec_sha256,
            )
        else:  # pragma: no cover - WorkerKind/MaterializedAttemptV2 fence this
            raise WaveMaterializationError("materialization contains an unknown worker")
        if loaded.manifest != item.manifest:
            raise ArtifactIntegrityError("materialized worker manifest changed before queueing")
    return store.queue_materialized_child_wave(
        parent_wave_id=persisted.plan.parent_wave_id,
        child_wave=persisted.child_wave_spec,
        experiments=[item.experiment_spec for item in persisted.experiments],
        attempts=[(item.manifest, item.worker_binding) for item in persisted.attempts],
        plan_hash=persisted.plan_hash,
        materialization_hash=persisted.materialization_hash,
        queued_at=queued_at,
        actor=actor,
    )
