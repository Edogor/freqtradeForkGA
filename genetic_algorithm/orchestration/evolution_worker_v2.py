"""Canonical immutable worker for standard GA evolution followed by V2 replay.

Legacy GA fitness is used only to search and shortlist genomes.  A successful
``AttemptResultV2`` is produced exclusively by replaying the generated,
executable candidates through :class:`ShadowReplayRunnerV2`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from freqtrade.exchange import timeframe_to_seconds
from pydantic import Field, model_validator

from genetic_algorithm.config.schema import validate_resolved_config_v2_or_raise
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
from genetic_algorithm.genome.codegen import StrategyGenerator
from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.code_manifest_v2 import CodeManifestV2
from genetic_algorithm.orchestration.data_manifest_v2 import DataManifestV2
from genetic_algorithm.orchestration.final_test_ledger_v2 import FinalTestUsageLedgerV2
from genetic_algorithm.orchestration.manifest_builder_v2 import AttemptManifestBundleV2
from genetic_algorithm.orchestration.output_layout_v2 import (
    AttemptOutputLayoutV2,
    attempt_output_scope,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ShadowGatePolicyV2,
    shadow_gate_policy_from_config,
)
from genetic_algorithm.orchestration.replay_runner_v2 import (
    FrozenCandidateV2,
    ShadowReplayRunnerV2,
    freeze_candidate,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptResultV2,
    ScenarioRole,
    StrictV2Model,
)
from genetic_algorithm.orchestration.split_contract_v2 import (
    EvaluationSplitPlanV2,
    build_evaluation_split_plan,
)


class EvolutionWorkerError(ValueError):
    """Raised when an evolution worker cannot prove its complete input contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"unsafe worker input path: {value!r}")
    return path


class FrozenEvolutionSeedV2(StrictV2Model):
    """A serialized genome tied to the exact executable phenotype it produced."""

    schema_version: Literal["2.0"] = "2.0"
    gene_schema_version: Literal["strategy-gene-v2"] = "strategy-gene-v2"
    candidate_id: str = Field(min_length=1)
    phenotype_hash: str = Field(min_length=64, max_length=64)
    gene_hash: str = Field(min_length=64, max_length=64)
    strategy_gene: dict[str, Any]
    frozen_candidate: FrozenCandidateV2

    @model_validator(mode="after")
    def _consistent_seed(self) -> FrozenEvolutionSeedV2:
        try:
            StrategyGene.from_dict_exact(self.strategy_gene)
        except Exception as exc:
            raise ValueError("evolution seed contains an invalid StrategyGene") from exc
        if canonical_config_hash(self.strategy_gene) != self.gene_hash:
            raise ValueError("gene_hash differs from serialized StrategyGene")
        if self.frozen_candidate.candidate_id != self.candidate_id:
            raise ValueError("evolution seed candidate_id differs from frozen candidate")
        if self.frozen_candidate.phenotype_hash != self.phenotype_hash:
            raise ValueError("evolution seed phenotype differs from frozen candidate")
        return self

    def fresh_individual(self) -> Individual:
        gene = StrategyGene.from_dict_exact(copy.deepcopy(self.strategy_gene))
        return Individual(strategy_gene=gene)


def freeze_evolution_seed(
    individual: Individual,
    *,
    resolved_config: Mapping[str, Any],
    candidate_id: str,
) -> FrozenEvolutionSeedV2:
    """Generate code once, then bind the post-codegen genome to that executable."""

    # A producer must already satisfy the V2 genome contract.  Normalizing an
    # evaluated legacy object here could bind stale fitness to a changed
    # phenotype, so migration has to happen upstream followed by re-evaluation.
    StrategyGene.from_dict_exact(individual.strategy_gene.to_dict())
    clone = Individual.from_dict(individual.to_dict())
    clone.strategy_gene.assign_instance_ids()
    generator = StrategyGenerator(copy.deepcopy(dict(resolved_config)))
    strategy_code = generator.generate_strategy_code(clone.strategy_gene)
    strategy_name = (
        f"GAStrategy_Gen{clone.strategy_gene.generation}_Ind{clone.strategy_gene.individual_id}"
    )
    frozen = freeze_candidate(
        candidate_id=candidate_id,
        strategy_name=strategy_name,
        strategy_code=strategy_code,
        timeframe=clone.strategy_gene.timeframe,
        max_open_trades=clone.strategy_gene.max_open_trades,
    )
    gene_payload = clone.strategy_gene.to_dict()
    return FrozenEvolutionSeedV2(
        candidate_id=candidate_id,
        phenotype_hash=frozen.phenotype_hash,
        gene_hash=canonical_config_hash(gene_payload),
        strategy_gene=gene_payload,
        frozen_candidate=frozen,
    )


def validate_evolution_seed(
    seed: FrozenEvolutionSeedV2,
    resolved_config: Mapping[str, Any],
) -> Individual:
    """Fail closed when the current generator/config cannot reproduce a parent seed."""

    reproduced = freeze_evolution_seed(
        seed.fresh_individual(),
        resolved_config=resolved_config,
        candidate_id=seed.candidate_id,
    )
    if reproduced != seed:
        raise EvolutionWorkerError(
            f"evolution seed {seed.candidate_id} is incompatible with the resolved config/code"
        )
    return seed.fresh_individual()


class EvolutionWorkerSeedInputV2(StrictV2Model):
    candidate_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    gene_hash: str = Field(min_length=64, max_length=64)
    phenotype_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _canonical_path(self) -> EvolutionWorkerSeedInputV2:
        expected = f"inputs/seeds/{self.candidate_id}.json"
        if self.relative_path != expected:
            raise ValueError("evolution seed input path is not canonical")
        _safe_relative_path(self.relative_path)
        return self


class EvolutionWorkerCheckpointInputV2(StrictV2Model):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _canonical_path(self) -> "EvolutionWorkerCheckpointInputV2":
        relative = _safe_relative_path(self.relative_path)
        if relative.parent != Path("inputs/checkpoints"):
            raise ValueError("resume checkpoint must use the canonical input directory")
        if not relative.name.startswith("island_checkpoint_gen") or not relative.name.endswith(
            ".json"
        ):
            raise ValueError("resume checkpoint filename is not canonical")
        return self


class EvolutionWorkerSpecV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    worker_kind: Literal[
        "STANDARD_EVOLUTION",
        "GENERIC_ISLAND_EVOLUTION",
    ] = "STANDARD_EVOLUTION"
    attempt_id: str = Field(min_length=1)
    created_at: datetime
    artifact_root: str = Field(min_length=1)
    repo_root: str = Field(min_length=1)
    final_test_ledger_path: str = Field(min_length=1)
    output_layout: AttemptOutputLayoutV2
    # RunEngine currently returns at most ten standard-GA finalists.
    top_n: int = Field(default=5, ge=1, le=12)
    seeds: list[EvolutionWorkerSeedInputV2] = Field(default_factory=list)
    resume_checkpoint: EvolutionWorkerCheckpointInputV2 | None = None
    replay_input_seeds: bool = False
    input_hashes: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_inputs(self) -> EvolutionWorkerSpecV2:
        if self.created_at.utcoffset() is None:
            raise ValueError("worker spec created_at must be timezone-aware")
        for field_name in ("artifact_root", "repo_root", "final_test_ledger_path"):
            if not Path(getattr(self, field_name)).is_absolute():
                raise ValueError(f"{field_name} must be absolute")
        if self.output_layout.artifact_root != str(Path(self.artifact_root).resolve()):
            raise ValueError("output layout differs from worker artifact_root")
        ids = [item.candidate_id for item in self.seeds]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("evolution seeds must be sorted and unique")
        if self.replay_input_seeds and len(self.seeds) != 1:
            raise ValueError("seed replay requires exactly one immutable parent seed")
        required = {
            V2ArtifactStore.MANIFEST_NAME,
            V2ArtifactStore.CONFIG_NAME,
            V2ArtifactStore.POLICY_NAME,
            V2ArtifactStore.DATA_MANIFEST_NAME,
            V2ArtifactStore.CODE_MANIFEST_NAME,
            V2ArtifactStore.SPLIT_MANIFEST_NAME,
            *(item.relative_path for item in self.seeds),
            *(
                [self.resume_checkpoint.relative_path]
                if self.resume_checkpoint is not None
                else []
            ),
        }
        if set(self.input_hashes) != required:
            raise ValueError("worker input hash matrix is incomplete or contains extras")
        for relative, digest in self.input_hashes.items():
            _safe_relative_path(relative)
            if len(digest) != 64:
                raise ValueError("worker input hash is not SHA-256")
        return self

    @property
    def spec_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


@dataclass(frozen=True)
class PreparedEvolutionWorkerV2:
    spec: EvolutionWorkerSpecV2
    spec_path: Path
    spec_file_sha256: str


@dataclass(frozen=True)
class LoadedEvolutionWorkerV2:
    spec: EvolutionWorkerSpecV2
    manifest: AttemptManifestV2
    resolved_config: dict[str, Any]
    policy: ShadowGatePolicyV2
    data_manifest: DataManifestV2
    code_manifest: CodeManifestV2
    split_manifest: EvaluationSplitPlanV2
    seeds: tuple[FrozenEvolutionSeedV2, ...]


def _base_input_paths() -> tuple[str, ...]:
    return (
        V2ArtifactStore.MANIFEST_NAME,
        V2ArtifactStore.CONFIG_NAME,
        V2ArtifactStore.POLICY_NAME,
        V2ArtifactStore.DATA_MANIFEST_NAME,
        V2ArtifactStore.CODE_MANIFEST_NAME,
        V2ArtifactStore.SPLIT_MANIFEST_NAME,
    )


def _reject_unexpected_preexecution_files(
    root: Path,
    spec: EvolutionWorkerSpecV2,
) -> None:
    allowed = set(spec.input_hashes) | {V2ArtifactStore.WORKER_SPEC_NAME}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.relative_to(root).as_posix().startswith("runtime/")
    }
    unexpected = sorted(actual - allowed)
    missing = sorted(allowed - actual)
    if missing or unexpected:
        raise EvolutionWorkerError(
            f"pre-execution artifact set differs; missing={missing}, unexpected={unexpected}"
        )


def _evolution_worker_kind(
    config: Mapping[str, Any],
) -> Literal["STANDARD_EVOLUTION", "GENERIC_ISLAND_EVOLUTION"]:
    if config.get("generic_island_model", {}).get("enabled", False):
        return "GENERIC_ISLAND_EVOLUTION"
    return "STANDARD_EVOLUTION"


def _validate_evolution_contract(
    manifest: AttemptManifestV2,
    config: Mapping[str, Any],
    policy: ShadowGatePolicyV2,
    data_manifest: DataManifestV2,
) -> None:
    if len(manifest.seeds) != 1:
        raise EvolutionWorkerError("evolution requires exactly one attempt seed")
    safety = config.get("safety_profile", {})
    ga = config.get("genetic_algorithm", {})
    backtesting = config.get("backtesting", {})
    if ga.get("mode") != "single_objective":
        raise EvolutionWorkerError("evolution currently supports single_objective only")
    if any(item.role == ScenarioRole.FINAL_TEST for item in policy.required_scenarios):
        raise EvolutionWorkerError(
            "evolution attempts cannot consume FINAL_TEST; use a later replay-validation wave"
        )
    generic = config.get("generic_island_model", {})
    generic_enabled = generic.get("enabled", False)
    classic_enabled = config.get("island_model", {}).get("enabled", False)
    if generic_enabled and classic_enabled:
        raise EvolutionWorkerError("only one island engine may be enabled")
    if generic_enabled:
        if (
            safety.get("name") not in {
                "automation_island_v2",
                "quality_experiment_v3",
                "hardcore_multipair_v1",
            }
            or not safety.get("enforce", False)
            or not safety.get("shadow_mode", False)
            or safety.get("automation_eligible", False)
        ):
            raise EvolutionWorkerError(
                "generic-island evolution requires an enforced shadow-only safety profile"
            )
        if generic.get("parallel_islands", False):
            raise EvolutionWorkerError(
                "automation_island_v2 forbids parallel_islands; use one shared worker pool"
            )
        if generic.get("specialization", {}).get("pair_rotation", False):
            raise EvolutionWorkerError(
                "automation_island_v2 uses fixed pair-validation, not island pair rotation"
            )
        if generic.get("external_migration", {}).get("enabled", False):
            raise EvolutionWorkerError(
                "automation_island_v2 forbids mutable external migration"
            )
        islands = generic.get("islands", [])
        if not isinstance(islands, list) or len(islands) < 2:
            raise EvolutionWorkerError(
                "automation_island_v2 requires at least two explicit islands"
            )
        island_names = [item.get("name") for item in islands if isinstance(item, Mapping)]
        if (
            len(island_names) != len(islands)
            or any(not isinstance(name, str) or not name for name in island_names)
            or len(island_names) != len(set(island_names))
        ):
            raise EvolutionWorkerError("explicit island names must be non-empty and unique")
        pair_validation = config.get("pair_validation", {})
        if not pair_validation.get("enabled", False):
            raise EvolutionWorkerError(
                "automation_island_v2 requires pair_validation.enabled"
            )
        if (
            safety.get("name") in {
                "automation_island_v2",
                "hardcore_multipair_v1",
            }
            and pair_validation.get("validate_top_n_only", 0) != 0
        ):
            raise EvolutionWorkerError(
                "automation_island_v2 must validate every candidate across the pair split"
            )
        training_pairs = set(pair_validation.get("training_pairs", []))
        validation_pairs = set(pair_validation.get("validation_pairs", []))
        configured_pairs = set(backtesting.get("pairs", []))
        if training_pairs | validation_pairs != configured_pairs:
            raise EvolutionWorkerError(
                "pair-validation train/validation union must equal backtesting.pairs"
            )
        declared_pair_validation = {
            item.pair
            for item in policy.required_scenarios
            if item.role == ScenarioRole.PAIR_VALIDATION
        }
        if not validation_pairs.issubset(declared_pair_validation):
            raise EvolutionWorkerError(
                "every validation pair needs a declared PAIR_VALIDATION replay scenario"
            )
    else:
        if classic_enabled:
            raise EvolutionWorkerError(
                "regime-locked island evolution has no canonical V2 worker"
            )
        if safety.get("name") != "safe_v2" or not safety.get("enforce", False):
            raise EvolutionWorkerError(
                "standard evolution requires the enforced safe_v2 profile"
            )
    timeframe = backtesting.get("timeframe")
    if not isinstance(timeframe, str) or not timeframe:
        raise EvolutionWorkerError("backtesting.timeframe must be explicit for evolution")
    covered = {(item.pair, item.timeframe) for item in data_manifest.files}
    missing = sorted(
        (pair, timeframe)
        for pair in backtesting.get("pairs", [])
        if (pair, timeframe) not in covered
    )
    if missing:
        raise EvolutionWorkerError(
            f"evolution OHLCV is absent from the immutable data manifest: {missing}"
        )
    if shadow_gate_policy_from_config(config) != policy:
        raise EvolutionWorkerError("promotion policy differs from resolved config")
    if build_evaluation_split_plan(config, policy).split_hash != manifest.split_manifest_hash:
        raise EvolutionWorkerError("runtime split plan differs from attempt manifest")


def _validate_evolution_seed_capacity(
    config: Mapping[str, Any], seed_ids: Sequence[str]
) -> None:
    """Validate seed capacity against the engine that will consume the seeds.

    Standard evolution has one population, so its global population size is the
    correct bound.  Generic-island archive seeding is different: immutable
    seeds are stored once by the worker and then assigned to individual archive
    islands by candidate ID.  Bounding that union by the compatibility
    ``genetic_algorithm.population_size`` incorrectly rejects valid campaigns
    whenever different bridge islands receive different seeds.
    """

    population_size = int(
        config.get("genetic_algorithm", {}).get("population_size", 0)
    )
    generic = config.get("generic_island_model", {})
    archive = generic.get("archive_seeding", {}) if isinstance(generic, Mapping) else {}
    assignments = archive.get("assignments", {}) if isinstance(archive, Mapping) else {}
    if not (
        isinstance(generic, Mapping)
        and generic.get("enabled") is True
        and isinstance(assignments, Mapping)
        and assignments
    ):
        if len(seed_ids) > population_size:
            raise EvolutionWorkerError("evolution seed count exceeds population size")
        return

    islands = {
        str(row.get("name")): row
        for row in generic.get("islands", [])
        if isinstance(row, Mapping) and row.get("name")
    }
    max_per_island = int(archive.get("max_seeds_per_island", 4))
    assigned_ids: set[str] = set()
    for island_name, rows in assignments.items():
        if island_name not in islands or not isinstance(rows, list):
            raise EvolutionWorkerError(
                "archive seed assignment references an invalid island"
            )
        row_ids = [
            str(row.get("candidate_id"))
            for row in rows
            if isinstance(row, Mapping) and row.get("candidate_id")
        ]
        if len(row_ids) != len(rows) or len(row_ids) != len(set(row_ids)):
            raise EvolutionWorkerError(
                "archive seed assignment IDs must be present and unique"
            )
        island_capacity = int(islands[island_name].get("population_size", 0))
        if len(row_ids) > min(max_per_island, island_capacity):
            raise EvolutionWorkerError(
                f"archive seed assignment exceeds capacity for {island_name}"
            )
        assigned_ids.update(row_ids)
    if set(seed_ids) != assigned_ids:
        raise EvolutionWorkerError(
            "immutable evolution seeds differ from archive seed assignments"
        )


def prepare_evolution_worker(
    *,
    bundle: AttemptManifestBundleV2,
    resolved_config: Mapping[str, Any],
    policy: ShadowGatePolicyV2,
    seeds: Sequence[FrozenEvolutionSeedV2],
    repo_root: str | Path,
    final_test_ledger_path: str | Path,
    created_at: datetime,
    top_n: int = 5,
    replay_input_seeds: bool = False,
    resume_checkpoint_path: str | Path | None = None,
) -> PreparedEvolutionWorkerV2:
    """Persist and hash every input before an evolution attempt can be queued."""

    manifest = bundle.manifest
    root = Path(manifest.artifact_root).resolve()
    resolved_repo = Path(repo_root).resolve()
    resolved_ledger = Path(final_test_ledger_path).resolve()
    config = dict(resolved_config)
    if not resolved_repo.is_dir():
        raise EvolutionWorkerError(f"repo_root does not exist: {resolved_repo}")
    if root == resolved_ledger or root in resolved_ledger.parents:
        raise EvolutionWorkerError("final-test ledger must live outside the artifact root")
    if canonical_config_hash(config) != manifest.config_hash:
        raise EvolutionWorkerError("resolved config differs from attempt manifest")
    if created_at.utcoffset() is None or created_at < manifest.created_at:
        raise EvolutionWorkerError("worker spec timestamp precedes attempt creation")
    if bundle.data_manifest.manifest_hash != manifest.data_manifest_hash:
        raise EvolutionWorkerError("data manifest differs from attempt manifest")
    if bundle.split_manifest.split_hash != manifest.split_manifest_hash:
        raise EvolutionWorkerError("split manifest differs from attempt manifest")
    _validate_evolution_contract(manifest, config, policy, bundle.data_manifest)

    seed_list = sorted(seeds, key=lambda item: item.candidate_id)
    seed_ids = [item.candidate_id for item in seed_list]
    if len(seed_ids) != len(set(seed_ids)):
        raise EvolutionWorkerError("evolution seed candidate IDs must be unique")
    if replay_input_seeds and len(seed_list) != 1:
        raise EvolutionWorkerError(
            "immutable parent replay requires exactly one evolution seed"
        )
    _validate_evolution_seed_capacity(config, seed_ids)
    for seed in seed_list:
        validate_evolution_seed(seed, config)

    store = V2ArtifactStore(root)
    store.write_manifest(manifest, config)
    store.write_policy(policy)
    store.write_data_manifest(bundle.data_manifest, manifest.data_manifest_hash)
    store.write_code_manifest(bundle.code_manifest, manifest)
    store.write_split_manifest(bundle.split_manifest, str(manifest.split_manifest_hash))
    worker_seeds: list[EvolutionWorkerSeedInputV2] = []
    for seed in seed_list:
        path = store.write_evolution_seed_input(seed)
        relative = path.relative_to(root).as_posix()
        worker_seeds.append(
            EvolutionWorkerSeedInputV2(
                candidate_id=seed.candidate_id,
                relative_path=relative,
                gene_hash=seed.gene_hash,
                phenotype_hash=seed.phenotype_hash,
            )
        )
    checkpoint_input: EvolutionWorkerCheckpointInputV2 | None = None
    if resume_checkpoint_path is not None:
        checkpoint_source = Path(resume_checkpoint_path).resolve()
        if not checkpoint_source.is_file():
            raise EvolutionWorkerError("resume checkpoint does not exist")
        checkpoint_target = store.write_evolution_checkpoint_input(
            filename=checkpoint_source.name,
            payload=checkpoint_source.read_bytes(),
        )
        checkpoint_input = EvolutionWorkerCheckpointInputV2(
            relative_path=checkpoint_target.relative_to(root).as_posix(),
            sha256=_sha256_file(checkpoint_target),
        )
    input_paths = [
        *_base_input_paths(),
        *(item.relative_path for item in worker_seeds),
        *(
            [checkpoint_input.relative_path]
            if checkpoint_input is not None
            else []
        ),
    ]
    input_hashes = {
        relative: _sha256_file(root / _safe_relative_path(relative))
        for relative in sorted(input_paths)
    }
    spec = EvolutionWorkerSpecV2(
        worker_kind=_evolution_worker_kind(config),
        attempt_id=manifest.attempt_id,
        created_at=created_at,
        artifact_root=str(root),
        repo_root=str(resolved_repo),
        final_test_ledger_path=str(resolved_ledger),
        output_layout=AttemptOutputLayoutV2.for_artifact_root(root),
        top_n=top_n,
        seeds=worker_seeds,
        resume_checkpoint=checkpoint_input,
        replay_input_seeds=replay_input_seeds,
        input_hashes=input_hashes,
    )
    spec_path = store.write_worker_spec(spec)
    _reject_unexpected_preexecution_files(root, spec)
    return PreparedEvolutionWorkerV2(
        spec=spec,
        spec_path=spec_path,
        spec_file_sha256=_sha256_file(spec_path),
    )


def evolution_worker_argv(
    prepared: PreparedEvolutionWorkerV2,
    *,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    executable = Path(python_executable).absolute()
    return [
        str(executable),
        "-m",
        "genetic_algorithm.orchestration.evolution_worker_v2",
        "--spec",
        str(prepared.spec_path.resolve()),
        "--spec-sha256",
        prepared.spec_file_sha256,
    ]


def _load_bound_spec(
    spec_path: str | Path,
    expected_spec_sha256: str,
) -> tuple[EvolutionWorkerSpecV2, Path]:
    path = Path(spec_path).resolve()
    if path.name != V2ArtifactStore.WORKER_SPEC_NAME or not path.is_file():
        raise EvolutionWorkerError("worker spec path is not canonical or does not exist")
    if len(expected_spec_sha256) != 64 or _sha256_file(path) != expected_spec_sha256:
        raise EvolutionWorkerError("worker spec SHA-256 differs from claimed command")
    try:
        spec = EvolutionWorkerSpecV2.model_validate_json(path.read_bytes())
    except Exception as exc:
        raise EvolutionWorkerError("worker spec violates the V2 contract") from exc
    root = path.parent
    if root != Path(spec.artifact_root).resolve():
        raise EvolutionWorkerError("worker spec artifact_root differs from its location")
    return spec, root


def _verify_worker_inputs(
    root: Path,
    spec: EvolutionWorkerSpecV2,
    *,
    require_pristine_artifact_root: bool,
) -> None:
    if require_pristine_artifact_root:
        _reject_unexpected_preexecution_files(root, spec)
    for relative, expected in spec.input_hashes.items():
        input_path = (root / _safe_relative_path(relative)).resolve()
        if root not in input_path.parents or not input_path.is_file():
            raise EvolutionWorkerError(f"worker input is missing or escapes root: {relative}")
        if _sha256_file(input_path) != expected:
            raise EvolutionWorkerError(f"worker input hash differs: {relative}")


def _load_core_inputs(
    root: Path,
    spec: EvolutionWorkerSpecV2,
) -> tuple[
    AttemptManifestV2,
    dict[str, Any],
    ShadowGatePolicyV2,
    DataManifestV2,
    CodeManifestV2,
    EvaluationSplitPlanV2,
]:
    manifest = AttemptManifestV2.model_validate_json(
        (root / V2ArtifactStore.MANIFEST_NAME).read_bytes()
    )
    if manifest.attempt_id != spec.attempt_id:
        raise EvolutionWorkerError("worker spec attempt_id differs from manifest")
    if Path(manifest.artifact_root).resolve() != root:
        raise EvolutionWorkerError("manifest artifact_root differs from worker root")
    if Path(manifest.resolved_config_path).resolve() != root / V2ArtifactStore.CONFIG_NAME:
        raise EvolutionWorkerError("manifest resolved_config_path is not canonical")
    try:
        raw_config = yaml.safe_load((root / V2ArtifactStore.CONFIG_NAME).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise EvolutionWorkerError("resolved config cannot be loaded") from exc
    if not isinstance(raw_config, Mapping):
        raise EvolutionWorkerError("resolved config must be a mapping")
    config = dict(raw_config)
    if canonical_config_hash(config) != manifest.config_hash:
        raise EvolutionWorkerError("resolved config semantic hash differs from manifest")
    try:
        validate_resolved_config_v2_or_raise(config)
    except ValueError as exc:
        raise EvolutionWorkerError("resolved config violates the V2 contract") from exc
    policy = ShadowGatePolicyV2.model_validate_json(
        (root / V2ArtifactStore.POLICY_NAME).read_bytes()
    )
    data_manifest = DataManifestV2.model_validate_json(
        (root / V2ArtifactStore.DATA_MANIFEST_NAME).read_bytes()
    )
    code_manifest = CodeManifestV2.model_validate_json(
        (root / V2ArtifactStore.CODE_MANIFEST_NAME).read_bytes()
    )
    split_manifest = EvaluationSplitPlanV2.model_validate_json(
        (root / V2ArtifactStore.SPLIT_MANIFEST_NAME).read_bytes()
    )
    if (
        data_manifest.manifest_hash != manifest.data_manifest_hash
        or split_manifest.split_hash != manifest.split_manifest_hash
        or code_manifest.code_version != manifest.code_version
        or code_manifest.dirty_patch_hash != manifest.dirty_patch_hash
    ):
        raise EvolutionWorkerError("persisted manifests differ from attempt provenance")
    return manifest, config, policy, data_manifest, code_manifest, split_manifest


def _load_seed_inputs(
    root: Path,
    spec: EvolutionWorkerSpecV2,
) -> tuple[FrozenEvolutionSeedV2, ...]:
    seeds = tuple(
        FrozenEvolutionSeedV2.model_validate_json((root / item.relative_path).read_bytes())
        for item in spec.seeds
    )
    for declared, seed in zip(spec.seeds, seeds, strict=True):
        if (
            declared.candidate_id != seed.candidate_id
            or declared.gene_hash != seed.gene_hash
            or declared.phenotype_hash != seed.phenotype_hash
        ):
            raise EvolutionWorkerError("evolution seed differs from worker spec")
    return seeds


def load_evolution_worker(
    spec_path: str | Path,
    *,
    expected_spec_sha256: str,
    require_pristine_artifact_root: bool = True,
) -> LoadedEvolutionWorkerV2:
    spec, root = _load_bound_spec(spec_path, expected_spec_sha256)
    _verify_worker_inputs(
        root,
        spec,
        require_pristine_artifact_root=require_pristine_artifact_root,
    )
    (
        manifest,
        config,
        policy,
        data_manifest,
        code_manifest,
        split_manifest,
    ) = _load_core_inputs(root, spec)
    seeds = _load_seed_inputs(root, spec)
    _validate_evolution_contract(manifest, config, policy, data_manifest)
    if spec.worker_kind != _evolution_worker_kind(config):
        raise EvolutionWorkerError("worker kind differs from resolved engine config")
    return LoadedEvolutionWorkerV2(
        spec=spec,
        manifest=manifest,
        resolved_config=config,
        policy=policy,
        data_manifest=data_manifest,
        code_manifest=code_manifest,
        split_manifest=split_manifest,
        seeds=seeds,
    )


def derive_search_seed(
    paired_evaluation_seed: int,
    search_seed_salt: int,
) -> int:
    """Derive the deterministic evolutionary RNG seed for one search arm.

    Replay keeps using ``paired_evaluation_seed`` so arms remain comparable.
    A non-zero salt selects a separate, deterministic search stream.  Keeping
    this derivation in one function also lets reporting verify the exact same
    contract instead of incorrectly comparing every arm to the replay seed.
    """

    if (
        not isinstance(paired_evaluation_seed, int)
        or isinstance(paired_evaluation_seed, bool)
    ):
        raise EvolutionWorkerError("paired evaluation seed must be an integer")
    if (
        not isinstance(search_seed_salt, int)
        or isinstance(search_seed_salt, bool)
        or search_seed_salt < 0
    ):
        raise EvolutionWorkerError("search seed salt must be an integer >= 0")

    paired_seed = paired_evaluation_seed % (2**32)
    if search_seed_salt == 0:
        return paired_seed
    return int(
        canonical_config_hash(
            {
                "contract": "ARM_SEARCH_SEED_V1",
                "paired_evaluation_seed": paired_seed,
                "search_seed_salt": search_seed_salt,
            }
        )[:8],
        16,
    )


def derive_engine_config(
    loaded: LoadedEvolutionWorkerV2,
) -> dict[str, Any]:
    """Change only seed/concurrency and operational paths under this attempt.

    ``manifest.seeds`` remains the paired evaluation/bootstrap seed consumed
    by strict V2 replay.  A non-zero ``search_seed_salt`` deliberately gives a
    repair/explore arm a different deterministic evolutionary RNG stream
    without sacrificing paired replay comparability.
    """

    config = copy.deepcopy(loaded.resolved_config)
    paired_evaluation_seed = int(loaded.manifest.seeds[0]) % (2**32)
    ga = config.setdefault("genetic_algorithm", {})
    search_seed_salt = ga.get("search_seed_salt", 0)
    if (
        not isinstance(search_seed_salt, int)
        or isinstance(search_seed_salt, bool)
        or search_seed_salt < 0
    ):
        raise EvolutionWorkerError(
            "genetic_algorithm.search_seed_salt must be an integer >= 0"
        )
    search_seed = derive_search_seed(
        paired_evaluation_seed,
        search_seed_salt,
    )
    ga["random_seed"] = search_seed
    parallel = config.setdefault("parallel_evaluation", {})
    parallel["enabled"] = loaded.manifest.worker_count > 1
    parallel["num_workers"] = loaded.manifest.worker_count
    backtesting = config.setdefault("backtesting", {})
    timeframe = str(backtesting["timeframe"])
    start = datetime.combine(
        loaded.split_manifest.evolution_period_start,
        time.min,
        tzinfo=UTC,
    )
    exclusive_end = datetime.combine(
        loaded.split_manifest.evolution_period_end_exclusive,
        time.min,
        tzinfo=UTC,
    )
    last_candle_open = exclusive_end - timedelta(
        seconds=timeframe_to_seconds(timeframe)
    )
    if last_candle_open < start:
        raise EvolutionWorkerError("evolution split has no complete timeframe candle")
    # Freqtrade treats the stop timestamp as inclusive. Date-only config
    # ranges therefore admit the first candle of the exclusive end day.
    # Bind search to the same exact candle cells as the split and V2 replay.
    backtesting["timerange"] = (
        f"{int(start.timestamp())}-{int(last_candle_open.timestamp())}"
    )
    loaded.spec.output_layout.apply_to_engine_config(config)
    config.setdefault("terminal_monitor", {})["enabled"] = False
    config.setdefault("warm_start", {})["enabled"] = False
    config["experiment_name"] = loaded.manifest.attempt_id
    generic_island = config.get("generic_island_model", {}).get("enabled", False)
    if generic_island:
        # Explicit island definitions historically carried fixed preset seeds
        # (42, 43, ...), which overrode the hash-bound attempt seed. As a
        # result, roots declared with different seeds reproduced the same
        # search population. Preserve deterministic island diversity while
        # making the attempt seed authoritative.
        for ordinal, island in enumerate(
            config["generic_island_model"].get("islands", [])
        ):
            island["seed"] = (search_seed + ordinal) % (2**32)
    island_names = (
        [
            str(item["name"])
            for item in config["generic_island_model"].get("islands", [])
        ]
        if generic_island
        else []
    )
    config["checkpoint_provenance"] = {
        "schema_version": "3.0",
        "engine_kind": "GENERIC_ISLAND" if generic_island else "STANDARD",
        "config_hash": loaded.manifest.config_hash,
        "code_manifest_hash": canonical_config_hash(
            loaded.code_manifest.model_dump(mode="json")
        ),
        "data_manifest_hash": loaded.data_manifest.manifest_hash,
        "genome_schema_version": "strategy-gene-v2",
        "island_names": island_names,
    }
    if (
        generic_island
        and config.get("safety_profile", {}).get("name")
        == "hardcore_multipair_v1"
    ):
        config["generic_island_model"]["graceful_stop_marker"] = str(
            Path(loaded.spec.output_layout.runtime_dir) / "graceful_stop.json"
        )
    loaded.spec.output_layout.assert_engine_config(config)
    return config


def _run_evolution_engine(
    config_path: Path,
    seeds: Sequence[Individual],
) -> Sequence[Individual]:
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, Mapping):
        raise EvolutionWorkerError("derived engine config must be a mapping")
    if config.get("generic_island_model", {}).get("enabled", False):
        from genetic_algorithm.core.generic_island_model import (
            GenericIslandModelEvolution,
        )
        from genetic_algorithm.engine.island_results import extract_island_finalists

        algorithm = GenericIslandModelEvolution(
            str(config_path),
            visualize=False,
            interactive=False,
        )
        if config.get("safety_profile", {}).get("name") == "hardcore_multipair_v1":
            # Across-run champions are intentionally confined to the three
            # configured archive islands.  The other nine islands remain
            # fully fresh; treating these as ordinary strict seeds would
            # inject every champion into every island and collapse diversity.
            algorithm.set_archive_initial_seeds(list(seeds))
        else:
            algorithm.set_strict_initial_seeds(list(seeds))
        results = algorithm.evolve()
        return extract_island_finalists(
            results,
            expected_islands=[
                island_config.name for island_config in algorithm.island_configs
            ],
            require_finalists=True,
        ).finalists

    from genetic_algorithm.core.evolution import GeneticAlgorithm

    algorithm = GeneticAlgorithm(str(config_path), visualize=False, interactive=False)
    algorithm.set_strict_initial_seeds(list(seeds))
    return algorithm.evolve()


def _attempt_backtester_factory(
    layout: AttemptOutputLayoutV2,
) -> Callable[[dict[str, Any]], DirectBacktester]:
    def factory(config: dict[str, Any]) -> DirectBacktester:
        isolated = layout.isolated_backtester_config(config, phase="replay")
        return DirectBacktester(isolated)

    return factory


def _candidate_id(attempt_id: str, rank: int) -> str:
    attempt_token = hashlib.sha256(attempt_id.encode()).hexdigest()[:20]
    return f"evo-{attempt_token}-rank-{rank:03d}"


def _usable_results(results: Sequence[Individual]) -> list[Individual]:
    usable = [
        item
        for item in results
        if item.has_measured_fitness
        and item.fitness is not None
        and math.isfinite(float(item.fitness))
    ]
    return sorted(usable, key=lambda item: float(item.fitness), reverse=True)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _failure_manifest(spec_path: str | Path) -> AttemptManifestV2:
    path = Path(spec_path).resolve()
    if path.name != V2ArtifactStore.WORKER_SPEC_NAME:
        raise EvolutionWorkerError("cannot recover manifest from a non-canonical spec path")
    manifest = AttemptManifestV2.model_validate_json(
        (path.parent / V2ArtifactStore.MANIFEST_NAME).read_bytes()
    )
    if Path(manifest.artifact_root).resolve() != path.parent:
        raise EvolutionWorkerError("failure manifest artifact_root differs from spec location")
    return manifest


def _finalize_worker_failure(
    spec_path: str | Path,
    error: Exception,
    *,
    finished_at: datetime,
) -> AttemptResultV2:
    manifest = _failure_manifest(spec_path)
    store = V2ArtifactStore(manifest.artifact_root)
    try:
        return store.read_verified_result()
    except ArtifactIntegrityError:
        pass
    error_code = (
        "WORKER_INPUT_INVALID"
        if isinstance(error, (EvolutionWorkerError, ArtifactIntegrityError))
        else "WORKER_EXECUTION_FAILED"
    )
    return store.finalize(
        {
            "attempt_id": manifest.attempt_id,
            "status": "INVALID_RESULT",
            "started_at": manifest.created_at,
            "finished_at": max(finished_at, manifest.created_at),
            "manifest": manifest.model_dump(mode="python"),
            "candidate_evaluations": [],
            "error_code": error_code,
            "error_detail": f"{type(error).__name__}: {error}"[:2000],
        }
    )


def run_evolution_worker(
    spec_path: str | Path,
    *,
    expected_spec_sha256: str,
    evolution_runner: Callable[
        [Path, Sequence[Individual]], Sequence[Individual]
    ] = _run_evolution_engine,
    backtester_factory: Callable[[dict[str, Any]], Any] | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> AttemptResultV2:
    """Search with configured fitness, then commit immutable six-pair replay evidence."""

    try:
        loaded = load_evolution_worker(
            spec_path,
            expected_spec_sha256=expected_spec_sha256,
        )
        # Perform all code/data/split checks before the engine creates outputs.
        runner = ShadowReplayRunnerV2(
            manifest=loaded.manifest,
            resolved_config=loaded.resolved_config,
            policy=loaded.policy,
            data_manifest=loaded.data_manifest,
            code_manifest=loaded.code_manifest,
            final_test_ledger=FinalTestUsageLedgerV2(loaded.spec.final_test_ledger_path),
            repo_root=loaded.spec.repo_root,
            backtester_factory=backtester_factory
            or _attempt_backtester_factory(loaded.spec.output_layout),
            clock=clock,
        )
        strict_seeds: list[Individual] = []
        quarantined_seeds: list[dict[str, str]] = []
        hardcore = (
            loaded.resolved_config.get("safety_profile", {}).get("name")
            == "hardcore_multipair_v1"
        )
        for seed in loaded.seeds:
            try:
                individual = validate_evolution_seed(seed, loaded.resolved_config)
            except EvolutionWorkerError as exc:
                if not hardcore:
                    raise
                quarantined_seeds.append(
                    {"candidate_id": seed.candidate_id, "reason": str(exc)}
                )
                continue
            individual.metrics["archive_candidate_id"] = seed.candidate_id
            strict_seeds.append(individual)
        if hardcore:
            quarantine_path = Path(loaded.spec.output_layout.runtime_dir) / "seed_quarantine.json"
            quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            preflight_quarantine: list[dict[str, str]] = []
            if quarantine_path.exists():
                try:
                    existing = json.loads(quarantine_path.read_text(encoding="utf-8"))
                    rows = existing.get("quarantined", []) if isinstance(existing, dict) else []
                    if isinstance(rows, list):
                        preflight_quarantine = [row for row in rows if isinstance(row, dict)]
                except (OSError, json.JSONDecodeError):
                    # The preflight artifact itself is evidence; an unreadable
                    # one is a real worker error and must not be hidden.
                    raise EvolutionWorkerError("preflight seed quarantine is unreadable")
            quarantine_path.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "quarantine_version": "hardcore-seed-quarantine-v3",
                        "quarantined": preflight_quarantine + quarantined_seeds,
                        "accepted_candidate_ids": [
                            item.metrics["archive_candidate_id"] for item in strict_seeds
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
        engine_config = derive_engine_config(loaded)
        store = V2ArtifactStore(loaded.spec.artifact_root)
        config_path = store.write_engine_config(engine_config)
        if config_path.resolve() != Path(loaded.spec.output_layout.engine_config_path).resolve():
            raise EvolutionWorkerError("engine config path differs from output layout")
        if loaded.spec.resume_checkpoint is not None:
            checkpoint_input = (
                Path(loaded.spec.artifact_root)
                / loaded.spec.resume_checkpoint.relative_path
            )
            if _sha256_file(checkpoint_input) != loaded.spec.resume_checkpoint.sha256:
                raise EvolutionWorkerError("resume checkpoint hash changed after input verification")
            checkpoint_target = (
                Path(loaded.spec.output_layout.checkpoint_dir)
                / checkpoint_input.name
            )
            checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint_target.exists():
                if _sha256_file(checkpoint_target) != loaded.spec.resume_checkpoint.sha256:
                    raise EvolutionWorkerError("materialized resume checkpoint differs")
            else:
                checkpoint_target.write_bytes(checkpoint_input.read_bytes())
        # Keep the attempt-local filesystem contract independent of tracker
        # implementation details. Generic-island sub-GAs are coordinated
        # directly and therefore must not initialise standalone run trackers.
        Path(loaded.spec.output_layout.runs_dir).mkdir(parents=True, exist_ok=True)
        with attempt_output_scope(loaded.spec.output_layout, phase="evolution"):
            results = _usable_results(evolution_runner(config_path, strict_seeds))
        if not results:
            raise EvolutionWorkerError("evolution produced no finite evaluated individual")
        replayed_parent: list[FrozenEvolutionSeedV2] = []
        evolved_capacity = loaded.spec.top_n
        if loaded.spec.replay_input_seeds:
            parent = loaded.seeds[0]
            reproduced_parent = freeze_evolution_seed(
                parent.fresh_individual(),
                resolved_config=loaded.resolved_config,
                candidate_id=_candidate_id(loaded.manifest.attempt_id, 0),
            )
            if reproduced_parent.phenotype_hash != parent.phenotype_hash:
                raise EvolutionWorkerError(
                    "immutable replay parent phenotype changed before strict replay"
                )
            replayed_parent.append(reproduced_parent)
            evolved_capacity -= 1
        selected = results[:evolved_capacity]
        evolved_seeds = [
            freeze_evolution_seed(
                individual,
                resolved_config=loaded.resolved_config,
                candidate_id=_candidate_id(loaded.manifest.attempt_id, rank),
            )
            for rank, individual in enumerate(selected, start=1)
        ]
        output_seeds = [*replayed_parent, *evolved_seeds]
        for seed in output_seeds:
            store.write_frozen_candidate(seed.frozen_candidate)
            store.write_evolution_seed(seed)
        with attempt_output_scope(loaded.spec.output_layout, phase="replay"):
            return runner.execute(
                [seed.frozen_candidate for seed in output_seeds]
            )
    except Exception as exc:
        return _finalize_worker_failure(spec_path, exc, finished_at=clock())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute one immutable standard GA V2 evolution worker spec"
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_evolution_worker(
            arguments.spec,
            expected_spec_sha256=arguments.spec_sha256,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "BOOTSTRAP_FAILED",
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {"attempt_id": result.attempt_id, "status": result.status.value},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
