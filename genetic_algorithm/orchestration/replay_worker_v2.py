"""Immutable subprocess contract for executing one canonical V2 shadow replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from genetic_algorithm.config.schema import validate_resolved_config_v2_or_raise
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
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
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptResultV2,
    StrictV2Model,
)
from genetic_algorithm.orchestration.split_contract_v2 import EvaluationSplitPlanV2


class ReplayWorkerError(ValueError):
    """Raised when persisted worker inputs are incomplete or inconsistent."""


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


class ReplayWorkerCandidateV2(StrictV2Model):
    candidate_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    phenotype_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _canonical_candidate_path(self) -> ReplayWorkerCandidateV2:
        expected = f"candidates/{self.candidate_id}/frozen_candidate.json"
        if self.relative_path != expected:
            raise ValueError("candidate worker input path is not canonical")
        _safe_relative_path(self.relative_path)
        return self


class ReplayWorkerSpecV2(StrictV2Model):
    schema_version: Literal["2.0"] = "2.0"
    worker_kind: Literal["SHADOW_REPLAY"] = "SHADOW_REPLAY"
    attempt_id: str = Field(min_length=1)
    created_at: datetime
    artifact_root: str = Field(min_length=1)
    repo_root: str = Field(min_length=1)
    final_test_ledger_path: str = Field(min_length=1)
    output_layout: AttemptOutputLayoutV2
    candidates: list[ReplayWorkerCandidateV2] = Field(min_length=1)
    input_hashes: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_inputs(self) -> ReplayWorkerSpecV2:
        if self.created_at.utcoffset() is None:
            raise ValueError("worker spec created_at must be timezone-aware")
        for field_name in ("artifact_root", "repo_root", "final_test_ledger_path"):
            if not Path(getattr(self, field_name)).is_absolute():
                raise ValueError(f"{field_name} must be absolute")
        if self.output_layout.artifact_root != str(Path(self.artifact_root).resolve()):
            raise ValueError("output layout differs from worker artifact_root")
        candidate_ids = [item.candidate_id for item in self.candidates]
        if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("worker candidates must be sorted and unique")
        required = {
            V2ArtifactStore.MANIFEST_NAME,
            V2ArtifactStore.CONFIG_NAME,
            V2ArtifactStore.POLICY_NAME,
            V2ArtifactStore.DATA_MANIFEST_NAME,
            V2ArtifactStore.CODE_MANIFEST_NAME,
            V2ArtifactStore.SPLIT_MANIFEST_NAME,
            *(item.relative_path for item in self.candidates),
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
class PreparedReplayWorkerV2:
    spec: ReplayWorkerSpecV2
    spec_path: Path
    spec_file_sha256: str


@dataclass(frozen=True)
class LoadedReplayWorkerV2:
    spec: ReplayWorkerSpecV2
    manifest: AttemptManifestV2
    resolved_config: dict[str, Any]
    policy: ShadowGatePolicyV2
    data_manifest: DataManifestV2
    code_manifest: CodeManifestV2
    split_manifest: EvaluationSplitPlanV2
    candidates: tuple[FrozenCandidateV2, ...]


def _base_input_paths() -> tuple[str, ...]:
    return (
        V2ArtifactStore.MANIFEST_NAME,
        V2ArtifactStore.CONFIG_NAME,
        V2ArtifactStore.POLICY_NAME,
        V2ArtifactStore.DATA_MANIFEST_NAME,
        V2ArtifactStore.CODE_MANIFEST_NAME,
        V2ArtifactStore.SPLIT_MANIFEST_NAME,
    )


def prepare_replay_worker(
    *,
    bundle: AttemptManifestBundleV2,
    resolved_config: Mapping[str, Any],
    policy: ShadowGatePolicyV2,
    candidates: Sequence[FrozenCandidateV2],
    repo_root: str | Path,
    final_test_ledger_path: str | Path,
    created_at: datetime,
) -> PreparedReplayWorkerV2:
    """Persist every immutable worker input before an attempt can be queued."""

    manifest = bundle.manifest
    root = Path(manifest.artifact_root).resolve()
    resolved_repo = Path(repo_root).resolve()
    resolved_ledger = Path(final_test_ledger_path).resolve()
    if not resolved_repo.is_dir():
        raise ReplayWorkerError(f"repo_root does not exist: {resolved_repo}")
    if root == resolved_ledger or root in resolved_ledger.parents:
        raise ReplayWorkerError("final-test ledger must live outside the attempt artifact root")
    if canonical_config_hash(dict(resolved_config)) != manifest.config_hash:
        raise ReplayWorkerError("resolved config differs from attempt manifest")
    if shadow_gate_policy_from_config(resolved_config) != policy:
        raise ReplayWorkerError("promotion policy differs from resolved config")
    if created_at.utcoffset() is None or created_at < manifest.created_at:
        raise ReplayWorkerError("worker spec timestamp precedes attempt creation")
    if bundle.data_manifest.manifest_hash != manifest.data_manifest_hash:
        raise ReplayWorkerError("data manifest differs from attempt manifest")
    if bundle.split_manifest.split_hash != manifest.split_manifest_hash:
        raise ReplayWorkerError("split manifest differs from attempt manifest")
    candidate_list = sorted(candidates, key=lambda item: item.candidate_id)
    if not candidate_list:
        raise ReplayWorkerError("replay worker requires at least one frozen candidate")
    candidate_ids = [item.candidate_id for item in candidate_list]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ReplayWorkerError("replay worker candidate IDs must be unique")

    store = V2ArtifactStore(root)
    store.write_manifest(manifest, resolved_config)
    store.write_policy(policy)
    store.write_data_manifest(bundle.data_manifest, manifest.data_manifest_hash)
    store.write_code_manifest(bundle.code_manifest, manifest)
    store.write_split_manifest(bundle.split_manifest, str(manifest.split_manifest_hash))
    for candidate in candidate_list:
        store.write_frozen_candidate(candidate)

    input_paths = [*_base_input_paths()]
    worker_candidates = []
    for candidate in candidate_list:
        relative = f"candidates/{candidate.candidate_id}/frozen_candidate.json"
        input_paths.append(relative)
        worker_candidates.append(
            ReplayWorkerCandidateV2(
                candidate_id=candidate.candidate_id,
                relative_path=relative,
                phenotype_hash=candidate.phenotype_hash,
            )
        )
    input_hashes = {
        relative: _sha256_file(root / _safe_relative_path(relative))
        for relative in sorted(input_paths)
    }
    spec = ReplayWorkerSpecV2(
        attempt_id=manifest.attempt_id,
        created_at=created_at,
        artifact_root=str(root),
        repo_root=str(resolved_repo),
        final_test_ledger_path=str(resolved_ledger),
        output_layout=AttemptOutputLayoutV2.for_artifact_root(root),
        candidates=worker_candidates,
        input_hashes=input_hashes,
    )
    spec_path = store.write_worker_spec(spec)
    _reject_unexpected_preexecution_files(root, spec)
    return PreparedReplayWorkerV2(
        spec=spec,
        spec_path=spec_path,
        spec_file_sha256=_sha256_file(spec_path),
    )


def replay_worker_argv(
    prepared: PreparedReplayWorkerV2,
    *,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    executable = Path(python_executable).absolute()
    return [
        str(executable),
        "-m",
        "genetic_algorithm.orchestration.replay_worker_v2",
        "--spec",
        str(prepared.spec_path.resolve()),
        "--spec-sha256",
        prepared.spec_file_sha256,
    ]


def _reject_unexpected_preexecution_files(root: Path, spec: ReplayWorkerSpecV2) -> None:
    allowed = set(spec.input_hashes) | {V2ArtifactStore.WORKER_SPEC_NAME}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.relative_to(root).as_posix().startswith("runtime/")
    }
    unexpected = sorted(actual - allowed)
    missing = sorted(allowed - actual)
    if missing or unexpected:
        raise ReplayWorkerError(
            f"pre-execution artifact set differs; missing={missing}, unexpected={unexpected}"
        )


def _load_bound_spec(
    spec_path: str | Path,
    expected_spec_sha256: str,
) -> tuple[ReplayWorkerSpecV2, Path]:
    path = Path(spec_path).resolve()
    if path.name != V2ArtifactStore.WORKER_SPEC_NAME or not path.is_file():
        raise ReplayWorkerError("worker spec path is not canonical or does not exist")
    if len(expected_spec_sha256) != 64 or _sha256_file(path) != expected_spec_sha256:
        raise ReplayWorkerError("worker spec SHA-256 differs from claimed command")
    try:
        spec = ReplayWorkerSpecV2.model_validate_json(path.read_bytes())
    except Exception as exc:
        raise ReplayWorkerError("worker spec violates the V2 contract") from exc
    root = path.parent
    if root != Path(spec.artifact_root).resolve():
        raise ReplayWorkerError("worker spec artifact_root differs from its location")
    return spec, root


def _verify_worker_inputs(root: Path, spec: ReplayWorkerSpecV2) -> None:
    _reject_unexpected_preexecution_files(root, spec)
    for relative, expected in spec.input_hashes.items():
        input_path = (root / _safe_relative_path(relative)).resolve()
        if root not in input_path.parents or not input_path.is_file():
            raise ReplayWorkerError(f"worker input is missing or escapes root: {relative}")
        if _sha256_file(input_path) != expected:
            raise ReplayWorkerError(f"worker input hash differs: {relative}")


def _load_resolved_config(root: Path, manifest: AttemptManifestV2) -> dict[str, Any]:
    try:
        raw_config = yaml.safe_load((root / V2ArtifactStore.CONFIG_NAME).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ReplayWorkerError("resolved config cannot be loaded") from exc
    if not isinstance(raw_config, Mapping):
        raise ReplayWorkerError("resolved config must be a mapping")
    resolved_config = dict(raw_config)
    if canonical_config_hash(resolved_config) != manifest.config_hash:
        raise ReplayWorkerError("resolved config semantic hash differs from manifest")
    try:
        validate_resolved_config_v2_or_raise(resolved_config)
    except ValueError as exc:
        raise ReplayWorkerError("resolved config violates the V2 contract") from exc
    return resolved_config


def _load_candidates(
    root: Path,
    spec: ReplayWorkerSpecV2,
) -> tuple[FrozenCandidateV2, ...]:
    candidates = tuple(
        FrozenCandidateV2.model_validate_json((root / item.relative_path).read_bytes())
        for item in spec.candidates
    )
    for declared, candidate in zip(spec.candidates, candidates, strict=True):
        if (
            declared.candidate_id != candidate.candidate_id
            or declared.phenotype_hash != candidate.phenotype_hash
        ):
            raise ReplayWorkerError("frozen candidate differs from worker spec")
    return candidates


def load_replay_worker(
    spec_path: str | Path,
    *,
    expected_spec_sha256: str,
) -> LoadedReplayWorkerV2:
    spec, root = _load_bound_spec(spec_path, expected_spec_sha256)
    _verify_worker_inputs(root, spec)

    manifest = AttemptManifestV2.model_validate_json(
        (root / V2ArtifactStore.MANIFEST_NAME).read_bytes()
    )
    if manifest.attempt_id != spec.attempt_id:
        raise ReplayWorkerError("worker spec attempt_id differs from manifest")
    if Path(manifest.artifact_root).resolve() != root:
        raise ReplayWorkerError("manifest artifact_root differs from worker root")
    if Path(manifest.resolved_config_path).resolve() != root / V2ArtifactStore.CONFIG_NAME:
        raise ReplayWorkerError("manifest resolved_config_path is not canonical")
    resolved_config = _load_resolved_config(root, manifest)
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
    candidates = _load_candidates(root, spec)
    return LoadedReplayWorkerV2(
        spec=spec,
        manifest=manifest,
        resolved_config=resolved_config,
        policy=policy,
        data_manifest=data_manifest,
        code_manifest=code_manifest,
        split_manifest=split_manifest,
        candidates=candidates,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _attempt_backtester_factory(
    layout: AttemptOutputLayoutV2,
    delegate: Callable[[dict[str, Any]], Any],
) -> Callable[[dict[str, Any]], Any]:
    def factory(config: dict[str, Any]) -> Any:
        return delegate(layout.isolated_backtester_config(config, phase="replay"))

    return factory


def _failure_manifest(spec_path: str | Path) -> AttemptManifestV2:
    path = Path(spec_path).resolve()
    if path.name != V2ArtifactStore.WORKER_SPEC_NAME:
        raise ReplayWorkerError("cannot recover manifest from a non-canonical spec path")
    manifest_path = path.parent / V2ArtifactStore.MANIFEST_NAME
    manifest = AttemptManifestV2.model_validate_json(manifest_path.read_bytes())
    if Path(manifest.artifact_root).resolve() != path.parent:
        raise ReplayWorkerError("failure manifest artifact_root differs from spec location")
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
        if isinstance(error, (ReplayWorkerError, ArtifactIntegrityError))
        else "WORKER_EXECUTION_FAILED"
    )
    detail = f"{type(error).__name__}: {error}"[:2000]
    return store.finalize(
        {
            "attempt_id": manifest.attempt_id,
            "status": "INVALID_RESULT",
            "started_at": manifest.created_at,
            "finished_at": max(finished_at, manifest.created_at),
            "manifest": manifest.model_dump(mode="python"),
            "candidate_evaluations": [],
            "error_code": error_code,
            "error_detail": detail,
        }
    )


def run_replay_worker(
    spec_path: str | Path,
    *,
    expected_spec_sha256: str,
    backtester_factory: Callable[[dict[str, Any]], Any] = DirectBacktester,
    clock: Callable[[], datetime] = _utc_now,
) -> AttemptResultV2:
    """Run one persisted spec and always commit a terminal result when bootstrap permits."""

    try:
        loaded = load_replay_worker(
            spec_path,
            expected_spec_sha256=expected_spec_sha256,
        )
        runner = ShadowReplayRunnerV2(
            manifest=loaded.manifest,
            resolved_config=loaded.resolved_config,
            policy=loaded.policy,
            data_manifest=loaded.data_manifest,
            code_manifest=loaded.code_manifest,
            final_test_ledger=FinalTestUsageLedgerV2(loaded.spec.final_test_ledger_path),
            repo_root=loaded.spec.repo_root,
            backtester_factory=_attempt_backtester_factory(
                loaded.spec.output_layout,
                backtester_factory,
            ),
            clock=clock,
        )
        with attempt_output_scope(loaded.spec.output_layout, phase="replay"):
            return runner.execute(loaded.candidates, finished_at=clock())
    except Exception as exc:
        return _finalize_worker_failure(spec_path, exc, finished_at=clock())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one immutable GA V2 replay worker spec")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = run_replay_worker(
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
