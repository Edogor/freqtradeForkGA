"""Build mutually consistent V2 attempt, code, and market-data manifests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import model_validator

from genetic_algorithm.config.schema import validate_resolved_config_v2_or_raise
from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.code_manifest_v2 import CodeManifestV2, capture_code_manifest
from genetic_algorithm.orchestration.data_manifest_v2 import DataManifestV2, build_data_manifest
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ShadowGatePolicyV2,
    shadow_gate_policy_from_config,
)
from genetic_algorithm.orchestration.result_contract import AttemptManifestV2, StrictV2Model
from genetic_algorithm.orchestration.split_contract_v2 import (
    EvaluationSplitPlanV2,
    build_evaluation_split_plan,
)


class AttemptManifestBundleV2(StrictV2Model):
    """Pre-execution provenance bundle consumed by the replay runner."""

    manifest: AttemptManifestV2
    code_manifest: CodeManifestV2
    data_manifest: DataManifestV2
    split_manifest: EvaluationSplitPlanV2

    @model_validator(mode="after")
    def _consistent_hashes(self) -> AttemptManifestBundleV2:
        if self.manifest.code_version != self.code_manifest.code_version:
            raise ValueError("attempt code_version differs from code manifest")
        if self.manifest.dirty_patch_hash != self.code_manifest.dirty_patch_hash:
            raise ValueError("attempt dirty_patch_hash differs from code manifest")
        if self.manifest.data_manifest_hash != self.data_manifest.manifest_hash:
            raise ValueError("attempt data_manifest_hash differs from data manifest")
        if self.manifest.split_manifest_hash != self.split_manifest.split_hash:
            raise ValueError("attempt split_manifest_hash differs from split manifest")
        return self


def build_attempt_manifest_bundle(
    *,
    attempt_id: str,
    wave_id: str,
    experiment_id: str,
    created_at: datetime,
    resolved_config: Mapping[str, Any],
    policy: ShadowGatePolicyV2,
    fitness_policy_version: str,
    seeds: list[int],
    worker_count: int,
    artifact_root: str | Path,
    repo_root: str | Path,
    parent_wave_id: str | None = None,
    data_root: str | Path | None = None,
) -> AttemptManifestBundleV2:
    """Resolve every hash before execution without writing or starting an attempt."""

    config = dict(resolved_config)
    validate_resolved_config_v2_or_raise(config)
    configured_policy = shadow_gate_policy_from_config(config)
    if configured_policy != policy:
        raise ValueError("policy differs from resolved promotion_v2 config")
    captured_code = capture_code_manifest(repo_root)
    captured_data = build_data_manifest(config, policy, data_root=data_root)
    split_manifest = build_evaluation_split_plan(config, policy)
    root = Path(artifact_root).resolve()
    manifest = AttemptManifestV2(
        attempt_id=attempt_id,
        wave_id=wave_id,
        parent_wave_id=parent_wave_id,
        experiment_id=experiment_id,
        created_at=created_at,
        config_hash=canonical_config_hash(config),
        code_version=captured_code.code_version,
        dirty_patch_hash=captured_code.dirty_patch_hash,
        data_manifest_hash=captured_data.manifest_hash,
        split_manifest_hash=split_manifest.split_hash,
        fitness_policy_version=fitness_policy_version,
        seeds=seeds,
        worker_count=worker_count,
        resolved_config_path=str(root / "resolved_config.yaml"),
        artifact_root=str(root),
    )
    return AttemptManifestBundleV2(
        manifest=manifest,
        code_manifest=captured_code,
        data_manifest=captured_data,
        split_manifest=split_manifest,
    )
