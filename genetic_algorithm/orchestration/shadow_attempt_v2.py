"""Runtime recorder joining V2 scenario records, gates, and terminal artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ShadowGatePolicyV2,
    evaluate_candidate_shadow,
    make_shadow_promotion_decision,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptResultV2,
    BacktestRecordV2,
)


class ShadowAttemptRecorderV2:
    """Collect immutable replay records and commit exactly one terminal result."""

    def __init__(
        self,
        manifest: AttemptManifestV2,
        resolved_config: Mapping[str, Any],
        policy: ShadowGatePolicyV2,
        data_manifest: BaseModel | None = None,
        code_manifest: BaseModel | None = None,
        split_manifest: BaseModel | None = None,
    ) -> None:
        self.manifest = manifest
        self.policy = policy
        self.store = V2ArtifactStore(manifest.artifact_root)
        self.store.write_manifest(manifest, resolved_config)
        self.store.write_policy(policy)
        if data_manifest is not None:
            self.store.write_data_manifest(data_manifest, manifest.data_manifest_hash)
        if code_manifest is not None:
            self.store.write_code_manifest(code_manifest, manifest)
        if split_manifest is not None:
            if manifest.split_manifest_hash is None:
                raise ArtifactIntegrityError("attempt manifest lacks split_manifest_hash")
            self.store.write_split_manifest(split_manifest, manifest.split_manifest_hash)
        self._records: dict[str, dict[str, BacktestRecordV2]] = {}
        self._finalized = False

    def add_frozen_candidate(self, candidate: BaseModel) -> None:
        if self._finalized:
            raise ArtifactIntegrityError("cannot add candidates after terminal result")
        self.store.write_frozen_candidate(candidate)

    def add_final_test_usage(self, usage: BaseModel) -> None:
        if self._finalized:
            raise ArtifactIntegrityError("cannot add final-test usage after terminal result")
        self.store.write_final_test_usage(usage, self.manifest)

    def add_backtest(self, record: BacktestRecordV2) -> None:
        if self._finalized:
            raise ArtifactIntegrityError("cannot add backtests after terminal result")
        expected = {
            "attempt_id": self.manifest.attempt_id,
            "wave_id": self.manifest.wave_id,
            "experiment_id": self.manifest.experiment_id,
            "config_hash": self.manifest.config_hash,
            "code_version": self.manifest.code_version,
            "data_manifest_hash": self.manifest.data_manifest_hash,
            "fitness_policy_version": self.manifest.fitness_policy_version,
            "worker_count": self.manifest.worker_count,
        }
        mismatches = {
            field: (value, getattr(record, field))
            for field, value in expected.items()
            if getattr(record, field) != value
        }
        if record.seed not in self.manifest.seeds:
            mismatches["seed"] = (self.manifest.seeds, record.seed)
        if mismatches:
            raise ArtifactIntegrityError(f"backtest provenance mismatch: {mismatches}")

        by_scenario = self._records.setdefault(record.candidate_id, {})
        existing = by_scenario.get(record.metrics.scenario_id)
        if existing is not None and existing != record:
            raise ArtifactIntegrityError(
                "candidate scenario was recorded twice with different content"
            )
        self.store.write_backtest(record)
        by_scenario[record.metrics.scenario_id] = record

    def finalize(self, *, finished_at: datetime) -> AttemptResultV2:
        if self._finalized:
            return self.store.read_verified_result()
        if finished_at < self.manifest.created_at:
            raise ArtifactIntegrityError("finished_at precedes manifest creation")

        candidates = []
        for candidate_id in sorted(self._records):
            records = [
                self._records[candidate_id][scenario_id]
                for scenario_id in sorted(self._records[candidate_id])
            ]
            candidate = evaluate_candidate_shadow(records, self.policy)
            decision = make_shadow_promotion_decision(
                candidate,
                wave_id=self.manifest.wave_id,
                policy_version=self.policy.policy_version,
                created_at=finished_at,
            )
            self.store.write_candidate(candidate)
            self.store.write_decision(decision)
            candidates.append(candidate)

        if candidates:
            status = "SUCCEEDED"
            error_code = None
            error_detail = None
        else:
            status = "INVALID_RESULT"
            error_code = "NO_SCENARIO_RECORDS"
            error_detail = "attempt completed without any V2 backtest record"

        draft = {
            "attempt_id": self.manifest.attempt_id,
            "status": status,
            "started_at": self.manifest.created_at,
            "finished_at": finished_at,
            "manifest": self.manifest.model_dump(mode="python"),
            "candidate_evaluations": [
                candidate.model_dump(mode="python") for candidate in candidates
            ],
            "error_code": error_code,
            "error_detail": error_detail,
        }
        result = self.store.finalize(draft)
        self._finalized = True
        return result
