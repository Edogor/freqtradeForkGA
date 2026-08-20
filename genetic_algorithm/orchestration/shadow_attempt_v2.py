"""Runtime recorder joining V2 scenario records, gates, and terminal artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    V2ArtifactStore,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ShadowGatePolicyV2,
    evaluate_candidate_shadow,
    make_shadow_promotion_decision,
)
from genetic_algorithm.orchestration.promotion_policy_v3 import (
    QualificationPolicyV3,
    evaluate_candidate_v3,
    qualification_policy_v3_from_config,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptResultV2,
    BacktestRecordV2,
    CandidateArtifactDispositionV2,
    CandidateEvaluationV2,
    EvaluationStatus,
)


def _scenario_behavior_sort_key(record: BacktestRecordV2) -> tuple[object, ...]:
    """Return the predeclared scenario identity, independent of candidate."""

    metrics = record.metrics
    return (
        metrics.scenario_id,
        metrics.pair,
        metrics.timeframe,
        metrics.role.value,
        metrics.period_start.isoformat(),
        metrics.period_end.isoformat(),
        metrics.cost_multiplier,
    )


def _exact_behavior_signature(candidate: CandidateEvaluationV2) -> str:
    """Hash complete replay behavior without using aggregate score proxies."""

    scenarios = []
    for record in sorted(candidate.scenarios, key=_scenario_behavior_sort_key):
        metrics = record.metrics
        scenarios.append(
            {
                "scenario_identity": {
                    "scenario_id": metrics.scenario_id,
                    "pair": metrics.pair,
                    "timeframe": metrics.timeframe,
                    "role": metrics.role.value,
                    "period_start": metrics.period_start.isoformat(),
                    "period_end": metrics.period_end.isoformat(),
                    "cost_multiplier": metrics.cost_multiplier,
                },
                "daily_net_returns": record.daily_net_returns,
                "equity_curve": record.equity_curve,
                "equity_method": record.equity_method,
                "trades": record.trades,
            }
        )
    return canonical_config_hash(
        {
            "contract": "EXACT_V2_REPLAY_BEHAVIOR_V1",
            "scenarios": scenarios,
        }
    )


def _deduplicate_valid_behaviors(
    candidates: list[CandidateEvaluationV2],
) -> list[CandidateEvaluationV2]:
    """Keep the lexicographically first candidate for each exact VALID behavior."""

    selected, _ = _classify_candidate_artifacts(candidates)
    return selected


def _classify_candidate_artifacts(
    candidates: list[CandidateEvaluationV2],
) -> tuple[list[CandidateEvaluationV2], list[CandidateArtifactDispositionV2]]:
    """Select authoritative results and retain explicit duplicate provenance."""

    selected: list[CandidateEvaluationV2] = []
    dispositions: list[CandidateArtifactDispositionV2] = []
    valid_signatures: dict[str, str] = {}
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        if candidate.status != EvaluationStatus.VALID:
            selected.append(candidate)
            dispositions.append(
                CandidateArtifactDispositionV2(
                    candidate_id=candidate.candidate_id,
                    phenotype_hash=candidate.phenotype_hash,
                    evaluation_status=candidate.status,
                    authoritative=True,
                    reason_code="AUTHORITATIVE_RESULT",
                    canonical_candidate_id=candidate.candidate_id,
                )
            )
            continue
        signature = _exact_behavior_signature(candidate)
        canonical_candidate_id = valid_signatures.get(signature)
        if canonical_candidate_id is not None:
            dispositions.append(
                CandidateArtifactDispositionV2(
                    candidate_id=candidate.candidate_id,
                    phenotype_hash=candidate.phenotype_hash,
                    evaluation_status=candidate.status,
                    authoritative=False,
                    reason_code="EXACT_BEHAVIOR_DUPLICATE",
                    canonical_candidate_id=canonical_candidate_id,
                    behavior_signature=signature,
                )
            )
            continue
        valid_signatures[signature] = candidate.candidate_id
        selected.append(candidate)
        dispositions.append(
            CandidateArtifactDispositionV2(
                candidate_id=candidate.candidate_id,
                phenotype_hash=candidate.phenotype_hash,
                evaluation_status=candidate.status,
                authoritative=True,
                reason_code="AUTHORITATIVE_RESULT",
                canonical_candidate_id=candidate.candidate_id,
                behavior_signature=signature,
            )
        )
    return selected, dispositions


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
        self.qualification_policy_v3: QualificationPolicyV3 | None = None
        qualification_config = resolved_config.get("qualification_v3")
        if isinstance(qualification_config, Mapping) and qualification_config.get(
            "enabled", False
        ):
            self.qualification_policy_v3 = qualification_policy_v3_from_config(
                resolved_config
            )
            v2_cells = {
                (
                    item.scenario_id,
                    item.pair,
                    item.timeframe,
                    item.role,
                    str(item.period_start),
                    str(item.period_end),
                    item.cost_multiplier,
                )
                for item in policy.required_scenarios
            }
            v3_cells = {
                item.record_key
                for item in self.qualification_policy_v3.required_scenarios
            }
            if v2_cells != v3_cells:
                raise ArtifactIntegrityError(
                    "qualification_v3 and promotion_v2 scenario matrices differ"
                )
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
        scenario_path = self.store.write_backtest(record)
        # Keep in-memory state in the exact JSON-normalized form persisted by
        # the immutable artifact. Raw trade dictionaries may contain values
        # such as pandas.Timestamp which serialize canonically to strings.
        by_scenario[record.metrics.scenario_id] = BacktestRecordV2.model_validate_json(
            scenario_path.read_bytes()
        )

    def finalize(self, *, finished_at: datetime) -> AttemptResultV2:
        if self._finalized:
            return self.store.read_verified_result()
        if finished_at < self.manifest.created_at:
            raise ArtifactIntegrityError("finished_at precedes manifest creation")

        evaluated_candidates = []
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
            if self.qualification_policy_v3 is not None:
                qualification = evaluate_candidate_v3(
                    records,
                    self.qualification_policy_v3,
                )
                self.store.write_qualification_v3(qualification)
            evaluated_candidates.append(candidate)

        candidates, dispositions = _classify_candidate_artifacts(evaluated_candidates)
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
            "candidate_artifact_dispositions": [
                disposition.model_dump(mode="python")
                for disposition in dispositions
            ],
            "error_code": error_code,
            "error_detail": error_detail,
        }
        result = self.store.finalize(draft)
        self._finalized = True
        return result
