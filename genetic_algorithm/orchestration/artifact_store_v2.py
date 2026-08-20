"""Immutable, atomic and hash-verified persistence for GA V2 attempts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from genetic_algorithm.orchestration.promotion_policy_v2 import ShadowGatePolicyV2
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptResultV2,
    BacktestRecordV2,
    CandidateEvaluationV2,
    PromotionDecisionV2,
)


class ArtifactIntegrityError(ValueError):
    """Raised when an artifact is missing, mutable, corrupt, or inconsistent."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_model_bytes(model: BaseModel) -> bytes:
    payload = model.model_dump(mode="json", exclude_none=False)
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _models_have_same_canonical_payload(left: BaseModel, right: BaseModel) -> bool:
    """Compare persisted contract content instead of Python runtime types.

    ``BacktestRecordV2.trades`` intentionally preserves the engine's raw JSON
    payload. Values typed as ``Any`` can therefore contain JSON-compatible
    Python subclasses such as ``pandas.Timestamp`` before persistence and
    plain strings after readback. Artifact integrity is defined by the
    immutable canonical bytes, so comparisons at that boundary must use the
    same representation.
    """

    return _canonical_model_bytes(left) == _canonical_model_bytes(right)


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _normalize_config_value(value.value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ArtifactIntegrityError("resolved config contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _normalize_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if hasattr(value, "item"):
        return _normalize_config_value(value.item())
    raise ArtifactIntegrityError(
        f"resolved config contains unsupported type: {type(value).__name__}"
    )


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    normalized = _normalize_config_value(payload)
    return (
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_id(value: str, field_name: str) -> str:
    if not value or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in value
    ):
        raise ArtifactIntegrityError(
            f"{field_name} may contain only letters, digits, dot, underscore, and dash"
        )
    if value in {".", ".."}:
        raise ArtifactIntegrityError(f"invalid {field_name}")
    return value


class V2ArtifactStore:
    """Persist one attempt with ``result.json`` as the final commit marker.

    Every write is atomic and immutable. Repeating a byte-identical write is
    idempotent; changing an already persisted artifact raises. A terminal
    result is accepted on read only when its checksum and every declared
    artifact hash still match.
    """

    MANIFEST_NAME = "manifest.json"
    CONFIG_NAME = "resolved_config.yaml"
    RESULT_NAME = "result.json"
    RESULT_CHECKSUM_NAME = "result.json.sha256"
    POLICY_NAME = "promotion_policy.json"
    DATA_MANIFEST_NAME = "data_manifest.json"
    CODE_MANIFEST_NAME = "code_manifest.json"
    FINAL_TEST_USAGE_NAME = "final_test_usage.json"
    SPLIT_MANIFEST_NAME = "split_manifest.json"
    WORKER_SPEC_NAME = "worker_spec.json"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, relative: str | Path) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ArtifactIntegrityError(f"unsafe artifact path: {relative_path}")
        path = self.root / relative_path
        if (
            path.resolve() == self.root.resolve()
            or self.root.resolve() not in path.resolve().parents
        ):
            raise ArtifactIntegrityError(f"artifact path escapes root: {relative_path}")
        return path

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        try:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            # Some filesystems do not support directory fsync. File fsync and
            # atomic replace still protect readers from partial JSON.
            pass

    def _write_immutable(self, relative: str | Path, payload: bytes) -> Path:
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() == payload:
                return path
            raise ArtifactIntegrityError(f"immutable artifact already differs: {relative}")

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
            # Refuse to overwrite a file that appeared after the existence check.
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ArtifactIntegrityError(f"concurrent artifact write differs: {relative}")
            self._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def write_manifest(
        self,
        manifest: AttemptManifestV2,
        resolved_config: Mapping[str, Any],
    ) -> dict[str, str]:
        declared_root = Path(manifest.artifact_root).resolve()
        if declared_root != self.root.resolve():
            raise ArtifactIntegrityError("manifest artifact_root differs from store root")
        if canonical_config_hash(resolved_config) != manifest.config_hash:
            raise ArtifactIntegrityError("resolved config hash differs from manifest config_hash")
        declared_config = Path(manifest.resolved_config_path).resolve()
        if declared_config != self._path(self.CONFIG_NAME).resolve():
            raise ArtifactIntegrityError(
                "manifest resolved_config_path differs from canonical artifact path"
            )

        normalized_config = _normalize_config_value(resolved_config)
        config_bytes = yaml.safe_dump(
            normalized_config,
            sort_keys=True,
            default_flow_style=False,
        ).encode("utf-8")
        config_path = self._write_immutable(self.CONFIG_NAME, config_bytes)
        manifest_path = self._write_immutable(self.MANIFEST_NAME, _canonical_model_bytes(manifest))
        return {
            self.CONFIG_NAME: _sha256_file(config_path),
            self.MANIFEST_NAME: _sha256_file(manifest_path),
        }

    def write_backtest(self, record: BacktestRecordV2) -> Path:
        self._validate_record_provenance(record, self._read_manifest())
        candidate = _safe_id(record.candidate_id, "candidate_id")
        scenario = _safe_id(record.metrics.scenario_id, "scenario_id")
        return self._write_immutable(
            Path("candidates") / candidate / "scenarios" / f"{scenario}.json",
            _canonical_model_bytes(record),
        )

    def write_frozen_candidate(self, candidate: BaseModel) -> Path:
        candidate_id = _safe_id(str(getattr(candidate, "candidate_id", "")), "candidate_id")
        phenotype_hash = getattr(candidate, "phenotype_hash", None)
        if not isinstance(phenotype_hash, str) or len(phenotype_hash) != 64:
            raise ArtifactIntegrityError("frozen candidate lacks a SHA-256 phenotype_hash")
        return self._write_immutable(
            Path("candidates") / candidate_id / "frozen_candidate.json",
            _canonical_model_bytes(candidate),
        )

    def write_evolution_seed(self, seed: BaseModel) -> Path:
        """Persist the executable-linked genome needed for a future warm start."""

        candidate_id = _safe_id(str(getattr(seed, "candidate_id", "")), "candidate_id")
        phenotype_hash = getattr(seed, "phenotype_hash", None)
        if not isinstance(phenotype_hash, str) or len(phenotype_hash) != 64:
            raise ArtifactIntegrityError("evolution seed lacks a SHA-256 phenotype_hash")
        return self._write_immutable(
            Path("candidates") / candidate_id / "evolution_seed.json",
            _canonical_model_bytes(seed),
        )

    def write_evolution_seed_input(self, seed: BaseModel) -> Path:
        """Copy a parent genome into the immutable worker-input namespace."""

        candidate_id = _safe_id(str(getattr(seed, "candidate_id", "")), "candidate_id")
        return self._write_immutable(
            Path("inputs") / "seeds" / f"{candidate_id}.json",
            _canonical_model_bytes(seed),
        )

    def write_evolution_checkpoint_input(
        self,
        *,
        filename: str,
        payload: bytes,
    ) -> Path:
        """Bind one verified Generic-Island resume checkpoint as worker input."""

        safe_name = _safe_id(filename, "checkpoint filename")
        if not safe_name.startswith("island_checkpoint_gen") or not safe_name.endswith(
            ".json"
        ):
            raise ArtifactIntegrityError("resume checkpoint filename is not canonical")
        return self._write_immutable(
            Path("inputs") / "checkpoints" / safe_name,
            payload,
        )

    def write_engine_config(self, config: Mapping[str, Any]) -> Path:
        """Persist the derived, attempt-local engine config as immutable evidence."""

        normalized = _normalize_config_value(config)
        payload = yaml.safe_dump(
            normalized,
            sort_keys=True,
            default_flow_style=False,
        ).encode("utf-8")
        return self._write_immutable(Path("evolution") / "engine_config.yaml", payload)

    def write_candidate(self, candidate: CandidateEvaluationV2) -> Path:
        candidate_id = _safe_id(candidate.candidate_id, "candidate_id")
        manifest = self._read_manifest()
        frozen_path = self._path(Path("candidates") / candidate_id / "frozen_candidate.json")
        if frozen_path.exists():
            try:
                frozen_payload = json.loads(frozen_path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactIntegrityError("frozen candidate artifact is invalid JSON") from exc
            if frozen_payload.get("candidate_id") != candidate.candidate_id:
                raise ArtifactIntegrityError("frozen candidate ID differs from evaluation")
            if frozen_payload.get("phenotype_hash") != candidate.phenotype_hash:
                raise ArtifactIntegrityError("frozen candidate phenotype differs from evaluation")
        for record in candidate.scenarios:
            self._validate_record_provenance(record, manifest)
            scenario = _safe_id(record.metrics.scenario_id, "scenario_id")
            scenario_path = self._path(
                Path("candidates") / candidate_id / "scenarios" / f"{scenario}.json"
            )
            if not scenario_path.exists():
                raise ArtifactIntegrityError(
                    f"candidate references missing scenario artifact: {scenario}"
                )
            persisted = BacktestRecordV2.model_validate_json(scenario_path.read_bytes())
            if not _models_have_same_canonical_payload(persisted, record):
                raise ArtifactIntegrityError(
                    f"candidate scenario differs from artifact: {scenario}"
                )
        return self._write_immutable(
            Path("candidates") / candidate_id / "candidate.json",
            _canonical_model_bytes(candidate),
        )

    def write_qualification_v3(self, qualification: BaseModel) -> Path:
        """Persist additive V3 qualification without mutating V2 result semantics."""

        from genetic_algorithm.orchestration.promotion_policy_v3 import (
            CandidateEvaluationV3,
        )

        candidate = CandidateEvaluationV3.model_validate(
            qualification.model_dump(mode="python")
        )
        candidate_id = _safe_id(candidate.candidate_id, "candidate_id")
        manifest = self._read_manifest()
        for record in candidate.scenarios:
            self._validate_record_provenance(record, manifest)
            scenario = _safe_id(record.metrics.scenario_id, "scenario_id")
            scenario_path = self._path(
                Path("candidates") / candidate_id / "scenarios" / f"{scenario}.json"
            )
            if not scenario_path.exists():
                raise ArtifactIntegrityError(
                    f"V3 qualification references missing scenario artifact: {scenario}"
                )
            persisted = BacktestRecordV2.model_validate_json(scenario_path.read_bytes())
            if not _models_have_same_canonical_payload(persisted, record):
                raise ArtifactIntegrityError(
                    f"V3 qualification scenario differs from artifact: {scenario}"
                )
        return self._write_immutable(
            Path("candidates") / candidate_id / "qualification_v3.json",
            _canonical_model_bytes(candidate),
        )

    def write_policy(self, policy: ShadowGatePolicyV2) -> Path:
        return self._write_immutable(self.POLICY_NAME, _canonical_model_bytes(policy))

    def write_data_manifest(self, data_manifest: BaseModel, expected_hash: str) -> Path:
        actual_hash = canonical_config_hash(data_manifest.model_dump(mode="json"))
        if actual_hash != expected_hash:
            raise ArtifactIntegrityError(
                "data manifest content differs from AttemptManifestV2.data_manifest_hash"
            )
        return self._write_immutable(
            self.DATA_MANIFEST_NAME,
            _canonical_model_bytes(data_manifest),
        )

    def write_code_manifest(
        self,
        code_manifest: BaseModel,
        manifest: AttemptManifestV2,
    ) -> Path:
        code_version = getattr(code_manifest, "code_version", None)
        dirty_patch_hash = getattr(code_manifest, "dirty_patch_hash", None)
        if code_version != manifest.code_version:
            raise ArtifactIntegrityError("code manifest differs from manifest code_version")
        if dirty_patch_hash != manifest.dirty_patch_hash:
            raise ArtifactIntegrityError("code manifest differs from manifest dirty_patch_hash")
        return self._write_immutable(
            self.CODE_MANIFEST_NAME,
            _canonical_model_bytes(code_manifest),
        )

    def write_final_test_usage(
        self,
        usage: BaseModel,
        manifest: AttemptManifestV2,
    ) -> Path:
        expected = {
            "attempt_id": manifest.attempt_id,
            "wave_id": manifest.wave_id,
            "data_manifest_hash": manifest.data_manifest_hash,
        }
        mismatches = {
            field: (value, getattr(usage, field, None))
            for field, value in expected.items()
            if getattr(usage, field, None) != value
        }
        policy_path = self._path(self.POLICY_NAME)
        if not policy_path.exists():
            raise ArtifactIntegrityError("promotion policy must precede final-test usage")
        policy = ShadowGatePolicyV2.model_validate_json(policy_path.read_bytes())
        if getattr(usage, "policy_version", None) != policy.policy_version:
            mismatches["policy_version"] = (
                policy.policy_version,
                getattr(usage, "policy_version", None),
            )
        if mismatches:
            raise ArtifactIntegrityError(f"final-test usage provenance mismatch: {mismatches}")
        return self._write_immutable(
            self.FINAL_TEST_USAGE_NAME,
            _canonical_model_bytes(usage),
        )

    def write_split_manifest(self, split_manifest: BaseModel, expected_hash: str) -> Path:
        actual_hash = canonical_config_hash(split_manifest.model_dump(mode="json"))
        if actual_hash != expected_hash:
            raise ArtifactIntegrityError(
                "split manifest content differs from AttemptManifestV2.split_manifest_hash"
            )
        return self._write_immutable(
            self.SPLIT_MANIFEST_NAME,
            _canonical_model_bytes(split_manifest),
        )

    def write_worker_spec(self, worker_spec: BaseModel) -> Path:
        attempt_id = getattr(worker_spec, "attempt_id", None)
        manifest = self._read_manifest()
        if attempt_id != manifest.attempt_id:
            raise ArtifactIntegrityError("worker spec attempt_id differs from manifest")
        return self._write_immutable(
            self.WORKER_SPEC_NAME,
            _canonical_model_bytes(worker_spec),
        )

    def write_decision(self, decision: PromotionDecisionV2) -> Path:
        candidate = _safe_id(decision.candidate_id, "candidate_id")
        manifest = self._read_manifest()
        if decision.wave_id != manifest.wave_id:
            raise ArtifactIntegrityError("decision wave_id differs from manifest")
        policy_path = self._path(self.POLICY_NAME)
        if not policy_path.exists():
            raise ArtifactIntegrityError("promotion policy must be persisted before decisions")
        policy = ShadowGatePolicyV2.model_validate_json(policy_path.read_bytes())
        if decision.policy_version != policy.policy_version:
            raise ArtifactIntegrityError("decision policy version differs from persisted policy")
        return self._write_immutable(
            Path("candidates") / candidate / "decision.json",
            _canonical_model_bytes(decision),
        )

    def _artifact_hashes(self) -> dict[str, str]:
        excluded = {self.RESULT_NAME, self.RESULT_CHECKSUM_NAME}
        hashes: dict[str, str] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            relative = path.relative_to(self.root).as_posix()
            # Runtime output remains mutable until the child has exited. It is
            # hashed into the terminal AttemptState instead of result.json so
            # late stdout flushes cannot invalidate otherwise immutable result
            # evidence.
            if relative in excluded or relative.startswith("runtime/"):
                continue
            hashes[relative] = _sha256_file(path)
        return hashes

    def _read_manifest(self) -> AttemptManifestV2:
        path = self._path(self.MANIFEST_NAME)
        if not path.exists():
            raise ArtifactIntegrityError("manifest.json must be written first")
        try:
            return AttemptManifestV2.model_validate_json(path.read_bytes())
        except Exception as exc:
            raise ArtifactIntegrityError("manifest.json violates the V2 contract") from exc

    @staticmethod
    def _validate_record_provenance(
        record: BacktestRecordV2,
        manifest: AttemptManifestV2,
    ) -> None:
        expected_provenance = {
            "attempt_id": manifest.attempt_id,
            "wave_id": manifest.wave_id,
            "experiment_id": manifest.experiment_id,
            "config_hash": manifest.config_hash,
            "code_version": manifest.code_version,
            "data_manifest_hash": manifest.data_manifest_hash,
            "fitness_policy_version": manifest.fitness_policy_version,
            "worker_count": manifest.worker_count,
        }
        mismatches = {
            field: (expected, getattr(record, field))
            for field, expected in expected_provenance.items()
            if getattr(record, field) != expected
        }
        if record.seed not in manifest.seeds:
            mismatches["seed"] = (manifest.seeds, record.seed)
        if mismatches:
            raise ArtifactIntegrityError(f"scenario provenance differs from manifest: {mismatches}")

    def finalize(self, result: AttemptResultV2 | Mapping[str, Any]) -> AttemptResultV2:
        manifest_path = self._path(self.MANIFEST_NAME)
        if not manifest_path.exists():
            raise ArtifactIntegrityError("manifest.json must be written before result.json")
        persisted_manifest = AttemptManifestV2.model_validate_json(manifest_path.read_bytes())
        payload = (
            result.model_dump(mode="json", exclude_none=False)
            if isinstance(result, AttemptResultV2)
            else dict(result)
        )
        draft_manifest = AttemptManifestV2.model_validate(payload.get("manifest"))
        if persisted_manifest != draft_manifest:
            raise ArtifactIntegrityError("terminal result manifest differs from persisted manifest")

        payload["artifact_hashes"] = self._artifact_hashes()
        finalized = AttemptResultV2.model_validate(payload)

        for candidate in finalized.candidate_evaluations:
            candidate_id = _safe_id(candidate.candidate_id, "candidate_id")
            candidate_path = self._path(Path("candidates") / candidate_id / "candidate.json")
            if not candidate_path.exists():
                raise ArtifactIntegrityError(
                    f"terminal result references missing candidate artifact: {candidate_id}"
                )
            persisted_candidate = CandidateEvaluationV2.model_validate_json(
                candidate_path.read_bytes()
            )
            if not _models_have_same_canonical_payload(persisted_candidate, candidate):
                raise ArtifactIntegrityError(
                    f"terminal result candidate differs from artifact: {candidate_id}"
                )
            for record in candidate.scenarios:
                self._validate_record_provenance(record, persisted_manifest)
                scenario = _safe_id(record.metrics.scenario_id, "scenario_id")
                scenario_path = self._path(
                    Path("candidates") / candidate_id / "scenarios" / f"{scenario}.json"
                )
                if not scenario_path.exists():
                    raise ArtifactIntegrityError(
                        f"candidate references missing scenario artifact: {scenario}"
                    )
                persisted_record = BacktestRecordV2.model_validate_json(scenario_path.read_bytes())
                if not _models_have_same_canonical_payload(persisted_record, record):
                    raise ArtifactIntegrityError(
                        f"candidate scenario differs from artifact: {scenario}"
                    )

        for disposition in finalized.candidate_artifact_dispositions:
            candidate_id = _safe_id(disposition.candidate_id, "candidate_id")
            candidate_path = self._path(
                Path("candidates") / candidate_id / "candidate.json"
            )
            if not candidate_path.exists():
                raise ArtifactIntegrityError(
                    "candidate disposition references missing artifact: "
                    f"{candidate_id}"
                )
            persisted_candidate = CandidateEvaluationV2.model_validate_json(
                candidate_path.read_bytes()
            )
            if (
                persisted_candidate.phenotype_hash != disposition.phenotype_hash
                or persisted_candidate.status != disposition.evaluation_status
            ):
                raise ArtifactIntegrityError(
                    "candidate disposition differs from persisted artifact: "
                    f"{candidate_id}"
                )

        result_bytes = _canonical_model_bytes(finalized)
        result_path = self._write_immutable(self.RESULT_NAME, result_bytes)
        checksum = (_sha256_file(result_path) + "\n").encode("ascii")
        self._write_immutable(self.RESULT_CHECKSUM_NAME, checksum)
        return finalized

    def read_verified_result(self) -> AttemptResultV2:
        result_path = self._path(self.RESULT_NAME)
        checksum_path = self._path(self.RESULT_CHECKSUM_NAME)
        if not result_path.exists() or not checksum_path.exists():
            raise ArtifactIntegrityError("attempt has no complete terminal result")
        expected_result_hash = checksum_path.read_text().strip()
        if _sha256_file(result_path) != expected_result_hash:
            raise ArtifactIntegrityError("result.json checksum mismatch")

        try:
            result = AttemptResultV2.model_validate_json(result_path.read_bytes())
        except Exception as exc:
            raise ArtifactIntegrityError("result.json violates the V2 contract") from exc
        actual_hashes = self._artifact_hashes()
        if actual_hashes != result.artifact_hashes:
            missing = sorted(set(result.artifact_hashes) - set(actual_hashes))
            unexpected = sorted(set(actual_hashes) - set(result.artifact_hashes))
            changed = sorted(
                key
                for key in set(actual_hashes) & set(result.artifact_hashes)
                if actual_hashes[key] != result.artifact_hashes[key]
            )
            raise ArtifactIntegrityError(
                "artifact manifest mismatch; "
                f"missing={missing}, unexpected={unexpected}, changed={changed}"
            )
        return result


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 for a fully resolved configuration mapping."""

    return _sha256_bytes(_canonical_json_bytes(config))
