"""Fail-closed checkpoint integrity and compatibility contract."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.result_contract import StrictV2Model


CHECKPOINT_VERSION = 3
GENOME_SCHEMA_VERSION = "strategy-gene-v2"


class CheckpointContractError(ValueError):
    """Base class for checkpoint contract failures."""


class CheckpointIntegrityError(CheckpointContractError):
    """Raised when a checkpoint is incomplete or has been modified."""


class CheckpointCompatibilityError(CheckpointContractError):
    """Raised when a checkpoint was produced by a different run contract."""


class CheckpointProvenanceV3(StrictV2Model):
    """Immutable inputs which must be identical before state can be resumed."""

    schema_version: Literal["3.0"] = "3.0"
    engine_kind: Literal["STANDARD", "GENERIC_ISLAND", "REGIME_ISLAND"]
    config_hash: str = Field(min_length=64, max_length=64)
    code_manifest_hash: str = Field(min_length=64, max_length=64)
    data_manifest_hash: str = Field(min_length=64, max_length=64)
    genome_schema_version: Literal["strategy-gene-v2"] = GENOME_SCHEMA_VERSION
    island_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_identity(self) -> CheckpointProvenanceV3:
        for field_name in ("config_hash", "code_manifest_hash", "data_manifest_hash"):
            value = getattr(self, field_name)
            if any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if any(not name.strip() or name != name.strip() for name in self.island_names):
            raise ValueError("island names must be non-empty and canonical")
        if len(self.island_names) != len(set(self.island_names)):
            raise ValueError("island names must be unique")
        if self.engine_kind == "STANDARD" and self.island_names:
            raise ValueError("standard checkpoints cannot declare islands")
        if self.engine_kind != "STANDARD" and not self.island_names:
            raise ValueError("island checkpoints must declare every island name")
        return self


def parse_checkpoint_provenance(value: Any) -> CheckpointProvenanceV3:
    """Parse an exact provenance declaration with a stable domain error."""

    if isinstance(value, CheckpointProvenanceV3):
        return value
    try:
        return CheckpointProvenanceV3.model_validate(value)
    except Exception as exc:
        raise CheckpointCompatibilityError("invalid checkpoint provenance") from exc


def checkpoint_provenance_from_config(
    config: Mapping[str, Any],
    *,
    engine_kind: Literal["STANDARD", "GENERIC_ISLAND", "REGIME_ISLAND"],
    island_names: Iterable[str] = (),
    required: bool,
) -> CheckpointProvenanceV3 | None:
    """Read and bind the provenance injected by an immutable worker."""

    raw = config.get("checkpoint_provenance")
    if raw is None:
        if required:
            raise CheckpointCompatibilityError(
                "resume requires checkpoint_provenance from an immutable worker"
            )
        return None
    provenance = parse_checkpoint_provenance(raw)
    expected_names = tuple(island_names)
    if provenance.engine_kind != engine_kind:
        raise CheckpointCompatibilityError(
            f"checkpoint engine differs: expected {engine_kind}, got {provenance.engine_kind}"
        )
    if provenance.island_names != expected_names:
        raise CheckpointCompatibilityError(
            "checkpoint island names/order differ from the current engine"
        )
    return provenance


def canonical_checkpoint_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return stable bytes for checksum calculation."""

    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointIntegrityError("checkpoint contains non-canonical values") from exc


_NONFINITE_TAG = "__ga_checkpoint_nonfinite_float__"


def _encode_nonfinite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {_NONFINITE_TAG: label}
    if isinstance(value, Mapping):
        if _NONFINITE_TAG in value:
            raise CheckpointIntegrityError(
                "checkpoint payload collides with the reserved non-finite float tag"
            )
        return {str(key): _encode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_nonfinite(item) for item in value]
    return value


def _decode_nonfinite(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {_NONFINITE_TAG}:
            label = value[_NONFINITE_TAG]
            if label == "nan":
                return float("nan")
            if label == "positive_infinity":
                return float("inf")
            if label == "negative_infinity":
                return float("-inf")
            raise CheckpointIntegrityError("checkpoint has an invalid non-finite float tag")
        return {key: _decode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nonfinite(item) for item in value]
    return value


def seal_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a mandatory checksum to a detached checkpoint payload."""

    sealed = _encode_nonfinite(copy.deepcopy(dict(payload)))
    if "checksum" in sealed:
        raise CheckpointIntegrityError("checkpoint payload already contains a checksum")
    sealed["checksum"] = hashlib.sha256(canonical_checkpoint_bytes(sealed)).hexdigest()
    return sealed


def verify_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    expected_provenance: CheckpointProvenanceV3 | Mapping[str, Any],
) -> dict[str, Any]:
    """Verify version, checksum and exact producer/consumer compatibility."""

    payload = verify_checkpoint_integrity(checkpoint)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError(
            f"checkpoint version {payload.get('version')!r} is not resume-compatible "
            f"with version {CHECKPOINT_VERSION}"
        )
    if payload.get("resume_eligible") is not True:
        raise CheckpointCompatibilityError(
            "checkpoint is diagnostic-only because immutable provenance was unavailable"
        )
    actual = parse_checkpoint_provenance(payload.get("provenance"))
    expected = parse_checkpoint_provenance(expected_provenance)
    if actual != expected:
        changed = [
            field
            for field in (
                "engine_kind",
                "config_hash",
                "code_manifest_hash",
                "data_manifest_hash",
                "genome_schema_version",
                "island_names",
            )
            if getattr(actual, field) != getattr(expected, field)
        ]
        raise CheckpointCompatibilityError(
            "checkpoint provenance differs from the current run: " + ", ".join(changed)
        )
    return payload


def verify_checkpoint_integrity(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a V3 checksum without requiring resume compatibility.

    This is used when a checkpoint is an explicitly hash-bound warm-start
    source for a new run rather than continuation of the producer run.
    """

    if not isinstance(checkpoint, Mapping):
        raise CheckpointIntegrityError("checkpoint root must be an object")
    payload = copy.deepcopy(dict(checkpoint))
    stored_checksum = payload.pop("checksum", None)
    if not isinstance(stored_checksum, str) or len(stored_checksum) != 64:
        raise CheckpointIntegrityError("resume checkpoint has no valid checksum")
    computed_checksum = hashlib.sha256(canonical_checkpoint_bytes(payload)).hexdigest()
    if not hmac.compare_digest(stored_checksum, computed_checksum):
        raise CheckpointIntegrityError("checkpoint checksum mismatch")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError(
            f"checkpoint version {payload.get('version')!r} has no V3 integrity contract"
        )
    return _decode_nonfinite(payload)


_GENERATION_PATTERN = re.compile(r"_gen(?P<generation>\d+)(?:_|\.json$)")


def checkpoint_generation_from_path(path: str | Path) -> int:
    """Read the numeric generation encoded in a checkpoint filename."""

    candidate = Path(path)
    match = _GENERATION_PATTERN.search(candidate.name)
    if match is None:
        raise CheckpointContractError(
            f"checkpoint filename has no numeric generation: {candidate.name}"
        )
    return int(match.group("generation"))


def latest_checkpoint_by_generation(paths: Iterable[str | Path]) -> Path | None:
    """Select the numerically newest checkpoint, with a deterministic tie-break."""

    candidates = [
        (checkpoint_generation_from_path(path), Path(path).name, Path(path)) for path in paths
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]
