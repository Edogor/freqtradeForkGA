"""Contract tests for deterministic, fail-closed checkpoint resume."""

import json
import logging
from pathlib import Path

import pytest

from genetic_algorithm.core.generic_island_model import (
    GenericIslandConfig,
    GenericIslandModelEvolution,
)
from genetic_algorithm.engine.checkpoint_contract import (
    CheckpointCompatibilityError,
    CheckpointContractError,
    CheckpointIntegrityError,
    CheckpointProvenanceV3,
    checkpoint_provenance_from_config,
    latest_checkpoint_by_generation,
    seal_checkpoint,
    verify_resume_checkpoint,
)


def _provenance(**overrides) -> CheckpointProvenanceV3:
    values = {
        "engine_kind": "STANDARD",
        "config_hash": "1" * 64,
        "code_manifest_hash": "2" * 64,
        "data_manifest_hash": "3" * 64,
    }
    values.update(overrides)
    return CheckpointProvenanceV3(**values)


def _checkpoint(provenance: CheckpointProvenanceV3, **overrides):
    payload = {
        "version": 3,
        "resume_eligible": True,
        "provenance": provenance.model_dump(mode="json"),
        "generation": 10,
    }
    payload.update(overrides)
    return seal_checkpoint(payload)


def test_latest_checkpoint_is_selected_numerically():
    paths = [
        Path("island_checkpoint_gen9_20260723_120000.json"),
        Path("island_checkpoint_gen10_20260723_110000.json"),
        Path("island_checkpoint_gen2_20260723_130000.json"),
    ]
    assert latest_checkpoint_by_generation(paths) == paths[1]


def test_latest_checkpoint_rejects_unparseable_candidate():
    with pytest.raises(CheckpointContractError, match="numeric generation"):
        latest_checkpoint_by_generation([Path("island_checkpoint_latest.json")])


def test_missing_checksum_blocks_resume():
    provenance = _provenance()
    payload = _checkpoint(provenance)
    payload.pop("checksum")
    with pytest.raises(CheckpointIntegrityError, match="no valid checksum"):
        verify_resume_checkpoint(payload, provenance)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("config_hash", "4" * 64),
        ("code_manifest_hash", "4" * 64),
        ("data_manifest_hash", "4" * 64),
    ],
)
def test_every_manifest_mismatch_blocks_resume(field, replacement):
    produced = _provenance()
    expected = _provenance(**{field: replacement})
    with pytest.raises(CheckpointCompatibilityError, match=field):
        verify_resume_checkpoint(_checkpoint(produced), expected)


def test_legacy_version_is_not_resume_compatible_even_with_valid_checksum():
    provenance = _provenance()
    checkpoint = _checkpoint(provenance, version=2)
    with pytest.raises(CheckpointCompatibilityError, match="version 2"):
        verify_resume_checkpoint(checkpoint, provenance)


def test_island_names_and_order_are_exact():
    produced = _provenance(
        engine_kind="GENERIC_ISLAND",
        island_names=("trend", "momentum"),
    )
    expected = _provenance(
        engine_kind="GENERIC_ISLAND",
        island_names=("momentum", "trend"),
    )
    with pytest.raises(CheckpointCompatibilityError, match="island_names"):
        verify_resume_checkpoint(_checkpoint(produced), expected)


def test_config_binding_rejects_wrong_engine_before_io():
    provenance = _provenance()
    config = {"checkpoint_provenance": provenance.model_dump(mode="json")}
    with pytest.raises(CheckpointCompatibilityError, match="engine differs"):
        checkpoint_provenance_from_config(
            config,
            engine_kind="GENERIC_ISLAND",
            island_names=("island-0",),
            required=True,
        )


def test_generic_loader_selects_newest_generation_then_blocks_corruption(tmp_path):
    provenance = _provenance(
        engine_kind="GENERIC_ISLAND",
        island_names=("trend", "momentum"),
    )
    model = object.__new__(GenericIslandModelEvolution)
    model.config = {
        "checkpoint_provenance": provenance.model_dump(mode="json"),
    }
    model.island_configs = [
        GenericIslandConfig(name="trend"),
        GenericIslandConfig(name="momentum"),
    ]
    model.checkpoint_dir = tmp_path
    model.logger = logging.getLogger("checkpoint-contract-test")

    older = tmp_path / "island_checkpoint_gen9_20260723_120000.json"
    newer = tmp_path / "island_checkpoint_gen10_20260723_110000.json"
    older.write_text(json.dumps(_checkpoint(provenance, generation=9)))
    corrupt = _checkpoint(provenance, generation=10)
    corrupt["generation"] = 11
    newer.write_text(json.dumps(corrupt))

    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        model.load_island_checkpoint()
