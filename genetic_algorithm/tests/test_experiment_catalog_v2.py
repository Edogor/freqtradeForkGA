"""SQLite experiment catalog, immutable legacy import, and export tests."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from genetic_algorithm.orchestration.attempt_state_v2 import AttemptStateStoreV2
from genetic_algorithm.orchestration.experiment_catalog_v2 import (
    ExperimentCatalogError,
    ExperimentCatalogV2,
    _select_catalog_winner,
)
from genetic_algorithm.orchestration.result_contract import AttemptManifestV2


NOW = datetime(2026, 7, 23, 18, 0, tzinfo=UTC)


def _manifest(tmp_path: Path, attempt_id: str, experiment_id: str) -> AttemptManifestV2:
    root = tmp_path / "attempts" / attempt_id
    return AttemptManifestV2(
        attempt_id=attempt_id,
        wave_id="wave-catalog",
        experiment_id=experiment_id,
        created_at=NOW,
        config_hash="c" * 64,
        code_version="catalog-test",
        data_manifest_hash="d" * 64,
        split_manifest_hash="s" * 64,
        fitness_policy_version="catalog-policy",
        seeds=[42],
        worker_count=1,
        resolved_config_path=str(root / "resolved_config.yaml"),
        artifact_root=str(root),
    )


def _legacy_file(tmp_path: Path, experiments: dict) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"version": 1, "experiments": experiments}))
    return path


def test_legacy_import_is_idempotent_and_never_creates_attempts(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "state.sqlite3")
    catalog = ExperimentCatalogV2(store)
    source = _legacy_file(
        tmp_path,
        {
            "legacy-one": {
                "experiment_id": "legacy-one",
                "status": "completed",
                "created_at": NOW.isoformat(),
                "updated_at": (NOW + timedelta(hours=1)).isoformat(),
                "best_fitness": 1.25,
                "best_profit": 12.0,
                "tags": ["old"],
            }
        },
    )

    first = catalog.import_legacy_registry(source, imported_at=NOW + timedelta(days=1))
    second = catalog.import_legacy_registry(source, imported_at=NOW + timedelta(days=2))

    assert first.import_id == second.import_id
    assert second.already_imported is True
    assert store.list_attempts() == []
    record = catalog.get_record("legacy-one")
    assert record is not None
    assert record.source == "LEGACY_IMPORT"
    assert record.best_net_return == 0.12
    assert record.attempt_ids == []


def test_catalog_winner_never_mixes_score_and_return_between_candidates():
    winner = _select_catalog_winner(
        [
            ("attempt-a", "high-score", 0.90, [0.40, -0.08, 0.12]),
            ("attempt-b", "high-return", 0.70, [0.80, 0.60]),
        ]
    )

    assert winner == ("attempt-a", "high-score", 0.90, -0.08)


def test_concurrent_legacy_import_claims_one_snapshot_version(tmp_path: Path):
    state_path = tmp_path / "state.sqlite3"
    AttemptStateStoreV2(state_path)
    source = _legacy_file(
        tmp_path,
        {"legacy-one": {"experiment_id": "legacy-one", "status": "completed"}},
    )

    def import_once(_: int):
        return ExperimentCatalogV2(AttemptStateStoreV2(state_path)).import_legacy_registry(
            source, imported_at=NOW
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(import_once, range(16)))

    assert {receipt.import_id for receipt in receipts} == {1}
    assert sum(not receipt.already_imported for receipt in receipts) == 1
    connection = sqlite3.connect(state_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM legacy_registry_imports").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM legacy_experiment_versions").fetchone()[0] == 1
        )
    finally:
        connection.close()


def test_new_legacy_snapshot_keeps_history_and_exposes_latest_version(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "state.sqlite3")
    catalog = ExperimentCatalogV2(store)
    source = _legacy_file(
        tmp_path,
        {"legacy-one": {"experiment_id": "legacy-one", "status": "running"}},
    )
    first = catalog.import_legacy_registry(source, imported_at=NOW)
    source.write_text(
        json.dumps(
            {
                "version": 1,
                "experiments": {
                    "legacy-one": {
                        "experiment_id": "legacy-one",
                        "status": "failed",
                        "error": "historic crash",
                    }
                },
            }
        )
    )
    second = catalog.import_legacy_registry(source, imported_at=NOW + timedelta(minutes=1))

    assert second.import_id > first.import_id
    assert catalog.get_record("legacy-one").status == "failed"
    connection = sqlite3.connect(store.path)
    try:
        assert (
            connection.execute("SELECT COUNT(*) FROM legacy_experiment_versions").fetchone()[0] == 2
        )
    finally:
        connection.close()


def test_canonical_attempt_wins_over_same_named_legacy_history(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "state.sqlite3")
    store.register(_manifest(tmp_path, "attempt-one", "shared-name"))
    catalog = ExperimentCatalogV2(store)
    source = _legacy_file(
        tmp_path,
        {
            "shared-name": {
                "experiment_id": "shared-name",
                "status": "completed",
                "best_fitness": 999.0,
            }
        },
    )
    catalog.import_legacy_registry(source, imported_at=NOW + timedelta(seconds=1))

    records = catalog.list_records()

    assert len(records) == 1
    assert records[0].source == "CANONICAL_V2"
    assert records[0].attempt_ids == ["attempt-one"]
    assert records[0].best_score is None


def test_export_contains_sorted_attempts_and_provenance(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "state.sqlite3")
    store.register(_manifest(tmp_path, "attempt-b", "experiment-b"))
    store.register(_manifest(tmp_path, "attempt-a", "experiment-a"))
    catalog = ExperimentCatalogV2(store)
    source = _legacy_file(
        tmp_path,
        {"legacy": {"experiment_id": "legacy", "status": "cancelled"}},
    )
    catalog.import_legacy_registry(source, imported_at=NOW + timedelta(seconds=1))
    destination = tmp_path / "exports" / "catalog.json"

    exported = catalog.export_json(destination, exported_at=NOW + timedelta(days=1))

    assert [item.attempt_id for item in exported.attempts] == [
        "attempt-a",
        "attempt-b",
    ]
    assert destination.is_file()
    persisted = json.loads(destination.read_text())
    assert persisted["state_schema_version"] == AttemptStateStoreV2.SCHEMA_VERSION
    assert persisted["legacy_imports"][0]["experiment_count"] == 1


def test_legacy_import_tables_are_immutable(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "state.sqlite3")
    catalog = ExperimentCatalogV2(store)
    source = _legacy_file(
        tmp_path,
        {"legacy": {"experiment_id": "legacy", "status": "completed"}},
    )
    receipt = catalog.import_legacy_registry(source, imported_at=NOW)

    connection = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE legacy_registry_imports SET source_path = ? WHERE import_id = ?",
                ("changed", receipt.import_id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "DELETE FROM legacy_experiment_versions WHERE import_id = ?",
                (receipt.import_id,),
            )
    finally:
        connection.close()


def test_non_finite_legacy_json_rejects_complete_snapshot(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "state.sqlite3")
    catalog = ExperimentCatalogV2(store)
    source = tmp_path / "registry.json"
    source.write_text(
        '{"version":1,"experiments":{"bad":{"experiment_id":"bad",'
        '"status":"completed","best_fitness":NaN}}}'
    )

    with pytest.raises(ExperimentCatalogError, match="non-finite"):
        catalog.import_legacy_registry(source, imported_at=NOW)

    connection = sqlite3.connect(store.path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM legacy_registry_imports").fetchone()[0] == 0
    finally:
        connection.close()


def test_malformed_optional_legacy_fields_are_safely_classified(tmp_path: Path):
    store = AttemptStateStoreV2(tmp_path / "state.sqlite3")
    catalog = ExperimentCatalogV2(store)
    source = _legacy_file(
        tmp_path,
        {
            "legacy": {
                "experiment_id": "legacy",
                "status": "invented",
                "pid": True,
                "tags": ["valid", 42, "", "valid"],
                "config_path": 123,
                "best_profit": "12.0",
            }
        },
    )

    catalog.import_legacy_registry(source, imported_at=NOW)
    record = catalog.get_record("legacy")

    assert record is not None
    assert record.status == "unknown"
    assert record.pid is None
    assert record.tags == ["valid"]
    assert record.config_path is None
    assert record.best_net_return is None
