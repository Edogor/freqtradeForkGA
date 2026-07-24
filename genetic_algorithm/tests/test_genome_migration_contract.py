"""Fail-closed tests for semantic legacy-genome migration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from genetic_algorithm.engine.hall_of_fame import HallOfFame, HallOfFameEntry
from genetic_algorithm.engine.warm_start import WarmStartError, WarmStartLoader
from genetic_algorithm.genome.gene import ConditionGene, IndicatorGene, StrategyGene
from genetic_algorithm.genome.migration import (
    GenomeMigrationError,
    migrate_genome_artifact_bytes,
    verify_migrated_genome_artifact,
)
from genetic_algorithm.scripts.migrate_genome_artifact import main as migration_main


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _gene() -> dict:
    return StrategyGene(
        generation=3,
        individual_id=7,
        indicators=[
            IndicatorGene(
                type="RSI",
                parameters={"period": 14},
                instance_id="RSI_0",
            )
        ],
        entry_conditions=[ConditionGene(indicator="RSI_0", operator="<", threshold=30.0)],
        exit_conditions=[ConditionGene(indicator="RSI_0", operator=">", threshold=70.0)],
    ).to_dict()


def _population(gene: dict | None = None) -> dict:
    return {
        "population": {
            "individuals": [
                {
                    "strategy_gene": gene or _gene(),
                    "fitness": 0.7,
                    "raw_fitness": 0.7,
                    "evaluated": True,
                }
            ]
        }
    }


def _encoded(payload: dict) -> tuple[bytes, str]:
    content = json.dumps(payload, sort_keys=True).encode()
    return content, hashlib.sha256(content).hexdigest()


def _migrate(payload: dict):
    content, digest = _encoded(payload)
    return migrate_genome_artifact_bytes(content, expected_sha256=digest)


def test_semantic_contract_rejects_unknown_indicator_parameters_operator_and_risk():
    invalid_cases = []

    unknown = _gene()
    unknown["indicators"][0]["type"] = "UNKNOWN"
    unknown["indicators"][0]["instance_id"] = "UNKNOWN_0"
    unknown["entry_conditions"][0]["indicator"] = "UNKNOWN_0"
    unknown["exit_conditions"][0]["indicator"] = "UNKNOWN_0"
    invalid_cases.append(unknown)

    wrong_parameters = _gene()
    wrong_parameters["indicators"][0]["parameters"] = {"kijun_period": 20}
    invalid_cases.append(wrong_parameters)

    wrong_operator = _gene()
    wrong_operator["entry_conditions"][0]["operator"] = "approximately"
    invalid_cases.append(wrong_operator)

    wrong_risk = _gene()
    wrong_risk["stoploss"] = 0.1
    invalid_cases.append(wrong_risk)

    for payload in invalid_cases:
        with pytest.raises(ValueError):
            StrategyGene.from_dict_exact(payload)


def test_unambiguous_legacy_reference_is_migrated_and_reported():
    legacy = _gene()
    legacy["entry_conditions"][0]["indicator"] = "RSI"

    artifact, report = _migrate(_population(legacy))

    migrated = artifact["population"]["individuals"][0]["strategy_gene"]
    assert migrated["entry_conditions"][0]["indicator"] == "RSI_0"
    assert report.entry_count == 1
    assert [change.kind.value for change in report.entries[0].changes] == [
        "NORMALIZED_CONDITION_REFERENCE"
    ]
    assert verify_migrated_genome_artifact(artifact) == report


def test_migration_is_deterministic_for_same_source_bytes():
    legacy = _gene()
    legacy["entry_conditions"][0]["indicator"] = "RSI"
    content, digest = _encoded(_population(legacy))

    first_artifact, first_report = migrate_genome_artifact_bytes(
        content,
        expected_sha256=digest,
    )
    second_artifact, second_report = migrate_genome_artifact_bytes(
        content,
        expected_sha256=digest,
    )

    assert first_artifact == second_artifact
    assert first_report.report_hash == second_report.report_hash


def test_ambiguous_bare_reference_blocks_complete_source():
    legacy = _gene()
    legacy["indicators"].append(
        {
            "type": "RSI",
            "parameters": {"period": 21},
            "weight": 1.0,
            "instance_id": "RSI_1",
            "timeframe": None,
            "param_bounds": None,
        }
    )
    legacy["entry_conditions"][0]["indicator"] = "RSI"

    with pytest.raises(GenomeMigrationError, match="ambiguously references"):
        _migrate(_population(legacy))


def test_semantically_corrupt_legacy_parameter_set_is_not_repaired():
    legacy = _gene()
    legacy["indicators"][0]["parameters"] = {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_b_period": 52,
    }

    with pytest.raises(GenomeMigrationError, match="target genome contract"):
        _migrate(_population(legacy))


def test_false_target_schema_and_source_hash_mismatch_block():
    payload = _population()
    payload["genome_schema_version"] = "strategy-gene-v2"
    content, digest = _encoded(payload)

    with pytest.raises(GenomeMigrationError, match="verified, not migrated"):
        migrate_genome_artifact_bytes(content, expected_sha256=digest)
    with pytest.raises(GenomeMigrationError, match="SHA-256 mismatch"):
        migrate_genome_artifact_bytes(content, expected_sha256="0" * 64)


def test_target_or_report_mutation_is_detected():
    artifact, _ = _migrate(_population())

    changed_gene = copy.deepcopy(artifact)
    changed_gene["population"]["individuals"][0]["strategy_gene"]["stoploss"] = -0.2
    with pytest.raises(GenomeMigrationError, match="genome hash mismatch"):
        verify_migrated_genome_artifact(changed_gene)

    changed_report = copy.deepcopy(artifact)
    changed_report["migration_report"]["entry_count"] = 2
    with pytest.raises(GenomeMigrationError, match="report is invalid"):
        verify_migrated_genome_artifact(changed_report)


def test_warm_start_accepts_verified_migration_and_records_report(tmp_path):
    legacy_content, legacy_digest = _encoded(_population())
    artifact, report = migrate_genome_artifact_bytes(
        legacy_content,
        expected_sha256=legacy_digest,
    )
    legacy_source = tmp_path / "legacy.json"
    legacy_source.write_bytes(legacy_content)
    source = tmp_path / "migrated.json"
    source.write_text(json.dumps(artifact, sort_keys=True))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    loaded = WarmStartLoader(
        {
            "warm_start": {
                "enabled": True,
                "source_type": "population",
                "top_n": 1,
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": digest,
                "source_checkpoint_migration_source": str(legacy_source),
                "source_checkpoint_migration_source_sha256": legacy_digest,
                "genome_schema_version": "strategy-gene-v2",
            }
        }
    ).load()

    assert loaded[0].metrics["warm_start_migration_reports"] == {"checkpoint": report.report_hash}
    assert loaded[0].metrics["warm_start_source_hashes"] == {
        "checkpoint": digest,
        "checkpoint_migration_source": legacy_digest,
    }


def test_warm_start_requires_legacy_source_to_rebuild_migration(tmp_path):
    artifact, _ = _migrate(_population())
    source = tmp_path / "migrated.json"
    source.write_text(json.dumps(artifact, sort_keys=True))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(WarmStartError, match="migration source path must be explicit"):
        WarmStartLoader(
            {
                "warm_start": {
                    "enabled": True,
                    "source_type": "population",
                    "top_n": 1,
                    "source_checkpoint": str(source),
                    "source_checkpoint_sha256": digest,
                    "genome_schema_version": "strategy-gene-v2",
                }
            }
        ).load()


def test_warm_start_rejects_migration_bound_to_another_legacy_source(tmp_path):
    first_content, first_digest = _encoded(_population())
    artifact, _ = migrate_genome_artifact_bytes(
        first_content,
        expected_sha256=first_digest,
    )
    migrated = tmp_path / "migrated.json"
    migrated.write_text(json.dumps(artifact, sort_keys=True))
    migrated_digest = hashlib.sha256(migrated.read_bytes()).hexdigest()

    other = _population()
    other["population"]["individuals"][0]["strategy_gene"]["stoploss"] = -0.2
    other_content, other_digest = _encoded(other)
    other_source = tmp_path / "other-legacy.json"
    other_source.write_bytes(other_content)

    with pytest.raises(WarmStartError, match="bound to another migration source"):
        WarmStartLoader(
            {
                "warm_start": {
                    "enabled": True,
                    "source_type": "population",
                    "top_n": 1,
                    "source_checkpoint": str(migrated),
                    "source_checkpoint_sha256": migrated_digest,
                    "source_checkpoint_migration_source": str(other_source),
                    "source_checkpoint_migration_source_sha256": other_digest,
                    "genome_schema_version": "strategy-gene-v2",
                }
            }
        ).load()


def test_warm_start_rejects_unversioned_native_source(tmp_path):
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(_population()))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(WarmStartError, match="no explicit strategy-gene-v2"):
        WarmStartLoader(
            {
                "warm_start": {
                    "enabled": True,
                    "source_type": "population",
                    "top_n": 1,
                    "source_checkpoint": str(source),
                    "source_checkpoint_sha256": digest,
                    "genome_schema_version": "strategy-gene-v2",
                }
            }
        ).load()


def test_native_hof_writer_declares_semantic_genome_schema(tmp_path):
    hall = HallOfFame(directory=str(tmp_path))
    hall.entries = [
        HallOfFameEntry(
            strategy_gene_dict=_gene(),
            fitness=0.5,
            metrics={"profit": 1.0},
            generation_found=3,
            run_timestamp=1.0,
        )
    ]

    hall._save()
    payload = json.loads(hall.filepath.read_text())

    assert payload["version"] == 3
    assert payload["artifact_schema_version"] == "hall-of-fame-v3"
    assert payload["genome_schema_version"] == "strategy-gene-v2"
    StrategyGene.from_dict_exact(payload["entries"][0]["strategy_gene"])


def test_active_hof_refuses_unmigrated_legacy_archive(tmp_path):
    legacy = {
        "version": 2,
        "entries": [
            {
                "strategy_gene": _gene(),
                "fitness": 0.5,
                "metrics": {"profit": 1.0},
            }
        ],
    }
    (tmp_path / "hall_of_fame.json").write_text(json.dumps(legacy))

    with pytest.raises(ValueError, match="requires explicit genome migration"):
        HallOfFame(directory=str(tmp_path))


@pytest.mark.parametrize(
    "relative_path",
    [
        "genetic_algorithm/data/hall_of_fame/hall_of_fame.json",
        "genetic_algorithm/data/checkpoints_wave44_seed_top30.json",
    ],
)
def test_real_project_seed_artifacts_migrate_to_exact_contract(relative_path):
    source = PROJECT_ROOT / relative_path
    content = source.read_bytes()

    artifact, report = migrate_genome_artifact_bytes(
        content,
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )

    assert report.entry_count > 0
    verify_migrated_genome_artifact(artifact)


def test_cli_writes_once_and_refuses_overwrite(tmp_path, capsys):
    source = tmp_path / "legacy.json"
    output = tmp_path / "migrated.json"
    content, digest = _encoded(_population())
    source.write_bytes(content)

    args = [str(source), str(output), "--source-sha256", digest]
    assert migration_main(args) == 0
    assert output.is_file()
    assert "Artifact SHA-256" in capsys.readouterr().out

    original = output.read_bytes()
    assert migration_main(args) == 2
    assert output.read_bytes() == original
    assert "refusing to overwrite" in capsys.readouterr().err
