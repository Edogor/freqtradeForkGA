"""Explicit, deterministic migration of legacy genome seed artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from genetic_algorithm.genome.contract import GENOME_SCHEMA_VERSION
from genetic_algorithm.genome.gene import StrategyGene
from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.result_contract import StrictV2Model


MIGRATION_POLICY_VERSION = "legacy-gene-to-strategy-gene-v2-v1"
MIGRATED_ARTIFACT_SCHEMA_VERSION = "genome-seed-migration-v1"


class GenomeMigrationError(ValueError):
    """Raised when legacy semantics cannot be migrated without guessing."""


class GenomeArtifactKind(StrEnum):
    POPULATION = "POPULATION"
    HALL_OF_FAME = "HALL_OF_FAME"


class GenomeMigrationChangeKind(StrEnum):
    ADDED_LEGACY_DEFAULT = "ADDED_LEGACY_DEFAULT"
    NORMALIZED_INSTANCE_ID = "NORMALIZED_INSTANCE_ID"
    NORMALIZED_CONDITION_REFERENCE = "NORMALIZED_CONDITION_REFERENCE"
    NORMALIZED_CDL_TYPE = "NORMALIZED_CDL_TYPE"
    NORMALIZED_EMPTY_BOUNDS = "NORMALIZED_EMPTY_BOUNDS"


class GenomeMigrationChangeV1(StrictV2Model):
    path: str = Field(min_length=1)
    kind: GenomeMigrationChangeKind


class GenomeMigrationEntryV1(StrictV2Model):
    locator: str = Field(min_length=1)
    source_gene_hash: str = Field(min_length=64, max_length=64)
    target_gene_hash: str = Field(min_length=64, max_length=64)
    changes: list[GenomeMigrationChangeV1] = Field(default_factory=list)


class GenomeMigrationReportV1(StrictV2Model):
    schema_version: Literal["1.0"] = "1.0"
    migration_policy_version: Literal["legacy-gene-to-strategy-gene-v2-v1"] = (
        MIGRATION_POLICY_VERSION
    )
    source_sha256: str = Field(min_length=64, max_length=64)
    source_artifact_kind: GenomeArtifactKind
    source_declared_genome_schema: str | None = None
    target_genome_schema: Literal["strategy-gene-v2"] = GENOME_SCHEMA_VERSION
    entry_count: int = Field(ge=1)
    entries: list[GenomeMigrationEntryV1] = Field(min_length=1)
    report_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_report(self) -> GenomeMigrationReportV1:
        if self.entry_count != len(self.entries):
            raise ValueError("migration report entry_count differs from entries")
        locators = [entry.locator for entry in self.entries]
        if len(locators) != len(set(locators)):
            raise ValueError("migration report contains duplicate locators")
        expected = canonical_config_hash(self.model_dump(mode="json", exclude={"report_hash"}))
        if self.report_hash != expected:
            raise ValueError("migration report hash mismatch")
        return self

    @classmethod
    def seal(cls, **values: Any) -> GenomeMigrationReportV1:
        payload = {
            "schema_version": "1.0",
            "migration_policy_version": MIGRATION_POLICY_VERSION,
            **values,
        }
        hash_payload = {
            key: ([item.model_dump(mode="json") for item in value] if key == "entries" else value)
            for key, value in payload.items()
        }
        payload["report_hash"] = canonical_config_hash(hash_payload)
        return cls.model_validate(payload)


_ROOT_DEFAULTS = {
    "exit_conditions",
    "short_entry_conditions",
    "short_exit_conditions",
    "timeframe",
    "informative_timeframes",
    "stoploss",
    "minimal_roi",
    "max_open_trades",
    "trailing_stop",
    "trailing_stop_positive",
    "trailing_stop_positive_offset",
    "can_short",
    "preferred_regime",
    "regime_mode",
    "regime_gene",
    "self_mutation_rate",
    "self_crossover_pref",
}
_INDICATOR_DEFAULTS = {"weight", "instance_id", "timeframe", "param_bounds"}
_CONDITION_DEFAULTS = {"logic", "threshold_upper", "lookback"}
_REGIME_DEFAULTS = {
    "enabled",
    "regime_timeframes",
    "entry_trend_min",
    "entry_trend_max",
    "exit_on_regime_change",
    "combination",
    "micro_regime",
}
_CONDITION_GROUPS = {
    "entry_conditions",
    "exit_conditions",
    "short_entry_conditions",
    "short_exit_conditions",
}


def _join_path(parent: str, part: str | int) -> str:
    escaped = str(part).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}" if parent else f"/{escaped}"


def _diff(source: Any, target: Any, path: str = "") -> list[tuple[str, str, Any, Any]]:
    changes: list[tuple[str, str, Any, Any]] = []
    if isinstance(source, dict) and isinstance(target, dict):
        for key in sorted(source.keys() - target.keys()):
            changes.append((_join_path(path, key), "REMOVED", source[key], None))
        for key in sorted(target.keys() - source.keys()):
            changes.append((_join_path(path, key), "ADDED", None, target[key]))
        for key in sorted(source.keys() & target.keys()):
            changes.extend(_diff(source[key], target[key], _join_path(path, key)))
        return changes
    if isinstance(source, list) and isinstance(target, list):
        if len(source) != len(target):
            changes.append((path, "LIST_LENGTH", len(source), len(target)))
            return changes
        for index, (source_item, target_item) in enumerate(zip(source, target)):
            changes.extend(_diff(source_item, target_item, _join_path(path, index)))
        return changes
    if source != target or type(source) is not type(target):
        changes.append((path, "CHANGED", source, target))
    return changes


def _path_parts(path: str) -> list[str]:
    return [part for part in path.split("/") if part]


def _classify_change(
    path: str,
    operation: str,
    source: Any,
    target: Any,
) -> GenomeMigrationChangeKind:
    parts = _path_parts(path)
    leaf = parts[-1] if parts else ""
    if operation == "ADDED":
        if len(parts) == 1 and leaf in _ROOT_DEFAULTS:
            return GenomeMigrationChangeKind.ADDED_LEGACY_DEFAULT
        if len(parts) == 3 and parts[0] == "indicators" and leaf in _INDICATOR_DEFAULTS:
            return GenomeMigrationChangeKind.ADDED_LEGACY_DEFAULT
        if len(parts) == 3 and parts[0] in _CONDITION_GROUPS and leaf in _CONDITION_DEFAULTS:
            return GenomeMigrationChangeKind.ADDED_LEGACY_DEFAULT
        if len(parts) == 2 and parts[0] == "regime_gene" and leaf in _REGIME_DEFAULTS:
            return GenomeMigrationChangeKind.ADDED_LEGACY_DEFAULT
    if (
        operation == "CHANGED"
        and len(parts) == 3
        and parts[0] == "indicators"
        and leaf == "instance_id"
    ):
        return GenomeMigrationChangeKind.NORMALIZED_INSTANCE_ID
    if (
        operation == "CHANGED"
        and len(parts) == 3
        and parts[0] in _CONDITION_GROUPS
        and leaf == "indicator"
    ):
        return GenomeMigrationChangeKind.NORMALIZED_CONDITION_REFERENCE
    if (
        operation == "CHANGED"
        and len(parts) == 3
        and parts[0] == "indicators"
        and leaf == "type"
        and isinstance(source, str)
        and isinstance(target, str)
        and source.startswith("CDL_")
        and target.startswith("CDL_")
    ):
        return GenomeMigrationChangeKind.NORMALIZED_CDL_TYPE
    if (
        operation == "CHANGED"
        and len(parts) == 3
        and parts[0] == "indicators"
        and leaf == "param_bounds"
        and source == {}
        and target is None
    ):
        return GenomeMigrationChangeKind.NORMALIZED_EMPTY_BOUNDS
    raise GenomeMigrationError(f"migration would change semantics at {path or '/'} ({operation})")


def _reject_ambiguous_references(source: dict[str, Any], locator: str) -> None:
    raw_indicators = source.get("indicators")
    if not isinstance(raw_indicators, list) or not raw_indicators:
        raise GenomeMigrationError(f"{locator} has no indicator list")
    type_counts: dict[str, int] = {}
    old_ids: list[str] = []
    for index, indicator in enumerate(raw_indicators):
        if not isinstance(indicator, dict):
            raise GenomeMigrationError(f"{locator}/indicators/{index} must be an object")
        indicator_type = indicator.get("type")
        if not isinstance(indicator_type, str) or not indicator_type:
            raise GenomeMigrationError(f"{locator}/indicators/{index} has no type")
        normalized_type = StrategyGene._strip_cdl_suffixes(indicator_type)
        type_counts[normalized_type] = type_counts.get(normalized_type, 0) + 1
        old_id = indicator.get("instance_id")
        if old_id is not None:
            if not isinstance(old_id, str) or not old_id:
                raise GenomeMigrationError(f"{locator}/indicators/{index}/instance_id is invalid")
            old_ids.append(old_id)
    if len(old_ids) != len(set(old_ids)):
        raise GenomeMigrationError(f"{locator} contains duplicate legacy instance IDs")

    for group in _CONDITION_GROUPS:
        raw_conditions = source.get(group, [])
        if not isinstance(raw_conditions, list):
            raise GenomeMigrationError(f"{locator}/{group} must be a list")
        for index, condition in enumerate(raw_conditions):
            if not isinstance(condition, dict):
                raise GenomeMigrationError(f"{locator}/{group}/{index} must be an object")
            reference = condition.get("indicator")
            if not isinstance(reference, str) or not reference:
                raise GenomeMigrationError(f"{locator}/{group}/{index}/indicator is invalid")
            normalized_reference = StrategyGene._strip_cdl_suffixes(reference)
            if type_counts.get(normalized_reference, 0) > 1:
                raise GenomeMigrationError(
                    f"{locator}/{group}/{index} ambiguously references "
                    f"{reference!r} with multiple instances"
                )


def migrate_legacy_gene(
    source: dict[str, Any],
    *,
    locator: str,
) -> tuple[dict[str, Any], GenomeMigrationEntryV1]:
    """Normalize only proven legacy defaults/references, then validate semantics."""

    if not isinstance(source, dict):
        raise GenomeMigrationError(f"{locator} strategy gene must be an object")
    detached = copy.deepcopy(source)
    _reject_ambiguous_references(detached, locator)
    try:
        target = StrategyGene.from_dict(detached).to_dict()
        StrategyGene.from_dict_exact(target)
    except Exception as exc:
        raise GenomeMigrationError(f"{locator} violates the target genome contract") from exc

    classified = []
    for path, operation, old_value, new_value in _diff(detached, target):
        classified.append(
            GenomeMigrationChangeV1(
                path=path,
                kind=_classify_change(path, operation, old_value, new_value),
            )
        )
    return target, GenomeMigrationEntryV1(
        locator=locator,
        source_gene_hash=canonical_config_hash(detached),
        target_gene_hash=canonical_config_hash(target),
        changes=classified,
    )


def _artifact_entries(
    artifact: dict[str, Any],
) -> tuple[GenomeArtifactKind, list[tuple[str, dict[str, Any], str]]]:
    population = artifact.get("population")
    entries = artifact.get("entries")
    has_population = isinstance(population, dict) and isinstance(
        population.get("individuals"),
        list,
    )
    has_hof = isinstance(entries, list)
    if has_population == has_hof:
        raise GenomeMigrationError(
            "source must contain exactly one population.individuals or entries list"
        )
    located: list[tuple[str, dict[str, Any], str]] = []
    if has_population:
        individuals = population["individuals"]
        if not individuals:
            raise GenomeMigrationError("source population is empty")
        for index, individual in enumerate(individuals):
            if not isinstance(individual, dict) or not isinstance(
                individual.get("strategy_gene"),
                dict,
            ):
                raise GenomeMigrationError(f"/population/individuals/{index} has no strategy_gene")
            located.append(
                (
                    f"/population/individuals/{index}/strategy_gene",
                    individual,
                    "strategy_gene",
                )
            )
        return GenomeArtifactKind.POPULATION, located

    if not entries:
        raise GenomeMigrationError("source hall of fame is empty")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise GenomeMigrationError(f"/entries/{index} must be an object")
        gene_key = (
            "strategy_gene"
            if isinstance(entry.get("strategy_gene"), dict)
            else "gene_dict"
            if isinstance(entry.get("gene_dict"), dict)
            else None
        )
        if gene_key is None:
            raise GenomeMigrationError(f"/entries/{index} has no strategy gene")
        located.append((f"/entries/{index}/{gene_key}", entry, gene_key))
    return GenomeArtifactKind.HALL_OF_FAME, located


def migrate_genome_artifact_bytes(
    content: bytes,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], GenomeMigrationReportV1]:
    """Migrate an explicitly hash-bound legacy artifact in one atomic operation."""

    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise GenomeMigrationError("source SHA-256 mismatch")
    try:
        source = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GenomeMigrationError("source is not valid JSON") from exc
    if not isinstance(source, dict):
        raise GenomeMigrationError("source root must be an object")
    declared_schema = source.get("genome_schema_version")
    if declared_schema is not None:
        raise GenomeMigrationError(
            "source already declares a genome schema and must be verified, not migrated"
        )

    target = copy.deepcopy(source)
    artifact_kind, located = _artifact_entries(target)
    migrated_entries = []
    for locator, container, gene_key in located:
        migrated, report_entry = migrate_legacy_gene(
            container[gene_key],
            locator=locator,
        )
        container[gene_key] = migrated
        migrated_entries.append(report_entry)

    report = GenomeMigrationReportV1.seal(
        source_sha256=actual_sha256,
        source_artifact_kind=artifact_kind,
        source_declared_genome_schema=None,
        target_genome_schema=GENOME_SCHEMA_VERSION,
        entry_count=len(migrated_entries),
        entries=migrated_entries,
    )
    target["artifact_schema_version"] = MIGRATED_ARTIFACT_SCHEMA_VERSION
    target["genome_schema_version"] = GENOME_SCHEMA_VERSION
    target["migration_report"] = report.model_dump(mode="json")
    verify_migrated_genome_artifact(target)
    return target, report


def verify_migrated_genome_artifact(
    artifact: dict[str, Any],
) -> GenomeMigrationReportV1:
    """Verify the embedded report against every target genome."""

    if artifact.get("artifact_schema_version") != MIGRATED_ARTIFACT_SCHEMA_VERSION:
        raise GenomeMigrationError("migrated artifact schema is invalid")
    if artifact.get("genome_schema_version") != GENOME_SCHEMA_VERSION:
        raise GenomeMigrationError("migrated artifact target genome schema is invalid")
    try:
        report = GenomeMigrationReportV1.model_validate(artifact.get("migration_report"))
    except Exception as exc:
        raise GenomeMigrationError("migration report is invalid") from exc
    artifact_kind, located = _artifact_entries(artifact)
    if report.source_artifact_kind != artifact_kind:
        raise GenomeMigrationError("migration report artifact kind mismatch")
    by_locator = {entry.locator: entry for entry in report.entries}
    if set(by_locator) != {locator for locator, _, _ in located}:
        raise GenomeMigrationError("migration report locator set mismatch")
    for locator, container, gene_key in located:
        gene_payload = container[gene_key]
        if canonical_config_hash(gene_payload) != by_locator[locator].target_gene_hash:
            raise GenomeMigrationError(f"migrated genome hash mismatch at {locator}")
        try:
            StrategyGene.from_dict_exact(gene_payload)
        except Exception as exc:
            raise GenomeMigrationError(
                f"migrated genome violates target contract at {locator}"
            ) from exc
    return report
