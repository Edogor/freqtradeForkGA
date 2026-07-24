"""
Warm-Start Module

Loads explicitly hash-bound populations or Hall of Fame archives to seed a new
evolution run.  Enabled warm-start is fail-closed: it never discovers a mutable
"latest" file and never degrades to a scratch population.

Config:
    warm_start:
        enabled: true
        source_experiment: "E170"          # Provenance label only
        source_type: "population"          # "population", "hof", or "both"
        top_n: 15                          # Max individuals to import
        source_checkpoint: path.json
        source_checkpoint_sha256: "<sha256>"
        source_hof: hall_of_fame.json
        source_hof_sha256: "<sha256>"
        genome_schema_version: strategy-gene-v2

Usage:
    loader = WarmStartLoader(config, logger)
    individuals = loader.load()
    # Inject into initialize_population()
"""

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.engine.checkpoint_contract import (
    CHECKPOINT_VERSION,
    GENOME_SCHEMA_VERSION,
    CheckpointContractError,
    verify_checkpoint_integrity,
)
from genetic_algorithm.genome.migration import (
    MIGRATED_ARTIFACT_SCHEMA_VERSION,
    migrate_genome_artifact_bytes,
    verify_migrated_genome_artifact,
)
from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash


logger = logging.getLogger(__name__)


class WarmStartError(ValueError):
    """Raised when an enabled warm-start cannot prove its complete source."""


class WarmStartLoader:
    """Loads seed individuals from previous experiment artifacts."""

    def __init__(self, config: Dict[str, Any], ga_logger: Optional[logging.Logger] = None):
        self.config = config
        self.ws_config = config.get("warm_start", {})
        self.logger = ga_logger or logger
        self.enabled = self.ws_config.get("enabled", False)
        self.source_type = self.ws_config.get("source_type", "population")
        self.top_n = self.ws_config.get("top_n", 15)
        self.source_experiment = self.ws_config.get("source_experiment", "")
        self.source_checkpoint = self.ws_config.get("source_checkpoint")
        self.source_checkpoint_sha256 = self.ws_config.get("source_checkpoint_sha256")
        self.source_checkpoint_migration_source = self.ws_config.get(
            "source_checkpoint_migration_source"
        )
        self.source_checkpoint_migration_source_sha256 = self.ws_config.get(
            "source_checkpoint_migration_source_sha256"
        )
        self.source_hof = self.ws_config.get("source_hof")
        self.source_hof_sha256 = self.ws_config.get("source_hof_sha256")
        self.source_hof_migration_source = self.ws_config.get("source_hof_migration_source")
        self.source_hof_migration_source_sha256 = self.ws_config.get(
            "source_hof_migration_source_sha256"
        )
        self.genome_schema_version = self.ws_config.get("genome_schema_version")
        self._migration_report_hashes: Dict[str, str] = {}

    def load(self) -> List[Individual]:
        """Load individuals according to configuration.

        Returns list of Individual objects with reset generation/id/fitness
        ready for evaluation in a new run.
        """
        if not self.enabled:
            return []

        self._validate_contract()
        individuals: List[Individual] = []
        if self.source_type in ("population", "both"):
            individuals.extend(self._load_from_checkpoint())
        if self.source_type in ("hof", "both"):
            individuals.extend(self._load_from_hof())
        if not individuals:
            raise WarmStartError("warm-start sources contain no individuals")

        # Deduplicate by gene dict hash
        seen = set()
        unique: List[Individual] = []
        for ind in individuals:
            h = canonical_config_hash(ind.strategy_gene.to_dict())
            if h not in seen:
                seen.add(h)
                unique.append(ind)
        if not unique:
            raise WarmStartError("warm-start sources contain no unique genomes")

        # Take top_n best (sorted by fitness descending, unknowns last)
        for individual in unique:
            if individual.fitness is not None:
                try:
                    finite_fitness = math.isfinite(float(individual.fitness))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise WarmStartError("warm-start source contains non-numeric fitness") from exc
                if not finite_fitness:
                    raise WarmStartError("warm-start source contains non-finite fitness")
        unique.sort(
            key=lambda i: i.fitness if i.fitness is not None else float("-inf"),
            reverse=True,
        )
        result = unique[: self.top_n]

        # Reset identifiers so they integrate cleanly into the new population
        for idx, ind in enumerate(result):
            ind.strategy_gene.generation = 0
            ind.strategy_gene.individual_id = 9000 + idx  # High IDs to avoid collisions
            ind.fitness = None
            ind.raw_fitness = None
            ind.evaluated = False
            if hasattr(ind, "metrics"):
                ind.metrics = {
                    "fitness_evidence": "UNMEASURED",
                    "origin": f"warm_start_{self.source_type}",
                    "warm_start_source_experiment": (self.source_experiment or "explicit_source"),
                    "warm_start_source_hashes": self._source_hashes(),
                    "warm_start_migration_reports": dict(self._migration_report_hashes),
                }
            ind.fitness_evidence = "UNMEASURED"
            ind.fitness_panel_id = None
            ind.fitness_panel_role = None
        self.logger.info(
            f"[WARM-START] Loaded {len(result)} individuals "
            f"(source_type={self.source_type}, experiment={self.source_experiment})"
        )
        return result

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_from_checkpoint(self) -> List[Individual]:
        """Load individuals from a checkpoint file."""
        filepath, data = self._verified_json_source(
            self.source_checkpoint,
            self.source_checkpoint_sha256,
            label="checkpoint",
        )

        self.logger.info(f"[WARM-START] Loading population from checkpoint: {filepath}")
        if data.get("version") == CHECKPOINT_VERSION:
            try:
                data = verify_checkpoint_integrity(data)
            except CheckpointContractError as exc:
                raise WarmStartError(
                    "warm-start checkpoint fails its internal integrity contract"
                ) from exc
            provenance = data.get("provenance") or {}
            if provenance.get("genome_schema_version") != GENOME_SCHEMA_VERSION:
                raise WarmStartError(
                    "warm-start checkpoint does not declare strategy-gene-v2 provenance"
                )
        else:
            self._verify_declared_genome_artifact(data, label="checkpoint")

        pop_data = data.get("population", {})
        if not isinstance(pop_data, dict):
            raise WarmStartError("warm-start checkpoint population must be an object")
        individuals_data = pop_data.get("individuals", [])
        return self._parse_individuals(
            individuals_data,
            label=f"checkpoint {filepath}",
        )

    def _load_from_hof(self) -> List[Individual]:
        """Load individuals from Hall of Fame JSON files."""
        filepath, data = self._verified_json_source(
            self.source_hof,
            self.source_hof_sha256,
            label="hall-of-fame",
        )
        self._verify_declared_genome_artifact(data, label="hall-of-fame")
        entries = data.get("entries")
        if not isinstance(entries, list) or not entries:
            raise WarmStartError("hall-of-fame source has no entries")

        individuals = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise WarmStartError(f"hall-of-fame entry {index} must be an object")
            gene_dict = entry.get("gene_dict") or entry.get("strategy_gene")
            if not isinstance(gene_dict, dict):
                raise WarmStartError(f"hall-of-fame entry {index} has no strategy gene")
            try:
                gene = StrategyGene.from_dict_exact(gene_dict)
            except Exception as exc:
                raise WarmStartError(
                    f"hall-of-fame entry {index} violates the genome contract"
                ) from exc
            individual = Individual(strategy_gene=gene)
            individual.fitness = entry.get("fitness")
            individuals.append(individual)
        return individuals

    def _validate_contract(self) -> None:
        if self.source_type not in {"population", "hof", "both"}:
            raise WarmStartError("warm_start.source_type must be population, hof, or both")
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, int) or self.top_n < 1:
            raise WarmStartError("warm_start.top_n must be a positive integer")
        if self.genome_schema_version != GENOME_SCHEMA_VERSION:
            raise WarmStartError(
                f"warm_start.genome_schema_version must be {GENOME_SCHEMA_VERSION}"
            )
        if self.source_type in {"population", "both"}:
            self._validate_source_declaration(
                self.source_checkpoint,
                self.source_checkpoint_sha256,
                label="checkpoint",
            )
        if self.source_type in {"hof", "both"}:
            self._validate_source_declaration(
                self.source_hof,
                self.source_hof_sha256,
                label="hall-of-fame",
            )

    @staticmethod
    def _validate_source_declaration(path: Any, digest: Any, *, label: str) -> None:
        if not isinstance(path, str) or not path.strip():
            raise WarmStartError(f"warm-start {label} path must be explicit")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise WarmStartError(f"warm-start {label} requires a lowercase SHA-256 digest")

    @classmethod
    def _verified_json_source(
        cls,
        path: Any,
        digest: Any,
        *,
        label: str,
    ) -> tuple[Path, Dict[str, Any]]:
        cls._validate_source_declaration(path, digest, label=label)
        source = Path(path)
        if not source.is_file():
            raise WarmStartError(f"warm-start {label} does not exist: {source}")
        try:
            content = source.read_bytes()
        except OSError as exc:
            raise WarmStartError(f"warm-start {label} cannot be read: {source}") from exc
        actual = hashlib.sha256(content).hexdigest()
        if actual != digest:
            raise WarmStartError(f"warm-start {label} SHA-256 mismatch")
        try:
            data = json.loads(content)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise WarmStartError(f"warm-start {label} is not valid JSON") from exc
        if not isinstance(data, dict):
            raise WarmStartError(f"warm-start {label} root must be an object")
        return source, data

    @staticmethod
    def _parse_individuals(raw: Any, *, label: str) -> List[Individual]:
        if not isinstance(raw, list) or not raw:
            raise WarmStartError(f"{label} has no individuals")
        result = []
        for index, ind_data in enumerate(raw):
            if not isinstance(ind_data, dict):
                raise WarmStartError(f"{label} individual {index} must be an object")
            gene_payload = ind_data.get("strategy_gene")
            if not isinstance(gene_payload, dict):
                raise WarmStartError(f"{label} individual {index} has no strategy_gene")
            try:
                StrategyGene.from_dict_exact(gene_payload)
                result.append(Individual.from_dict(ind_data))
            except Exception as exc:
                raise WarmStartError(
                    f"{label} individual {index} violates the genome contract"
                ) from exc
        return result

    def _source_hashes(self) -> Dict[str, str]:
        hashes = {}
        if self.source_type in {"population", "both"}:
            hashes["checkpoint"] = self.source_checkpoint_sha256
            if self.source_checkpoint_migration_source:
                hashes["checkpoint_migration_source"] = (
                    self.source_checkpoint_migration_source_sha256
                )
        if self.source_type in {"hof", "both"}:
            hashes["hall_of_fame"] = self.source_hof_sha256
            if self.source_hof_migration_source:
                hashes["hall_of_fame_migration_source"] = self.source_hof_migration_source_sha256
        return hashes

    def _verify_declared_genome_artifact(
        self,
        data: Dict[str, Any],
        *,
        label: str,
    ) -> None:
        if data.get("genome_schema_version") != GENOME_SCHEMA_VERSION:
            raise WarmStartError(
                f"warm-start {label} has no explicit {GENOME_SCHEMA_VERSION} schema"
            )
        artifact_schema = data.get("artifact_schema_version")
        if artifact_schema == MIGRATED_ARTIFACT_SCHEMA_VERSION:
            try:
                report = verify_migrated_genome_artifact(data)
                source_path, source_digest = {
                    "checkpoint": (
                        self.source_checkpoint_migration_source,
                        self.source_checkpoint_migration_source_sha256,
                    ),
                    "hall-of-fame": (
                        self.source_hof_migration_source,
                        self.source_hof_migration_source_sha256,
                    ),
                }[label]
                self._validate_source_declaration(
                    source_path,
                    source_digest,
                    label=f"{label} migration source",
                )
                legacy_path = Path(source_path)
                if not legacy_path.is_file():
                    raise WarmStartError(
                        f"warm-start {label} migration source does not exist: {legacy_path}"
                    )
                try:
                    legacy_content = legacy_path.read_bytes()
                except OSError as exc:
                    raise WarmStartError(
                        f"warm-start {label} migration source cannot be read: {legacy_path}"
                    ) from exc
                if hashlib.sha256(legacy_content).hexdigest() != source_digest:
                    raise WarmStartError(f"warm-start {label} migration source SHA-256 mismatch")
                if report.source_sha256 != source_digest:
                    raise WarmStartError(
                        f"warm-start {label} report is bound to another migration source"
                    )
                rebuilt, rebuilt_report = migrate_genome_artifact_bytes(
                    legacy_content,
                    expected_sha256=source_digest,
                )
                if rebuilt != data or rebuilt_report != report:
                    raise WarmStartError(f"warm-start {label} migration is not reproducible")
            except Exception as exc:
                if isinstance(exc, WarmStartError):
                    raise
                raise WarmStartError(f"warm-start {label} migration report is invalid") from exc
            self._migration_report_hashes[label] = report.report_hash
            return
        expected_native_schema = {
            "checkpoint": "population-seed-v1",
            "hall-of-fame": "hall-of-fame-v3",
        }[label]
        if artifact_schema != expected_native_schema:
            raise WarmStartError(
                f"warm-start {label} artifact_schema_version must be "
                f"{expected_native_schema} or {MIGRATED_ARTIFACT_SCHEMA_VERSION}"
            )
