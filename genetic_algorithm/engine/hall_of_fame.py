"""
Strategy Hall of Fame

Persistent archive of the best strategies discovered across all GA runs.
Saves top strategies to disk so good genetic material is never lost.
Hall of fame members can be re-injected into future evolution runs.
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from genetic_algorithm.core.individual import (
    FITNESS_EVIDENCE_BACKTEST_VALID,
    VALID_FITNESS_EVIDENCE,
    Individual,
    infer_fitness_evidence,
)
from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.engine.checkpoint_contract import GENOME_SCHEMA_VERSION
from genetic_algorithm.evaluation.panel_contract import validate_panel_identity
from genetic_algorithm.strategies.operator_registry import (
    get_standard_operators,
    is_valid_operator,
    resolve_indicator_type,
)


logger = logging.getLogger(__name__)

DEFAULT_HOF_DIR = "genetic_algorithm/data/hall_of_fame"
DEFAULT_MAX_SIZE = 50


@dataclass
class HallOfFameEntry:
    """A single hall of fame entry."""

    strategy_gene_dict: Dict[str, Any]
    fitness: float
    metrics: Dict[str, Any]
    generation_found: int
    run_timestamp: float
    run_id: str = ""
    individual_id: int = 0
    entry_id: str = ""
    fitness_evidence: Optional[str] = None
    fitness_panel_id: Optional[str] = None
    fitness_panel_role: Optional[str] = None

    def __post_init__(self):
        """Generate a stable entry_id if not provided."""
        if self.fitness_evidence is None:
            self.fitness_evidence = infer_fitness_evidence(
                metrics=self.metrics,
                evaluated=True,
                fitness=self.fitness,
                legacy_unknown=True,
            )
        if self.fitness_evidence not in VALID_FITNESS_EVIDENCE:
            raise ValueError(f"Unknown hall-of-fame fitness evidence: {self.fitness_evidence!r}")
        self.metrics = dict(self.metrics or {})
        self.metrics["fitness_evidence"] = self.fitness_evidence
        if self.fitness_panel_id is None:
            self.fitness_panel_id = self.metrics.get("fitness_panel_id")
        if self.fitness_panel_role is None:
            self.fitness_panel_role = self.metrics.get("fitness_panel_role")
        validate_panel_identity(self.fitness_panel_id, self.fitness_panel_role)
        if self.fitness_panel_id is not None:
            self.metrics["fitness_panel_id"] = self.fitness_panel_id
        if self.fitness_panel_role is not None:
            self.metrics["fitness_panel_role"] = self.fitness_panel_role
        if not self.entry_id:
            import hashlib

            fp = json.dumps(self.strategy_gene_dict, sort_keys=True, default=str)
            self.entry_id = f"hof_{hashlib.sha256(fp.encode()).hexdigest()[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
            "strategy_gene": self.strategy_gene_dict,
            "fitness": self.fitness,
            "metrics": self.metrics,
            "generation_found": self.generation_found,
            "individual_id": self.individual_id,
            "run_timestamp": self.run_timestamp,
            "run_id": self.run_id,
            "fitness_evidence": self.fitness_evidence,
            "fitness_panel_id": self.fitness_panel_id,
            "fitness_panel_role": self.fitness_panel_role,
        }

    @property
    def has_measured_fitness(self) -> bool:
        """Whether this archive entry has successful real-backtest evidence."""
        try:
            finite = math.isfinite(float(self.fitness))
        except (TypeError, ValueError, OverflowError):
            finite = False
        return (
            finite
            and self.fitness_evidence == FITNESS_EVIDENCE_BACKTEST_VALID
            and not self.metrics.get("error")
        )

    def has_comparable_fitness(self, panel_id: str) -> bool:
        """Whether this measured entry belongs to an exact panel."""
        return self.has_measured_fitness and self.fitness_panel_id == panel_id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HallOfFameEntry":
        return cls(
            strategy_gene_dict=data["strategy_gene"],
            fitness=data["fitness"],
            metrics=data.get("metrics", {}),
            generation_found=data.get("generation_found", 0),
            run_timestamp=data.get("run_timestamp", 0),
            run_id=data.get("run_id", ""),
            individual_id=data.get("individual_id", 0),
            entry_id=data.get("id", ""),
            fitness_evidence=data.get("fitness_evidence"),
            fitness_panel_id=data.get("fitness_panel_id"),
            fitness_panel_role=data.get("fitness_panel_role"),
        )


class HallOfFame:
    """
    Persistent archive of top-performing strategies.

    Maintains a ranked list of the best strategies ever discovered,
    persisted to a JSON file on disk. Supports:
    - Adding new candidates after each generation
    - Re-injecting hall of fame members into new GA runs
    - Deduplication by strategy structure similarity
    """

    def __init__(
        self,
        directory: str = DEFAULT_HOF_DIR,
        max_size: int = DEFAULT_MAX_SIZE,
        min_fitness: float = 0.0,
        run_id: Optional[str] = None,
        required_panel_id: Optional[str] = None,
        required_panel_role: Optional[str] = None,
        panel_data_identity_verified: bool = False,
    ):
        """
        Args:
            directory: Directory to store hall of fame files.
            max_size: Maximum number of strategies to keep.
            min_fitness: Minimum fitness threshold to enter the hall.
            run_id: Optional run identifier; auto-generated from timestamp if omitted.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.min_fitness = min_fitness
        self.entries: List[HallOfFameEntry] = []
        self._fingerprint_cache: set = set()
        self.run_id = run_id or f"run_{time.time_ns()}"
        validate_panel_identity(required_panel_id, required_panel_role)
        self.required_panel_id = required_panel_id
        self.required_panel_role = required_panel_role
        self.panel_data_identity_verified = bool(panel_data_identity_verified)

        # Load existing hall of fame
        self._load()
        # Build fingerprint cache from loaded entries
        self._rebuild_fingerprint_cache()

    @property
    def filepath(self) -> Path:
        if self.required_panel_id is not None:
            scope = self.required_panel_id
            if not self.panel_data_identity_verified:
                # Without a content-addressed data manifest, equal config does
                # not prove equal candles across runs. Keep the archive useful
                # inside this run but never compare its score cross-run.
                scope = f"{scope}_{self.run_id}"
            return self.directory / f"hall_of_fame_{scope}.json"
        return self.directory / "hall_of_fame.json"

    def _load(self) -> None:
        """Load hall of fame from disk."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                if (
                    data.get("artifact_schema_version") != "hall-of-fame-v3"
                    or data.get("genome_schema_version") != GENOME_SCHEMA_VERSION
                ):
                    raise ValueError("legacy hall of fame requires explicit genome migration")
                for index, raw_entry in enumerate(data.get("entries", [])):
                    try:
                        StrategyGene.from_dict_exact(raw_entry["strategy_gene"])
                    except Exception as exc:
                        raise ValueError(
                            f"hall-of-fame entry {index} violates {GENOME_SCHEMA_VERSION}"
                        ) from exc
                loaded = [HallOfFameEntry.from_dict(e) for e in data.get("entries", [])]
                self.entries = [
                    entry
                    for entry in loaded
                    if entry.has_measured_fitness
                    and (
                        self.required_panel_id is None
                        or entry.fitness_panel_id == self.required_panel_id
                    )
                    and (
                        self.required_panel_role is None
                        or entry.fitness_panel_role == self.required_panel_role
                    )
                ]
                rejected = len(loaded) - len(self.entries)
                if rejected:
                    logger.warning(
                        "[HALL OF FAME] Ignored %d entries without successful "
                        "real-backtest evidence",
                        rejected,
                    )
                logger.info(
                    f"Loaded hall of fame with {len(self.entries)} entries from {self.filepath}"
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise ValueError(f"Refusing unsafe hall of fame {self.filepath}: {e}") from e
        else:
            logger.info("No existing hall of fame found. Starting fresh.")

    def bind_panel(
        self,
        panel_id: str,
        panel_role: Optional[str] = None,
        *,
        data_identity_verified: bool = False,
    ) -> None:
        """Scope this archive to one comparable panel and reload fail-closed."""
        validate_panel_identity(panel_id, panel_role)
        self.required_panel_id = panel_id
        self.required_panel_role = panel_role
        self.panel_data_identity_verified = bool(data_identity_verified)
        self.entries = []
        self._load()
        self._rebuild_fingerprint_cache()

    def _rebuild_fingerprint_cache(self) -> None:
        """Rebuild the O(1) fingerprint cache from current entries."""
        self._fingerprint_cache = {self._fingerprint(e.strategy_gene_dict) for e in self.entries}

    def _save(self) -> None:
        """Save hall of fame to disk (atomic write via tmp+rename)."""
        for index, entry in enumerate(self.entries):
            try:
                StrategyGene.from_dict_exact(entry.strategy_gene_dict)
            except Exception as exc:
                raise ValueError(
                    f"hall-of-fame entry {index} violates {GENOME_SCHEMA_VERSION}"
                ) from exc
        data = {
            "version": 3,
            "artifact_schema_version": "hall-of-fame-v3",
            "genome_schema_version": GENOME_SCHEMA_VERSION,
            "last_updated": time.time(),
            "total_entries": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }
        try:
            tmp_path = self.filepath.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            os.replace(tmp_path, self.filepath)
        except IOError as e:
            logger.error(f"Failed to save hall of fame: {e}")
            raise

    def _is_duplicate(self, gene_dict: Dict[str, Any]) -> bool:
        """
        Check if a strategy is structurally similar to an existing entry.

        Uses a detailed fingerprint including indicator types + parameters +
        timeframes, and condition operators + thresholds + indicator instance
        references. O(1) lookup via cached fingerprint set.
        """
        new_fp = self._fingerprint(gene_dict)
        return new_fp in self._fingerprint_cache

    def _fingerprint(self, gene_dict: Dict[str, Any]) -> str:
        """Create a structural fingerprint of a strategy gene dict.

        Includes indicator types + parameters, condition operators +
        thresholds + indicator refs, and timeframes so that strategies
        differing only in numeric values are NOT treated as duplicates.
        """
        # Indicator: type + sorted params + timeframe
        ind_parts = []
        for ind in gene_dict.get("indicators", []):
            params = ind.get("parameters", {})
            param_str = str(sorted(params.items())) if params else ""
            tf = ind.get("timeframe", "")
            ind_parts.append(f"{ind.get('type', '')}({param_str})@{tf}")
        ind_parts.sort()

        # Conditions: indicator ref + operator + threshold (rounded)
        # Use 1-decimal rounding so near-identical strategies (30.0 vs 30.0001)
        # are properly deduplicated while meaningfully different ones stay separate.
        def _cond_key(c):
            thr = round(c.get("threshold", 0), 2)
            return f"{c.get('indicator', '')}:{c.get('operator', '')}:{thr}"

        entry_keys = sorted(_cond_key(c) for c in gene_dict.get("entry_conditions", []))
        exit_keys = sorted(_cond_key(c) for c in gene_dict.get("exit_conditions", []))

        return f"{'|'.join(ind_parts)}::{'|'.join(entry_keys)}::{'|'.join(exit_keys)}"

    def update(self, population, generation: int) -> int:
        """
        Consider top individuals from a population for hall of fame induction.

        Args:
            population: Evaluated Population object.
            generation: Current generation number.

        Returns:
            Number of new entries added.
        """
        added = 0
        candidates = sorted(
            [
                ind
                for ind in population
                if ind.has_measured_fitness
                and ind.fitness is not None
                and (
                    self.required_panel_id is None or ind.fitness_panel_id == self.required_panel_id
                )
                and (
                    self.required_panel_role is None
                    or ind.fitness_panel_role == self.required_panel_role
                )
            ],
            key=lambda x: x.fitness or 0,
            reverse=True,
        )

        # Consider top 20% as candidates
        n_candidates = max(1, int(len(candidates) * 0.2))

        for ind in candidates[:n_candidates]:
            if ind.fitness is None or ind.fitness < self.min_fitness:
                continue

            gene_dict = ind.strategy_gene.to_dict()

            if self._is_duplicate(gene_dict):
                continue

            # Check if it qualifies — use raw_fitness (pre-sharing) for true performance
            actual_fitness = getattr(ind, "raw_fitness", None)
            if actual_fitness is None:
                actual_fitness = ind.fitness
            if actual_fitness is None or not math.isfinite(float(actual_fitness)):
                continue
            if len(self.entries) < self.max_size or actual_fitness > self.entries[-1].fitness:
                entry_metrics = dict(ind.metrics or {})
                entry_metrics["fitness_evidence"] = FITNESS_EVIDENCE_BACKTEST_VALID
                entry = HallOfFameEntry(
                    strategy_gene_dict=gene_dict,
                    fitness=actual_fitness,
                    metrics=entry_metrics,
                    generation_found=generation,
                    run_timestamp=time.time(),
                    run_id=self.run_id,
                    individual_id=getattr(ind, "id", 0) if hasattr(ind, "id") else 0,
                    fitness_evidence=FITNESS_EVIDENCE_BACKTEST_VALID,
                    fitness_panel_id=ind.fitness_panel_id,
                    fitness_panel_role=ind.fitness_panel_role,
                )
                self.entries.append(entry)
                self._fingerprint_cache.add(self._fingerprint(gene_dict))
                added += 1

        if added > 0:
            # Sort by fitness descending, trim to max_size
            self.entries.sort(key=lambda e: e.fitness, reverse=True)
            self.entries = self.entries[: self.max_size]
            # Rebuild cache after truncation may have removed entries
            self._rebuild_fingerprint_cache()
            self._save()
            logger.info(f"[HALL OF FAME] Added {added} new entries. Total: {len(self.entries)}")

        return added

    def get_individuals(self, count: int) -> List[Individual]:
        """
        Create Individual objects from hall of fame entries for re-injection.

        Requires the same exact semantic contract used when the archive was
        written. Persisted strategies are never repaired during injection.

        Args:
            count: Maximum number of individuals to return.

        Returns:
            List of Individual objects with strategy genes from the hall of fame.
        """
        individuals = []
        for entry in self.entries[:count]:
            gene = StrategyGene.from_dict_exact(entry.strategy_gene_dict)
            ind = Individual(strategy_gene=gene)
            ind.metrics = {
                "source": "hall_of_fame",
                "original_fitness": entry.fitness,
                "source_fitness_evidence": entry.fitness_evidence,
                "fitness_evidence": ind.fitness_evidence,
            }
            individuals.append(ind)

        return individuals

    @staticmethod
    def _fix_invalid_operators(gene: StrategyGene) -> None:
        """Fix any invalid operator/indicator combos in a restored gene."""
        import random

        indicator_map = {ind.instance_id: ind for ind in gene.indicators if ind.instance_id}

        for cond_list in (gene.entry_conditions, gene.exit_conditions):
            for cond in cond_list:
                ind_obj = indicator_map.get(cond.indicator)
                if ind_obj is not None:
                    ind_type = ind_obj.type
                else:
                    # Fall back to parsing the indicator reference string
                    ind_type = resolve_indicator_type(cond.indicator)
                if not is_valid_operator(ind_type, cond.operator):
                    valid_ops = get_standard_operators(ind_type)
                    if not valid_ops:
                        continue
                    old_op = cond.operator
                    cond.operator = random.choice(valid_ops)
                    logger.debug(f"[HOF FIX] {ind_type} operator '{old_op}' -> '{cond.operator}'")

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the hall of fame."""
        if not self.entries:
            return {"size": 0, "entries": []}

        return {
            "size": len(self.entries),
            "best_fitness": self.entries[0].fitness,
            "worst_fitness": self.entries[-1].fitness,
            "avg_fitness": sum(e.fitness for e in self.entries) / len(self.entries),
            "unique_runs": len(set(e.run_id for e in self.entries if e.run_id)),
            "top_5": [
                {
                    "fitness": round(e.fitness, 4),
                    "profit": round(e.metrics.get("profit", 0), 2),
                    "sharpe": round(e.metrics.get("sharpe_ratio", 0), 2),
                    "generation": e.generation_found,
                    "run_id": e.run_id[:16],
                }
                for e in self.entries[:5]
            ],
        }
