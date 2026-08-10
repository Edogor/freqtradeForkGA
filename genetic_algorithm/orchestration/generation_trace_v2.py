"""Hash-chained, append-only generation diagnostics for GA island runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field, model_validator

from genetic_algorithm.core.individual import (
    FITNESS_EVIDENCE_BACKTEST_FAILED,
    FITNESS_EVIDENCE_BACKTEST_VALID,
    Individual,
)
from genetic_algorithm.core.population import PopulationStats
from genetic_algorithm.orchestration.result_contract import StrictV2Model


TRACE_SCHEMA_VERSION = "generation-trace-v2"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _finite_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if math.isfinite(resolved) else None


def _genome_signature(individual: Individual) -> str:
    payload = individual.strategy_gene.to_dict()
    payload.pop("generation", None)
    payload.pop("individual_id", None)
    return _sha256(payload)


def _behavior_signature(individual: Individual) -> str | None:
    """Return a conservative signature only when measured behavior exists."""

    if individual.fitness_evidence != FITNESS_EVIDENCE_BACKTEST_VALID:
        return None
    metrics = individual.metrics or {}
    per_pair = metrics.get("per_pair_profit")
    monthly = metrics.get("monthly_profits")
    trade_count = metrics.get("num_trades")
    drawdown = _finite_or_none(metrics.get("max_drawdown"))
    if not per_pair and not monthly and trade_count is None:
        return None
    payload = {
        "per_pair_profit": per_pair or {},
        "monthly_profits": monthly or [],
        "num_trades": trade_count,
        "max_drawdown": drawdown,
    }
    try:
        return _sha256(payload)
    except (TypeError, ValueError):
        return None


class GenerationTraceRecordV2(StrictV2Model):
    """One immutable island observation after evaluating a generation."""

    schema_version: Literal["generation-trace-v2"] = TRACE_SCHEMA_VERSION
    created_at: datetime
    generation: int = Field(ge=0)
    island_name: str = Field(min_length=1)
    panel_id: str | None = None
    population_size: int = Field(ge=0)
    measured_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    unique_genome_count: int = Field(ge=0)
    behavior_observed_count: int = Field(ge=0)
    unique_behavior_count: int = Field(ge=0)
    exact_genome_duplicate_fraction: float = Field(ge=0, le=1)
    exact_behavior_duplicate_fraction: float | None = Field(default=None, ge=0, le=1)
    best_fitness: float | None = None
    median_fitness: float | None = None
    avg_fitness: float | None = None
    best_raw_fitness: float | None = None
    avg_raw_fitness: float | None = None
    fitness_dispersion: float | None = Field(default=None, ge=0)
    genetic_diversity: float | None = Field(default=None, ge=0)
    previous_record_hash: str | None = Field(default=None, min_length=64, max_length=64)
    record_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _coherent_record(self) -> "GenerationTraceRecordV2":
        if self.created_at.utcoffset() is None:
            raise ValueError("generation trace timestamp must be timezone-aware")
        if self.measured_count + self.failed_count > self.population_size:
            raise ValueError("trace evidence counts exceed population")
        if self.unique_genome_count > self.population_size:
            raise ValueError("unique genomes exceed population")
        if self.unique_behavior_count > self.behavior_observed_count:
            raise ValueError("unique behaviors exceed observed behaviors")
        if self.record_hash != self.expected_hash:
            raise ValueError("generation trace record hash differs from content")
        return self

    @property
    def expected_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"record_hash"})
        return _sha256(payload)

    @property
    def identity(self) -> tuple[int, str]:
        return (self.generation, self.island_name)


def build_generation_trace_record(
    *,
    generation: int,
    island_name: str,
    panel_id: str | None,
    individuals: Iterable[Individual],
    stats: PopulationStats,
    created_at: datetime | None = None,
    previous_record_hash: str | None = None,
) -> GenerationTraceRecordV2:
    population = list(individuals)
    genome_signatures = [_genome_signature(item) for item in population]
    behavior_signatures = [
        signature
        for item in population
        if (signature := _behavior_signature(item)) is not None
    ]
    measured_count = sum(
        item.fitness_evidence == FITNESS_EVIDENCE_BACKTEST_VALID for item in population
    )
    failed_count = sum(
        item.fitness_evidence == FITNESS_EVIDENCE_BACKTEST_FAILED for item in population
    )
    genome_unique = len(set(genome_signatures))
    behavior_unique = len(set(behavior_signatures))
    payload = {
        "created_at": created_at or datetime.now(UTC),
        "generation": generation,
        "island_name": island_name,
        "panel_id": panel_id,
        "population_size": len(population),
        "measured_count": measured_count,
        "failed_count": failed_count,
        "unique_genome_count": genome_unique,
        "behavior_observed_count": len(behavior_signatures),
        "unique_behavior_count": behavior_unique,
        "exact_genome_duplicate_fraction": (
            0.0 if not population else 1.0 - (genome_unique / len(population))
        ),
        "exact_behavior_duplicate_fraction": (
            None
            if not behavior_signatures
            else 1.0 - (behavior_unique / len(behavior_signatures))
        ),
        "best_fitness": _finite_or_none(stats.best_fitness),
        "median_fitness": _finite_or_none(stats.median_fitness),
        "avg_fitness": _finite_or_none(stats.avg_fitness),
        "best_raw_fitness": _finite_or_none(stats.best_raw_fitness),
        "avg_raw_fitness": _finite_or_none(stats.avg_raw_fitness),
        "fitness_dispersion": _finite_or_none(stats.diversity_score),
        "genetic_diversity": _finite_or_none(stats.genetic_diversity),
        "previous_record_hash": previous_record_hash,
    }
    unvalidated = GenerationTraceRecordV2.model_construct(
        **payload,
        record_hash="0" * 64,
    )
    record_hash = _sha256(
        unvalidated.model_dump(mode="json", exclude={"record_hash"})
    )
    return GenerationTraceRecordV2(**payload, record_hash=record_hash)


class GenerationTraceWriterV2:
    """Append idempotent records while verifying the complete hash chain."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read_verified(self) -> list[GenerationTraceRecordV2]:
        if not self.path.exists():
            return []
        records: list[GenerationTraceRecordV2] = []
        previous: str | None = None
        identities: set[tuple[int, str]] = set()
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"blank generation trace line: {line_number}")
            record = GenerationTraceRecordV2.model_validate_json(line)
            if record.previous_record_hash != previous:
                raise ValueError(f"broken generation trace chain at line {line_number}")
            if record.identity in identities:
                raise ValueError(f"duplicate generation trace identity: {record.identity}")
            identities.add(record.identity)
            records.append(record)
            previous = record.record_hash
        return records

    def append(
        self,
        *,
        generation: int,
        island_name: str,
        panel_id: str | None,
        individuals: Iterable[Individual],
        stats: PopulationStats,
        created_at: datetime | None = None,
    ) -> GenerationTraceRecordV2:
        existing = self.read_verified()
        identity = (generation, island_name)
        matches = [record for record in existing if record.identity == identity]
        if matches:
            return matches[0]
        record = build_generation_trace_record(
            generation=generation,
            island_name=island_name,
            panel_id=panel_id,
            individuals=individuals,
            stats=stats,
            created_at=created_at,
            previous_record_hash=(existing[-1].record_hash if existing else None),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(_canonical_json(record.model_dump(mode="json")) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record
