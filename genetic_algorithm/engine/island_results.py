"""Fail-closed result contract shared by every island-model consumer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.evaluation.panel_contract import (
    PANEL_ROLE_COMMON_REPLAY,
    phenotype_fingerprint,
)


class IslandResultContractError(ValueError):
    """Raised when island output cannot prove one comparable final ranking."""


@dataclass(frozen=True)
class IslandFinalistBatch:
    """Validated deterministic view over a raw per-island result mapping."""

    island_names: tuple[str, ...]
    per_island: tuple[tuple[Individual, ...], ...]
    flattened_candidates: tuple[Individual, ...]
    finalists: tuple[Individual, ...]
    common_panel_id: str

    def candidates_for(self, island_name: str) -> tuple[Individual, ...]:
        try:
            index = self.island_names.index(island_name)
        except ValueError as exc:
            raise KeyError(island_name) from exc
        return self.per_island[index]

    def top(self, count: int) -> tuple[Individual, ...]:
        if count < 0:
            raise ValueError("finalist count cannot be negative")
        return self.finalists[:count]


def _coerce_individual_list(value: Any, *, key: str) -> tuple[Individual, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise IslandResultContractError(f"island result {key!r} must be a sequence")
    items = tuple(value)
    invalid = [
        f"{index}:{type(item).__name__}"
        for index, item in enumerate(items)
        if not isinstance(item, Individual)
    ]
    if invalid:
        raise IslandResultContractError(
            f"island result {key!r} contains non-Individual values: {invalid}"
        )
    return items


def _required_metric(candidate: Individual, key: str) -> Real:
    value = candidate.metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise IslandResultContractError(f"finalist {candidate.id} lacks numeric metric {key!r}")
    if not math.isfinite(float(value)):
        raise IslandResultContractError(f"finalist {candidate.id} has non-finite metric {key!r}")
    return value


def extract_island_finalists(
    results: Mapping[str, Sequence[Individual]],
    *,
    expected_islands: Sequence[str],
    require_finalists: bool = True,
) -> IslandFinalistBatch:
    """Flatten local lists and prove that ``__global__`` is a real replay rank.

    Local search scores may come from different panels and are never ranked
    here.  Only the explicitly replayed ``__global__`` list can become a final
    result.
    """

    if not isinstance(results, Mapping):
        raise IslandResultContractError("island evolution result must be a mapping")
    island_names = tuple(expected_islands)
    if (
        not island_names
        or len(set(island_names)) != len(island_names)
        or any(
            not isinstance(name, str) or not name or name == "__global__" for name in island_names
        )
    ):
        raise IslandResultContractError("expected island names must be non-empty and unique")

    expected_keys = {*island_names, "__global__"}
    actual_keys = set(results)
    if actual_keys != expected_keys:
        raise IslandResultContractError(
            "island result key set differs; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )

    per_island = tuple(_coerce_individual_list(results[name], key=name) for name in island_names)
    flattened = tuple(candidate for items in per_island for candidate in items)
    finalists = _coerce_individual_list(results["__global__"], key="__global__")
    if require_finalists and not finalists:
        raise IslandResultContractError("island evolution produced no common-replay finalist")
    if not finalists:
        return IslandFinalistBatch(
            island_names=island_names,
            per_island=per_island,
            flattened_candidates=flattened,
            finalists=(),
            common_panel_id="",
        )

    local_fingerprints = {phenotype_fingerprint(candidate.strategy_gene) for candidate in flattened}
    finalist_fingerprints: list[str] = []
    panel_ids: set[str] = set()
    previous_fitness = math.inf
    for candidate in finalists:
        fingerprint = phenotype_fingerprint(candidate.strategy_gene)
        if fingerprint not in local_fingerprints:
            raise IslandResultContractError(
                f"global finalist {candidate.id} is absent from all island lists"
            )
        if fingerprint in finalist_fingerprints:
            raise IslandResultContractError(f"duplicate global finalist phenotype: {candidate.id}")
        finalist_fingerprints.append(fingerprint)

        if (
            not candidate.has_measured_fitness
            or candidate.fitness_panel_role != PANEL_ROLE_COMMON_REPLAY
            or candidate.metrics.get("common_replay") is not True
            or not candidate.fitness_panel_id
        ):
            raise IslandResultContractError(
                f"global finalist {candidate.id} lacks COMMON_REPLAY evidence"
            )
        panel_ids.add(candidate.fitness_panel_id)
        fitness = float(candidate.raw_fitness)
        if fitness > previous_fitness:
            raise IslandResultContractError("__global__ finalists are not sorted by replay fitness")
        previous_fitness = fitness

        _required_metric(candidate, "profit")
        trades = _required_metric(candidate, "num_trades")
        if trades < 0 or float(trades) != int(trades):
            raise IslandResultContractError(f"finalist {candidate.id} has invalid num_trades")

    if len(panel_ids) != 1:
        raise IslandResultContractError("global finalists do not share one exact replay panel")

    return IslandFinalistBatch(
        island_names=island_names,
        per_island=per_island,
        flattened_candidates=flattened,
        finalists=finalists,
        common_panel_id=next(iter(panel_ids)),
    )
