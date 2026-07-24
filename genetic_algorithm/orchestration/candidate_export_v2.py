"""Fail-closed export of legacy GA objects into frozen replay candidates."""

from __future__ import annotations

import copy
import math
import random
import re
from collections.abc import Mapping
from typing import Any, Literal, Optional

from genetic_algorithm.core.strategy_gene import StrategyGene, timeframe_to_minutes
from genetic_algorithm.engine.hall_of_fame import HallOfFameEntry
from genetic_algorithm.genome.codegen import StrategyGenerator
from genetic_algorithm.genome.indicators import is_valid_operator, resolve_indicator_type
from genetic_algorithm.genome.individual import (
    FITNESS_EVIDENCE_BACKTEST_VALID,
    Individual,
)
from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.replay_runner_v2 import FrozenCandidateV2, freeze_candidate
from genetic_algorithm.orchestration.result_contract import StrictV2Model


class CandidateExportError(ValueError):
    """Raised when a legacy candidate cannot be frozen without repair or ambiguity."""


class CandidateExportV2(StrictV2Model):
    """Frozen executable plus provenance of the legacy object it came from."""

    source_kind: Literal["INDIVIDUAL", "HALL_OF_FAME"]
    source_id: str
    source_gene_hash: str
    normalized_gene_hash: str
    source_fitness: float
    source_fitness_evidence: Literal["BACKTEST_VALID"]
    source_fitness_panel_id: Optional[str] = None
    source_fitness_panel_role: Optional[str] = None
    candidate: FrozenCandidateV2


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _require_safe_id(value: str, field: str) -> str:
    if not _SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise CandidateExportError(f"{field} is not a safe artifact identifier: {value!r}")
    return value


def _normalized_gene(source: Mapping[str, Any]) -> tuple[StrategyGene, str, str]:
    try:
        source_payload = copy.deepcopy(dict(source))
        source_hash = canonical_config_hash(source_payload)
        gene = StrategyGene.from_dict(source_payload)
        gene.assign_instance_ids()
        normalized_hash = canonical_config_hash(gene.to_dict())
    except Exception as exc:
        raise CandidateExportError("legacy strategy gene cannot be normalized exactly") from exc
    if normalized_hash != source_hash:
        raise CandidateExportError(
            "legacy strategy gene requires migration; re-evaluate the canonical gene "
            "before V2 export"
        )
    return gene, source_hash, normalized_hash


def _condition_indicator_type(gene: StrategyGene, indicator_ref: str) -> str | None:
    by_id = {indicator.instance_id: indicator.type for indicator in gene.indicators}
    if indicator_ref in by_id:
        return by_id[indicator_ref]
    matching_types = [
        indicator.type for indicator in gene.indicators if indicator.type == indicator_ref
    ]
    if len(matching_types) == 1:
        return matching_types[0]
    resolved = resolve_indicator_type(indicator_ref)
    matching_resolved = [
        indicator.type for indicator in gene.indicators if indicator.type == resolved
    ]
    return matching_resolved[0] if len(matching_resolved) == 1 else None


def _validate_conditions(gene: StrategyGene, config: Mapping[str, Any]) -> None:
    indicator_config = config.get("indicators", {})
    minimums = {
        "entry": int(indicator_config.get("min_entry_conditions", 2)),
        "exit": int(indicator_config.get("min_exit_conditions", 1)),
    }
    condition_groups = {
        "entry": gene.entry_conditions,
        "exit": gene.exit_conditions,
    }
    for group_name, conditions in condition_groups.items():
        if len(conditions) < minimums[group_name]:
            raise CandidateExportError(
                f"{group_name} conditions are below the configured minimum; "
                "legacy code generation would repair the candidate randomly"
            )
        for condition in conditions:
            indicator_type = _condition_indicator_type(gene, condition.indicator)
            if indicator_type is None:
                raise CandidateExportError(
                    f"{group_name} condition references missing or ambiguous indicator "
                    f"{condition.indicator!r}"
                )
            if not is_valid_operator(indicator_type, condition.operator):
                raise CandidateExportError(
                    f"invalid operator {condition.operator!r} for {indicator_type}; "
                    "automatic Hall-of-Fame repair is forbidden in V2"
                )
            if condition.logic not in {"AND", "OR"}:
                raise CandidateExportError(f"invalid condition logic: {condition.logic!r}")
            if not math.isfinite(float(condition.threshold)):
                raise CandidateExportError("condition threshold must be finite")
            if not math.isfinite(float(condition.threshold_upper)):
                raise CandidateExportError("condition threshold_upper must be finite")
            if condition.lookback < 1:
                raise CandidateExportError("condition lookback must be positive")


def _validate_safe_v2_gene(gene: StrategyGene, config: Mapping[str, Any]) -> None:
    if timeframe_to_minutes(gene.timeframe) <= 0:
        raise CandidateExportError(f"unsupported strategy timeframe: {gene.timeframe!r}")
    allowed_timeframes = config.get("strategy_constraints", {}).get("timeframes")
    if allowed_timeframes and gene.timeframe not in allowed_timeframes:
        raise CandidateExportError("candidate timeframe is outside strategy_constraints.timeframes")
    if gene.max_open_trades < 1:
        raise CandidateExportError("max_open_trades must be positive")
    if gene.can_short or config.get("short_selling", {}).get("enabled", False):
        raise CandidateExportError("safe V2 export currently supports spot-long candidates only")

    informative = set(gene.informative_timeframes)
    informative.update(
        indicator.timeframe for indicator in gene.indicators if indicator.timeframe is not None
    )
    if informative:
        raise CandidateExportError(
            "safe V2 export does not yet support informative-timeframe data provenance"
        )
    if config.get("multi_timeframe", {}).get("enabled", False):
        raise CandidateExportError("safe V2 export requires multi_timeframe.enabled=false")
    if gene.regime_gene is not None and gene.regime_gene.enabled:
        raise CandidateExportError(
            "safe V2 export does not yet support runtime regime data provenance"
        )
    if config.get("regime_aware", {}).get("enabled", False):
        raise CandidateExportError("safe V2 export requires regime_aware.enabled=false")
    _validate_conditions(gene, config)


def _generate_once(
    normalized: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    gene = StrategyGene.from_dict_exact(copy.deepcopy(dict(normalized)))
    code = StrategyGenerator(copy.deepcopy(dict(config))).generate_strategy_code(gene)
    return code, gene.to_dict()


def _freeze_legacy_gene(
    *,
    source_kind: Literal["INDIVIDUAL", "HALL_OF_FAME"],
    source_id: str,
    source_fitness: float,
    source_fitness_evidence: str,
    source_fitness_panel_id: Optional[str],
    source_fitness_panel_role: Optional[str],
    source_gene: Mapping[str, Any],
    config: Mapping[str, Any],
) -> CandidateExportV2:
    source_id = _require_safe_id(source_id, "source_id")
    if not math.isfinite(float(source_fitness)):
        raise CandidateExportError("source candidate must have finite evaluated fitness")
    if source_fitness_evidence != FITNESS_EVIDENCE_BACKTEST_VALID:
        raise CandidateExportError(
            "source candidate lacks successful real-backtest fitness evidence"
        )
    gene, source_hash, normalized_hash = _normalized_gene(source_gene)
    _validate_safe_v2_gene(gene, config)
    normalized = gene.to_dict()

    original_random_state = random.getstate()
    try:
        first_code, first_gene = _generate_once(normalized, config)
        second_code, second_gene = _generate_once(normalized, config)
    finally:
        random.setstate(original_random_state)

    if first_gene != normalized or second_gene != normalized:
        raise CandidateExportError(
            "legacy code generation mutates or repairs this candidate; export refused"
        )
    if first_code != second_code:
        raise CandidateExportError("legacy code generation is not deterministic for this candidate")
    if "Auto-generated FALLBACK strategy" in first_code:
        raise CandidateExportError(
            "legacy code generation returned a non-trading fallback strategy"
        )

    strategy_name = f"GAStrategy_Gen{gene.generation}_Ind{gene.individual_id}"
    candidate = freeze_candidate(
        candidate_id=source_id,
        strategy_name=strategy_name,
        strategy_code=first_code,
        timeframe=gene.timeframe,
        max_open_trades=gene.max_open_trades,
        execution_parameters={
            "trading_mode": "spot",
            "position_adjustment": False,
            "can_short": False,
            "informative_timeframes": [],
        },
    )
    return CandidateExportV2(
        source_kind=source_kind,
        source_id=source_id,
        source_gene_hash=source_hash,
        normalized_gene_hash=normalized_hash,
        source_fitness=float(source_fitness),
        source_fitness_evidence=FITNESS_EVIDENCE_BACKTEST_VALID,
        source_fitness_panel_id=source_fitness_panel_id,
        source_fitness_panel_role=source_fitness_panel_role,
        candidate=candidate,
    )


def freeze_individual_v2(
    individual: Individual,
    config: Mapping[str, Any],
) -> CandidateExportV2:
    """Freeze an evaluated legacy Individual without invoking repair behavior."""

    fitness = individual.raw_fitness
    if fitness is None:
        fitness = individual.fitness
    if not individual.has_measured_fitness or fitness is None:
        raise CandidateExportError(
            "Individual needs successful real-backtest evidence before V2 export"
        )
    return _freeze_legacy_gene(
        source_kind="INDIVIDUAL",
        source_id=individual.id,
        source_fitness=fitness,
        source_fitness_evidence=individual.fitness_evidence,
        source_fitness_panel_id=individual.fitness_panel_id,
        source_fitness_panel_role=individual.fitness_panel_role,
        source_gene=individual.strategy_gene.to_dict(),
        config=config,
    )


def freeze_hof_entry_v2(
    entry: HallOfFameEntry,
    config: Mapping[str, Any],
) -> CandidateExportV2:
    """Freeze a Hall-of-Fame entry without its random operator-repair path."""

    if not entry.has_measured_fitness:
        raise CandidateExportError(
            "Hall-of-Fame entry lacks successful real-backtest evidence"
        )
    return _freeze_legacy_gene(
        source_kind="HALL_OF_FAME",
        source_id=entry.entry_id,
        source_fitness=entry.fitness,
        source_fitness_evidence=entry.fitness_evidence,
        source_fitness_panel_id=entry.fitness_panel_id,
        source_fitness_panel_role=entry.fitness_panel_role,
        source_gene=entry.strategy_gene_dict,
        config=config,
    )
