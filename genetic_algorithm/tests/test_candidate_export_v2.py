"""Regression tests for fail-closed legacy candidate export."""

from __future__ import annotations

import copy
import time

import pytest

from genetic_algorithm.core.strategy_gene import ConditionGene, IndicatorGene, StrategyGene
from genetic_algorithm.engine.hall_of_fame import HallOfFameEntry
from genetic_algorithm.genome.codegen import StrategyGenerator
from genetic_algorithm.genome.individual import Individual
from genetic_algorithm.orchestration.candidate_export_v2 import (
    CandidateExportError,
    freeze_hof_entry_v2,
    freeze_individual_v2,
)


def _config(*, min_entry: int = 1) -> dict:
    return {
        "indicators": {
            "available": ["RSI"],
            "min_entry_conditions": min_entry,
            "min_exit_conditions": 1,
        },
        "strategy_constraints": {"timeframes": ["1h"]},
        "multi_timeframe": {"enabled": False},
        "short_selling": {"enabled": False},
        "regime_aware": {"enabled": False},
    }


def _gene() -> StrategyGene:
    gene = StrategyGene(
        generation=7,
        individual_id=11,
        indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
        entry_conditions=[ConditionGene(indicator="RSI", operator="<", threshold=30)],
        exit_conditions=[ConditionGene(indicator="RSI", operator=">", threshold=70)],
        timeframe="1h",
        max_open_trades=3,
    )
    gene.assign_instance_ids()
    return gene


def _individual() -> Individual:
    individual = Individual(strategy_gene=_gene())
    individual.set_fitness(1.25, {"profit": 12.0})
    return individual


def test_individual_export_is_deterministic_and_does_not_mutate_source():
    individual = _individual()
    before = copy.deepcopy(individual.strategy_gene.to_dict())

    first = freeze_individual_v2(individual, _config())
    second = freeze_individual_v2(individual, _config())

    assert first == second
    assert individual.strategy_gene.to_dict() == before
    assert first.source_kind == "INDIVIDUAL"
    assert first.candidate.candidate_id == "Gen7_Ind11"
    assert first.candidate.strategy_name == "GAStrategy_Gen7_Ind11"
    assert first.candidate.timeframe == "1h"
    assert len(first.candidate.phenotype_hash) == 64


def test_hof_export_uses_exact_entry_without_random_repair():
    gene_dict = _gene().to_dict()
    entry = HallOfFameEntry(
        strategy_gene_dict=gene_dict,
        fitness=2.5,
        metrics={"profit": 10.0, "num_trades": 20},
        generation_found=7,
        run_timestamp=time.time(),
        entry_id="hof_test_001",
    )

    exported = freeze_hof_entry_v2(entry, _config())

    assert exported.source_kind == "HALL_OF_FAME"
    assert exported.source_fitness_evidence == "BACKTEST_VALID"
    assert exported.source_gene_hash == freeze_hof_entry_v2(entry, _config()).source_gene_hash
    assert exported.candidate.candidate_id == "hof_test_001"


def test_export_rejects_legacy_random_top_up():
    with pytest.raises(CandidateExportError, match="configured minimum"):
        freeze_individual_v2(_individual(), _config(min_entry=2))


def test_export_rejects_gene_migration_with_stale_fitness():
    individual = _individual()
    individual.strategy_gene.indicators[0].instance_id = "custom_rsi"
    individual.strategy_gene.entry_conditions[0].indicator = "custom_rsi"
    individual.strategy_gene.exit_conditions[0].indicator = "custom_rsi"

    with pytest.raises(CandidateExportError, match="requires migration"):
        freeze_individual_v2(individual, _config())


def test_export_rejects_invalid_operator_instead_of_hof_repair():
    individual = _individual()
    individual.strategy_gene.entry_conditions[0].operator = "definitely_invalid"

    with pytest.raises(CandidateExportError, match="automatic Hall-of-Fame repair"):
        freeze_individual_v2(individual, _config())


def test_export_rejects_missing_indicator_instead_of_codegen_repair():
    individual = _individual()
    individual.strategy_gene.entry_conditions[0].indicator = "EMA_0"

    with pytest.raises(CandidateExportError, match="missing or ambiguous indicator"):
        freeze_individual_v2(individual, _config())


def test_export_detects_codegen_mutation(monkeypatch: pytest.MonkeyPatch):
    def mutating_codegen(self, gene):
        gene.entry_conditions[0].threshold += 1
        return (
            f"class GAStrategy_Gen{gene.generation}_Ind{gene.individual_id}:\n"
            f"    timeframe = {gene.timeframe!r}\n"
        )

    monkeypatch.setattr(StrategyGenerator, "generate_strategy_code", mutating_codegen)

    with pytest.raises(CandidateExportError, match="mutates or repairs"):
        freeze_individual_v2(_individual(), _config())


def test_export_rejects_informative_timeframe_until_manifest_supports_it():
    individual = _individual()
    individual.strategy_gene.indicators[0].timeframe = "4h"
    individual.strategy_gene.informative_timeframes = ["4h"]
    individual.strategy_gene.assign_instance_ids()

    with pytest.raises(CandidateExportError, match="informative-timeframe data provenance"):
        freeze_individual_v2(individual, _config())
