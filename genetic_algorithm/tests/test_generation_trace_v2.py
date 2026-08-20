"""Generation trace integrity and diagnostic coverage tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import PopulationStats
from genetic_algorithm.core.strategy_gene import ConditionGene, IndicatorGene, StrategyGene
from genetic_algorithm.orchestration.generation_trace_v2 import GenerationTraceWriterV2


NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
PANEL_ID = "panel_v1_" + "a" * 24


def _individual(individual_id: int, profit: float) -> Individual:
    gene = StrategyGene(
        generation=0,
        individual_id=individual_id,
        indicators=[IndicatorGene(type="RSI", parameters={"period": 14 + individual_id})],
        entry_conditions=[ConditionGene(indicator="RSI", operator="<", threshold=30)],
        exit_conditions=[ConditionGene(indicator="RSI", operator=">", threshold=70)],
        timeframe="1h",
        stoploss=-0.08,
        minimal_roi={"0": 0.03},
        max_open_trades=3,
    )
    individual = Individual(strategy_gene=gene)
    individual.set_fitness(
        0.5 + profit,
        {
            "profit": profit,
            "num_trades": 20,
            "max_drawdown": 0.1,
            "per_pair_profit": {"BTC/USDT": profit},
            "monthly_profits": [profit, profit / 2],
            "fitness_panel_id": PANEL_ID,
            "fitness_panel_role": "OPTIMIZATION",
        },
    )
    individual.assign_fitness_panel(PANEL_ID, "OPTIMIZATION")
    return individual


def _stats() -> PopulationStats:
    return PopulationStats(
        generation=0,
        size=2,
        best_fitness=0.7,
        median_fitness=0.6,
        avg_fitness=0.65,
        best_raw_fitness=0.7,
        avg_raw_fitness=0.65,
        diversity_score=0.05,
        genetic_diversity=0.4,
    )


def test_trace_is_hash_chained_and_idempotent(tmp_path):
    writer = GenerationTraceWriterV2(tmp_path / "trace.jsonl")
    population = [_individual(1, 0.1), _individual(2, 0.2)]

    first = writer.append(
        generation=0,
        island_name="momentum",
        panel_id=PANEL_ID,
        individuals=population,
        stats=_stats(),
        created_at=NOW,
    )
    repeated = writer.append(
        generation=0,
        island_name="momentum",
        panel_id=PANEL_ID,
        individuals=population,
        stats=_stats(),
        created_at=NOW,
    )
    second = writer.append(
        generation=0,
        island_name="trend",
        panel_id=PANEL_ID,
        individuals=population,
        stats=_stats(),
        created_at=NOW,
    )

    assert repeated == first
    assert second.previous_record_hash == first.record_hash
    assert len(writer.read_verified()) == 2
    assert first.unique_genome_count == 2
    assert first.unique_behavior_count == 2
    assert first.exact_behavior_duplicate_fraction == 0.0


def test_trace_rejects_tampering(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = GenerationTraceWriterV2(path)
    writer.append(
        generation=0,
        island_name="momentum",
        panel_id=PANEL_ID,
        individuals=[_individual(1, 0.1)],
        stats=_stats(),
        created_at=NOW,
    )
    payload = json.loads(path.read_text())
    payload["population_size"] = 99
    path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="hash differs"):
        writer.read_verified()
