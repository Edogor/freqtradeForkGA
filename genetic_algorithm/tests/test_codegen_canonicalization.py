"""Regressions for code-generation repairs at the StrategyGene boundary."""

from __future__ import annotations

import random

from genetic_algorithm.engine.operators.mutation import _mutate_condition_threshold
from genetic_algorithm.genome.codegen import StrategyGenerator
from genetic_algorithm.genome.gene import ConditionGene, IndicatorGene, StrategyGene


def _config() -> dict:
    return {
        "indicators": {
            "available": ["OBV", "VWAP", "SUPERTREND"],
            "min_entry_conditions": 2,
            "min_exit_conditions": 1,
        },
        "strategy_constraints": {"timeframes": ["1h"]},
        "multi_timeframe": {"enabled": False},
        "short_selling": {"enabled": False},
        "regime_aware": {"enabled": False},
    }


def test_codegen_condition_top_up_keeps_canonical_instance_reference(monkeypatch):
    """Reproduce the week-run OBV payload that broke strict V2 export."""

    gene = StrategyGene(
        generation=11,
        individual_id=2,
        indicators=[
            IndicatorGene("OBV", {}, instance_id="OBV_0"),
            IndicatorGene("VWAP", {"period": 22}, instance_id="VWAP_0"),
        ],
        entry_conditions=[ConditionGene("VWAP_0", "cross_above", 0)],
        exit_conditions=[ConditionGene("VWAP_0", "cross_below", 0)],
        timeframe="1h",
    )
    monkeypatch.setattr(
        "genetic_algorithm.genome.codegen.random.choice",
        lambda items: items[0],
    )

    StrategyGenerator(_config()).generate_strategy_code(gene)

    assert [condition.indicator for condition in gene.entry_conditions] == [
        "VWAP_0",
        "OBV_0",
    ]
    payload = gene.to_dict()
    assert StrategyGene.from_dict_exact(payload).to_dict() == payload


def test_codegen_cache_hit_repairs_genome_before_fingerprinting(monkeypatch):
    """Cached code and the persisted genome must describe one phenotype."""

    first = StrategyGene(
        generation=4,
        individual_id=7,
        indicators=[
            IndicatorGene("OBV", {}, instance_id="OBV_0"),
            IndicatorGene("VWAP", {"period": 22}, instance_id="VWAP_0"),
        ],
        entry_conditions=[ConditionGene("VWAP_0", "cross_above", 0)],
        exit_conditions=[ConditionGene("VWAP_0", "cross_below", 0)],
        timeframe="1h",
    )
    second = StrategyGene.from_dict_exact(first.to_dict())
    monkeypatch.setattr(
        "genetic_algorithm.genome.codegen.random.choice",
        lambda items: items[0],
    )
    generator = StrategyGenerator(_config())

    first_code = generator.generate_strategy_code(first)
    second_code = generator.generate_strategy_code(second)

    assert generator._code_cache_hits == 1
    assert len(first.entry_conditions) == len(second.entry_conditions) == 2
    assert first.strategy_fingerprint() == second.strategy_fingerprint()
    assert first_code == second_code
    assert StrategyGene.from_dict_exact(second.to_dict()).to_dict() == second.to_dict()


def test_codegen_repair_is_deterministic_across_replay_rng_states():
    """Worker replay must compile the same repaired executable phenotype."""
    original = StrategyGene(
        generation=4,
        individual_id=7,
        indicators=[
            IndicatorGene("OBV", {}, instance_id="OBV_0"),
            IndicatorGene("VWAP", {"period": 22}, instance_id="VWAP_0"),
        ],
        entry_conditions=[ConditionGene("VWAP_0", "cross_above", 0)],
        exit_conditions=[ConditionGene("VWAP_0", "cross_below", 0)],
        timeframe="1h",
    )
    first = StrategyGene.from_dict_exact(original.to_dict())
    second = StrategyGene.from_dict_exact(original.to_dict())

    random.seed(11)
    StrategyGenerator(_config()).generate_strategy_code(first)
    random.seed(982451653)
    StrategyGenerator(_config()).generate_strategy_code(second)

    assert first.to_dict() == second.to_dict()
    assert first.strategy_fingerprint() == second.strategy_fingerprint()


def test_codegen_eliminates_duplicate_compiled_supertrend_terms():
    """Neutral SuperTrend thresholds must not duplicate an executable term."""

    gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(
                "SUPERTREND",
                {"period": 10, "multiplier": 3.9920561105623342},
                instance_id="SUPERTREND_0",
            )
        ],
        entry_conditions=[
            ConditionGene(
                "SUPERTREND_0",
                "cross_above",
                -0.03366318983445016,
                logic="OR",
            ),
            ConditionGene("SUPERTREND_0", "cross_above", 0, logic="AND"),
        ],
        exit_conditions=[ConditionGene("SUPERTREND_0", "cross_below", 0)],
        timeframe="1h",
    )

    code = StrategyGenerator(_config()).generate_strategy_code(gene)

    assert code.count("(dataframe['supertrend'] == True)") == 1


def test_supertrend_threshold_mutation_removes_neutral_drift():
    condition = ConditionGene(
        "SUPERTREND_0",
        "cross_above",
        -0.75,
        threshold_upper=4.0,
    )
    mutations_applied: list[str] = []

    _mutate_condition_threshold(
        condition,
        {},
        True,
        0,
        mutations_applied,
    )

    assert condition.threshold == 0
    assert condition.threshold_upper == 0.0
    assert mutations_applied == []
