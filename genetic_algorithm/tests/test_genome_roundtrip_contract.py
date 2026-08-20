"""Regression contract for lossless StrategyGene serialization."""

from __future__ import annotations

import copy
import json
import random

from genetic_algorithm.genome.codegen import StrategyGenerator
from genetic_algorithm.genome.gene import (
    ConditionGene,
    IndicatorGene,
    RegimeGene,
    StrategyGene,
)
from genetic_algorithm.genome.individual import Individual


def _condition(indicator: str, operator: str, threshold: float) -> ConditionGene:
    return ConditionGene(indicator=indicator, operator=operator, threshold=threshold)


def _roundtrip_gene() -> StrategyGene:
    gene = StrategyGene(
        generation=4,
        individual_id=17,
        indicators=[
            IndicatorGene(type="RSI", parameters={"period": 7}),
            IndicatorGene(type="RSI", parameters={"period": 21}),
            IndicatorGene(type="EMA", parameters={"period": 50}, timeframe="1h"),
            IndicatorGene(type="CDL_HAMMER", parameters={}),
        ],
        entry_conditions=[
            _condition("RSI_0", "<", 25),
            _condition("RSI_1", ">", 65),
            _condition("EMA_1h_0", ">", 0),
            _condition("CDL_HAMMER_0", ">", 0),
        ],
        exit_conditions=[_condition("RSI_1", "<", 50)],
        short_entry_conditions=[_condition("RSI_1", ">", 75)],
        short_exit_conditions=[_condition("CDL_HAMMER_0", ">", 0)],
        timeframe="5m",
        informative_timeframes=["1h"],
        can_short=True,
    )
    gene.assign_instance_ids()
    return gene


def _codegen_config() -> dict:
    return {
        "indicators": {
            "available": ["RSI", "EMA", "CDL_HAMMER"],
            "min_entry_conditions": 1,
            "min_exit_conditions": 1,
        },
        "strategy_constraints": {"timeframes": ["5m"]},
        "multi_timeframe": {"enabled": True},
        "short_selling": {"enabled": True},
        "regime_aware": {"enabled": False},
    }


def test_canonical_gene_roundtrip_is_lossless_for_all_condition_groups():
    original = _roundtrip_gene()
    payload = json.loads(json.dumps(original.to_dict()))

    restored = StrategyGene.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.strategy_fingerprint() == original.strategy_fingerprint()
    assert restored.get_missing_indicators() == []


def test_individual_roundtrip_preserves_gene_and_metadata():
    original = Individual(strategy_gene=_roundtrip_gene())
    original.set_fitness(1.5, {"profit": 12.0, "max_drawdown": 3.0})
    payload = json.loads(json.dumps(original.to_dict()))

    restored = Individual.from_dict(payload)

    assert restored.to_dict() == payload


def test_individual_legacy_migration_invalidates_stale_evaluation():
    original = Individual(strategy_gene=_roundtrip_gene())
    original.set_fitness(3.5, {"profit": 99.0})
    original.objectives = [3.5, -0.1]
    original.rank = 1
    original.crowding_distance = 8.0
    payload = original.to_dict()
    payload["strategy_gene"]["indicators"][0]["instance_id"] = "fast_rsi"
    payload["strategy_gene"]["entry_conditions"][0]["indicator"] = "fast_rsi"

    restored = Individual.from_dict(payload)

    assert restored.strategy_gene.indicators[0].instance_id == "RSI_0"
    assert restored.fitness is None
    assert restored.raw_fitness is None
    assert restored.objectives is None
    assert restored.metrics == {"fitness_evidence": "UNMEASURED"}
    assert restored.evaluated is False
    assert restored.rank == 0
    assert restored.crowding_distance == 0.0


def test_serialized_and_restored_values_do_not_share_mutable_state():
    original = Individual(strategy_gene=_roundtrip_gene())
    original.strategy_gene.regime_gene = RegimeGene(
        enabled=True,
        regime_timeframes=["1h", "4h"],
    )
    original.objectives = [1.0, 2.0]
    original.metrics = {"nested": {"profit": 12.0}}
    original.parent_ids = ["parent"]
    original.mutations = [{"type": "indicator", "values": [1]}]

    payload = original.to_dict()
    payload["strategy_gene"]["indicators"][0]["parameters"]["period"] = 999
    payload["strategy_gene"]["minimal_roi"]["0"] = 999
    payload["strategy_gene"]["regime_gene"]["regime_timeframes"].append("1d")
    payload["metrics"]["nested"]["profit"] = 999
    payload["mutations"][0]["values"].append(2)

    assert original.strategy_gene.indicators[0].parameters["period"] == 7
    assert original.strategy_gene.minimal_roi["0"] != 999
    assert original.strategy_gene.regime_gene.regime_timeframes == ["1h", "4h"]
    assert original.metrics["nested"]["profit"] == 12.0
    assert original.mutations[0]["values"] == [1]

    restored = Individual.from_dict(payload)
    restored.strategy_gene.indicators[0].parameters["period"] = 123
    restored.strategy_gene.regime_gene.regime_timeframes.append("1w")
    restored.metrics["nested"]["profit"] = 123
    restored.mutations[0]["values"].append(3)

    assert payload["strategy_gene"]["indicators"][0]["parameters"]["period"] == 999
    assert payload["strategy_gene"]["regime_gene"]["regime_timeframes"] == [
        "1h",
        "4h",
        "1d",
    ]
    assert payload["metrics"]["nested"]["profit"] == 999
    assert payload["mutations"][0]["values"] == [1, 2]


def test_noncanonical_ids_are_remapped_without_retargeting_conditions():
    payload = _roundtrip_gene().to_dict()
    payload["indicators"] = [
        {
            "type": "RSI",
            "parameters": {"period": 7},
            "weight": 1.0,
            "instance_id": "fast_rsi",
            "timeframe": None,
            "param_bounds": None,
        },
        {
            "type": "RSI",
            "parameters": {"period": 21},
            "weight": 1.0,
            "instance_id": "slow_rsi",
            "timeframe": None,
            "param_bounds": None,
        },
    ]
    payload["entry_conditions"] = [
        {"indicator": "fast_rsi", "operator": "<", "threshold": 25},
        {"indicator": "slow_rsi", "operator": ">", "threshold": 65},
    ]
    payload["exit_conditions"] = []
    payload["short_entry_conditions"] = []
    payload["short_exit_conditions"] = []
    payload["informative_timeframes"] = []
    payload["can_short"] = False

    restored = StrategyGene.from_dict(payload)

    assert [indicator.instance_id for indicator in restored.indicators] == ["RSI_0", "RSI_1"]
    assert [condition.indicator for condition in restored.entry_conditions] == ["RSI_0", "RSI_1"]
    assert restored.get_missing_indicators() == []


def test_exact_loader_rejects_every_legacy_repair():
    canonical = _roundtrip_gene().to_dict()
    assert StrategyGene.from_dict_exact(canonical).to_dict() == canonical

    noncanonical = copy.deepcopy(canonical)
    noncanonical["indicators"][0]["instance_id"] = "fast_rsi"
    noncanonical["entry_conditions"][0]["indicator"] = "fast_rsi"
    try:
        StrategyGene.from_dict_exact(noncanonical)
    except ValueError as exc:
        assert "requires migration" in str(exc)
    else:
        raise AssertionError("noncanonical instance IDs must be rejected")

    orphaned = copy.deepcopy(canonical)
    orphaned["entry_conditions"][0]["indicator"] = "MISSING_0"
    try:
        StrategyGene.from_dict_exact(orphaned)
    except ValueError as exc:
        assert "unresolved condition references" in str(exc)
    else:
        raise AssertionError("orphaned condition references must be rejected")


def test_legacy_cascaded_cdl_reference_is_repaired_without_adding_indicator():
    payload = _roundtrip_gene().to_dict()
    payload["indicators"] = [
        {
            "type": "CDL_HAMMER_0_0",
            "parameters": {},
            "weight": 1.0,
            "instance_id": "CDL_HAMMER_0_0_0",
            "timeframe": None,
            "param_bounds": None,
        }
    ]
    payload["entry_conditions"] = [
        {"indicator": "CDL_HAMMER_0_0_0", "operator": ">", "threshold": 0}
    ]
    payload["exit_conditions"] = []
    payload["short_entry_conditions"] = []
    payload["short_exit_conditions"] = []
    payload["informative_timeframes"] = []
    payload["can_short"] = False

    restored = StrategyGene.from_dict(payload)

    assert len(restored.indicators) == 1
    assert restored.indicators[0].type == "CDL_HAMMER"
    assert restored.indicators[0].instance_id == "CDL_HAMMER_0"
    assert restored.entry_conditions[0].indicator == "CDL_HAMMER_0"
    assert restored.get_missing_indicators() == []


def test_legacy_missing_informative_indicator_is_repaired_on_its_timeframe():
    gene = StrategyGene(
        generation=1,
        individual_id=2,
        indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
        entry_conditions=[_condition("RSI", "<", 30)],
        exit_conditions=[_condition("EMA_1h_7", ">", 0)],
        timeframe="5m",
    )
    gene.assign_instance_ids()

    gene.ensure_indicators_for_conditions(_codegen_config()["indicators"])
    gene.assign_instance_ids()

    assert [(item.type, item.timeframe) for item in gene.indicators] == [
        ("RSI", None),
        ("EMA", "1h"),
    ]
    assert gene.exit_conditions[0].indicator == "EMA_1h_0"
    assert gene.informative_timeframes == ["1h"]
    assert gene.get_missing_indicators() == []


def test_codegen_is_idempotent_across_serialization_roundtrips():
    generator = StrategyGenerator(_codegen_config())
    original = _roundtrip_gene()
    before = copy.deepcopy(original.to_dict())

    first_code = generator.generate_strategy_code(original)
    restored = StrategyGene.from_dict(json.loads(json.dumps(original.to_dict())))
    second_code = generator.generate_strategy_code(restored)

    assert first_code == second_code
    assert original.to_dict() == before
    assert restored.to_dict() == before
    assert len(restored.indicators) == len(before["indicators"])


def test_random_generator_outputs_are_canonical_and_lossless():
    config = _codegen_config()
    config["indicators"].update(
        {
            "min_per_strategy": 1,
            "max_per_strategy": 3,
            "min_entry_conditions": 4,
            "max_entry_conditions": 4,
        }
    )
    generator = StrategyGenerator(config)

    for seed in range(100):
        random.seed(seed)
        gene = generator.generate_random_strategy(generation=0, individual_id=seed)
        payload = gene.to_dict()
        assert StrategyGene.from_dict(payload).to_dict() == payload, f"seed={seed}"
        assert gene.get_missing_indicators() == [], f"seed={seed}"
