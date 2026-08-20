"""Fail-closed tests for the legacy warm-start boundary."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from genetic_algorithm.config.schema import DEFAULTS, validate_config
from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import ConditionGene, IndicatorGene, StrategyGene
from genetic_algorithm.engine.checkpoint_contract import seal_checkpoint
from genetic_algorithm.engine.warm_start import WarmStartError, WarmStartLoader


def _individual(identifier: int = 0, fitness: float = 0.5) -> Individual:
    indicator = IndicatorGene(
        type="RSI",
        parameters={"period": 14},
        instance_id="RSI_0",
    )
    gene = StrategyGene(
        generation=3,
        individual_id=identifier,
        indicators=[indicator],
        entry_conditions=[
            ConditionGene(
                indicator="RSI_0",
                operator="<",
                threshold=30.0,
            )
        ],
        exit_conditions=[
            ConditionGene(
                indicator="RSI_0",
                operator=">",
                threshold=70.0,
            )
        ],
    )
    individual = Individual(strategy_gene=gene)
    individual.fitness = fitness
    individual.raw_fitness = fitness
    individual.evaluated = True
    return individual


def _write_json(path, payload) -> str:
    path.write_text(json.dumps(payload, indent=2))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_population(individuals):
    return {
        "artifact_schema_version": "population-seed-v1",
        "genome_schema_version": "strategy-gene-v2",
        "population": {"individuals": individuals},
    }


def _native_hof(entries):
    return {
        "version": 3,
        "artifact_schema_version": "hall-of-fame-v3",
        "genome_schema_version": "strategy-gene-v2",
        "entries": entries,
    }


def _config(**warm_overrides):
    warm = {
        "enabled": True,
        "source_type": "population",
        "top_n": 2,
        "source_experiment": "parent-wave",
        "genome_schema_version": "strategy-gene-v2",
    }
    warm.update(warm_overrides)
    return {"warm_start": warm}


def test_disabled_warm_start_is_the_only_scratch_fallback():
    assert WarmStartLoader({"warm_start": {"enabled": False}}).load() == []


def test_enabled_warm_start_requires_explicit_hash_bound_source():
    with pytest.raises(WarmStartError, match="path must be explicit"):
        WarmStartLoader(_config()).load()


def test_missing_source_blocks_instead_of_discovery(tmp_path):
    config = _config(
        source_checkpoint=str(tmp_path / "missing.json"),
        source_checkpoint_sha256="1" * 64,
    )
    with pytest.raises(WarmStartError, match="does not exist"):
        WarmStartLoader(config).load()


def test_source_file_hash_mismatch_blocks(tmp_path):
    source = tmp_path / "population.json"
    _write_json(source, {"population": {"individuals": [_individual().to_dict()]}})
    config = _config(
        source_checkpoint=str(source),
        source_checkpoint_sha256="1" * 64,
    )
    with pytest.raises(WarmStartError, match="SHA-256 mismatch"):
        WarmStartLoader(config).load()


def test_valid_population_source_is_reset_and_provenance_marked(tmp_path):
    source = tmp_path / "population.json"
    digest = _write_json(
        source,
        _native_population(
            [
                _individual(identifier=1, fitness=0.2).to_dict(),
                _individual(identifier=2, fitness=0.8).to_dict(),
            ]
        ),
    )
    loaded = WarmStartLoader(
        _config(
            source_checkpoint=str(source),
            source_checkpoint_sha256=digest,
        )
    ).load()

    assert [item.strategy_gene.individual_id for item in loaded] == [9000, 9001]
    assert all(item.strategy_gene.generation == 0 for item in loaded)
    assert all(item.fitness is None and not item.evaluated for item in loaded)
    assert loaded[0].metrics["warm_start_source_hashes"] == {"checkpoint": digest}


def test_one_invalid_genome_blocks_the_complete_source(tmp_path):
    invalid = _individual(identifier=2).to_dict()
    invalid["strategy_gene"]["entry_conditions"][0]["indicator"] = "MISSING_0"
    source = tmp_path / "population.json"
    digest = _write_json(
        source,
        _native_population([_individual(identifier=1).to_dict(), invalid]),
    )
    loader = WarmStartLoader(
        _config(
            source_checkpoint=str(source),
            source_checkpoint_sha256=digest,
        )
    )
    with pytest.raises(WarmStartError, match="individual 1"):
        loader.load()


def test_v3_checkpoint_requires_its_internal_checksum_too(tmp_path):
    source = tmp_path / "checkpoint_gen4.json"
    checkpoint = seal_checkpoint(
        {
            "version": 3,
            "resume_eligible": False,
            "provenance": None,
            "generation": 4,
            "population": {"individuals": [_individual().to_dict()]},
        }
    )
    checkpoint["generation"] = 5
    digest = _write_json(source, checkpoint)
    loader = WarmStartLoader(
        _config(
            source_checkpoint=str(source),
            source_checkpoint_sha256=digest,
        )
    )
    with pytest.raises(WarmStartError, match="internal integrity"):
        loader.load()


def test_both_sources_are_atomic_when_hof_is_invalid(tmp_path):
    population = tmp_path / "population.json"
    population_hash = _write_json(
        population,
        _native_population([_individual().to_dict()]),
    )
    hof = tmp_path / "hof.json"
    hof_hash = _write_json(hof, _native_hof([]))
    loader = WarmStartLoader(
        _config(
            source_type="both",
            source_checkpoint=str(population),
            source_checkpoint_sha256=population_hash,
            source_hof=str(hof),
            source_hof_sha256=hof_hash,
        )
    )
    with pytest.raises(WarmStartError, match="no entries"):
        loader.load()


def test_enabled_loader_error_propagates_through_population_initialization():
    algorithm = object.__new__(GeneticAlgorithm)
    algorithm.population_size = 2
    algorithm._strict_initial_seeds = []
    algorithm.config = _config()
    algorithm.logger = WarmStartLoader({}).logger

    with pytest.raises(WarmStartError, match="path must be explicit"):
        algorithm.initialize_population()


def test_population_initialization_rejects_partial_warm_start_injection(monkeypatch):
    algorithm = object.__new__(GeneticAlgorithm)
    algorithm.population_size = 1
    algorithm._strict_initial_seeds = []
    algorithm.config = {"warm_start": {"enabled": True}}
    algorithm.logger = WarmStartLoader({}).logger
    monkeypatch.setattr(
        WarmStartLoader,
        "load",
        lambda self: [_individual(identifier=1), _individual(identifier=2)],
    )

    with pytest.raises(WarmStartError, match="only 1 population slots"):
        algorithm.initialize_population()


def test_schema_rejects_unbound_enabled_warm_start():
    config = copy.deepcopy(DEFAULTS)
    config["warm_start"] = {
        "enabled": True,
        "source_type": "population",
        "top_n": 1,
    }
    errors, _ = validate_config(config)
    assert "warm_start.source_checkpoint must be an explicit path" in errors
    assert "warm_start.source_checkpoint_sha256 must be a lowercase SHA-256 digest" in errors
    assert "warm_start.genome_schema_version must be strategy-gene-v2" in errors
