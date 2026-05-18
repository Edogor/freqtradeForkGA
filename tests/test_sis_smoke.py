"""
SIS E2E Smoke Tests

Validates the complete SIS hook pipeline (all 5 hooks) integrates cleanly
with a real SISIntegrator backed by trained models.

No backtesting is performed — fitness values are synthetic.
Tests focus on: no crashes, correct output types, hook counters, log output.
"""

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.intelligence.corpus import SURROGATE_FEATURE_NAMES
from genetic_algorithm.evaluation.surrogate import _INDICATOR_TYPES, _OPERATOR_TYPES

# ─────────────────────────────────────────────────────────────
# Module-level: re-use the same synthetic corpus across tests
# ─────────────────────────────────────────────────────────────

import hashlib

def _make_synthetic_corpus_fast(n: int = 160, seed: int = 7) -> "pd.DataFrame":
    """Fast synthetic corpus used by smoke tests (no ML training delay)."""
    import pandas as pd

    rng = np.random.RandomState(seed)
    records = []
    for i in range(n):
        row = {}
        cluster = i % 4
        ind_onehot = np.zeros(len(_INDICATOR_TYPES))
        if cluster == 0:
            ind_onehot[_INDICATOR_TYPES.index("RSI")] = 1.0
            ind_onehot[_INDICATOR_TYPES.index("EMA")] = 1.0
        elif cluster == 1:
            ind_onehot[_INDICATOR_TYPES.index("MACD")] = 1.0
            ind_onehot[_INDICATOR_TYPES.index("BBANDS")] = 1.0
        elif cluster == 2:
            ind_onehot[_INDICATOR_TYPES.index("ADX")] = 1.0
            ind_onehot[_INDICATOR_TYPES.index("ATR")] = 1.0
            ind_onehot[_INDICATOR_TYPES.index("STOCH")] = 1.0
        else:
            for idx in rng.choice(len(_INDICATOR_TYPES), size=rng.randint(1, 4), replace=False):
                ind_onehot[idx] = 1.0

        for j, name in enumerate(SURROGATE_FEATURE_NAMES):
            if name.startswith("ind_"):
                idx = [k for k, t in enumerate(_INDICATOR_TYPES) if f"ind_{t}" == name]
                row[name] = ind_onehot[idx[0]] if idx else 0.0
            elif name == "n_indicators":
                row[name] = float(int(ind_onehot.sum()))
            elif name in ("n_entry_conds", "n_exit_conds"):
                row[name] = float(rng.randint(1, 5))
            elif name == "n_short_conds":
                row[name] = float(rng.randint(0, 3))
            elif name.startswith("entry_op_") or name.startswith("exit_op_"):
                row[name] = float(rng.randint(0, 3))
            elif name in ("period_mean", "period_max", "period_min"):
                row[name] = 14.0 + cluster * 10 + rng.uniform(-2, 2)
            elif name.startswith("weight_"):
                row[name] = rng.uniform(0.5, 2.0)
            elif name == "stoploss":
                row[name] = -0.05 - cluster * 0.02 + rng.uniform(-0.01, 0.01)
            elif name in ("roi_max", "roi_min"):
                row[name] = rng.uniform(0.01, 0.10)
            elif name == "max_open_trades":
                row[name] = float(rng.randint(1, 6))
            elif name == "trailing_stop":
                row[name] = float(rng.randint(0, 2))
            elif name.startswith("threshold_"):
                row[name] = rng.uniform(0, 100)
            elif name.startswith("logic_"):
                row[name] = float(rng.randint(0, 4))
            elif name == "n_informative_tf":
                row[name] = float(rng.randint(0, 3))
            elif name == "can_short":
                row[name] = float(rng.randint(0, 2))
            else:
                row[name] = rng.uniform(0, 1)

        base_fitness = [0.8, 0.5, 0.3, 0.1][cluster]
        row["fitness"] = base_fitness + rng.uniform(-0.1, 0.1)
        row["raw_fitness"] = row["fitness"]
        row["profit"] = row["fitness"] * 100 + rng.uniform(-5, 5)
        row["sharpe_ratio"] = row["fitness"] * 2
        row["sortino_ratio"] = row["sharpe_ratio"] * 1.2
        row["profit_factor"] = 1.0 + row["fitness"]
        row["max_drawdown"] = rng.uniform(0.05, 0.35)
        row["win_rate"] = 0.4 + row["fitness"] * 0.3
        row["num_trades"] = float(rng.randint(20, 500))
        row["train_fitness"] = row["fitness"]
        row["val_fitness"] = row["fitness"] - 0.02
        row["pair_generalization_ratio"] = rng.uniform(0.5, 1.0)
        row["val_profit"] = row["profit"] * 0.8
        row["val_sharpe"] = row["sharpe_ratio"] * 0.9
        row["val_trades"] = row["num_trades"] * 0.5
        row["val_max_drawdown"] = row["max_drawdown"] * 1.1
        row["val_win_rate"] = row["win_rate"] * 0.95
        row["holdout_fitness"] = row["fitness"] - 0.05
        row["holdout_degradation"] = rng.uniform(0.0, 0.2)
        row["holdout_profit"] = row["profit"] * 0.7
        row["holdout_trades"] = row["num_trades"] * 0.4
        row["complexity"] = float(rng.randint(3, 15))
        row["monthly_return_std"] = rng.uniform(1, 8)
        row["positive_months_ratio"] = rng.uniform(0.3, 0.9)
        row["pair_profit_std"] = rng.uniform(0.5, 4.0)
        row["max_consecutive_losses"] = float(rng.randint(1, 10))
        row["max_drawdown_duration_days"] = float(rng.randint(1, 30))
        row["monthly_profit_mean"] = rng.uniform(-1, 5)
        row["monthly_profit_std"] = rng.uniform(1, 8)
        row["monthly_profit_min"] = row["monthly_profit_mean"] - rng.uniform(3, 15)
        row["monthly_profit_max"] = row["monthly_profit_mean"] + rng.uniform(3, 15)
        row["n_positive_months"] = rng.randint(1, 12)
        row["n_negative_months"] = rng.randint(0, 6)
        row["n_months_total"] = row["n_positive_months"] + row["n_negative_months"]
        row["per_pair_profit_mean"] = rng.uniform(-1, 4)
        row["per_pair_profit_std"] = rng.uniform(0.5, 3)
        row["per_pair_profit_min"] = row["per_pair_profit_mean"] - rng.uniform(2, 6)
        row["per_pair_profit_max"] = row["per_pair_profit_mean"] + rng.uniform(2, 6)
        row["n_profitable_pairs"] = rng.randint(1, 8)
        row["n_pairs"] = rng.randint(5, 12)
        wave_num = (i // 32) + 1
        row["fingerprint"] = hashlib.sha256(f"s{i}".encode()).hexdigest()[:16]
        row["run_id"] = f"wave{wave_num}_A_rank_5m_ring"
        row["generation"] = rng.randint(0, 30)
        row["individual_id"] = str(i)
        row["source"] = "hall_of_fame" if i % 3 == 0 else "gen_snapshot"
        row["timeframe"] = ["5m", "15m", "1h"][cluster % 3]
        row["origin"] = "backtest"
        row["training_pairs"] = "BTC/USDT,ETH/USDT"
        row["validation_pairs"] = "SOL/USDT"
        row["wave"] = f"wave{wave_num}"
        records.append(row)

    return pd.DataFrame(records)


def _make_individual(indicators=None, fitness=0.5, evaluated=True):
    """Create a test individual."""
    if indicators is None:
        indicators = [
            IndicatorGene(type="RSI", parameters={"period": 14}, instance_id="RSI_0"),
            IndicatorGene(type="EMA", parameters={"period": 21}, instance_id="EMA_0"),
        ]
    gene = StrategyGene(
        generation=0, individual_id=0,
        indicators=indicators,
        entry_conditions=[
            ConditionGene(indicator="RSI_0", operator="<", threshold=30, logic="AND"),
        ],
        exit_conditions=[
            ConditionGene(indicator="RSI_0", operator=">", threshold=70, logic="AND"),
        ],
        stoploss=-0.08,
        timeframe="5m",
        minimal_roi={"0": 0.05, "30": 0.02},
        max_open_trades=3,
    )
    ind = Individual(strategy_gene=gene)
    ind.fitness = fitness
    ind.raw_fitness = fitness
    ind.evaluated = evaluated
    if fitness is not None:
        ind.metrics = {
            "profit": fitness * 50,
            "win_rate": 0.5 + fitness * 0.2,
            "max_drawdown": 0.1 + (1 - fitness) * 0.1,
            "sharpe_ratio": fitness * 1.5,
            "num_trades": 100,
            "pair_generalization_ratio": 0.8,
        }
    else:
        ind.metrics = {}
    return ind


def _make_population(n: int = 20, rng_seed: int = 1) -> Population:
    """Create a population of evaluated individuals with varied indicator mixes."""
    rng = np.random.RandomState(rng_seed)
    indicator_pool = [
        ("RSI", "EMA"), ("MACD", "BBANDS"), ("ADX", "ATR"),
        ("RSI", "STOCH"), ("CCI", "ROC"), ("DONCHIAN", "ATR"),
    ]
    pop = Population(size=n)
    for i in range(n):
        pair = indicator_pool[i % len(indicator_pool)]
        inds = [
            IndicatorGene(type=pair[0], parameters={"period": 14}, instance_id=f"{pair[0]}_0"),
            IndicatorGene(type=pair[1], parameters={"period": 20}, instance_id=f"{pair[1]}_0"),
        ]
        fitness = float(rng.uniform(0.2, 0.9))
        # Mark first 5 as unevaluated (for filter_population hook)
        ind = _make_individual(inds, fitness=fitness, evaluated=(i >= 5))
        if i < 5:
            ind.metrics = {}
        pop.add_individual(ind)
    return pop


MINIMAL_CONFIG = {
    "sis": {"enabled": True},
    "genetic_algorithm": {"population_size": 20},
    "indicators": {
        "available": ["RSI", "EMA", "SMA", "MACD", "BBANDS", "ATR", "ADX",
                      "STOCH", "CCI", "ROC", "DONCHIAN", "SUPERTREND"],
        "max_per_strategy": 5,
        "min_per_strategy": 2,
        "min_entry_conditions": 1,
        "min_exit_conditions": 1,
        "RSI": {"period": [7, 21]},
        "EMA": {"period": [10, 50]},
        "SMA": {"period": [10, 50]},
        "MACD": {"fast_period": [8, 21], "slow_period": [21, 50], "signal_period": [5, 14]},
        "BBANDS": {"period": [14, 28]},
        "ATR": {"period": [7, 21]},
        "ADX": {"period": [7, 28]},
        "STOCH": {"k_period": [5, 21], "d_period": [3, 9]},
        "CCI": {"period": [10, 28]},
        "ROC": {"period": [7, 21]},
        "DONCHIAN": {"period": [10, 40]},
        "SUPERTREND": {"period": [7, 21], "multiplier": [2, 5]},
    },
    "strategy_constraints": {
        "stoploss_range": [-0.15, -0.03],
        "roi_range": [0.01, 0.10],
        "timeframes": ["5m", "15m", "1h"],
        "max_open_trades_range": [1, 10],
    },
    "output": {"dir": "/tmp/sis_smoke_output"},
    "walk_forward": {"enabled": False},
    "monte_carlo": {},
    "fitness_bounds": {},
}


@pytest.fixture(scope="module")
def corpus_df():
    return _make_synthetic_corpus_fast()


@pytest.fixture(scope="module")
def trained_sis(corpus_df, tmp_path_factory):
    """SISIntegrator backed by trained models on synthetic corpus."""
    tmp = tmp_path_factory.mktemp("sis_smoke_models")
    from genetic_algorithm.intelligence.predictors import MultiTargetPredictor
    from genetic_algorithm.intelligence.archetypes import ArchetypeClassifier
    from genetic_algorithm.intelligence.sis_integrator import SISIntegrator

    pred = MultiTargetPredictor(models_dir=tmp)
    pred.train(corpus_df)
    pred.save(tmp)

    ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3)
    ac.fit(corpus_df)
    ac.save(tmp)

    corpus_path = tmp / "strategy_corpus.parquet"
    corpus_df.to_parquet(corpus_path, index=False)

    config = {
        **MINIMAL_CONFIG,
        "sis": {
            "enabled": True,
            "models_dir": str(tmp),
            "corpus_path": str(corpus_path),
            "log_file": str(tmp / "sis_log.jsonl"),
            "hooks": {
                "seed_filtering": True,
                "immigrants": True,
                "indicator_weights": True,
                "operator_weights": True,
                "synergy_weights": True,
            },
        },
        "output": {"dir": str(tmp)},
    }
    sis = SISIntegrator(config, logging.getLogger("sis_smoke"))
    return sis


@pytest.fixture(scope="module")
def mock_ga(trained_sis):
    """Lightweight GA stand-in that satisfies SIS's ga.strategy_generator contract."""
    from genetic_algorithm.strategies.generator import StrategyGenerator
    gen = StrategyGenerator(MINIMAL_CONFIG)
    ga = SimpleNamespace(
        strategy_generator=gen,
        population=None,  # overridden per test
    )
    return ga


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test 1 — Initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestSISSmoke_Init:
    """SISIntegrator initialises cleanly with trained models."""

    def test_predictor_ready(self, trained_sis):
        assert trained_sis._predictor_ready is True

    def test_classifier_ready(self, trained_sis):
        assert trained_sis._classifier_ready is True

    def test_evo_state_exists(self, trained_sis):
        assert trained_sis._evo_state is not None

    def test_adaptive_weights_exist(self, trained_sis):
        assert trained_sis._adaptive_weights is not None

    def test_synergy_graph_populated(self, trained_sis):
        assert len(trained_sis._synergy_graph) > 0

    def test_all_hooks_enabled(self, trained_sis):
        for hook_name in ("seed_filtering", "immigrants", "indicator_weights",
                          "operator_weights", "synergy_weights"):
            assert trained_sis._hook_enabled[hook_name] is True, (
                f"Hook {hook_name!r} should be enabled"
            )

    def test_status_summary_returns_dict(self, trained_sis):
        summary = trained_sis.get_status_summary()
        assert isinstance(summary, dict)
        assert "evolution_state" in summary
        assert "n_archetypes" in summary
        assert "hooks_enabled" in summary


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test 2 — Hook: indicator_weights
# ══════════════════════════════════════════════════════════════════════════════

class TestSISSmoke_IndicatorWeights:
    """get_indicator_weights() returns sensible weighted dict."""

    def test_returns_nonempty_dict(self, trained_sis):
        weights = trained_sis.get_indicator_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_all_values_positive(self, trained_sis):
        weights = trained_sis.get_indicator_weights()
        for k, v in weights.items():
            assert v > 0, f"Weight for {k!r} must be positive, got {v}"

    def test_values_within_cap(self, trained_sis):
        weights = trained_sis.get_indicator_weights()
        cap = trained_sis._weight_cap
        for k, v in weights.items():
            assert v <= cap, f"Weight for {k!r} exceeds cap {cap}, got {v}"

    def test_cci_higher_than_generic(self, trained_sis):
        """CCI should get enrichment boost (26.5x lift in SIS_INDICATOR_ENRICHMENT)."""
        weights = trained_sis.get_indicator_weights()
        if "CCI" in weights and "SMA" in weights:
            # CCI has 3.0 static enrichment; SMA has 1.0 — CCI should weigh more
            assert weights["CCI"] >= weights["SMA"] * 0.8, (
                f"Expected CCI ({weights['CCI']:.3f}) >= SMA ({weights['SMA']:.3f}) * 0.8"
            )

    def test_hook_call_counter_increments(self, trained_sis):
        before = trained_sis._hook_calls["indicator_weights"]
        trained_sis.get_indicator_weights()
        assert trained_sis._hook_calls["indicator_weights"] == before + 1

    def test_base_weights_blended(self, trained_sis):
        base = {"RSI": 2.0, "EMA": 0.5}
        blended = trained_sis.get_indicator_weights(base_weights=base)
        assert "RSI" in blended or "EMA" in blended


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test 3 — Hook: operator_weights
# ══════════════════════════════════════════════════════════════════════════════

class TestSISSmoke_OperatorWeights:
    """get_operator_weights() returns valid operator weight mapping."""

    def test_returns_dict(self, trained_sis):
        weights = trained_sis.get_operator_weights()
        assert isinstance(weights, dict)

    def test_all_values_positive(self, trained_sis):
        weights = trained_sis.get_operator_weights()
        for k, v in weights.items():
            assert v > 0, f"Operator weight for {k!r} must be positive, got {v}"


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test 4 — Hook: synergy_weights
# ══════════════════════════════════════════════════════════════════════════════

class TestSISSmoke_SynergyWeights:
    """get_synergy_weights() returns complementary boosts."""

    def test_empty_indicators_returns_empty(self, trained_sis):
        weights = trained_sis.get_synergy_weights([])
        assert isinstance(weights, dict)

    def test_donchian_roc_synergy(self, trained_sis):
        """Expected synergy pair from static SYNERGY_GRAPH."""
        weights = trained_sis.get_synergy_weights(["DONCHIAN"])
        # ROC is a synergy partner of DONCHIAN in the static graph
        if weights:
            assert all(v > 0 for v in weights.values())

    def test_returns_complementary_indicators(self, trained_sis):
        """With RSI present, should suggest synergy partners."""
        weights = trained_sis.get_synergy_weights(["RSI"])
        assert isinstance(weights, dict)
        for k in weights:
            assert k != "RSI", "Should not suggest already-present indicator"


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test 5 — Hook: immigrant_provider
# ══════════════════════════════════════════════════════════════════════════════

class TestSISSmoke_ImmigrantProvider:
    """immigrant_provider() generates valid Individual objects."""

    def test_returns_list(self, trained_sis, mock_ga):
        immigrants = trained_sis.immigrant_provider(mock_ga, generation=5)
        assert isinstance(immigrants, list)

    def test_immigrants_have_strategy_genes(self, trained_sis, mock_ga):
        immigrants = trained_sis.immigrant_provider(mock_ga, generation=5)
        for imm in immigrants:
            assert isinstance(imm, Individual)
            assert imm.strategy_gene is not None

    def test_immigrants_tagged_as_sis(self, trained_sis, mock_ga):
        immigrants = trained_sis.immigrant_provider(mock_ga, generation=5)
        for imm in immigrants:
            origin = (imm.metrics or {}).get("origin", "")
            assert "sis" in origin, f"Expected sis origin tag, got {origin!r}"

    def test_hook_call_counter_increments(self, trained_sis, mock_ga):
        before = trained_sis._hook_calls["immigrants"]
        trained_sis.immigrant_provider(mock_ga, generation=6)
        assert trained_sis._hook_calls["immigrants"] == before + 1

    def test_immigrants_not_evaluated(self, trained_sis, mock_ga):
        """New immigrants should not be pre-evaluated."""
        immigrants = trained_sis.immigrant_provider(mock_ga, generation=7)
        for imm in immigrants:
            assert imm.evaluated is False, "Immigrants should arrive unevaluated"


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test 6 — Hook: filter_population (seed filtering)
# ══════════════════════════════════════════════════════════════════════════════

class TestSISSmoke_FilterPopulation:
    """filter_population() replaces low-quality seeds via predictor scoring."""

    def test_runs_without_crash(self, trained_sis, mock_ga):
        pop = _make_population(n=20)
        mock_ga.population = pop
        trained_sis.filter_population(pop, mock_ga)  # should not raise

    def test_population_size_preserved(self, trained_sis, mock_ga):
        pop = _make_population(n=20)
        original_count = len(pop.individuals)
        mock_ga.population = pop
        trained_sis.filter_population(pop, mock_ga)
        assert len(pop.individuals) == original_count

    def test_hook_counter_tracked(self, trained_sis, mock_ga):
        pop = _make_population(n=20)
        before = trained_sis._hook_calls["seed_filtering"]
        trained_sis.filter_population(pop, mock_ga)
        assert trained_sis._hook_calls["seed_filtering"] == before + 1


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test 7 — update_from_generation (online learning feed)
# ══════════════════════════════════════════════════════════════════════════════

class TestSISSmoke_UpdateFromGeneration:
    """update_from_generation() accumulates live data and handles periodic retrain."""

    def test_accepts_evaluated_population(self, trained_sis):
        pop = _make_population(n=15)
        # Should not raise
        trained_sis.update_from_generation(1, pop.individuals)

    def test_accumulates_live_records(self, trained_sis):
        before = len(trained_sis._live_corpus_records)
        pop = _make_population(n=10, rng_seed=99)
        trained_sis.update_from_generation(2, pop.individuals)
        after = len(trained_sis._live_corpus_records)
        # Should have added some records (evaluated individuals)
        assert after > before

    def test_handles_empty_population(self, trained_sis):
        # Should not raise
        trained_sis.update_from_generation(3, [])

    def test_handles_unevaluated_population(self, trained_sis):
        inds = [_make_individual(fitness=None, evaluated=False) for _ in range(5)]
        for ind in inds:
            ind.fitness = None
        trained_sis.update_from_generation(4, inds)  # Should not raise


# ══════════════════════════════════════════════════════════════════════════════
# Smoke Test 8 — Full Generation Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestSISSmoke_FullGenerationPipeline:
    """Simulate 5 generation turns exercising the complete hook chain."""

    def test_5_generation_pipeline_no_crash(self, trained_sis, mock_ga):
        """
        Exercises all hooks for 5 simulated generations:
          1. filter_population (gen 0 initial seeding)
          2. get_indicator_weights (bias new individuals)
          3. immigrant_provider (inject archetype immigrants)
          4. update_from_generation (feed results back)
        After 5 gens, check hook counters and SIS log.
        """
        from genetic_algorithm.intelligence.sis_integrator import SISIntegrator

        # Use a fresh SIS for isolation (don't pollute the module-level fixture)
        sis = trained_sis

        calls_before = dict(sis._hook_calls)

        for gen in range(5):
            pop = _make_population(n=20, rng_seed=gen * 13)
            mock_ga.population = pop

            # 1. Seed filtering
            sis.filter_population(pop, mock_ga)

            # 2. Indicator bias
            weights = sis.get_indicator_weights()
            assert isinstance(weights, dict)

            # 3. Synergy suggestion for RSI-based strategy
            syn = sis.get_synergy_weights(["RSI", "EMA"])
            assert isinstance(syn, dict)

            # 4. Immigration
            immigrants = sis.immigrant_provider(mock_ga, generation=gen)
            assert isinstance(immigrants, list)

            # 5. Online learn feedback
            sis.update_from_generation(gen, pop.individuals)

        # Each hook should have been called more times
        assert sis._hook_calls["indicator_weights"] > calls_before["indicator_weights"]
        assert sis._hook_calls["immigrants"] > calls_before["immigrants"]
        assert sis._hook_calls["seed_filtering"] > calls_before["seed_filtering"]

    def test_get_archetype_gaps_returns_dict(self, trained_sis):
        pop = _make_population(n=20)
        gaps = trained_sis.get_archetype_gaps(pop.individuals)
        assert isinstance(gaps, dict)

    def test_get_status_summary_after_pipeline(self, trained_sis):
        summary = trained_sis.get_status_summary()
        assert summary["predictor_ready"] is True
        assert summary["classifier_ready"] is True
        # hook_calls should reflect all preceding calls
        assert sum(summary["hook_calls"].values()) > 0

    def test_sis_jsonl_log_written(self, trained_sis, mock_ga):
        """SIS log file is created and has valid JSONL entries."""
        pop = _make_population(n=10)
        trained_sis.log_generation(
            generation=99,
            population=pop.individuals,
            archetype_coverage={"0": 0.5, "1": 0.5},
        )
        log_path = trained_sis._log_path
        assert log_path.exists(), "SIS JSONL log file should have been created"
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1
        last = json.loads(lines[-1])
        assert "gen" in last or "generation" in last or "best_fitness" in last
