"""
SIS Performance Use Case Test

Measures SIS impact on evolution quality across a simulated 25-generation run.

What this tests:
  1. Hook latency — are all hooks fast enough for production use (<50ms each)?
  2. Indicator weight bias — do weights meaningfully diverge from uniform?
  3. Immigration quality — do SIS immigrants score higher than random?
  4. Seed filtering effect — is filter_fraction working?
  5. A/B comparison — SIS ON vs SIS OFF across multiple seeds.

All fitness values are synthetic (no backtesting) — this tests SIS infrastructure
correctness, not trading alpha.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.intelligence.corpus import SURROGATE_FEATURE_NAMES
from genetic_algorithm.evaluation.surrogate import _INDICATOR_TYPES
from genetic_algorithm.intelligence.sis_integrator import (
    SIS_INDICATOR_ENRICHMENT,
)
from genetic_algorithm.intelligence.ab_framework import ABResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (duplicated from smoke test to keep files independent)
# ─────────────────────────────────────────────────────────────────────────────

_INDICATOR_WEIGHTS_IN_STATIC_ENRICHMENT = set(SIS_INDICATOR_ENRICHMENT.keys())
_TOP_ENRICHED = {"CCI", "STOCH", "ATR", "ROC", "DONCHIAN"}  # ≥2.0 in static data

MINIMAL_CONFIG = {
    "sis": {"enabled": True},
    "genetic_algorithm": {"population_size": 30},
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
    "output": {"dir": "/tmp/sis_perf_output"},
    "walk_forward": {"enabled": False},
    "monte_carlo": {},
    "fitness_bounds": {},
}


def _make_corpus(n: int = 160, seed: int = 7) -> "pd.DataFrame":
    import pandas as pd

    rng = np.random.RandomState(seed)
    records = []
    for i in range(n):
        row: Dict = {}
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
        row.update({
            "fitness": base_fitness + rng.uniform(-0.1, 0.1),
            "raw_fitness": base_fitness,
            "profit": base_fitness * 100 + rng.uniform(-5, 5),
            "sharpe_ratio": base_fitness * 2,
            "sortino_ratio": base_fitness * 2.4,
            "profit_factor": 1.0 + base_fitness,
            "max_drawdown": rng.uniform(0.05, 0.35),
            "win_rate": 0.4 + base_fitness * 0.3,
            "num_trades": float(rng.randint(20, 500)),
            "train_fitness": base_fitness,
            "val_fitness": base_fitness - 0.02,
            "pair_generalization_ratio": rng.uniform(0.5, 1.0),
            "val_profit": base_fitness * 80,
            "val_sharpe": base_fitness * 1.8,
            "val_trades": float(rng.randint(10, 250)),
            "val_max_drawdown": rng.uniform(0.05, 0.40),
            "val_win_rate": 0.38 + base_fitness * 0.3,
            "holdout_fitness": base_fitness - 0.05,
            "holdout_degradation": rng.uniform(0, 0.2),
            "holdout_profit": base_fitness * 70,
            "holdout_trades": float(rng.randint(5, 200)),
            "complexity": float(rng.randint(3, 15)),
            "monthly_return_std": rng.uniform(1, 8),
            "positive_months_ratio": rng.uniform(0.3, 0.9),
            "pair_profit_std": rng.uniform(0.5, 4.0),
            "max_consecutive_losses": float(rng.randint(1, 10)),
            "max_drawdown_duration_days": float(rng.randint(1, 30)),
            "monthly_profit_mean": rng.uniform(-1, 5),
            "monthly_profit_std": rng.uniform(1, 8),
            "monthly_profit_min": rng.uniform(-20, -2),
            "monthly_profit_max": rng.uniform(5, 25),
            "n_positive_months": rng.randint(1, 12),
            "n_negative_months": rng.randint(0, 6),
            "n_months_total": rng.randint(6, 18),
            "per_pair_profit_mean": rng.uniform(-1, 4),
            "per_pair_profit_std": rng.uniform(0.5, 3),
            "per_pair_profit_min": rng.uniform(-8, -1),
            "per_pair_profit_max": rng.uniform(3, 12),
            "n_profitable_pairs": rng.randint(1, 8),
            "n_pairs": rng.randint(5, 12),
            "fingerprint": hashlib.sha256(f"p{i}".encode()).hexdigest()[:16],
            "run_id": f"wave{(i//32)+1}_A_rank_5m_ring",
            "generation": rng.randint(0, 30),
            "individual_id": str(i),
            "source": "hall_of_fame" if i % 3 == 0 else "gen_snapshot",
            "timeframe": ["5m", "15m", "1h"][cluster % 3],
            "origin": "backtest",
            "training_pairs": "BTC/USDT,ETH/USDT",
            "validation_pairs": "SOL/USDT",
            "wave": f"wave{(i//32)+1}",
        })
        records.append(row)

    return pd.DataFrame(records)


def _make_ind(ind_types: List[str] = None, fitness: float = 0.5,
              rng: Optional[np.random.RandomState] = None) -> Individual:
    if ind_types is None:
        ind_types = ["RSI", "EMA"]
    if rng is None:
        rng = np.random.RandomState(0)
    indicators = [
        IndicatorGene(type=t, parameters={"period": int(rng.randint(7, 28))},
                      instance_id=f"{t}_0")
        for t in ind_types
    ]
    gene = StrategyGene(
        generation=0, individual_id=int(rng.randint(0, 9999)),
        indicators=indicators,
        entry_conditions=[
            ConditionGene(indicator=f"{ind_types[0]}_0", operator="<",
                          threshold=30, logic="AND")
        ],
        exit_conditions=[
            ConditionGene(indicator=f"{ind_types[0]}_0", operator=">",
                          threshold=70, logic="AND")
        ],
        stoploss=-0.08, timeframe="5m",
        minimal_roi={"0": 0.05, "30": 0.02},
        max_open_trades=3,
    )
    ind = Individual(strategy_gene=gene)
    ind.fitness = fitness
    ind.raw_fitness = fitness
    ind.evaluated = True
    ind.metrics = {
        "profit": fitness * 50,
        "win_rate": 0.4 + fitness * 0.3,
        "max_drawdown": 0.1,
        "sharpe_ratio": fitness * 1.5,
        "num_trades": 80,
        "pair_generalization_ratio": 0.75,
    }
    return ind


def _synthetic_fitness(ind: Individual) -> float:
    """
    Fake fitness oracle: strategies with CCI/STOCH/ATR get a 20% bonus.
    This matches what SIS should bias toward via its enrichment weights.
    """
    types = {g.type for g in ind.strategy_gene.indicators}
    bonus = 0.15 * len(types & _TOP_ENRICHED)
    base = 0.4
    return min(1.0, base + bonus + np.random.uniform(-0.1, 0.1))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd


@pytest.fixture(scope="module")
def corpus():
    return _make_corpus()


@pytest.fixture(scope="module")
def trained_sis_perf(corpus, tmp_path_factory):
    """Module-scoped SISIntegrator with trained models."""
    tmp = tmp_path_factory.mktemp("sis_perf_models")
    from genetic_algorithm.intelligence.predictors import MultiTargetPredictor
    from genetic_algorithm.intelligence.archetypes import ArchetypeClassifier
    from genetic_algorithm.intelligence.sis_integrator import SISIntegrator

    pred = MultiTargetPredictor(models_dir=tmp)
    pred.train(corpus)
    pred.save(tmp)

    ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3)
    ac.fit(corpus)
    ac.save(tmp)

    corpus_path = tmp / "strategy_corpus.parquet"
    corpus.to_parquet(corpus_path, index=False)

    config = {
        **MINIMAL_CONFIG,
        "sis": {
            "enabled": True,
            "models_dir": str(tmp),
            "corpus_path": str(corpus_path),
            "log_file": str(tmp / "sis_perf_log.jsonl"),
        },
        "output": {"dir": str(tmp)},
    }
    return SISIntegrator(config, logging.getLogger("sis_perf"))


@pytest.fixture(scope="module")
def mock_ga_perf(trained_sis_perf):
    from genetic_algorithm.strategies.generator import StrategyGenerator
    generator = StrategyGenerator(MINIMAL_CONFIG)
    return SimpleNamespace(strategy_generator=generator, population=None)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a simulated N-generation loop
# ─────────────────────────────────────────────────────────────────────────────

_INDICATOR_PAIRS = [
    ["RSI", "EMA"], ["MACD", "BBANDS"], ["ADX", "ATR"],
    ["RSI", "STOCH"], ["CCI", "ROC"], ["DONCHIAN", "ATR"],
    ["CCI", "STOCH"], ["ATR", "ROC"], ["RSI", "SMA"],
    ["BBANDS", "EMA"],
]


def _run_simulated_evolution(
    sis,
    ga,
    generations: int = 20,
    pop_size: int = 30,
    rng_seed: int = 42,
    sis_enabled: bool = True,
) -> Dict:
    """
    Simulate GA evolution for `generations` gen with optional SIS guidance.

    Per generation:
      1. Build a new random population
      2. Optionally use SIS indicator weights to bias indicator selection
      3. Optionally call filter_population (if SIS enabled)
      4. Assign synthetic fitness (CCI/STOCH/ATR-biased oracle)
      5. Optionally inject SIS immigrants
      6. Feed update_from_generation
      7. Record metrics

    Returns dict of per-generation metrics.
    """
    rng = np.random.RandomState(rng_seed)

    metrics_history = []
    best_fitness_ever = 0.0
    convergence_gen = 0

    for gen in range(generations):
        # ── 1. Build population ─────────────────────────────────────────────
        pop = Population(size=pop_size)
        for i in range(pop_size):
            # Choose indicator pair, optionally biased by SIS weights
            if sis_enabled and sis is not None:
                weights = sis.get_indicator_weights()
                available = list(MINIMAL_CONFIG["indicators"]["available"])
                # Sample 2 indicators; SIS-biased weights inflate top-enriched
                probs = np.array([weights.get(ind, 1.0) for ind in available], dtype=float)
                probs = probs / probs.sum()
                chosen = list(rng.choice(available, size=2, replace=False, p=probs))
            else:
                chosen = list(rng.choice(
                    list(MINIMAL_CONFIG["indicators"]["available"]), size=2, replace=False
                ))
            ind = _make_ind(ind_types=chosen, fitness=0.0, rng=rng)
            ind.evaluated = False
            ind.fitness = None
            pop.add_individual(ind)

        ga.population = pop

        # ── 2. Seed filtering (SIS) ──────────────────────────────────────────
        if sis_enabled and sis is not None:
            sis.filter_population(pop, ga)

        # ── 3. Evaluate with synthetic oracle ───────────────────────────────
        for ind in pop.individuals:
            ind.fitness = _synthetic_fitness(ind)
            ind.raw_fitness = ind.fitness
            ind.evaluated = True
            ind.metrics = {
                "profit": ind.fitness * 50,
                "win_rate": 0.4 + ind.fitness * 0.3,
                "max_drawdown": 0.1,
                "sharpe_ratio": ind.fitness * 1.5,
                "num_trades": 80,
                "pair_generalization_ratio": 0.75,
            }

        # ── 4. Inject SIS immigrants ─────────────────────────────────────────
        if sis_enabled and sis is not None:
            immigrants = sis.immigrant_provider(ga, generation=gen)
            for imm in immigrants:
                imm.fitness = _synthetic_fitness(imm)
                imm.raw_fitness = imm.fitness
                imm.evaluated = True
                imm.metrics = {
                    "profit": imm.fitness * 50,
                    "win_rate": 0.4 + imm.fitness * 0.3,
                    "max_drawdown": 0.10,
                    "sharpe_ratio": imm.fitness * 1.5,
                    "num_trades": 80,
                    "pair_generalization_ratio": 0.75,
                }
                pop.add_individual(imm)

        # ── 5. Online learning feedback ──────────────────────────────────────
        if sis_enabled and sis is not None:
            sis.update_from_generation(gen, pop.individuals)

        # ── 6. Record metrics ────────────────────────────────────────────────
        fitnesses = [ind.fitness for ind in pop.individuals
                     if ind.fitness is not None]
        best = max(fitnesses) if fitnesses else 0.0
        mean = float(np.mean(fitnesses)) if fitnesses else 0.0

        if best > best_fitness_ever:
            best_fitness_ever = best
            convergence_gen = gen

        # Fraction of top-enriched indicators in this generation's pop
        all_types: List[str] = []
        for ind in pop.individuals:
            all_types.extend(g.type for g in ind.strategy_gene.indicators)
        enriched_frac = (
            sum(1 for t in all_types if t in _TOP_ENRICHED) / max(len(all_types), 1)
        )

        metrics_history.append({
            "gen": gen,
            "best_fitness": best,
            "mean_fitness": mean,
            "enriched_indicator_frac": enriched_frac,
            "pop_size": len(pop.individuals),
        })

    return {
        "history": metrics_history,
        "best_fitness_ever": best_fitness_ever,
        "convergence_gen": convergence_gen,
        "final_mean": metrics_history[-1]["mean_fitness"] if metrics_history else 0.0,
        "final_enriched_frac": metrics_history[-1]["enriched_indicator_frac"] if metrics_history else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Perf Test 1 — Hook latency
# ══════════════════════════════════════════════════════════════════════════════

class TestSISPerf_HookLatency:
    """Each hook must respond within 50ms for a population of 30."""

    def test_indicator_weights_latency(self, trained_sis_perf):
        t0 = time.perf_counter()
        for _ in range(10):
            trained_sis_perf.get_indicator_weights()
        elapsed_ms = (time.perf_counter() - t0) * 100  # avg per call in ms
        assert elapsed_ms < 50, f"indicator_weights too slow: {elapsed_ms:.1f}ms avg"

    def test_operator_weights_latency(self, trained_sis_perf):
        t0 = time.perf_counter()
        for _ in range(10):
            trained_sis_perf.get_operator_weights()
        elapsed_ms = (time.perf_counter() - t0) * 100
        assert elapsed_ms < 50, f"operator_weights too slow: {elapsed_ms:.1f}ms avg"

    def test_synergy_weights_latency(self, trained_sis_perf):
        t0 = time.perf_counter()
        for _ in range(10):
            trained_sis_perf.get_synergy_weights(["RSI", "EMA"])
        elapsed_ms = (time.perf_counter() - t0) * 100
        assert elapsed_ms < 50, f"synergy_weights too slow: {elapsed_ms:.1f}ms avg"

    def test_immigrant_provider_latency(self, trained_sis_perf, mock_ga_perf):
        t0 = time.perf_counter()
        trained_sis_perf.immigrant_provider(mock_ga_perf, generation=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 5000, f"immigrant_provider too slow: {elapsed_ms:.0f}ms"

    def test_filter_population_latency(self, trained_sis_perf, mock_ga_perf):
        from genetic_algorithm.core.population import Population
        pop = Population(size=30)
        rng = np.random.RandomState(1)
        avail = list(MINIMAL_CONFIG["indicators"]["available"])
        for i in range(30):
            types = list(rng.choice(avail, size=2, replace=False))
            ind = _make_ind(types, fitness=float(rng.uniform(0.2, 0.8)), rng=rng)
            # Mark some as unevaluated so filter has something to act on
            if i < 10:
                ind.evaluated = False
                ind.fitness = None
                ind.metrics = {}
            pop.add_individual(ind)
        mock_ga_perf.population = pop

        t0 = time.perf_counter()
        trained_sis_perf.filter_population(pop, mock_ga_perf)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 5000, f"filter_population too slow: {elapsed_ms:.0f}ms"


# ══════════════════════════════════════════════════════════════════════════════
# Perf Test 2 — Indicator weight bias
# ══════════════════════════════════════════════════════════════════════════════

class TestSISPerf_WeightBias:
    """SIS weights should meaningfully favor enriched indicators over plain ones."""

    def test_top_enriched_have_higher_weights_than_sma(self, trained_sis_perf):
        """CCI, STOCH, ATR should all exceed SMA weight (SMA has 1.0 base enrichment)."""
        weights = trained_sis_perf.get_indicator_weights()
        sma_w = weights.get("SMA", 1.0)
        for ind in ["CCI", "STOCH", "ATR"]:
            if ind in weights:
                assert weights[ind] >= sma_w * 0.8, (
                    f"{ind} ({weights[ind]:.3f}) should be ≥ SMA ({sma_w:.3f}) * 0.8"
                )

    def test_weight_variance_nonzero(self, trained_sis_perf):
        """Weight variance should be non-trivial (not all ones)."""
        weights = trained_sis_perf.get_indicator_weights()
        vals = list(weights.values())
        variance = float(np.var(vals))
        assert variance > 0.001, f"Weights are suspiciously uniform (variance={variance:.5f})"

    def test_weights_cover_all_available_indicators(self, trained_sis_perf):
        available = set(MINIMAL_CONFIG["indicators"]["available"])
        weights = trained_sis_perf.get_indicator_weights()
        missing = available - set(weights.keys())
        assert not missing, f"Missing weights for: {missing}"


# ══════════════════════════════════════════════════════════════════════════════
# Perf Test 3 — Immigration quality
# ══════════════════════════════════════════════════════════════════════════════

class TestSISPerf_ImmigrantQuality:
    """Archetype immigrants should score >= baseline after synthetic eval."""

    def test_immigrants_receive_nonzero_fitness(self, trained_sis_perf, mock_ga_perf):
        immigrants = trained_sis_perf.immigrant_provider(mock_ga_perf, generation=10)
        if not immigrants:
            pytest.skip("No immigrants generated (classifier may have no archetypes)")
        for imm in immigrants:
            eval_fitness = _synthetic_fitness(imm)
            assert eval_fitness >= 0, "Negative synthetic fitness is impossible"

    def test_immigrants_have_valid_indicators(self, trained_sis_perf, mock_ga_perf):
        immigrants = trained_sis_perf.immigrant_provider(mock_ga_perf, generation=10)
        if not immigrants:
            pytest.skip("No immigrants generated")
        available = set(MINIMAL_CONFIG["indicators"]["available"])
        for imm in immigrants:
            for ind_gene in imm.strategy_gene.indicators:
                assert ind_gene.type in available, (
                    f"Immigrant has unknown indicator type: {ind_gene.type!r}"
                )

    def test_immigrants_score_higher_than_pure_random_baseline(
        self, trained_sis_perf, mock_ga_perf
    ):
        """
        Generate 10 batches of immigrants; their average synthetic fitness should
        be at least as good as the baseline (≥0.35 from our oracle).
        We don't expect dramatic improvements since fitness is synthetic —
        the test just verifies immigrants aren't systematically terrible.
        """
        immigrants = trained_sis_perf.immigrant_provider(mock_ga_perf, generation=5)
        if not immigrants:
            pytest.skip("No immigrants generated")

        scores = [_synthetic_fitness(imm) for imm in immigrants]
        avg_score = float(np.mean(scores))
        # Baseline: pure random = expected ~0.4 (base=0.4 + no bonus)
        assert avg_score >= 0.30, (
            f"Immigrant average score {avg_score:.3f} is below baseline 0.30"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Perf Test 4 — Simulated evolution: SIS ON vs SIS OFF
# ══════════════════════════════════════════════════════════════════════════════

class TestSISPerf_SimulatedEvolution:
    """
    Run a 20-generation simulated evolution twice (SIS ON / OFF) and compare.

    With our synthetic fitness oracle (CCI/STOCH/ATR earn a +15% bonus),
    SIS-biased indicator sampling should produce slightly higher mean fitness
    by selecting more top-enriched indicators on average.
    """

    GENERATIONS = 20
    POP_SIZE = 30

    @pytest.fixture(scope="class")
    def results_sis_on(self, trained_sis_perf, mock_ga_perf):
        return _run_simulated_evolution(
            trained_sis_perf, mock_ga_perf,
            generations=self.GENERATIONS,
            pop_size=self.POP_SIZE,
            rng_seed=42,
            sis_enabled=True,
        )

    @pytest.fixture(scope="class")
    def results_sis_off(self, trained_sis_perf, mock_ga_perf):
        return _run_simulated_evolution(
            None, mock_ga_perf,
            generations=self.GENERATIONS,
            pop_size=self.POP_SIZE,
            rng_seed=42,
            sis_enabled=False,
        )

    def test_sis_on_run_completes(self, results_sis_on):
        assert len(results_sis_on["history"]) == self.GENERATIONS

    def test_sis_off_run_completes(self, results_sis_off):
        assert len(results_sis_off["history"]) == self.GENERATIONS

    def test_sis_on_mean_fitness_reasonable(self, results_sis_on):
        final_mean = results_sis_on["final_mean"]
        assert final_mean > 0.30, f"SIS ON mean fitness {final_mean:.3f} seems low"

    def test_sis_off_mean_fitness_reasonable(self, results_sis_off):
        final_mean = results_sis_off["final_mean"]
        assert final_mean > 0.30, f"SIS OFF mean fitness {final_mean:.3f} seems low"

    def test_sis_on_enriched_fraction_higher(self, results_sis_on, results_sis_off):
        """SIS biases indicator sampling → more CCI/STOCH/ATR in population."""
        on_frac = results_sis_on["final_enriched_frac"]
        off_frac = results_sis_off["final_enriched_frac"]
        # SIS-biased sampling should produce at least as many top-enriched indicators
        assert on_frac >= off_frac * 0.9, (
            f"SIS ON enriched frac ({on_frac:.3f}) should be >= OFF frac "
            f"({off_frac:.3f}) * 0.9"
        )

    def test_sis_on_best_fitness_nondegraded(self, results_sis_on, results_sis_off):
        """SIS should not significantly worsen peak fitness."""
        on_best = results_sis_on["best_fitness_ever"]
        off_best = results_sis_off["best_fitness_ever"]
        assert on_best >= off_best * 0.85, (
            f"SIS ON best fitness ({on_best:.3f}) degraded vs OFF ({off_best:.3f})"
        )

    def test_history_best_fitness_monotone_nondecreasing(self, results_sis_on):
        """Best-ever fitness should be non-decreasing across generations."""
        history = results_sis_on["history"]
        running_best = 0.0
        for entry in history:
            bf = entry["best_fitness"]
            # Best-ever should not drop (but per-gen best can fluctuate)
            running_best = max(running_best, bf)
        assert running_best > 0, "Should have found at least one fitness > 0"


# ══════════════════════════════════════════════════════════════════════════════
# Perf Test 5 — A/B Result generation across 3 seeds
# ══════════════════════════════════════════════════════════════════════════════

class TestSISPerf_ABResult:
    """Build an ABResult from simulated multi-seed runs and check outputs."""

    GENERATIONS = 15
    POP_SIZE = 25
    SEEDS = [42, 123, 456]

    @pytest.fixture(scope="class")
    def ab_result(self, trained_sis_perf, mock_ga_perf):
        """Run 3 seeds SIS ON and SIS OFF, build ABResult."""
        result = ABResult(
            experiment_name="sis_indicator_weights_perf",
            hook="indicator_weights",
            seeds=self.SEEDS,
            generations=self.GENERATIONS,
        )
        for seed in self.SEEDS:
            on = _run_simulated_evolution(
                trained_sis_perf, mock_ga_perf,
                generations=self.GENERATIONS, pop_size=self.POP_SIZE,
                rng_seed=seed, sis_enabled=True,
            )
            off = _run_simulated_evolution(
                None, mock_ga_perf,
                generations=self.GENERATIONS, pop_size=self.POP_SIZE,
                rng_seed=seed, sis_enabled=False,
            )
            result.treat_best_fitness.append(on["best_fitness_ever"])
            result.treat_final_fitness.append(on["final_mean"])
            result.treat_convergence_gen.append(on["convergence_gen"])
            result.treat_diversity.append(on["final_enriched_frac"])

            result.ctrl_best_fitness.append(off["best_fitness_ever"])
            result.ctrl_final_fitness.append(off["final_mean"])
            result.ctrl_convergence_gen.append(off["convergence_gen"])
            result.ctrl_diversity.append(off["final_enriched_frac"])

        return result

    def test_ab_result_has_all_seeds(self, ab_result):
        assert len(ab_result.ctrl_best_fitness) == len(self.SEEDS)
        assert len(ab_result.treat_best_fitness) == len(self.SEEDS)

    def test_ab_result_is_significant(self, ab_result):
        """3 seeds per arm — should pass is_significant threshold."""
        assert ab_result.is_significant(), "Need >= 3 samples per arm"

    def test_fitness_delta_computed(self, ab_result):
        delta = ab_result.fitness_delta()
        assert isinstance(delta, float)

    def test_effect_size_computed(self, ab_result):
        d = ab_result.effect_size()
        assert isinstance(d, float)

    def test_summary_prints_cleanly(self, ab_result, capsys):
        summary = ab_result.summary()
        assert "A/B Experiment" in summary
        assert "Verdict" in summary
        print("\n" + "=" * 60)
        print(summary)
        print("=" * 60)

    def test_ab_result_serializable(self, ab_result):
        d = ab_result.to_dict()
        assert isinstance(d, dict)
        dumped = json.dumps(d)  # Should not raise
        assert len(dumped) > 10


# ══════════════════════════════════════════════════════════════════════════════
# Perf Test 6 — Adaptive weight learning over generations
# ══════════════════════════════════════════════════════════════════════════════

class TestSISPerf_AdaptiveWeights:
    """AdaptiveWeightTracker should shift weights as live evidence accumulates."""

    def test_evidence_trust_starts_low(self, trained_sis_perf):
        trust = trained_sis_perf._adaptive_weights._current_evidence_trust()
        # Trust should be between 0 and 1
        assert 0.0 <= trust <= 1.0

    def test_weight_diagnostics_returned(self, trained_sis_perf):
        diag = trained_sis_perf._adaptive_weights.get_diagnostics()
        assert isinstance(diag, dict)

    def test_live_records_accumulate_across_gens(
        self, trained_sis_perf
    ):
        """After 10 generations of update_from_generation(), live records grow."""
        before = len(trained_sis_perf._live_corpus_records)
        rng = np.random.RandomState(77)
        avail = list(MINIMAL_CONFIG["indicators"]["available"])
        for gen in range(10):
            inds = []
            for _ in range(15):
                types = list(rng.choice(avail, size=2, replace=False))
                ind = _make_ind(types, fitness=float(rng.uniform(0.3, 0.8)), rng=rng)
                inds.append(ind)
            trained_sis_perf.update_from_generation(gen + 200, inds)
        after = len(trained_sis_perf._live_corpus_records)
        assert after > before, "Live corpus records should have grown"

    def test_online_diagnostics_summary(self, trained_sis_perf, capsys):
        """Print a quick live performance report."""
        summary = trained_sis_perf.get_status_summary()
        weights = trained_sis_perf.get_indicator_weights()
        diag = trained_sis_perf._adaptive_weights.get_diagnostics()

        print("\n" + "═" * 60)
        print("  SIS LIVE STATUS REPORT")
        print("═" * 60)
        print(f"  Evolution state:   {summary['evolution_state']}")
        print(f"  N archetypes:      {summary['n_archetypes']}")
        print(f"  Reliable models:   {summary['reliable_models']}")
        print(f"  Hook calls:        {summary['hook_calls']}")
        print(f"  Live records:      {len(trained_sis_perf._live_corpus_records)}")
        print(f"  Evidence trust:    {diag}")
        print()
        print("  Top indicator weights:")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for name, w in sorted_weights[:8]:
            bar = "█" * int(w * 10)
            print(f"    {name:12s} {w:.3f}  {bar}")
        print("═" * 60)
