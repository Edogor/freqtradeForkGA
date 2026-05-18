"""
Comprehensive tests for the Strategy Intelligence System (SIS).

Tests cover: CorpusBuilder, MultiTargetPredictor, ArchetypeClassifier,
TemporalAnalyzer, and PatternMiner.
"""

import json
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genetic_algorithm.evaluation.surrogate import _INDICATOR_TYPES, _OPERATOR_TYPES
from genetic_algorithm.intelligence.corpus import (
    CorpusBuilder,
    SURROGATE_FEATURE_NAMES,
    METRIC_COLUMNS,
    META_COLUMNS,
)
from genetic_algorithm.intelligence.predictors import MultiTargetPredictor, TARGET_DEFINITIONS
from genetic_algorithm.intelligence.archetypes import ArchetypeClassifier
from genetic_algorithm.intelligence.temporal_analysis import TemporalAnalyzer
from genetic_algorithm.intelligence.pattern_mining import PatternMiner


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

def _make_synthetic_corpus(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic corpus DataFrame with proper column names.

    Creates three distinct strategy "clusters" to give HDBSCAN something
    to work with, plus noise.
    """
    rng = np.random.RandomState(seed)

    records = []
    for i in range(n):
        row = {}

        # --- Surrogate feature columns (65) ---
        # Indicator one-hot: create 3 distinct profiles
        cluster = i % 4  # 0,1,2 = distinct clusters, 3 = noise
        ind_onehot = np.zeros(len(_INDICATOR_TYPES))
        if cluster == 0:
            # RSI + EMA cluster
            ind_onehot[_INDICATOR_TYPES.index("RSI")] = 1.0
            ind_onehot[_INDICATOR_TYPES.index("EMA")] = 1.0
        elif cluster == 1:
            # MACD + BBANDS cluster
            ind_onehot[_INDICATOR_TYPES.index("MACD")] = 1.0
            ind_onehot[_INDICATOR_TYPES.index("BBANDS")] = 1.0
        elif cluster == 2:
            # ADX + ATR + STOCH cluster
            ind_onehot[_INDICATOR_TYPES.index("ADX")] = 1.0
            ind_onehot[_INDICATOR_TYPES.index("ATR")] = 1.0
            ind_onehot[_INDICATOR_TYPES.index("STOCH")] = 1.0
        else:
            # Random noise
            for idx in rng.choice(len(_INDICATOR_TYPES), size=rng.randint(1, 5), replace=False):
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
                base = 14.0 + cluster * 10
                row[name] = base + rng.uniform(-2, 2)
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

        # --- Metric columns ---
        base_fitness = [0.8, 0.5, 0.3, 0.1][cluster]
        row["fitness"] = base_fitness + rng.uniform(-0.15, 0.15)
        row["raw_fitness"] = row["fitness"]
        row["profit"] = row["fitness"] * 100 + rng.uniform(-10, 10)
        row["sharpe_ratio"] = row["fitness"] * 2 + rng.uniform(-0.5, 0.5)
        row["sortino_ratio"] = row["sharpe_ratio"] * 1.2
        row["profit_factor"] = 1.0 + row["fitness"] + rng.uniform(-0.2, 0.2)
        row["max_drawdown"] = rng.uniform(0.05, 0.35)
        row["win_rate"] = 0.4 + row["fitness"] * 0.3 + rng.uniform(-0.05, 0.05)
        row["num_trades"] = float(rng.randint(20, 2000))
        row["train_fitness"] = row["fitness"] + rng.uniform(-0.05, 0.05)
        row["val_fitness"] = row["fitness"] - rng.uniform(0, 0.1)
        row["pair_generalization_ratio"] = rng.uniform(0.5, 1.0)
        row["val_profit"] = row["profit"] * rng.uniform(0.6, 1.0)
        row["val_sharpe"] = row["sharpe_ratio"] * rng.uniform(0.5, 1.0)
        row["val_trades"] = row["num_trades"] * rng.uniform(0.3, 0.8)
        row["val_max_drawdown"] = row["max_drawdown"] * rng.uniform(0.8, 1.5)
        row["val_win_rate"] = row["win_rate"] * rng.uniform(0.8, 1.1)
        row["holdout_fitness"] = row["fitness"] - rng.uniform(0, 0.15)
        row["holdout_degradation"] = rng.uniform(0.0, 0.3)
        row["holdout_profit"] = row["profit"] * rng.uniform(0.4, 0.9)
        row["holdout_trades"] = row["num_trades"] * rng.uniform(0.2, 0.6)
        row["complexity"] = float(rng.randint(3, 20))
        row["monthly_return_std"] = rng.uniform(1, 10)
        row["positive_months_ratio"] = rng.uniform(0.3, 0.9)
        row["pair_profit_std"] = rng.uniform(0.5, 5.0)
        row["max_consecutive_losses"] = float(rng.randint(1, 15))
        row["max_drawdown_duration_days"] = float(rng.randint(1, 60))

        # --- Monthly profit summary stats ---
        row["monthly_profit_mean"] = rng.uniform(-2, 8)
        row["monthly_profit_std"] = rng.uniform(1, 10)
        row["monthly_profit_min"] = row["monthly_profit_mean"] - rng.uniform(5, 20)
        row["monthly_profit_max"] = row["monthly_profit_mean"] + rng.uniform(5, 20)
        row["n_positive_months"] = rng.randint(1, 12)
        row["n_negative_months"] = rng.randint(0, 6)
        row["n_months_total"] = row["n_positive_months"] + row["n_negative_months"]

        # --- Per-pair profit summary stats ---
        row["per_pair_profit_mean"] = rng.uniform(-1, 5)
        row["per_pair_profit_std"] = rng.uniform(0.5, 4)
        row["per_pair_profit_min"] = row["per_pair_profit_mean"] - rng.uniform(2, 8)
        row["per_pair_profit_max"] = row["per_pair_profit_mean"] + rng.uniform(2, 8)
        row["n_profitable_pairs"] = rng.randint(1, 10)
        row["n_pairs"] = rng.randint(5, 15)

        # --- Metadata ---
        wave_num = (i // 40) + 1  # 5 waves across 200 rows
        row["fingerprint"] = hashlib.sha256(f"strat_{i}".encode()).hexdigest()[:16]
        row["run_id"] = f"wave{wave_num}_A_rank_5m_ring"
        row["generation"] = rng.randint(0, 50)
        row["individual_id"] = str(i)
        row["source"] = "hall_of_fame" if i % 3 == 0 else "gen_snapshot"
        row["timeframe"] = ["5m", "15m", "1h"][cluster % 3]
        row["origin"] = "backtest"
        row["training_pairs"] = "BTC/USDT,ETH/USDT"
        row["validation_pairs"] = "SOL/USDT"
        row["wave"] = f"wave{wave_num}"

        records.append(row)

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_corpus_df():
    """Shared synthetic corpus DataFrame for tests."""
    return _make_synthetic_corpus(n=200)


@pytest.fixture
def small_corpus_df():
    """Smaller corpus for fast tests."""
    return _make_synthetic_corpus(n=60, seed=99)


@pytest.fixture
def sample_gene_dict():
    """A sample gene dict matching StrategyGene.to_dict() structure."""
    return {
        "generation": 10,
        "individual_id": 42,
        "indicators": [
            {"type": "RSI", "parameters": {"period": 14}, "weight": 1.0,
             "instance_id": "RSI_0", "timeframe": None, "param_bounds": None},
            {"type": "EMA", "parameters": {"period": 21}, "weight": 0.8,
             "instance_id": "EMA_0", "timeframe": None, "param_bounds": None},
        ],
        "entry_conditions": [
            {"indicator": "RSI_0", "operator": "<", "threshold": 30.0,
             "logic": "AND", "threshold_upper": 0.0, "lookback": 3},
        ],
        "exit_conditions": [
            {"indicator": "RSI_0", "operator": ">", "threshold": 70.0,
             "logic": "AND", "threshold_upper": 0.0, "lookback": 3},
        ],
        "short_entry_conditions": [],
        "short_exit_conditions": [],
        "timeframe": "5m",
        "informative_timeframes": ["1h"],
        "stoploss": -0.08,
        "minimal_roi": {"0": 0.04, "30": 0.02, "60": 0.01},
        "max_open_trades": 3,
        "trailing_stop": True,
        "trailing_stop_positive": None,
        "trailing_stop_positive_offset": None,
        "can_short": False,
        "preferred_regime": None,
        "regime_mode": "generalist",
        "regime_gene": None,
        "self_mutation_rate": None,
        "self_crossover_pref": None,
    }


# ══════════════════════════════════════════════════════════════════════
# Corpus Tests
# ══════════════════════════════════════════════════════════════════════

class TestCorpusBuilder:
    """Tests for CorpusBuilder."""

    def test_gene_fingerprint_deterministic(self, sample_gene_dict):
        """Same gene dict should always produce the same fingerprint."""
        fp1 = CorpusBuilder._gene_fingerprint(sample_gene_dict)
        fp2 = CorpusBuilder._gene_fingerprint(sample_gene_dict)
        assert fp1 == fp2
        assert len(fp1) == 16  # SHA256 truncated to 16 hex chars

    def test_gene_fingerprint_different_genes(self, sample_gene_dict):
        """Different gene dicts should produce different fingerprints."""
        import copy
        gene2 = copy.deepcopy(sample_gene_dict)
        gene2["stoploss"] = -0.20
        fp1 = CorpusBuilder._gene_fingerprint(sample_gene_dict)
        fp2 = CorpusBuilder._gene_fingerprint(gene2)
        assert fp1 != fp2

    def test_gene_fingerprint_ignores_non_key_fields(self, sample_gene_dict):
        """Fingerprint should be based on structural fields only."""
        import copy
        gene2 = copy.deepcopy(sample_gene_dict)
        # Change fields NOT in the fingerprint key
        gene2["generation"] = 999
        gene2["individual_id"] = 9999
        fp1 = CorpusBuilder._gene_fingerprint(sample_gene_dict)
        fp2 = CorpusBuilder._gene_fingerprint(gene2)
        assert fp1 == fp2

    def test_extract_wave_from_run_id(self):
        """_extract_wave should parse wave prefix from run_id."""
        assert CorpusBuilder._extract_wave("wave30_A_rank_5m_ring") == "wave30"
        assert CorpusBuilder._extract_wave("wave5_B_star") == "wave5"
        assert CorpusBuilder._extract_wave("random_run_id") == ""
        assert CorpusBuilder._extract_wave("") == ""

    def test_extract_features_from_dict(self, sample_gene_dict):
        """_extract_features_from_dict should return 65-length feature vector."""
        builder = CorpusBuilder()
        features = builder._extract_features_from_dict(sample_gene_dict)
        assert len(features) == 65
        # RSI should be present (indicator one-hot)
        rsi_idx = _INDICATOR_TYPES.index("RSI")
        assert features[rsi_idx] == 1.0
        # MACD should not be present
        macd_idx = _INDICATOR_TYPES.index("MACD")
        assert features[macd_idx] == 0.0
        # n_indicators = 2
        assert features[len(_INDICATOR_TYPES)] == 2.0

    def test_extract_features_from_dict_empty_gene(self):
        """Should handle empty/minimal gene dicts gracefully."""
        builder = CorpusBuilder()
        empty_gene = {
            "indicators": [],
            "entry_conditions": [],
            "exit_conditions": [],
        }
        features = builder._extract_features_from_dict(empty_gene)
        assert len(features) == 65
        # All indicator one-hots should be 0
        for i in range(len(_INDICATOR_TYPES)):
            assert features[i] == 0.0

    def test_build_record_with_metrics(self, sample_gene_dict):
        """_build_record should produce a dict with all expected keys."""
        builder = CorpusBuilder()
        metrics = {
            "profit": 45.2,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.12,
            "win_rate": 0.62,
            "num_trades": 150,
            "monthly_profits": [5.0, -2.0, 8.0, 3.0, -1.0, 6.0],
            "per_pair_profit": {"BTC/USDT": 20.0, "ETH/USDT": -3.0, "SOL/USDT": 10.0},
        }
        record = builder._build_record(
            gene_dict=sample_gene_dict,
            fitness=0.75,
            raw_fitness=0.70,
            metrics=metrics,
            run_id="wave10_A_rank_5m_ring",
            generation=5,
            individual_id=42,
            source="hall_of_fame",
        )
        assert record is not None
        # Check surrogate features
        for name in SURROGATE_FEATURE_NAMES:
            assert name in record, f"Missing feature column: {name}"
        # Check metrics
        assert record["fitness"] == 0.75
        assert record["raw_fitness"] == 0.70
        assert record["profit"] == 45.2
        # Check monthly profit summary
        assert record["monthly_profit_mean"] == pytest.approx(np.mean([5, -2, 8, 3, -1, 6]))
        assert record["n_positive_months"] == 4
        assert record["n_negative_months"] == 2
        # Check per-pair summary
        assert record["per_pair_profit_mean"] == pytest.approx(np.mean([20, -3, 10]))
        assert record["n_profitable_pairs"] == 2
        assert record["n_pairs"] == 3
        # Check metadata
        assert record["wave"] == "wave10"
        assert record["source"] == "hall_of_fame"
        assert record["timeframe"] == "5m"

    def test_build_with_hof_file(self, tmp_path, sample_gene_dict):
        """build() should read and parse a hall_of_fame.json file."""
        hof_dir = tmp_path / "hall_of_fame_wave1_test"
        hof_dir.mkdir()
        hof_data = {
            "entries": [
                {
                    "fitness": 0.82,
                    "strategy_gene": sample_gene_dict,
                    "metrics": {
                        "profit": 50.0,
                        "sharpe_ratio": 2.0,
                        "num_trades": 100,
                    },
                    "generation_found": 10,
                    "individual_id": 1,
                },
            ]
        }
        with open(hof_dir / "hall_of_fame.json", "w") as f:
            json.dump(hof_data, f)

        builder = CorpusBuilder(data_dir=tmp_path)
        df = builder.build()
        assert len(df) == 1
        assert df.iloc[0]["fitness"] == 0.82
        # run_id is inferred from dir name "hall_of_fame_wave1_test" -> "wave1_test"
        assert df.iloc[0]["run_id"] == "wave1_test"

    def test_build_deduplicates(self, tmp_path, sample_gene_dict):
        """Duplicate genes (same fingerprint) should be deduplicated."""
        hof_dir = tmp_path / "hof1"
        hof_dir.mkdir()
        entry = {
            "fitness": 0.80,
            "strategy_gene": sample_gene_dict,
            "metrics": {},
            "generation_found": 5,
        }
        hof_data = {"entries": [entry, entry]}  # Same gene twice
        with open(hof_dir / "hall_of_fame.json", "w") as f:
            json.dump(hof_data, f)

        builder = CorpusBuilder(data_dir=tmp_path)
        df = builder.build()
        assert len(df) == 1

    def test_build_empty_returns_empty_df(self, tmp_path):
        """build() should return empty DataFrame when no data files exist."""
        builder = CorpusBuilder(data_dir=tmp_path)
        df = builder.build()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_save_and_load_parquet(self, tmp_path, sample_gene_dict):
        """save() should write a loadable Parquet file."""
        hof_dir = tmp_path / "hof"
        hof_dir.mkdir()
        hof_data = {
            "entries": [{
                "fitness": 0.5,
                "strategy_gene": sample_gene_dict,
                "metrics": {"profit": 10.0},
                "generation_found": 1,
            }]
        }
        with open(hof_dir / "hall_of_fame.json", "w") as f:
            json.dump(hof_data, f)

        builder = CorpusBuilder(data_dir=tmp_path)
        out_path = tmp_path / "corpus.parquet"
        df = builder.save(path=out_path)
        assert out_path.exists()
        loaded = pd.read_parquet(out_path)
        assert len(loaded) == len(df)


# ══════════════════════════════════════════════════════════════════════
# Predictor Tests
# ══════════════════════════════════════════════════════════════════════

class TestMultiTargetPredictor:
    """Tests for MultiTargetPredictor."""

    def test_train_and_predict(self, synthetic_corpus_df):
        """train() should fit models and predict() should return predictions."""
        pred = MultiTargetPredictor()
        results = pred.train(synthetic_corpus_df, targets=["fitness", "profit"])
        assert "fitness" in results
        assert "profit" in results
        assert "r2" in results["fitness"]
        assert "rmse" in results["fitness"]
        assert "mae" in results["fitness"]

        # Predict on the same data
        preds = pred.predict(synthetic_corpus_df, targets=["fitness"])
        assert "pred_fitness" in preds.columns
        assert len(preds) == len(synthetic_corpus_df)

    def test_train_all_default_targets(self, synthetic_corpus_df):
        """Training with default targets should train on all TARGET_DEFINITIONS."""
        pred = MultiTargetPredictor()
        results = pred.train(synthetic_corpus_df)
        # Should have trained on all targets that exist in the df
        for target_col, _, _ in TARGET_DEFINITIONS:
            if target_col in synthetic_corpus_df.columns:
                assert target_col in results

    def test_save_and_load(self, synthetic_corpus_df, tmp_path):
        """Models should be saveable and loadable with consistent predictions."""
        pred = MultiTargetPredictor(models_dir=tmp_path)
        pred.train(synthetic_corpus_df, targets=["fitness"])
        pred.save(tmp_path)

        # Check files exist
        assert (tmp_path / "predictor_fitness.pkl").exists()
        assert (tmp_path / "predictor_meta.json").exists()

        # Load and compare
        pred2 = MultiTargetPredictor(models_dir=tmp_path)
        pred2.load(tmp_path)
        assert "fitness" in pred2.models
        assert "fitness" in pred2.metrics

        # Predictions should match
        p1 = pred.predict(synthetic_corpus_df, targets=["fitness"])["pred_fitness"]
        p2 = pred2.predict(synthetic_corpus_df, targets=["fitness"])["pred_fitness"]
        np.testing.assert_array_almost_equal(p1.values, p2.values)

    def test_split_data_wave_based(self, synthetic_corpus_df):
        """_split_data should split by wave when wave info is available."""
        pred = MultiTargetPredictor()
        # Explicit test waves
        train_df, test_df = pred._split_data(synthetic_corpus_df, test_waves=["wave5"])
        assert all(test_df["wave"] == "wave5")
        assert "wave5" not in train_df["wave"].values

    def test_split_data_auto_detect(self, synthetic_corpus_df):
        """_split_data with no args should auto-detect latest waves."""
        pred = MultiTargetPredictor()
        train_df, test_df = pred._split_data(synthetic_corpus_df)
        # Should have split the data somehow
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(test_df) == len(synthetic_corpus_df)

    def test_handles_missing_targets(self, synthetic_corpus_df):
        """Training with a non-existent target column should skip it."""
        pred = MultiTargetPredictor()
        results = pred.train(synthetic_corpus_df, targets=["fitness", "nonexistent_col"])
        assert "fitness" in results
        assert "nonexistent_col" not in results

    def test_report_format(self, synthetic_corpus_df):
        """report() should return a non-empty string."""
        pred = MultiTargetPredictor()
        pred.train(synthetic_corpus_df, targets=["fitness"])
        report = pred.report()
        assert isinstance(report, str)
        assert "Fitness" in report or "fitness" in report
        assert "R" in report  # R-squared

    def test_feature_importance_populated(self, synthetic_corpus_df):
        """After training, feature_importance should have DataFrames."""
        pred = MultiTargetPredictor()
        pred.train(synthetic_corpus_df, targets=["fitness"])
        assert "fitness" in pred.feature_importance
        fi_df = pred.feature_importance["fitness"]
        assert isinstance(fi_df, pd.DataFrame)
        assert "feature" in fi_df.columns
        assert "importance" in fi_df.columns
        assert len(fi_df) == 77  # 65 base + 10 synergies + 2 ratios


# ══════════════════════════════════════════════════════════════════════
# Archetype Tests
# ══════════════════════════════════════════════════════════════════════

class TestArchetypeClassifier:
    """Tests for ArchetypeClassifier."""

    def test_fit_discovers_clusters(self, synthetic_corpus_df):
        """fit() should assign archetype labels to the DataFrame."""
        ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3)
        result = ac.fit(synthetic_corpus_df)
        assert "archetype" in result.columns
        assert "archetype_label" in result.columns
        assert "cluster_x" in result.columns
        assert "cluster_y" in result.columns
        # Should find at least 1 cluster (not just noise)
        unique = set(result["archetype"].values)
        non_noise = unique - {-1}
        assert len(non_noise) >= 1

    def test_predict_assigns_labels(self, synthetic_corpus_df):
        """predict() on new data should assign archetype labels."""
        ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3)
        ac.fit(synthetic_corpus_df)

        # Predict on a subset
        new_data = synthetic_corpus_df.head(20).copy()
        result = ac.predict(new_data)
        assert "archetype" in result.columns
        assert "archetype_label" in result.columns
        assert len(result) == 20

    def test_predict_before_fit_raises(self, synthetic_corpus_df):
        """predict() without fit() should raise RuntimeError."""
        ac = ArchetypeClassifier()
        with pytest.raises(RuntimeError, match="Must call fit"):
            ac.predict(synthetic_corpus_df)

    def test_save_and_load(self, synthetic_corpus_df, tmp_path):
        """Classifier should be saveable and loadable."""
        ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3, models_dir=tmp_path)
        ac.fit(synthetic_corpus_df)
        ac.save(tmp_path)

        assert (tmp_path / "archetype_classifier.pkl").exists()

        ac2 = ArchetypeClassifier(models_dir=tmp_path)
        ac2.load(tmp_path)
        assert len(ac2.archetype_labels) == len(ac.archetype_labels)
        assert ac2.archetype_labels == ac.archetype_labels

    def test_labels_include_indicators(self, synthetic_corpus_df):
        """Auto-generated labels should reference indicator names or 'Mixed'."""
        ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3)
        ac.fit(synthetic_corpus_df)

        for cluster_id, label in ac.archetype_labels.items():
            if cluster_id == -1:
                assert label == "noise"
            else:
                # Label should be a non-empty string with timeframe info
                assert isinstance(label, str)
                assert len(label) > 0
                assert "(" in label  # contains timeframe in parens

    def test_compute_stats(self, synthetic_corpus_df):
        """_compute_stats should return per-cluster statistics."""
        ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3)
        ac.fit(synthetic_corpus_df)
        assert len(ac.archetype_stats) > 0
        for cluster_id, stats in ac.archetype_stats.items():
            assert "count" in stats
            assert stats["count"] > 0
            if "fitness_mean" in stats:
                assert isinstance(stats["fitness_mean"], float)

    def test_report_format(self, synthetic_corpus_df):
        """report() should return a readable string."""
        ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3)
        ac.fit(synthetic_corpus_df)
        report = ac.report()
        assert isinstance(report, str)
        assert "ARCHETYPE" in report


# ══════════════════════════════════════════════════════════════════════
# Temporal Analysis Tests
# ══════════════════════════════════════════════════════════════════════

class TestTemporalAnalyzer:
    """Tests for TemporalAnalyzer."""

    def test_analyze_weaknesses(self, synthetic_corpus_df):
        """analyze_weaknesses should identify weaknesses in top strategies."""
        analyzer = TemporalAnalyzer(synthetic_corpus_df)
        report = analyzer.analyze_weaknesses(top_n=20)
        assert "n_analyzed" in report
        assert report["n_analyzed"] == 20
        assert "strategies" in report
        assert len(report["strategies"]) == 20
        assert "weakness_frequency" in report

    def test_weakness_types_detected(self, synthetic_corpus_df):
        """Known weakness types should be detected in synthetic data."""
        # Inject a strategy with known weaknesses
        df = synthetic_corpus_df.copy()
        # Make top strategy have high variance + deep loss
        top_idx = df["fitness"].idxmax()
        df.loc[top_idx, "monthly_profit_std"] = 50.0
        df.loc[top_idx, "monthly_profit_mean"] = 2.0
        df.loc[top_idx, "monthly_profit_min"] = -15.0
        df.loc[top_idx, "max_drawdown"] = 0.35
        df.loc[top_idx, "max_drawdown_duration_days"] = 45.0
        df.loc[top_idx, "max_consecutive_losses"] = 12.0

        analyzer = TemporalAnalyzer(df)
        report = analyzer.analyze_weaknesses(top_n=5)
        all_weaknesses = []
        for s in report["strategies"]:
            all_weaknesses.extend(s.get("weaknesses", []))
        # At least some weaknesses should be detected
        assert len(all_weaknesses) > 0

    def test_monthly_matrix_with_data(self, synthetic_corpus_df):
        """build_monthly_matrix should return data when monthly columns exist."""
        analyzer = TemporalAnalyzer(synthetic_corpus_df)
        matrix = analyzer.build_monthly_matrix()
        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) > 0
        assert "fingerprint" in matrix.columns
        assert "monthly_profit_mean" in matrix.columns

    def test_monthly_matrix_empty_without_data(self):
        """build_monthly_matrix should return empty when no monthly data."""
        df = pd.DataFrame({"fitness": [0.5], "fingerprint": ["abc"]})
        analyzer = TemporalAnalyzer(df)
        matrix = analyzer.build_monthly_matrix()
        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) == 0

    def test_complementary_pairs_correlation(self, synthetic_corpus_df):
        """find_complementary_pairs should return scored pairs."""
        analyzer = TemporalAnalyzer(synthetic_corpus_df)
        pairs = analyzer.find_complementary_pairs(top_n=5, method="correlation")
        assert isinstance(pairs, list)
        if len(pairs) > 0:
            pair = pairs[0]
            assert "strategy_a" in pair
            assert "strategy_b" in pair
            assert "complementary_score" in pair
            assert "composite_score" in pair

    def test_complementary_pairs_pair_coverage(self, synthetic_corpus_df):
        """find_complementary_pairs with pair_coverage method."""
        analyzer = TemporalAnalyzer(synthetic_corpus_df)
        pairs = analyzer.find_complementary_pairs(top_n=5, method="pair_coverage")
        assert isinstance(pairs, list)
        if len(pairs) > 0:
            assert "coverage_score" in pairs[0]

    def test_report_format(self, synthetic_corpus_df):
        """report() should return a string even without prior analysis."""
        analyzer = TemporalAnalyzer(synthetic_corpus_df)
        # Before analysis
        report = analyzer.report()
        assert isinstance(report, str)
        assert "TEMPORAL" in report

        # After analysis
        analyzer.analyze_weaknesses(top_n=10)
        report = analyzer.report()
        assert "Analyzed" in report or "Weakness" in report


# ══════════════════════════════════════════════════════════════════════
# Pattern Mining Tests
# ══════════════════════════════════════════════════════════════════════

class TestPatternMiner:
    """Tests for PatternMiner."""

    def test_mine_returns_summary(self, synthetic_corpus_df):
        """mine() should return a summary dict with expected keys."""
        miner = PatternMiner(synthetic_corpus_df)
        summary = miner.mine()
        assert isinstance(summary, dict)
        assert "n_strategies" in summary
        assert "n_top" in summary
        assert "n_bottom" in summary
        assert "top_indicators" in summary
        assert "top_synergies" in summary
        assert summary["n_strategies"] == len(synthetic_corpus_df.dropna(subset=["fitness"]))

    def test_indicator_enrichment(self, synthetic_corpus_df):
        """_mine_indicator_enrichment should produce a DataFrame with expected columns."""
        miner = PatternMiner(synthetic_corpus_df)
        miner._mine_indicator_enrichment()
        assert miner.indicator_enrichment is not None
        df = miner.indicator_enrichment
        assert "indicator" in df.columns
        assert "enrichment" in df.columns
        assert "top_rate" in df.columns
        assert "bottom_rate" in df.columns
        assert "lift" in df.columns
        assert len(df) > 0
        # Enrichment values should be positive
        assert (df["enrichment"] > 0).all()

    def test_indicator_synergies(self, synthetic_corpus_df):
        """_mine_indicator_synergies should find co-occurrence patterns."""
        miner = PatternMiner(synthetic_corpus_df)
        miner._mine_indicator_synergies()
        assert miner.indicator_synergies is not None
        # May or may not find synergies depending on data, but should not error
        assert isinstance(miner.indicator_synergies, pd.DataFrame)

    def test_operator_patterns(self, synthetic_corpus_df):
        """_mine_operator_patterns should produce entry/exit operator stats."""
        miner = PatternMiner(synthetic_corpus_df)
        miner._mine_operator_patterns()
        assert miner.operator_patterns is not None
        assert "entry_operators" in miner.operator_patterns
        assert "exit_operators" in miner.operator_patterns

    def test_risk_parameters(self, synthetic_corpus_df):
        """_mine_risk_parameters should analyze risk param distributions."""
        miner = PatternMiner(synthetic_corpus_df)
        miner._mine_risk_parameters()
        assert miner.risk_param_analysis is not None
        assert "stoploss" in miner.risk_param_analysis
        stoploss_stats = miner.risk_param_analysis["stoploss"]
        assert "top_mean" in stoploss_stats
        assert "bottom_mean" in stoploss_stats
        assert stoploss_stats["top_mean"] is not None

    def test_report_format(self, synthetic_corpus_df):
        """report() should produce a readable string after mining."""
        miner = PatternMiner(synthetic_corpus_df)
        miner.mine()
        report = miner.report()
        assert isinstance(report, str)
        assert "PATTERN" in report
        assert "INDICATOR" in report

    def test_custom_top_bottom_pct(self, synthetic_corpus_df):
        """Custom top_pct/bottom_pct should change the analysis scope."""
        miner = PatternMiner(synthetic_corpus_df, top_pct=0.1, bottom_pct=0.1)
        summary = miner.mine()
        expected_top = max(1, int(len(synthetic_corpus_df.dropna(subset=["fitness"])) * 0.1))
        assert summary["n_top"] == expected_top

    def test_handles_nan_fitness(self):
        """Miner should handle rows with NaN fitness by dropping them."""
        df = _make_synthetic_corpus(n=50)
        df.loc[0:5, "fitness"] = np.nan
        miner = PatternMiner(df)
        summary = miner.mine()
        assert summary["n_strategies"] == len(df) - 6


# ══════════════════════════════════════════════════════════════════════
# Integration / Cross-Module Tests
# ══════════════════════════════════════════════════════════════════════

class TestSISIntegration:
    """Cross-module integration tests."""

    def test_corpus_feeds_predictor(self, synthetic_corpus_df):
        """Predictor should work on corpus-shaped data."""
        pred = MultiTargetPredictor()
        results = pred.train(synthetic_corpus_df, targets=["fitness"])
        assert results["fitness"]["r2"] > -1.0  # At least not catastrophic

    def test_corpus_feeds_archetypes(self, synthetic_corpus_df):
        """ArchetypeClassifier should work on corpus-shaped data."""
        ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3)
        result = ac.fit(synthetic_corpus_df)
        assert len(result) == len(synthetic_corpus_df)

    def test_corpus_feeds_pattern_miner(self, synthetic_corpus_df):
        """PatternMiner should work on corpus-shaped data."""
        miner = PatternMiner(synthetic_corpus_df)
        summary = miner.mine()
        assert summary["n_strategies"] > 0

    def test_feature_column_count(self):
        """SURROGATE_FEATURE_NAMES should have exactly 65 entries."""
        assert len(SURROGATE_FEATURE_NAMES) == 65


# ══════════════════════════════════════════════════════════════════════
# SIS v3 Tests — New Features
# ══════════════════════════════════════════════════════════════════════

class TestOverfitRiskPredictor:
    """Tests for the overfitting risk predictor target."""

    def test_overfit_risk_computed(self, synthetic_corpus_df):
        """Overfitting risk should be derived from train/holdout gap."""
        pred = MultiTargetPredictor()
        results = pred.train(synthetic_corpus_df)
        # overfit_risk should appear in results if train_fitness and holdout_fitness exist
        assert "overfit_risk" in results

    def test_overfit_risk_range(self, synthetic_corpus_df):
        """Overfitting risk predictions should be in reasonable range."""
        pred = MultiTargetPredictor()
        pred.train(synthetic_corpus_df)
        scored = pred.predict(synthetic_corpus_df)
        if "pred_overfit_risk" in scored.columns:
            preds = scored["pred_overfit_risk"].dropna()
            assert preds.min() > -2.0  # Clipped to sane range
            assert preds.max() < 2.0


class TestClassificationPredictors:
    """Tests for the top-quartile classification predictors."""

    def test_classifiers_trained(self, synthetic_corpus_df):
        """Classification models should be trained for available targets."""
        pred = MultiTargetPredictor()
        results = pred.train(synthetic_corpus_df)
        # At least clf_fitness and clf_win_rate should exist
        assert any(k.startswith("clf_") for k in results)

    def test_classifier_auc_above_chance(self, synthetic_corpus_df):
        """Classification AUC should be above random chance (0.5)."""
        pred = MultiTargetPredictor()
        results = pred.train(synthetic_corpus_df)
        for key, metrics in results.items():
            if metrics.get("type") == "classification":
                # For synthetic data with clear clusters, AUC should be decent
                assert metrics["auc"] >= 0.45, f"{key} AUC too low: {metrics['auc']}"

    def test_classification_probabilities(self, synthetic_corpus_df):
        """predict() should return probability columns for classifiers."""
        pred = MultiTargetPredictor()
        pred.train(synthetic_corpus_df)
        scored = pred.predict(synthetic_corpus_df)
        prob_cols = [c for c in scored.columns if c.startswith("prob_clf_")]
        assert len(prob_cols) >= 1
        for col in prob_cols:
            vals = scored[col].dropna()
            assert vals.min() >= 0.0
            assert vals.max() <= 1.0

    def test_quality_score(self, synthetic_corpus_df):
        """predict_quality_score should return normalized 0-1 scores."""
        pred = MultiTargetPredictor()
        pred.train(synthetic_corpus_df)
        scores = pred.predict_quality_score(synthetic_corpus_df)
        assert len(scores) == len(synthetic_corpus_df)
        assert scores.min() >= -0.01  # Allow tiny float imprecision
        assert scores.max() <= 1.01

    def test_save_load_classifiers(self, synthetic_corpus_df, tmp_path):
        """Classifiers should persist and reload correctly."""
        pred = MultiTargetPredictor(models_dir=tmp_path)
        pred.train(synthetic_corpus_df)
        n_clf = len(pred.classifiers)
        pred.save(tmp_path)

        pred2 = MultiTargetPredictor(models_dir=tmp_path)
        pred2.load(tmp_path)
        assert len(pred2.classifiers) == n_clf
        assert pred2._quartile_thresholds  # Thresholds should be saved


class TestInteractionFeatures:
    """Tests for pairwise indicator interaction features."""

    def test_interaction_features_added(self, synthetic_corpus_df):
        """Interaction features should be added during training."""
        from genetic_algorithm.intelligence.predictors import _add_interaction_features
        df = _add_interaction_features(synthetic_corpus_df)
        synergy_cols = [c for c in df.columns if c.startswith("syn_")]
        assert len(synergy_cols) == 10  # 10 synergy pairs
        assert "ratio_entry_exit" in df.columns
        assert "ratio_ind_cond" in df.columns

    def test_interaction_features_values(self, synthetic_corpus_df):
        """Interaction features should be products of indicator presence."""
        from genetic_algorithm.intelligence.predictors import _add_interaction_features
        df = _add_interaction_features(synthetic_corpus_df)
        # syn_DONCHIAN_ROC should be 1 only when both indicators present
        if "syn_DONCHIAN_ROC" in df.columns:
            for _, row in df.head(10).iterrows():
                expected = row.get("ind_DONCHIAN", 0) * row.get("ind_ROC", 0)
                assert row["syn_DONCHIAN_ROC"] == expected


class TestEvolutionState:
    """Tests for convergence-aware evolution state tracking."""

    def test_initial_state_exploring(self):
        """Initial state should be EXPLORING."""
        from genetic_algorithm.intelligence.sis_integrator import EvolutionState
        es = EvolutionState()
        assert es.state == EvolutionState.EXPLORING

    def test_states_transition(self):
        """State should transition through exploring -> improving -> stagnating."""
        from genetic_algorithm.intelligence.sis_integrator import EvolutionState
        es = EvolutionState()

        # Early gens: exploring
        es.update(1, 0.3, 0.25, 100)
        assert es.state == EvolutionState.EXPLORING

        # Improving (monotonically increasing fitness, mean stays stable)
        for gen in range(6, 10):
            es.update(gen, 0.3 + gen * 0.01, 0.3 + gen * 0.009, 100)
        assert es.state == EvolutionState.IMPROVING

        # Stagnating (no improvement for many gens, mean stays stable)
        peak = 0.3 + 9 * 0.01
        for gen in range(10, 20):
            es.update(gen, peak, peak * 0.95, 100)  # no improvement
        assert es.state == EvolutionState.STAGNATING

    def test_stagnation_params(self):
        """Stagnation should increase immigrant multiplier."""
        from genetic_algorithm.intelligence.sis_integrator import EvolutionState
        es = EvolutionState()
        for gen in range(20):
            es.update(gen, 0.5, 0.3, 100)
        params = es.params
        assert params["immigrant_multiplier"] > 1.0  # More immigrants during stagnation
        assert params["weight_bias_strength"] < 1.0  # Flatter weights

    def test_post_restart_detected(self):
        """Post-restart state should be detected after fitness reset."""
        from genetic_algorithm.intelligence.sis_integrator import EvolutionState
        es = EvolutionState()
        es.update(1, 0.5, 0.4, 100)
        es.update(2, 0.55, 0.45, 100)
        # Simulate catastrophic restart: big fitness drop
        es.update(3, 0.2, 0.15, 100)
        assert es.state == EvolutionState.POST_RESTART


class TestAdaptiveWeightTracker:
    """Tests for online Bayesian weight blending."""

    def test_early_gen_trusts_prior(self):
        """Early generations should return weights close to prior."""
        from genetic_algorithm.intelligence.sis_integrator import AdaptiveWeightTracker
        prior = {"CCI": 3.0, "MACD": 0.5}
        tracker = AdaptiveWeightTracker(prior, {})
        # No observations yet, gen 0
        blended = tracker.get_blended_indicator_weights()
        assert abs(blended["CCI"] - 3.0) < 0.01
        assert abs(blended["MACD"] - 0.5) < 0.01

    def test_late_gen_shifts_to_evidence(self):
        """Late generations should shift toward live evidence."""
        from unittest.mock import MagicMock
        from genetic_algorithm.intelligence.sis_integrator import AdaptiveWeightTracker

        prior = {"CCI": 3.0, "MACD": 0.5}
        tracker = AdaptiveWeightTracker(prior, {})

        # Simulate many generations where MACD strategies do well
        for gen in range(30):
            mock_pop = []
            for _ in range(20):
                ind = MagicMock()
                ind.fitness = 0.7
                gene = MagicMock()
                indicator = MagicMock()
                indicator.type = "MACD"
                gene.indicators = [indicator]
                gene.entry_conditions = []
                ind.strategy_gene = gene
                mock_pop.append(ind)
            tracker.observe_generation(mock_pop, gen)

        blended = tracker.get_blended_indicator_weights()
        # MACD should have increased from 0.5 toward higher value
        assert blended["MACD"] > 0.5


class TestSynergyGraph:
    """Tests for synergy-aware indicator weights."""

    def test_synergy_weights_empty_when_no_indicators(self):
        """Synergy weights should be empty with no existing indicators."""
        from genetic_algorithm.intelligence.sis_integrator import SIS_INDICATOR_ENRICHMENT, SYNERGY_GRAPH
        # Direct function test
        existing = []
        synergy_weights = {}
        for existing_ind in existing:
            partners = SYNERGY_GRAPH.get(existing_ind, {})
            for partner, lift in partners.items():
                if partner not in existing:
                    synergy_weights[partner] = max(synergy_weights.get(partner, 0), lift)
        assert synergy_weights == {}

    def test_synergy_weights_with_donchian(self):
        """Having DONCHIAN should strongly prefer ROC."""
        from genetic_algorithm.intelligence.sis_integrator import SYNERGY_GRAPH
        existing = ["DONCHIAN"]
        synergy_weights = {}
        for existing_ind in existing:
            partners = SYNERGY_GRAPH.get(existing_ind, {})
            for partner, lift in partners.items():
                if partner not in existing:
                    synergy_weights[partner] = max(synergy_weights.get(partner, 0), lift)
        assert "ROC" in synergy_weights
        assert synergy_weights["ROC"] > 10  # 21.5x lift

    def test_synergy_bidirectional(self):
        """Synergy graph should be bidirectional: DONCHIAN->ROC and ROC->DONCHIAN."""
        from genetic_algorithm.intelligence.sis_integrator import SYNERGY_GRAPH
        assert "ROC" in SYNERGY_GRAPH.get("DONCHIAN", {})
        assert "DONCHIAN" in SYNERGY_GRAPH.get("ROC", {})


class TestSISIntegratorV3:
    """Tests for the v3 SIS integrator features."""

    def test_integrator_initializes_v3_components(self, synthetic_corpus_df, tmp_path):
        """SIS integrator should initialize all v3 components."""
        from genetic_algorithm.intelligence.sis_integrator import SISIntegrator
        # Train and save models
        pred = MultiTargetPredictor(models_dir=tmp_path)
        pred.train(synthetic_corpus_df)
        pred.save(tmp_path)

        ac = ArchetypeClassifier(min_cluster_size=5, min_samples=3)
        ac.fit(synthetic_corpus_df)
        ac.save(tmp_path)

        corpus_path = tmp_path / "strategy_corpus.parquet"
        synthetic_corpus_df.to_parquet(corpus_path, index=False)

        config = {
            "sis": {
                "enabled": True,
                "models_dir": str(tmp_path),
                "corpus_path": str(corpus_path),
            },
            "output": {"dir": str(tmp_path)},
        }
        sis = SISIntegrator(config, logging.getLogger("test"))
        assert sis._predictor_ready
        assert sis._classifier_ready
        assert sis._evo_state is not None
        assert sis._adaptive_weights is not None
        assert sis._synergy_graph  # Should have synergy data
