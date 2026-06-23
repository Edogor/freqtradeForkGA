"""Tests for T4.1 — data-driven SIS prior rebuilder + loader."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from genetic_algorithm.intelligence.prior_rebuilder import (
    IndicatorStat,
    RebuildResult,
    _shrink_lift,
    _split_indicators,
    infer_tier,
    rebuild_from_csv,
    rebuild_from_rows,
    write_result,
)
from genetic_algorithm.intelligence.prior_loader import load_data_driven_priors


# ---------------------------------------------------------------------------
# Indicator parsing
# ---------------------------------------------------------------------------


class TestSplitIndicators:
    def test_empty_returns_empty_list(self):
        assert _split_indicators(None) == []
        assert _split_indicators("") == []
        assert _split_indicators(42) == []

    def test_splits_and_uppercases(self):
        assert _split_indicators("rsi|MACD|adx") == ["RSI", "MACD", "ADX"]

    def test_deduplicates(self):
        assert _split_indicators("RSI|RSI|MACD") == ["RSI", "MACD"]


# ---------------------------------------------------------------------------
# Tier inference
# ---------------------------------------------------------------------------


class TestInferTier:
    def test_explicit_tier_wins(self):
        assert infer_tier({"tier": 1}) == 1
        assert infer_tier({"tier": "3"}) == 3

    def test_missing_fitness_is_tier3(self):
        assert infer_tier({}) == 3

    def test_high_fitness_high_holdout_is_t1(self):
        assert (
            infer_tier({"fitness": 0.8, "holdout_fitness": 0.5}) == 1
        )

    def test_high_fitness_no_holdout_optimistic_t1(self):
        # No validation columns present at all → optimistic T1
        assert infer_tier({"fitness": 0.8}) == 1

    def test_high_fitness_bad_holdout_demotes_to_t2(self):
        assert (
            infer_tier({"fitness": 0.8, "holdout_fitness": 0.1}) == 2
        )

    def test_mid_fitness_is_t2(self):
        assert infer_tier({"fitness": 0.5}) == 2

    def test_low_fitness_is_t3(self):
        assert infer_tier({"fitness": 0.2}) == 3


# ---------------------------------------------------------------------------
# Shrinkage math
# ---------------------------------------------------------------------------


class TestShrinkage:
    def test_zero_support_returns_neutral(self):
        # 0 tier1 of 0 with base 0.1 — posterior collapses to prior → ratio 1
        assert _shrink_lift(0, 0, 0.1) == pytest.approx(1.0)

    def test_high_support_dominates_prior(self):
        # 9 of 10 with base 0.1 → lift much > 1
        out = _shrink_lift(9, 10, 0.1, prior_strength=5.0)
        assert out > 3.0

    def test_small_support_shrinks_toward_one(self):
        # 1 of 1 with base 0.1 → without shrinkage lift would be 10
        out = _shrink_lift(1, 1, 0.1, prior_strength=20.0)
        assert out < 3.0  # heavy shrinkage


# ---------------------------------------------------------------------------
# Rebuilder end-to-end
# ---------------------------------------------------------------------------


def _make_rows():
    """Synthetic corpus with a clear signal:
    - GOOD indicator → 8/10 T1
    - BAD  indicator → 0/10 T1, 8/10 T3
    - NEUTRAL → matches base rate
    Also: pair GOOD+SUPPORTING → strong synergy.
           pair BAD+TRAP → strong anti-pattern.
    """
    rows = []
    # GOOD: 10 strategies, 8 are T1
    for i in range(8):
        rows.append({"indicators": "GOOD|SUPPORTING", "fitness": 0.9, "holdout_fitness": 0.7})
    for i in range(2):
        rows.append({"indicators": "GOOD", "fitness": 0.3})

    # BAD+TRAP: 40 strategies, all T3 (zero T1) → strong anti-pattern signal
    for i in range(40):
        rows.append({"indicators": "BAD|TRAP", "fitness": 0.1})
    for i in range(2):
        rows.append({"indicators": "BAD", "fitness": 0.4})

    # NEUTRAL: 10 strategies, mix
    for i in range(3):
        rows.append({"indicators": "NEUTRAL", "fitness": 0.8, "holdout_fitness": 0.5})
    for i in range(7):
        rows.append({"indicators": "NEUTRAL", "fitness": 0.3})

    return rows


class TestRebuilderEndToEnd:
    def test_good_indicator_has_high_weight(self):
        result = rebuild_from_rows(_make_rows(), min_support=2)
        assert "GOOD" in result.indicator_weights
        assert "BAD" in result.indicator_weights
        assert result.indicator_weights["GOOD"] > result.indicator_weights["BAD"]
        assert result.indicator_weights["GOOD"] > 1.0

    def test_bad_indicator_has_low_weight(self):
        result = rebuild_from_rows(_make_rows(), min_support=2)
        assert result.indicator_weights["BAD"] < 1.0

    def test_weights_respect_bounds(self):
        result = rebuild_from_rows(_make_rows(), min_support=2, weight_lo=0.3, weight_hi=2.0)
        for w in result.indicator_weights.values():
            assert 0.3 <= w <= 2.0

    def test_min_support_filters_rare(self):
        result = rebuild_from_rows(_make_rows(), min_support=1000)
        # No indicator has 1000+ rows → empty
        assert result.indicator_weights == {}

    def test_synergy_graph_contains_good_pair(self):
        result = rebuild_from_rows(_make_rows(), min_support=2)
        assert "GOOD" in result.synergy_graph
        assert "SUPPORTING" in result.synergy_graph.get("GOOD", {})

    def test_anti_patterns_contains_bad_pair(self):
        result = rebuild_from_rows(_make_rows(), min_support=2)
        assert "BAD" in result.anti_patterns
        assert "TRAP" in result.anti_patterns.get("BAD", {})

    def test_metadata_populated(self):
        result = rebuild_from_rows(_make_rows(), min_support=2)
        assert result.metadata["n_total"] == 62
        assert result.metadata["n_t1"] >= 8

    def test_empty_corpus_returns_empty_result(self):
        result = rebuild_from_rows([], min_support=2)
        assert result.indicator_weights == {}
        assert result.metadata["n_total"] == 0


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------


class TestCsvRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        csv_path = tmp_path / "strategies.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["indicators", "fitness", "holdout_fitness"]
            )
            writer.writeheader()
            for row in _make_rows():
                writer.writerow(
                    {
                        "indicators": row.get("indicators", ""),
                        "fitness": row.get("fitness", ""),
                        "holdout_fitness": row.get("holdout_fitness", ""),
                    }
                )
        result = rebuild_from_csv(str(csv_path), min_support=2)
        assert result.indicator_weights["GOOD"] > result.indicator_weights["BAD"]

    def test_write_json_when_no_yaml(self, tmp_path: Path):
        result = rebuild_from_rows(_make_rows(), min_support=2)
        out = tmp_path / "priors.json"
        write_result(result, str(out))
        assert out.exists()
        payload = json.loads(out.read_text())
        assert "indicator_enrichment" in payload
        assert payload["indicator_enrichment"]["GOOD"] > 1.0


# ---------------------------------------------------------------------------
# Loader merge behavior
# ---------------------------------------------------------------------------


class TestLoader:
    def test_no_path_returns_hardcoded(self):
        base_ind = {"RSI": 1.0, "CCI": 1.5}
        base_op = {"<": 1.0}
        base_syn = {"RSI": {"STOCH": 3.5}}
        ind, op, syn, anti = load_data_driven_priors(
            None, base_indicator=base_ind, base_operator=base_op, base_synergy=base_syn
        )
        assert ind == base_ind
        assert op == base_op
        assert syn == base_syn
        assert anti == {}

    def test_missing_file_returns_hardcoded(self, tmp_path: Path):
        ind, op, syn, anti = load_data_driven_priors(
            str(tmp_path / "missing.yaml"),
            base_indicator={"RSI": 1.0},
            base_operator={},
            base_synergy={},
        )
        assert ind == {"RSI": 1.0}
        assert anti == {}

    def test_partial_override(self, tmp_path: Path):
        payload = {
            "indicator_enrichment": {"CCI": 0.5},  # was 1.5
            "anti_patterns": {"BAD": {"TRAP": 0.8}},
        }
        path = tmp_path / "priors.json"
        path.write_text(json.dumps(payload))
        ind, op, syn, anti = load_data_driven_priors(
            str(path),
            base_indicator={"RSI": 1.0, "CCI": 1.5},
            base_operator={},
            base_synergy={"RSI": {"STOCH": 3.5}},
        )
        assert ind["CCI"] == 0.5  # overridden
        assert ind["RSI"] == 1.0  # untouched
        assert syn["RSI"]["STOCH"] == 3.5  # untouched
        assert anti["BAD"]["TRAP"] == 0.8

    def test_corrupt_file_falls_back_safely(self, tmp_path: Path):
        path = tmp_path / "garbage.json"
        path.write_text("{not json")
        ind, op, syn, anti = load_data_driven_priors(
            str(path),
            base_indicator={"RSI": 1.0},
            base_operator={},
            base_synergy={},
        )
        assert ind == {"RSI": 1.0}
        assert anti == {}

    def test_real_generated_priors_load(self):
        """The JSON priors we ship at intelligence/priors/ must load cleanly
        and override at least a few hardcoded indicator weights."""
        from genetic_algorithm.intelligence.sis_integrator import (
            SIS_INDICATOR_ENRICHMENT,
            SIS_OPERATOR_ENRICHMENT,
            SYNERGY_GRAPH,
        )
        repo_root = Path(__file__).resolve().parents[2]
        priors = repo_root / "genetic_algorithm" / "intelligence" / "priors" / "data_driven_enrichment.json"
        if not priors.exists():
            pytest.skip("data_driven_enrichment.json not generated")
        ind, op, syn, anti = load_data_driven_priors(
            str(priors),
            base_indicator=dict(SIS_INDICATOR_ENRICHMENT),
            base_operator=dict(SIS_OPERATOR_ENRICHMENT),
            base_synergy=dict(SYNERGY_GRAPH),
        )
        # At least one indicator should be overridden by data-driven value
        overridden = sum(
            1 for k, v in ind.items()
            if k in SIS_INDICATOR_ENRICHMENT and v != SIS_INDICATOR_ENRICHMENT[k]
        )
        assert overridden >= 1
        # Anti-patterns should be non-empty for the real corpus
        assert anti  # at least one src
