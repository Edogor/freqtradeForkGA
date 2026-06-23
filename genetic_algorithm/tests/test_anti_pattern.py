"""Tests for T4.3 — anti-pattern penalty hook."""

from __future__ import annotations

import pytest

from genetic_algorithm.intelligence.anti_pattern import (
    anti_pattern_multiplier,
    matched_anti_pattern_pairs,
    summarise_anti_patterns,
    would_introduce_anti_pattern,
)


GRAPH = {
    "ADX": {"MFI": 0.5, "STOCH": 0.4},
    "MFI": {"ADX": 0.5},
    "STOCH": {"ADX": 0.4},
    "ATR": {"PSAR": 0.6},
    "PSAR": {"ATR": 0.6},
}


class TestMatchedPairs:
    def test_no_indicators_returns_empty(self):
        assert matched_anti_pattern_pairs([], GRAPH) == []
        assert matched_anti_pattern_pairs(["RSI"], GRAPH) == []

    def test_no_graph_returns_empty(self):
        assert matched_anti_pattern_pairs(["ADX", "MFI"], None) == []
        assert matched_anti_pattern_pairs(["ADX", "MFI"], {}) == []

    def test_single_pair_match(self):
        pairs = matched_anti_pattern_pairs(["ADX", "MFI"], GRAPH)
        assert pairs == [("ADX", "MFI", 0.5)]

    def test_pairs_deduped(self):
        # ADX, MFI both list each other — should still be one pair.
        pairs = matched_anti_pattern_pairs(["ADX", "MFI", "STOCH"], GRAPH)
        # ADX×MFI and ADX×STOCH should both show, MFI×STOCH is not in graph
        pair_keys = {(a, b) for a, b, _ in pairs}
        assert pair_keys == {("ADX", "MFI"), ("ADX", "STOCH")}

    def test_uppercases_and_dedupes_input(self):
        pairs = matched_anti_pattern_pairs(
            ["adx", "MFI", "adx", "mfi"], GRAPH
        )
        assert len(pairs) == 1

    def test_unrelated_indicators_no_match(self):
        assert matched_anti_pattern_pairs(["RSI", "MACD", "EMA"], GRAPH) == []


class TestMultiplier:
    def test_no_match_returns_one(self):
        assert anti_pattern_multiplier(["RSI", "MACD"], GRAPH) == 1.0

    def test_no_graph_returns_one(self):
        assert anti_pattern_multiplier(["ADX", "MFI"], None) == 1.0

    def test_one_edge_reduces(self):
        # weight 0.5 * scale 0.5 = 0.25 → multiplier 0.75
        m = anti_pattern_multiplier(["ADX", "MFI"], GRAPH)
        assert m == pytest.approx(0.75)

    def test_two_edges_compound(self):
        # ADX×MFI (0.5) + ADX×STOCH (0.4) = 0.9 × scale 0.5 = 0.45 → 0.55, floored to 0.6
        m = anti_pattern_multiplier(["ADX", "MFI", "STOCH"], GRAPH)
        assert m == pytest.approx(0.6)

    def test_floor_respected(self):
        # Force many edges via custom graph
        big = {"A": {"B": 1.0, "C": 1.0, "D": 1.0}, "B": {"A": 1.0}, "C": {"A": 1.0}, "D": {"A": 1.0}}
        m = anti_pattern_multiplier(["A", "B", "C", "D"], big, floor=0.4)
        assert m == 0.4

    def test_custom_scale(self):
        m = anti_pattern_multiplier(["ADX", "MFI"], GRAPH, scale=1.0)
        # 0.5 * 1.0 = 0.5 → 0.5, floored at 0.6 by default
        assert m == pytest.approx(0.6)


class TestVeto:
    def test_no_conflict_returns_none(self):
        assert would_introduce_anti_pattern(["RSI"], "MACD", GRAPH) is None

    def test_no_graph_returns_none(self):
        assert would_introduce_anti_pattern(["ADX"], "MFI", None) is None

    def test_detects_conflict(self):
        out = would_introduce_anti_pattern(["ADX"], "MFI", GRAPH)
        assert out == ("ADX", 0.5)

    def test_picks_highest_weight_when_multiple(self):
        # PSAR conflicts with ATR (0.6); also fake an extra
        extended = dict(GRAPH)
        extended["PSAR"] = {"ATR": 0.6, "ADX": 0.3}
        extended["ADX"] = dict(GRAPH["ADX"])
        extended["ADX"]["PSAR"] = 0.3
        out = would_introduce_anti_pattern(["ATR", "ADX"], "PSAR", extended)
        assert out == ("ATR", 0.6)

    def test_case_insensitive(self):
        out = would_introduce_anti_pattern(["adx"], "mfi", GRAPH)
        assert out == ("ADX", 0.5)


class TestSummary:
    def test_summary_record_shape(self):
        rec = summarise_anti_patterns(["ADX", "MFI"], GRAPH)
        assert rec["n_pairs"] == 1
        assert rec["pairs"][0] == {"a": "ADX", "b": "MFI", "weight": 0.5}
        assert rec["multiplier"] == pytest.approx(0.75)

    def test_summary_empty_when_no_match(self):
        rec = summarise_anti_patterns(["RSI"], GRAPH)
        assert rec["multiplier"] == 1.0
        assert rec["pairs"] == []
        assert rec["n_pairs"] == 0
