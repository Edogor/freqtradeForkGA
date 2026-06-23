"""Tests for T4.8 — strategy embedding space + mode-collapse diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from genetic_algorithm.intelligence.embedding import (
    EmbeddingResult,
    embed_population,
    extract_tokens,
    jaccard_distance,
    population_diversity,
)


# ---------------------------------------------------------------------------
# Lightweight gene fixtures
# ---------------------------------------------------------------------------


@dataclass
class _Ind:
    type: str


@dataclass
class _Cond:
    operator: Optional[str] = None
    logic: Optional[str] = None


@dataclass
class _Gene:
    indicators: List[_Ind] = field(default_factory=list)
    entry_conditions: List[_Cond] = field(default_factory=list)
    exit_conditions: List[_Cond] = field(default_factory=list)
    short_entry_conditions: List[_Cond] = field(default_factory=list)
    short_exit_conditions: List[_Cond] = field(default_factory=list)
    informative_timeframes: List[str] = field(default_factory=list)
    can_short: bool = False


def _make(*indicators: str, ops: List[str] = None, can_short: bool = False, tfs: List[str] = None) -> _Gene:
    return _Gene(
        indicators=[_Ind(i) for i in indicators],
        entry_conditions=[_Cond(operator=op, logic="and") for op in (ops or [])],
        informative_timeframes=tfs or [],
        can_short=can_short,
    )


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


class TestExtractTokens:
    def test_empty_gene(self):
        assert extract_tokens(None) == []
        assert extract_tokens(_Gene()) == []

    def test_indicators_uppercased(self):
        toks = extract_tokens(_make("rsi", "macd"))
        assert "ind:RSI" in toks and "ind:MACD" in toks

    def test_dedup(self):
        toks = extract_tokens(_make("RSI", "RSI", "MACD"))
        assert toks.count("ind:RSI") == 1

    def test_operators_and_logic(self):
        toks = extract_tokens(_make("RSI", ops=["<", ">"]))
        assert "op:<" in toks and "op:>" in toks and "logic:and" in toks

    def test_can_short_and_tfs(self):
        toks = extract_tokens(_make("RSI", can_short=True, tfs=["1h", "4h"]))
        assert "can_short" in toks
        assert "tf:1h" in toks and "tf:4h" in toks

    def test_works_with_dict_gene(self):
        gene = {"indicators": [{"type": "RSI"}], "can_short": False}
        toks = extract_tokens(gene)
        assert "ind:RSI" in toks


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_zero(self):
        assert jaccard_distance(["a", "b"], ["a", "b"]) == 0.0

    def test_disjoint_one(self):
        assert jaccard_distance(["a"], ["b"]) == 1.0

    def test_empty_both_zero(self):
        assert jaccard_distance([], []) == 0.0

    def test_partial(self):
        d = jaccard_distance(["a", "b"], ["b", "c"])
        assert d == pytest.approx(1 - 1 / 3)


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_empty_population(self):
        r = embed_population([])
        assert r.tokens == []

    def test_returns_token_sets(self):
        pop = [_make("RSI", "MACD"), _make("EMA", "SMA")]
        r = embed_population(pop, project_2d=False)
        assert len(r.tokens) == 2
        assert "ind:RSI" in r.tokens[0]

    def test_tfidf_backend_when_sklearn_present(self):
        pytest.importorskip("sklearn")
        pop = [_make("RSI", "MACD"), _make("EMA", "SMA"), _make("RSI", "EMA")]
        r = embed_population(pop, project_2d=True)
        assert r.backend == "tfidf"
        assert r.n_features >= 1
        assert r.vectors is not None
        # 2D projection should produce one (x, y) per strategy
        assert r.coords_2d is not None
        assert len(r.coords_2d) == 3
        for x, y in r.coords_2d:
            assert isinstance(x, float) and isinstance(y, float)

    def test_handles_empty_gene_in_population(self):
        pytest.importorskip("sklearn")
        pop = [_make("RSI"), _Gene()]
        r = embed_population(pop, project_2d=False)
        # Empty gene contributes zero row, not a crash
        assert len(r.tokens) == 2
        assert r.tokens[1] == []

    def test_2d_disabled(self):
        pytest.importorskip("sklearn")
        pop = [_make("RSI"), _make("EMA"), _make("MACD")]
        r = embed_population(pop, project_2d=False)
        assert r.coords_2d is None


# ---------------------------------------------------------------------------
# Diversity diagnostics
# ---------------------------------------------------------------------------


class TestDiversity:
    def test_empty_population(self):
        r = EmbeddingResult(tokens=[])
        d = population_diversity(r)
        assert d["mean_pairwise_distance"] == 0.0

    def test_single_strategy(self):
        r = EmbeddingResult(tokens=[["ind:RSI"]])
        d = population_diversity(r)
        assert d["effective_unique_ratio"] == 1.0
        assert d["mean_pairwise_distance"] == 0.0

    def test_identical_population_collapse(self):
        # All identical → distance 0, unique-ratio 1/N, single cluster
        tokens = [["ind:RSI", "ind:MACD"]] * 5
        r = EmbeddingResult(tokens=tokens)
        d = population_diversity(r)
        assert d["mean_pairwise_distance"] == 0.0
        assert d["effective_unique_ratio"] == 0.2
        assert d["n_clusters_estimate"] == 1

    def test_disjoint_population_high_diversity(self):
        tokens = [["ind:A"], ["ind:B"], ["ind:C"], ["ind:D"]]
        r = EmbeddingResult(tokens=tokens)
        d = population_diversity(r)
        assert d["mean_pairwise_distance"] == 1.0
        assert d["effective_unique_ratio"] == 1.0
        assert d["n_clusters_estimate"] == 4

    def test_two_clear_clusters(self):
        # Two groups of two identical genes → 2 clusters
        tokens = [
            ["ind:RSI", "ind:MACD"],
            ["ind:RSI", "ind:MACD"],
            ["ind:EMA", "ind:SMA"],
            ["ind:EMA", "ind:SMA"],
        ]
        r = EmbeddingResult(tokens=tokens)
        d = population_diversity(r)
        assert d["n_clusters_estimate"] == 2

    def test_works_on_real_embedding(self):
        pytest.importorskip("sklearn")
        pop = [_make("RSI", "MACD"), _make("RSI", "MACD"), _make("EMA", "SMA")]
        r = embed_population(pop)
        d = population_diversity(r)
        assert 0.0 <= d["mean_pairwise_distance"] <= 1.0
        assert d["n_clusters_estimate"] in (2, )  # exactly two unique sets
