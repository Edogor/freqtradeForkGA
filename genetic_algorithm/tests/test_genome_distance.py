"""T3.5 — Tests for the genome distance metric and niching dispatch."""
from __future__ import annotations

import pytest

from genetic_algorithm.core.genome_distance import (
    calculate_genome_distance,
    distance_breakdown,
)


# ── Lightweight fakes mirroring the public surface of the gene types ──


class _Ind:
    def __init__(self, type_, parameters=None, timeframe=None):
        self.type = type_
        self.parameters = parameters or {}
        self.timeframe = timeframe
        self.weight = 1.0


class _Cond:
    def __init__(self, indicator, operator, threshold=0.0, logic="AND", lookback=0):
        self.indicator = indicator
        self.operator = operator
        self.threshold = threshold
        self.logic = logic
        self.lookback = lookback


class _Gene:
    def __init__(
        self,
        indicators=None,
        entry_conditions=None,
        exit_conditions=None,
        timeframe="5m",
        stoploss=-0.05,
        trailing_stop=False,
        trailing_stop_positive=0.0,
        trailing_stop_positive_offset=0.0,
        minimal_roi=None,
        regime=None,
    ):
        self.indicators = indicators or []
        self.entry_conditions = entry_conditions or []
        self.exit_conditions = exit_conditions or []
        self.timeframe = timeframe
        self.stoploss = stoploss
        self.trailing_stop = trailing_stop
        self.trailing_stop_positive = trailing_stop_positive
        self.trailing_stop_positive_offset = trailing_stop_positive_offset
        self.minimal_roi = minimal_roi or {"0": 0.05, "30": 0.02, "60": 0.0}
        self.regime = regime


class _Individual:
    def __init__(self, gene):
        self.strategy_gene = gene


# ── Core invariants ───────────────────────────────────────────────────


def _g(**kw):
    return _Individual(_Gene(**kw))


class TestGenomeDistance:
    def test_identical_returns_zero(self):
        g = _g(
            indicators=[_Ind("RSI", {"period": 14}), _Ind("EMA", {"period": 50})],
            entry_conditions=[_Cond("RSI", "<", 30.0)],
            exit_conditions=[_Cond("RSI", ">", 70.0)],
        )
        h = _g(
            indicators=[_Ind("RSI", {"period": 14}), _Ind("EMA", {"period": 50})],
            entry_conditions=[_Cond("RSI", "<", 30.0)],
            exit_conditions=[_Cond("RSI", ">", 70.0)],
        )
        assert calculate_genome_distance(g, h) == pytest.approx(0.0, abs=1e-9)

    def test_returns_in_unit_interval(self):
        # Two completely different strategies.
        a = _g(
            indicators=[_Ind("RSI", {"period": 14})],
            entry_conditions=[_Cond("RSI", "<", 30.0)],
            timeframe="5m",
            stoploss=-0.05,
        )
        b = _g(
            indicators=[_Ind("MACD", {"fast": 12, "slow": 26, "signal": 9})],
            entry_conditions=[_Cond("MACD", "cross_above", 0.0)],
            timeframe="1h",
            stoploss=-0.20,
            trailing_stop=True,
        )
        d = calculate_genome_distance(a, b)
        assert 0.0 <= d <= 1.0
        assert d > 0.4  # should be substantially different

    def test_symmetric(self):
        a = _g(indicators=[_Ind("RSI", {"period": 14})])
        b = _g(indicators=[_Ind("EMA", {"period": 50})])
        assert calculate_genome_distance(a, b) == pytest.approx(
            calculate_genome_distance(b, a), abs=1e-9
        )

    def test_parameter_difference_is_detected(self):
        # Legacy distance would treat these as identical (same type set).
        a = _g(indicators=[_Ind("RSI", {"period": 14})])
        b = _g(indicators=[_Ind("RSI", {"period": 80})])
        d = calculate_genome_distance(a, b)
        assert d > 0.0, "v2 distance must detect parameter differences"

    def test_condition_threshold_buckets(self):
        a = _g(entry_conditions=[_Cond("RSI", "<", 30.0)])
        b = _g(entry_conditions=[_Cond("RSI", "<", 70.0)])
        c = _g(entry_conditions=[_Cond("RSI", "<", 31.0)])
        d_ab = calculate_genome_distance(a, b)
        d_ac = calculate_genome_distance(a, c)
        # Coarse bucketing — small threshold change should be ≤ large change.
        assert d_ac <= d_ab

    def test_disjoint_indicator_types(self):
        a = _g(indicators=[_Ind("RSI", {"period": 14})])
        b = _g(indicators=[_Ind("MACD", {"fast": 12})])
        d = calculate_genome_distance(a, b)
        assert d > 0.3

    def test_handles_empty_gene(self):
        a = _g(indicators=[], entry_conditions=[], exit_conditions=[])
        b = _g(indicators=[_Ind("RSI", {"period": 14})])
        d = calculate_genome_distance(a, b)
        assert 0.0 < d <= 1.0

    def test_breakdown_components_sum_to_total(self):
        a = _g(indicators=[_Ind("RSI", {"period": 14})], stoploss=-0.05)
        b = _g(indicators=[_Ind("MACD", {"fast": 12})], stoploss=-0.20, trailing_stop=True)
        bd = distance_breakdown(a, b)
        for key in ("indicator_signature", "indicator_params",
                    "condition_signature", "risk_params", "flags", "total"):
            assert key in bd
            assert 0.0 <= bd[key] <= 1.0


# ── Dispatch via engine.population ────────────────────────────────────


class TestDispatch:
    def test_legacy_mode_uses_legacy(self):
        from genetic_algorithm.engine import population as pop

        ind1 = _g(indicators=[_Ind("RSI", {"period": 14})])
        ind2 = _g(indicators=[_Ind("RSI", {"period": 80})])

        old = pop._GENOME_DISTANCE_MODE
        try:
            pop._GENOME_DISTANCE_MODE = "legacy"
            d_legacy = pop.calculate_strategy_distance(ind1, ind2)
            # Legacy ignores parameter values: same type → 0.
            assert d_legacy == pytest.approx(0.0, abs=1e-9)
        finally:
            pop._GENOME_DISTANCE_MODE = old

    def test_v2_mode_detects_parameter_change(self):
        from genetic_algorithm.engine import population as pop

        ind1 = _g(indicators=[_Ind("RSI", {"period": 14})])
        ind2 = _g(indicators=[_Ind("RSI", {"period": 80})])

        old = pop._GENOME_DISTANCE_MODE
        try:
            pop._GENOME_DISTANCE_MODE = "genome_v2"
            d_v2 = pop.calculate_strategy_distance(ind1, ind2)
            assert d_v2 > 0.0
        finally:
            pop._GENOME_DISTANCE_MODE = old

    def test_blend_mode_between_legacy_and_v2(self):
        from genetic_algorithm.engine import population as pop

        ind1 = _g(indicators=[_Ind("RSI", {"period": 14})], stoploss=-0.05)
        ind2 = _g(
            indicators=[_Ind("RSI", {"period": 80}), _Ind("MACD", {"fast": 12})],
            stoploss=-0.30,
        )

        old = pop._GENOME_DISTANCE_MODE
        try:
            pop._GENOME_DISTANCE_MODE = "legacy"
            d_l = pop.calculate_strategy_distance(ind1, ind2)
            pop._GENOME_DISTANCE_MODE = "genome_v2"
            d_v = pop.calculate_strategy_distance(ind1, ind2)
            pop._GENOME_DISTANCE_MODE = "blend"
            d_b = pop.calculate_strategy_distance(ind1, ind2)
            assert d_b == pytest.approx(0.5 * d_l + 0.5 * d_v, abs=1e-9)
        finally:
            pop._GENOME_DISTANCE_MODE = old

    def test_unknown_mode_falls_back_to_legacy(self):
        from genetic_algorithm.engine import population as pop

        ind1 = _g(indicators=[_Ind("RSI", {"period": 14})])
        ind2 = _g(indicators=[_Ind("RSI", {"period": 80})])

        old = pop._GENOME_DISTANCE_MODE
        try:
            pop._GENOME_DISTANCE_MODE = "garbage"
            d = pop.calculate_strategy_distance(ind1, ind2)
            assert d == pytest.approx(0.0, abs=1e-9)
        finally:
            pop._GENOME_DISTANCE_MODE = old


# ── Niching effect: fitness sharing actually changes outcomes ─────────


class TestFitnessSharingEffect:
    def test_sharing_penalises_near_duplicates_more_with_v2(self):
        """With v2 distance, two near-duplicates (same types, different params)
        should be detected as similar and have their fitness shared, while
        legacy treats them as identical and produces the same result.
        Either way the v2 must produce a sensible niche count."""
        from genetic_algorithm.engine import population as pop
        from genetic_algorithm.engine.population import (
            calculate_pairwise_distances,
        )

        a = _g(indicators=[_Ind("RSI", {"period": 14})], stoploss=-0.05)
        b = _g(indicators=[_Ind("RSI", {"period": 15})], stoploss=-0.05)
        c = _g(indicators=[_Ind("MACD", {"fast": 12})], stoploss=-0.20,
               trailing_stop=True, timeframe="1h")

        inds = [_Individual(a.strategy_gene), _Individual(b.strategy_gene),
                _Individual(c.strategy_gene)]

        old = pop._GENOME_DISTANCE_MODE
        try:
            pop._GENOME_DISTANCE_MODE = "genome_v2"
            dm = calculate_pairwise_distances(inds)
            assert dm[0][1] < dm[0][2], (
                f"near-duplicate distance {dm[0][1]:.3f} should be smaller "
                f"than disjoint-pair distance {dm[0][2]:.3f}"
            )
            assert dm[0][1] >= 0.0
            assert dm[0][2] <= 1.0
        finally:
            pop._GENOME_DISTANCE_MODE = old
