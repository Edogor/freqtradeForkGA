"""Tests for T4.5 — rule-based structured failure diagnostics."""

from __future__ import annotations

import pytest

from genetic_algorithm.intelligence.diagnostics import (
    DEFAULT_THRESHOLDS,
    DiagnosticReport,
    diagnose,
    diagnose_to_text,
)


# ---------------------------------------------------------------------------
# Healthy strategy
# ---------------------------------------------------------------------------


def test_healthy_strategy_no_findings():
    report = diagnose(
        {
            "num_trades": 100,
            "max_drawdown": 0.10,
            "win_rate": 55.0,
            "sharpe_ratio": 1.5,
            "profit": 25.0,
            "avg_profit": 0.25,
            "indicator_count": 3,
            "condition_count": 3,
        }
    )
    assert report.findings == []
    assert report.score == 1.0
    assert report.summary == "healthy"


# ---------------------------------------------------------------------------
# Trading findings
# ---------------------------------------------------------------------------


class TestTrading:
    def test_no_trades_is_critical(self):
        r = diagnose({"num_trades": 0})
        assert any(f.code == "no_trades" and f.severity == "critical" for f in r.findings)

    def test_too_few_trades_is_high(self):
        r = diagnose({"num_trades": 2})
        assert any(f.code == "too_few_trades" and f.severity == "high" for f in r.findings)

    def test_low_trade_count_is_low(self):
        r = diagnose({"num_trades": 10})
        assert any(f.code == "low_trade_count" and f.severity == "low" for f in r.findings)


# ---------------------------------------------------------------------------
# Risk findings
# ---------------------------------------------------------------------------


class TestRisk:
    def test_critical_drawdown(self):
        r = diagnose({"num_trades": 100, "max_drawdown": 0.50})
        codes = {f.code for f in r.findings}
        assert "critical_drawdown" in codes
        # Should not also report high_drawdown for same dd
        assert "high_drawdown" not in codes

    def test_high_drawdown(self):
        r = diagnose({"num_trades": 100, "max_drawdown": 0.25})
        assert any(f.code == "high_drawdown" for f in r.findings)

    def test_acceptable_drawdown_no_finding(self):
        r = diagnose({"num_trades": 100, "max_drawdown": 0.10})
        codes = {f.code for f in r.findings}
        assert "high_drawdown" not in codes
        assert "critical_drawdown" not in codes


# ---------------------------------------------------------------------------
# Efficiency findings
# ---------------------------------------------------------------------------


class TestEfficiency:
    def test_negative_sharpe(self):
        r = diagnose({"num_trades": 100, "sharpe_ratio": -0.5})
        assert any(f.code == "negative_sharpe" and f.severity == "high" for f in r.findings)

    def test_weak_sharpe(self):
        r = diagnose({"num_trades": 100, "sharpe_ratio": 0.2})
        assert any(f.code == "weak_sharpe" for f in r.findings)

    def test_low_win_rate(self):
        r = diagnose({"num_trades": 100, "win_rate": 20.0})
        assert any(f.code == "low_win_rate" for f in r.findings)

    def test_win_rate_ignored_when_too_few_trades(self):
        # Below min_trades, win-rate is meaningless — should not fire
        r = diagnose({"num_trades": 2, "win_rate": 5.0})
        assert not any(f.code == "low_win_rate" for f in r.findings)

    def test_tiny_edge(self):
        r = diagnose({"num_trades": 100, "avg_profit": 0.0001})
        assert any(f.code == "tiny_edge" for f in r.findings)


# ---------------------------------------------------------------------------
# Robustness findings
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_walkforward_gap(self):
        r = diagnose({"num_trades": 100, "wf_gap": 0.3})
        assert any(f.code == "walkforward_gap" for f in r.findings)

    def test_holdout_degradation(self):
        r = diagnose({"num_trades": 100, "holdout_degradation": 0.5})
        assert any(f.code == "holdout_degradation" for f in r.findings)

    def test_cross_pair_instability(self):
        r = diagnose({"num_trades": 100, "cross_pair_cov": 2.0})
        assert any(f.code == "cross_pair_instability" for f in r.findings)


# ---------------------------------------------------------------------------
# Structure / anti-pattern hookup
# ---------------------------------------------------------------------------


class TestStructure:
    def test_excessive_complexity(self):
        r = diagnose(
            {"num_trades": 100, "indicator_count": 6, "condition_count": 5}
        )
        assert any(f.code == "excessive_complexity" for f in r.findings)

    def test_anti_pattern_finding(self):
        report = diagnose(
            {"num_trades": 100},
            anti_pattern_report={
                "n_pairs": 1,
                "multiplier": 0.75,
                "pairs": [{"a": "ADX", "b": "MFI", "weight": 0.5}],
            },
        )
        f = next(f for f in report.findings if f.code == "anti_pattern_pair")
        assert "ADX×MFI" in f.message
        assert f.detail["multiplier"] == 0.75

    def test_anti_pattern_empty_does_not_fire(self):
        report = diagnose(
            {"num_trades": 100},
            anti_pattern_report={"n_pairs": 0, "multiplier": 1.0, "pairs": []},
        )
        assert not any(f.code == "anti_pattern_pair" for f in report.findings)


# ---------------------------------------------------------------------------
# Score, ordering, rendering
# ---------------------------------------------------------------------------


class TestScoreAndRendering:
    def test_severity_ordering(self):
        r = diagnose(
            {
                "num_trades": 0,  # critical
                "max_drawdown": 0.5,  # critical too
                "win_rate": 20.0,  # but ignored: num_trades < min_trades
                "wf_gap": 0.3,  # high
            }
        )
        sev = [f.severity for f in r.findings]
        # critical findings come first
        assert sev[0] == "critical"

    def test_score_drops_with_findings(self):
        r = diagnose({"num_trades": 0})
        assert r.score < 1.0

    def test_score_floor_zero(self):
        r = diagnose(
            {
                "num_trades": 0,
                "max_drawdown": 0.9,
                "wf_gap": 0.9,
                "holdout_degradation": 0.9,
                "cross_pair_cov": 5.0,
            }
        )
        assert r.score >= 0.0
        assert r.score < 0.5

    def test_to_dict_round_trip(self):
        r = diagnose({"num_trades": 0})
        d = r.to_dict()
        assert "findings" in d and "score" in d
        assert d["findings"][0]["code"] == "no_trades"

    def test_diagnose_to_text_healthy(self):
        r = DiagnosticReport()
        assert diagnose_to_text(r) == "✓ healthy"

    def test_diagnose_to_text_renders_findings(self):
        r = diagnose({"num_trades": 0, "max_drawdown": 0.5})
        text = diagnose_to_text(r)
        assert "no_trades" in text or "critical_drawdown" in text

    def test_by_severity_filter(self):
        r = diagnose({"num_trades": 0, "max_drawdown": 0.25})
        crits = r.by_severity("critical")
        assert all(f.severity == "critical" for f in crits)
        assert len(crits) >= 1

    def test_by_category_filter(self):
        r = diagnose({"num_trades": 0, "max_drawdown": 0.5})
        cats = {f.category for f in r.findings}
        assert "trading" in cats and "risk" in cats
        assert all(f.category == "risk" for f in r.by_category("risk"))


# ---------------------------------------------------------------------------
# Threshold override
# ---------------------------------------------------------------------------


def test_threshold_override():
    # Raise min_trades to 50 so a 30-trade strategy fires too_few_trades
    r = diagnose(
        {"num_trades": 30}, thresholds={"min_trades": 50, "low_trades": 60}
    )
    assert any(f.code == "too_few_trades" for f in r.findings)


def test_default_thresholds_exposed():
    assert "min_trades" in DEFAULT_THRESHOLDS
    assert DEFAULT_THRESHOLDS["min_trades"] > 0
