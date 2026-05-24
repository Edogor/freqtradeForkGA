"""Tests for T1.2 — Out-of-Time Holdout (Locked-Box)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from genetic_algorithm.tools import holdout as hm


# ── Date parsers ──────────────────────────────────────────────────────


def test_parse_date_compact_and_dashed():
    assert hm.parse_date("20250101") == datetime(2025, 1, 1)
    assert hm.parse_date("2025-01-01") == datetime(2025, 1, 1)


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError):
        hm.parse_date("nope")
    with pytest.raises(ValueError):
        hm.parse_date("")


def test_parse_timerange_full_and_partial():
    s, e = hm.parse_timerange("20240101-20250101")
    assert s == datetime(2024, 1, 1)
    assert e == datetime(2025, 1, 1)
    assert hm.parse_timerange("20240101-") == (datetime(2024, 1, 1), None)
    assert hm.parse_timerange("-20250101") == (None, datetime(2025, 1, 1))
    assert hm.parse_timerange("") == (None, None)


def test_format_timerange_round_trip():
    s, e = datetime(2024, 1, 1), datetime(2025, 1, 1)
    assert hm.format_timerange(s, e) == "20240101-20250101"
    assert hm.format_timerange(None, e) == "-20250101"
    assert hm.format_timerange(s, None) == "20240101-"


# ── enforce_lockbox ───────────────────────────────────────────────────


def test_enforce_lockbox_disabled_is_noop():
    cfg = {"holdout": {"enabled": False},
           "backtesting": {"timerange": "20240101-20250601"}}
    result = hm.enforce_lockbox(cfg)
    assert result.applied is False
    assert cfg["backtesting"]["timerange"] == "20240101-20250601"


def test_enforce_lockbox_clips_training_timerange():
    cfg = {"holdout": {"enabled": True, "lock_start": "20250101"},
           "backtesting": {"timerange": "20240101-20250601"}}
    result = hm.enforce_lockbox(cfg)
    assert result.applied is True
    assert result.clipped_timerange == "20240101-20250101"
    assert result.holdout_timerange == "20250101-20250601"
    # Mutation visible on the config.
    assert cfg["backtesting"]["timerange"] == "20240101-20250101"


def test_enforce_lockbox_open_ended_training():
    cfg = {"holdout": {"enabled": True, "lock_start": "20250101"},
           "backtesting": {"timerange": "20240101-"}}
    result = hm.enforce_lockbox(cfg)
    assert result.clipped_timerange == "20240101-20250101"
    # Holdout end is open-ended too.
    assert result.holdout_timerange == "20250101-"


def test_enforce_lockbox_no_clip_when_already_inside():
    cfg = {"holdout": {"enabled": True, "lock_start": "20260101"},
           "backtesting": {"timerange": "20240101-20250101"}}
    result = hm.enforce_lockbox(cfg)
    assert result.applied is True
    assert cfg["backtesting"]["timerange"] == "20240101-20250101"
    assert "already inside" in result.note


def test_enforce_lockbox_missing_lock_start_warns_only():
    cfg = {"holdout": {"enabled": True},
           "backtesting": {"timerange": "20240101-20250601"}}
    result = hm.enforce_lockbox(cfg)
    assert result.applied is False
    # Training timerange must not be touched if lock_start is missing.
    assert cfg["backtesting"]["timerange"] == "20240101-20250601"


def test_enforce_lockbox_invalid_lock_start():
    cfg = {"holdout": {"enabled": True, "lock_start": "bogus"},
           "backtesting": {"timerange": "20240101-20250601"}}
    result = hm.enforce_lockbox(cfg)
    assert result.applied is False
    assert cfg["backtesting"]["timerange"] == "20240101-20250601"


def test_check_lockbox_does_not_mutate():
    cfg = {"holdout": {"enabled": True, "lock_start": "20250101"},
           "backtesting": {"timerange": "20240101-20250601"}}
    info = hm.check_lockbox(cfg)
    assert info["clipped_training_timerange"] == "20240101-20250101"
    # Original untouched.
    assert cfg["backtesting"]["timerange"] == "20240101-20250601"


# ── run_holdout (mocked backtester) ───────────────────────────────────


def _write_hof_file(tmp_path: Path, *, with_code: bool = True) -> Path:
    pinned = {"strategy_code": "class Pinned: pass",
              "code_hash": "abc", "code_pinned_version": "1.0"} if with_code else {}
    entries = [
        {
            "id": "e1",
            "fitness": 0.9,
            "metrics": {"profit": 20.0},
            "strategy_gene": {"foo": "bar"},
            **pinned,
        },
        {
            "id": "e2",
            "fitness": 0.4,
            "metrics": {"profit": -5.0},
            "strategy_gene": {"baz": "qux"},
            **pinned,
        },
    ]
    hof = tmp_path / "hall_of_fame.json"
    hof.write_text(json.dumps({"version": 1, "entries": entries}))
    return hof


class _StubBacktestResult:
    success = True
    error_message = None

    def __init__(self, profit, sharpe=1.0, dd=5.0, trades=20, winrate=0.55):
        self.profit_percent = profit
        self.sharpe_ratio = sharpe
        self.max_drawdown = dd
        self.total_trades = trades
        self.win_rate = winrate


def test_run_holdout_uses_pinned_code_when_available(tmp_path):
    hof = _write_hof_file(tmp_path, with_code=True)

    class _FakeBT:
        def __init__(self, _cfg):
            pass

        def backtest(self, **kwargs):
            # Returns negative for first entry, positive for second — random.
            return _StubBacktestResult(profit=12.0 if "e1" in kwargs["strategy_name"] else 3.0)

    with patch("genetic_algorithm.evaluation.direct_backtester.DirectBacktester", _FakeBT):
        report = hm.run_holdout(hof, "20250101-20250601", config={}, top_n=2)

    assert len(report.scores) == 2
    e1 = next(s for s in report.scores if s.entry_id == "e1")
    assert e1.code_was_regenerated is False  # pinned was used
    assert e1.profit_holdout == 12.0
    # overfit ratio = (train - holdout)/|train| = (20-12)/20 = 0.4
    assert e1.overfit_ratio == pytest.approx(0.4)
    assert report.summary["total"] == 2
    assert report.summary["profitable_on_holdout"] == 2


def test_run_holdout_regenerates_code_when_not_pinned(tmp_path):
    hof = _write_hof_file(tmp_path, with_code=False)

    class _FakeBT:
        def __init__(self, _cfg):
            pass

        def backtest(self, **kwargs):
            return _StubBacktestResult(profit=1.0)

    class _FakeGen:
        def __init__(self, _cfg):
            pass

        def generate_strategy_code(self, _gene):
            return "class Regen: pass"

    with patch("genetic_algorithm.evaluation.direct_backtester.DirectBacktester", _FakeBT), \
         patch("genetic_algorithm.genome.codegen.StrategyGenerator", _FakeGen):
        report = hm.run_holdout(hof, "20250101-20250601", config={}, top_n=2)

    for s in report.scores:
        assert s.code_was_regenerated is True


def test_run_holdout_handles_backtest_failure(tmp_path):
    hof = _write_hof_file(tmp_path)

    class _FakeBT:
        def __init__(self, _cfg):
            pass

        def backtest(self, **_):
            raise RuntimeError("boom")

    with patch("genetic_algorithm.evaluation.direct_backtester.DirectBacktester", _FakeBT):
        report = hm.run_holdout(hof, "20250101-", config={}, top_n=2)

    assert all(s.error == "boom" for s in report.scores)
    assert report.summary["profitable_on_holdout"] == 0


def test_write_report_round_trip(tmp_path):
    report = hm.HoldoutReport(
        hof_file="/tmp/x", holdout_timerange="20250101-20250601",
        scores=[hm.HoldoutScore(entry_id="e1", profit_holdout=10.0)],
        summary={"total": 1}, generated_at=1.0,
    )
    path = hm.write_report(report, tmp_path / "report.json")
    data = json.loads(path.read_text())
    assert data["holdout_timerange"] == "20250101-20250601"
    assert data["scores"][0]["entry_id"] == "e1"


def test_write_report_csv(tmp_path):
    report = hm.HoldoutReport(
        hof_file="/tmp/x", holdout_timerange="20250101-",
        scores=[hm.HoldoutScore(entry_id="e1", profit_holdout=10.0,
                                overfit_ratio=0.2)],
    )
    path = hm.write_report_csv(report, tmp_path / "report.csv")
    text = path.read_text()
    assert "entry_id" in text
    assert "e1" in text


def test_default_report_path_is_next_to_hof(tmp_path):
    hof = tmp_path / "subdir" / "hall_of_fame.json"
    hof.parent.mkdir()
    hof.write_text("{}")
    assert hm.default_report_path(hof) == hof.parent / "holdout_report.json"
