"""Tests for T1.1 — Hall of Fame Replay tool.

Heavy parts of the GA stack (FreqTrade backtester, StrategyGenerator) are
mocked so the tests stay fast and don't require market data.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

from genetic_algorithm.tools import hof_replay as hr


# ----- Fixtures ---------------------------------------------------------

_PINNED_CODE = "class GAStrategy_Pinned:\n    pass\n"

_HOF_DATA = {
    "version": 1,
    "entries": [
        {
            "id": "hof_aaa111",
            "strategy_gene": {"indicators": [{"type": "RSI"}]},
            "fitness": 2.5,
            "metrics": {"profit": 30.0, "sharpe_ratio": 1.8, "max_drawdown": 8.0},
            "generation_found": 12,
            "run_timestamp": 0,
            "run_id": "wave_test_A",
            "strategy_code": _PINNED_CODE,
            "code_hash": "deadbeef" * 8,
            "code_pinned_version": "v1",
        },
        {
            "id": "hof_bbb222",
            "strategy_gene": {"indicators": [{"type": "MACD"}]},
            "fitness": 1.5,
            "metrics": {"profit": 12.0, "sharpe_ratio": 1.0, "max_drawdown": 15.0},
            "generation_found": 7,
            "run_timestamp": 0,
            "run_id": "wave_test_A",
            # no pinned code -> must regenerate
        },
    ],
}


@dataclass
class _FakeBTResult:
    success: bool = True
    strategy_name: str = "x"
    total_profit: float = 0.0
    profit_percent: float = 25.0
    total_trades: int = 42
    win_rate: float = 0.55
    max_drawdown: float = 9.0
    sharpe_ratio: float = 1.7
    error_message: str | None = None


class _FakeBacktester:
    def __init__(self, config):
        self.config = config
        self.backtest_config = config.get("backtesting", {})
        self.calls = []

    def backtest_strategy(self, strategy_code, strategy_name, **kw):
        # Record what was actually used so we can assert overrides took effect.
        self.calls.append({
            "strategy_name": strategy_name,
            "code_len": len(strategy_code),
            "fee": self.backtest_config.get("fee"),
            "slippage": self.backtest_config.get("slippage"),
            "max_open_trades": kw.get("strategy_max_open_trades"),
            "timerange_override": kw.get("timerange_override"),
            "pairs_override": kw.get("pairs_override"),
        })
        return _FakeBTResult()


def _write_hof_file(tmp: Path, data=None) -> Path:
    p = tmp / "hall_of_fame.json"
    p.write_text(json.dumps(data or _HOF_DATA))
    return p


_BASE_CONFIG = {
    "backtesting": {
        "fee": 0.001, "slippage": 0.0005,
        "max_open_trades": 5,
        "pairs": ["BTC/USDT"],
        "timerange": "20240101-20240601",
    },
}


# ----- Tests ------------------------------------------------------------

class TestSelectEntries(unittest.TestCase):
    def test_by_top_n(self):
        out = hr._select_entries(_HOF_DATA, top_n=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], "hof_aaa111")

    def test_by_ids(self):
        out = hr._select_entries(_HOF_DATA, entry_ids=["hof_bbb222"])
        self.assertEqual([e["id"] for e in out], ["hof_bbb222"])

    def test_by_min_fitness(self):
        out = hr._select_entries(_HOF_DATA, min_fitness=2.0)
        self.assertEqual([e["id"] for e in out], ["hof_aaa111"])


class TestBuildReplayConfig(unittest.TestCase):
    def test_overrides_applied(self):
        cfg = hr._build_replay_config(_BASE_CONFIG, fee=0.003, slippage=0.002,
                                       max_open_trades=10, timerange="20240601-20240801",
                                       pairs=["ETH/USDT"])
        self.assertEqual(cfg["backtesting"]["fee"], 0.003)
        self.assertEqual(cfg["backtesting"]["slippage"], 0.002)
        self.assertEqual(cfg["backtesting"]["max_open_trades"], 10)
        self.assertEqual(cfg["backtesting"]["pairs"], ["ETH/USDT"])
        self.assertFalse(cfg["backtesting"]["enable_cache"])
        # Original untouched
        self.assertEqual(_BASE_CONFIG["backtesting"]["fee"], 0.001)


class TestReplayEntry(unittest.TestCase):
    def test_uses_pinned_code(self):
        bt = _FakeBacktester(_BASE_CONFIG)
        r = hr.replay_entry(
            _HOF_DATA["entries"][0], _BASE_CONFIG, bt,
            fee=0.002, slippage=0.001,
        )
        self.assertTrue(r.success)
        self.assertFalse(r.code_was_regenerated)
        self.assertEqual(r.profit_replay, 25.0)
        self.assertAlmostEqual(r.profit_diff, 25.0 - 30.0)
        self.assertEqual(r.fee, 0.002)
        # The fake backtester saw the overridden fee
        self.assertEqual(bt.calls[-1]["fee"], 0.002)
        # Code hash recorded
        self.assertIsNotNone(r.code_hash_replay)

    def test_regenerates_when_not_pinned(self):
        bt = _FakeBacktester(_BASE_CONFIG)
        # Patch the codegen import sites used by _resolve_strategy_code.
        import sys, types
        fake_codegen_mod = types.ModuleType("genetic_algorithm.genome.codegen")
        class _FakeGen:
            def __init__(self, cfg): pass
            def generate_strategy_code(self, gene):
                return "class Regenerated: pass\n"
        fake_codegen_mod.StrategyGenerator = _FakeGen  # type: ignore
        sys.modules["genetic_algorithm.genome.codegen"] = fake_codegen_mod

        fake_sg_mod = types.ModuleType("genetic_algorithm.core.strategy_gene")
        class _FakeGene:
            @classmethod
            def from_dict(cls, d): return object()
        fake_sg_mod.StrategyGene = _FakeGene  # type: ignore
        sys.modules["genetic_algorithm.core.strategy_gene"] = fake_sg_mod

        try:
            r = hr.replay_entry(
                _HOF_DATA["entries"][1], _BASE_CONFIG, bt,
                fee=0.0015, slippage=0.0005,
            )
        finally:
            sys.modules.pop("genetic_algorithm.genome.codegen", None)
            sys.modules.pop("genetic_algorithm.core.strategy_gene", None)
        self.assertTrue(r.success)
        self.assertTrue(r.code_was_regenerated)


class TestStressEntry(unittest.TestCase):
    def test_grid_enumerated(self):
        bt = _FakeBacktester(_BASE_CONFIG)
        rep = hr.stress_entry(
            _HOF_DATA["entries"][0], _BASE_CONFIG, bt,
            fees=[0.001, 0.002], slippages=[0.0, 0.001],
            max_open_trades_grid=[3, 5],
        )
        self.assertEqual(len(rep.grid_results), 2 * 2 * 2)  # 8 points
        self.assertEqual(rep.entry_id, "hof_aaa111")
        # Survival = all positive (FakeBacktester returns +25 always)
        self.assertEqual(rep.survival_rate, 1.0)
        self.assertGreater(rep.robustness_score, 0)


class TestReplayHofFile(unittest.TestCase):
    def test_top_n_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            hof_path = _write_hof_file(tmp)
            bt = _FakeBacktester(_BASE_CONFIG)
            results = hr.replay_hof_file(
                hof_path, _BASE_CONFIG,
                fee=0.0015, top_n=1, backtester=bt,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].entry_id, "hof_aaa111")
            # CSV writer
            out = tmp / "replay.csv"
            hr.write_replay_csv(results, out)
            self.assertTrue(out.exists())
            content = out.read_text()
            self.assertIn("entry_id", content)
            self.assertIn("hof_aaa111", content)


class TestListAndShow(unittest.TestCase):
    def test_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hof_path = _write_hof_file(Path(tmpdir))
            txt = hr.list_hof_entries(hof_path, limit=10)
            self.assertIn("hof_aaa111", txt)
            self.assertIn("yes", txt)  # pinned column for entry 1
            self.assertIn(" no", txt)  # not-pinned for entry 2

    def test_show_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hof_path = _write_hof_file(Path(tmpdir))
            txt = hr.show_hof_entry(hof_path, "hof_aaa111")
            self.assertIn("hof_aaa111", txt)
            self.assertIn("pinned", txt)
            # Code body is replaced with size annotation
            self.assertNotIn("class GAStrategy_Pinned", txt)

    def test_show_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hof_path = _write_hof_file(Path(tmpdir))
            txt = hr.show_hof_entry(hof_path, "hof_zzz999")
            self.assertIn("not found", txt)


class TestRobustness(unittest.TestCase):
    def test_uniform_high_robustness(self):
        score = hr._compute_robustness([10, 10, 10, 10])
        self.assertAlmostEqual(score, 10.0)

    def test_swings_kill_robustness(self):
        score = hr._compute_robustness([10, -10, 10, -10])
        self.assertEqual(score, 0.0)

    def test_empty(self):
        self.assertEqual(hr._compute_robustness([]), 0.0)


if __name__ == "__main__":
    unittest.main()
