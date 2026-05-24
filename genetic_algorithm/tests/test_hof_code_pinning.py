"""Tests for T1.5 — Hall of Fame strategy-code pinning.

Verifies that:
  1. Legacy HoF entries (no code) still round-trip.
  2. New entries store ``strategy_code`` + ``code_hash`` + version.
  3. ``pin_codes_for_all_entries`` backfills legacy entries.
  4. ``get_summary`` reports the pinned-entry count.

The strategy generator is mocked so the test does not depend on the heavy
``StrategyGenerator``/freqtrade import chain.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from genetic_algorithm.engine.hall_of_fame import HallOfFame, HallOfFameEntry


_SAMPLE_GENE = {
    "indicators": [{"type": "RSI", "parameters": {"period": 14}, "timeframe": "5m"}],
    "entry_conditions": [{"indicator": "RSI_0", "operator": "<", "threshold": 30.0}],
    "exit_conditions": [{"indicator": "RSI_0", "operator": ">", "threshold": 70.0}],
    "timeframe": "5m",
    "stoploss": -0.05,
    "roi_table": {"0": 0.05},
}


def _make_individual(fitness: float = 1.23, gene_dict=None):
    gene_dict = gene_dict or _SAMPLE_GENE
    ind = MagicMock()
    ind.evaluated = True
    ind.fitness = fitness
    ind.raw_fitness = fitness
    ind.metrics = {"profit": 5.0, "sharpe_ratio": 1.5}
    ind.id = 42
    ind.strategy_gene = MagicMock()
    ind.strategy_gene.to_dict.return_value = gene_dict
    return ind


class TestCodePinning(unittest.TestCase):
    def test_legacy_entry_roundtrip_without_code(self):
        e = HallOfFameEntry(
            strategy_gene_dict=_SAMPLE_GENE, fitness=1.0, metrics={},
            generation_found=0, run_timestamp=0.0,
        )
        self.assertFalse(e.has_pinned_code())
        d = e.to_dict()
        self.assertNotIn("strategy_code", d)
        # Round-trip
        e2 = HallOfFameEntry.from_dict(d)
        self.assertEqual(e.entry_id, e2.entry_id)
        self.assertFalse(e2.has_pinned_code())

    def test_entry_with_pinned_code(self):
        code = "class Foo: pass\n"
        e = HallOfFameEntry(
            strategy_gene_dict=_SAMPLE_GENE, fitness=2.0, metrics={},
            generation_found=1, run_timestamp=0.0, strategy_code=code,
        )
        self.assertTrue(e.has_pinned_code())
        self.assertEqual(len(e.code_hash), 64)  # sha256 hex
        d = e.to_dict()
        self.assertEqual(d["strategy_code"], code)
        self.assertEqual(d["code_pinned_version"], "v1")
        # Round-trip preserves everything
        e2 = HallOfFameEntry.from_dict(d)
        self.assertTrue(e2.has_pinned_code())
        self.assertEqual(e2.strategy_code, code)
        self.assertEqual(e2.code_hash, e.code_hash)

    def test_update_pins_code_when_generator_present(self):
        gen = MagicMock()
        gen.generate_strategy_code.return_value = "class Pinned: pass\n"
        with tempfile.TemporaryDirectory() as tmp:
            hof = HallOfFame(directory=tmp, max_size=5, min_fitness=0.0,
                             strategy_generator=gen)
            pop = [_make_individual(fitness=1.5)]
            added = hof.update(pop, generation=3)
            self.assertEqual(added, 1)
            self.assertTrue(hof.entries[0].has_pinned_code())
            self.assertEqual(hof.entries[0].code_pinned_version, "v1")
            # Persisted to disk
            data = json.loads((Path(tmp) / "hall_of_fame.json").read_text())
            self.assertIn("strategy_code", data["entries"][0])

    def test_update_without_generator_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            hof = HallOfFame(directory=tmp, max_size=5, min_fitness=0.0)
            added = hof.update([_make_individual(1.1)], generation=0)
            self.assertEqual(added, 1)
            self.assertFalse(hof.entries[0].has_pinned_code())

    def test_backfill_pin_codes_for_legacy_entries(self):
        # Pre-populate disk file with an un-pinned entry.
        with tempfile.TemporaryDirectory() as tmp:
            hof = HallOfFame(directory=tmp, max_size=5, min_fitness=0.0)
            hof.update([_make_individual(1.1)], generation=0)
            self.assertFalse(hof.entries[0].has_pinned_code())

            # Now backfill with a generator.  We have to mock at the right
            # import site because the method imports StrategyGene lazily.
            gen = MagicMock()
            gen.generate_strategy_code.return_value = "class B: pass\n"
            import genetic_algorithm.core.strategy_gene as _sg
            orig = _sg.StrategyGene.from_dict
            _sg.StrategyGene.from_dict = classmethod(lambda cls, d: object())  # type: ignore
            try:
                n = hof.pin_codes_for_all_entries(gen)
            finally:
                _sg.StrategyGene.from_dict = orig
            self.assertEqual(n, 1)
            self.assertTrue(hof.entries[0].has_pinned_code())
            self.assertEqual(hof.entries[0].code_pinned_version, "v1-backfill")

    def test_summary_reports_pinned_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            hof = HallOfFame(directory=tmp, max_size=5, min_fitness=0.0)
            # one pinned, one not
            hof.entries.append(HallOfFameEntry(
                strategy_gene_dict=_SAMPLE_GENE, fitness=2.0, metrics={},
                generation_found=0, run_timestamp=0.0,
                strategy_code="class X: pass\n",
            ))
            gene2 = dict(_SAMPLE_GENE); gene2["stoploss"] = -0.10
            hof.entries.append(HallOfFameEntry(
                strategy_gene_dict=gene2, fitness=1.0, metrics={},
                generation_found=0, run_timestamp=0.0,
            ))
            s = hof.get_summary()
            self.assertEqual(s["size"], 2)
            self.assertEqual(s["pinned_entries"], 1)


if __name__ == "__main__":
    unittest.main()
