#!/usr/bin/env python3
"""Quick Hall-of-Fame inspector with robust schema handling."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def summarize_file(path: Path, top_n: int):
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"ERROR reading {path}: {e}")
        return

    entries = data.get("entries", [])
    if not isinstance(entries, list):
        print(f"INVALID schema in {path} (missing list: entries)")
        return

    entries.sort(key=lambda x: float(x.get("fitness", 0.0) or 0.0), reverse=True)
    print(f"\n{path}")
    print(f"  total_entries={len(entries)}")
    for i, e in enumerate(entries[:top_n], start=1):
        m = e.get("metrics", {}) or {}
        print(
            f"  #{i:02d} fit={float(e.get('fitness', 0.0)):.4f} "
            f"profit={float(m.get('profit', 0.0)):.2f} "
            f"dd={float(m.get('max_drawdown', 0.0)):.4f} "
            f"trades={int(float(m.get('num_trades', 0)))} "
            f"pgr={float(m.get('pair_generalization_ratio', 0.0)):.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="genetic_algorithm/data/hall_of_fame*/hall_of_fame.json")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    for p in paths:
        summarize_file(p, args.top)


if __name__ == "__main__":
    main()
