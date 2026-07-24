#!/usr/bin/env python3
"""Retired log-based GA comparison entry point.

The former implementation combined unrelated "last" regex matches from logs
and arbitrary JSON files, then recommended a winning run. That evidence has no
candidate/scenario identity and is unsafe for comparison.
"""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(
        "ERROR: aggregate_comparison.py is retired because log-derived metrics "
        "cannot form a candidate-bound decision.\n"
        "Use: python -m genetic_algorithm.scripts.wave_comparison "
        "<wave_id> --state-path <orchestration.sqlite3>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
