#!/usr/bin/env python3
"""Sequential runner for phone SIS experiments.

Runs one config at a time and records a compact queue status JSON.  This
intentionally does not use the server queue daemon and never starts multiple
GA processes concurrently.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "genetic_algorithm/config/phone_sis"
STATUS_PATH = REPO / "genetic_algorithm/output/phone_sis/queue_status.json"
LOG_DIR = REPO / "genetic_algorithm/logs/phone_sis"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_status() -> Dict[str, Any]:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {"runs": []}


def save_status(status: Dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def configs() -> List[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


def run_config(cfg: Path, *, dry_run: bool = False) -> int:
    rel = cfg.relative_to(REPO)
    cmd = [sys.executable, "genetic_algorithm/run_ga.py", "--config", str(rel), "--yes"]
    log_file = LOG_DIR / f"{cfg.stem}.runner.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    status = load_status()
    rec: Dict[str, Any] = {
        "config": str(rel),
        "log_file": str(log_file.relative_to(REPO)),
        "started_at": utc_now(),
        "status": "dry_run" if dry_run else "running",
        "returncode": None,
    }
    status["runs"].append(rec)
    save_status(status)

    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        rec["finished_at"] = utc_now()
        rec["returncode"] = 0
        save_status(status)
        return 0

    print(f"[phone-sis] starting {rel}")
    with log_file.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    rec["finished_at"] = utc_now()
    rec["returncode"] = proc.returncode
    rec["status"] = "done" if proc.returncode == 0 else "failed"
    save_status(status)
    print(f"[phone-sis] finished {rel} rc={proc.returncode} log={log_file.relative_to(REPO)}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run phone SIS experiments sequentially")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write queue status only")
    parser.add_argument("--from-index", type=int, default=0, help="Start from config index")
    parser.add_argument("--limit", type=int, default=None, help="Run at most N configs")
    parser.add_argument("--continue-on-fail", action="store_true", help="Continue queue after failed config")
    args = parser.parse_args()

    selected = configs()[args.from_index:]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        print(f"No configs found in {CONFIG_DIR}")
        return 1

    rc = 0
    for cfg in selected:
        rc = run_config(cfg, dry_run=args.dry_run)
        if rc != 0 and not args.continue_on_fail:
            return rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
