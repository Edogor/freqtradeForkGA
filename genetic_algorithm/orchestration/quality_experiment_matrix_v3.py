"""Reproducible 30-run diagnosis plus 10-run winner confirmation matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Literal, Mapping

import yaml

from genetic_algorithm.config.schema import load_config, validate_config_or_raise


TIMEFRAMES = ("1h", "15m")
DIAGNOSTIC_SEEDS = (41001, 42001, 43001)
CONFIRMATION_SEEDS = (51001, 52001, 53001, 54001, 55001)
ARM_PRESETS = {
    "A0": "quality_v3_a0_historical.yaml",
    "A1": "quality_v3_a1_balanced.yaml",
    "A2": "quality_v3_a2_separated.yaml",
    "A3": "quality_v3_a3_feasibility.yaml",
    "A4": "quality_v3_a4_diversity.yaml",
}


@dataclass(frozen=True)
class QualityRunSpecV3:
    run_id: str
    phase: Literal["DIAGNOSTIC", "CONFIRMATION"]
    timeframe: Literal["1h", "15m"]
    arm_id: str
    paired_seed: int
    preset_path: str


@dataclass(frozen=True)
class ArmRunOutcomeV3:
    timeframe: Literal["1h", "15m"]
    arm_id: str
    paired_seed: int
    completed: bool
    v3_pass_candidate_count: int
    max_feasibility_stage: int
    best_robust_score: float
    behavior_duplicate_fraction: float


def diagnostic_specs() -> list[QualityRunSpecV3]:
    return [
        QualityRunSpecV3(
            run_id=f"diag-{timeframe}-{arm.lower()}-{seed}",
            phase="DIAGNOSTIC",
            timeframe=timeframe,
            arm_id=arm,
            paired_seed=seed,
            preset_path=f"genetic_algorithm/config/presets/{preset}",
        )
        for timeframe in TIMEFRAMES
        for seed in DIAGNOSTIC_SEEDS
        for arm, preset in ARM_PRESETS.items()
    ]


def confirmation_specs(
    winners: Mapping[str, str],
) -> list[QualityRunSpecV3]:
    if set(winners) != set(TIMEFRAMES):
        raise ValueError("one winning arm is required for 1h and 15m")
    specs = []
    for timeframe in TIMEFRAMES:
        arm = winners[timeframe]
        if arm not in ARM_PRESETS:
            raise ValueError(f"unknown winner arm: {arm}")
        for seed in CONFIRMATION_SEEDS:
            specs.append(
                QualityRunSpecV3(
                    run_id=f"confirm-{timeframe}-{arm.lower()}-{seed}",
                    phase="CONFIRMATION",
                    timeframe=timeframe,
                    arm_id=arm,
                    paired_seed=seed,
                    preset_path=(
                        "genetic_algorithm/config/presets/"
                        f"{ARM_PRESETS[arm]}"
                    ),
                )
            )
    return specs


def full_specs(winners: Mapping[str, str]) -> list[QualityRunSpecV3]:
    specs = diagnostic_specs() + confirmation_specs(winners)
    if len(specs) != 40 or len({item.run_id for item in specs}) != 40:
        raise AssertionError("quality experiment matrix must contain 40 unique runs")
    return specs


def choose_winner(
    outcomes: Iterable[ArmRunOutcomeV3], timeframe: str
) -> str:
    """Select by qualification yield, feasibility progress, then diversity.

    Each arm must complete all three paired seeds. Profit alone never breaks a
    tie ahead of qualification yield or feasibility stage.
    """
    relevant = [item for item in outcomes if item.timeframe == timeframe]
    grouped = {
        arm: [item for item in relevant if item.arm_id == arm]
        for arm in ARM_PRESETS
    }
    eligible = {
        arm: rows
        for arm, rows in grouped.items()
        if len(rows) == len(DIAGNOSTIC_SEEDS)
        and {item.paired_seed for item in rows} == set(DIAGNOSTIC_SEEDS)
        and all(item.completed for item in rows)
    }
    if not eligible:
        raise ValueError(f"no complete three-seed arm for {timeframe}")

    def rank(rows: list[ArmRunOutcomeV3]) -> tuple[float, ...]:
        return (
            float(sum(item.v3_pass_candidate_count for item in rows)),
            float(sum(item.v3_pass_candidate_count > 0 for item in rows)),
            median(item.max_feasibility_stage for item in rows),
            median(item.best_robust_score for item in rows),
            -median(item.behavior_duplicate_fraction for item in rows),
        )

    # Stable arm ID is the final deterministic tie-breaker.
    return max(sorted(eligible), key=lambda arm: rank(eligible[arm]))


def resolved_run_config(spec: QualityRunSpecV3) -> dict:
    config = copy.deepcopy(load_config(spec.preset_path))
    config["backtesting"]["timeframe"] = spec.timeframe
    config["strategy_constraints"]["timeframes"] = [spec.timeframe]
    config["genetic_algorithm"]["random_seed"] = spec.paired_seed
    config["automation_controller"]["root_seeds"] = [spec.paired_seed]
    for index, island in enumerate(config["generic_island_model"].get("islands", [])):
        island["seed"] = spec.paired_seed + index
    for section in ("promotion_v2", "qualification_v3"):
        for scenario in config[section]["required_scenarios"]:
            scenario["timeframe"] = spec.timeframe
    validate_config_or_raise(config)
    return config


def materialize_matrix(specs: Iterable[QualityRunSpecV3], output_root: Path) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in specs:
        config = resolved_run_config(spec)
        run_root = output_root / spec.run_id
        run_root.mkdir(parents=True, exist_ok=False)
        config["output"]["dir"] = str(run_root / "output")
        config["storage"]["checkpoint_dir"] = str(run_root / "checkpoints")
        config["storage"]["runs_dir"] = str(run_root / "runs")
        config.setdefault("hall_of_fame", {})["directory"] = str(
            run_root / "hall_of_fame"
        )
        config_path = run_root / f"{spec.run_id}.resolved.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
        manifest.append(
            {
                **asdict(spec),
                "config_path": str(config_path),
                "config_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "command": [
                    ".venv/bin/python",
                    "-m",
                    "genetic_algorithm.run_ga",
                    "--config",
                    str(config_path),
                    "--yes",
                    "--no-monitor",
                ],
            }
        )
    manifest_path = output_root / "matrix_manifest_v3.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "3.0", "runs": manifest}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--winner-1h", choices=sorted(ARM_PRESETS))
    parser.add_argument("--winner-15m", choices=sorted(ARM_PRESETS))
    args = parser.parse_args(argv)
    if bool(args.winner_1h) != bool(args.winner_15m):
        parser.error("both timeframe winners must be supplied together")
    specs = diagnostic_specs()
    if args.winner_1h:
        specs = full_specs({"1h": args.winner_1h, "15m": args.winner_15m})
    print(materialize_matrix(specs, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
