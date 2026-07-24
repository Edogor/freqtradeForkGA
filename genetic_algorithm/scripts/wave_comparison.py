#!/usr/bin/env python3
"""Render a verified comparison of one analyzed GA V2 wave.

Logs are deliberately not accepted.  Every displayed economic, risk and
activity metric comes from the same candidate aggregation in the recorded
ANALYSIS decision and is rebuilt from hash-verified result artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

from genetic_algorithm.orchestration.runner_v2 import default_state_path
from genetic_algorithm.orchestration.wave_analyzer_v2 import CandidateEligibilityStatus
from genetic_algorithm.orchestration.wave_comparison_v2 import (
    VerifiedWaveComparisonV2,
    build_verified_wave_comparison,
)
from genetic_algorithm.orchestration.wave_state_v2 import WaveStateStoreV2


def _number(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    scaled = value * 100.0 if percent else value
    suffix = "%" if percent else ""
    return f"{scaled:.3f}{suffix}"


def format_verified_comparison(comparison: VerifiedWaveComparisonV2) -> str:
    """Render only candidate-bound, structured V2 evidence."""

    analysis = comparison.analysis
    lines = [
        "=" * 118,
        f"VERIFIED WAVE COMPARISON — {comparison.wave_id}",
        "=" * 118,
        (
            f"Lifecycle: {comparison.lifecycle_status.value} | "
            f"Planning allowed: {analysis.planning_allowed} | "
            f"Eligible candidates: {analysis.has_eligible_candidates}"
        ),
        f"Snapshot: {comparison.snapshot_hash}",
        f"Analysis: {analysis.analysis_hash}",
        f"Comparison: {comparison.comparison_hash}",
        "",
        "EXPERIMENT HEALTH",
        "-" * 118,
        (
            f"{'Experiment':<28} {'Arm':<12} {'Health':<9} {'Success':>9} "
            f"{'Results':>9} {'Aborts':>8} {'Candidates':>11}  Reasons"
        ),
    ]
    for experiment in analysis.experiments:
        lines.append(
            f"{experiment.experiment_id[:28]:<28} "
            f"{experiment.arm_type.value:<12} "
            f"{experiment.health_status.value:<9} "
            f"{experiment.successful_attempt_count:>4}/"
            f"{experiment.expected_attempt_count:<4} "
            f"{experiment.verified_result_count:>4}/"
            f"{experiment.expected_attempt_count:<4} "
            f"{experiment.documented_abort_count:>8} "
            f"{experiment.candidate_observation_count:>11}  "
            f"{','.join(experiment.reason_codes)}"
        )

    lines.extend(
        [
            "",
            "CANDIDATE EVIDENCE",
            "-" * 118,
            (
                f"{'Experiment / phenotype':<42} {'Status':<10} {'Seeds':>5} "
                f"{'Worst LCB':>10} {'Median LCB':>11} {'DD UCB':>9} "
                f"{'ES5 UCB':>9} {'Exp LCB':>9} {'PF':>7} {'Win':>8} {'Trades':>8}"
            ),
        ]
    )
    for candidate in analysis.candidates:
        identity = f"{candidate.experiment_id}/{candidate.phenotype_hash[:12]}"
        lines.append(
            f"{identity[:42]:<42} "
            f"{candidate.eligibility_status.value:<10} "
            f"{candidate.observation_count:>5} "
            f"{_number(candidate.worst_annualized_return_lcb, percent=True):>10} "
            f"{_number(candidate.median_annualized_return_lcb, percent=True):>11} "
            f"{_number(candidate.worst_max_drawdown_ucb, percent=True):>9} "
            f"{_number(candidate.worst_daily_es5_ucb, percent=True):>9} "
            f"{_number(candidate.worst_net_expectancy_lcb, percent=True):>9} "
            f"{_number(candidate.median_profit_factor):>7} "
            f"{_number(candidate.median_win_rate, percent=True):>8} "
            f"{candidate.scenario_trade_count_sum:>8}"
        )
        if candidate.eligibility_status != CandidateEligibilityStatus.ELIGIBLE:
            reasons = sorted(set(candidate.reason_codes + candidate.failed_gate_reason_codes))
            lines.append(f"  Ineligible reasons: {','.join(reasons)}")

    lines.extend(["", "Wave reasons: " + ",".join(analysis.reason_codes)])
    return "\n".join(lines)


def _write_csv(path: Path, comparison: VerifiedWaveComparisonV2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wave_id",
                "experiment_id",
                "phenotype_hash",
                "eligibility_status",
                "seed_count",
                "valid_observation_count",
                "worst_annualized_return_lcb",
                "median_annualized_return_lcb",
                "worst_max_drawdown_ucb",
                "worst_daily_es5_ucb",
                "worst_net_expectancy_lcb",
                "median_profit_factor",
                "median_win_rate",
                "min_effective_sample_size",
                "min_trades_per_active_month",
                "scenario_trade_count_sum",
                "reason_codes",
                "failed_gate_reason_codes",
                "snapshot_hash",
                "analysis_hash",
                "comparison_hash",
            ]
        )
        for candidate in comparison.analysis.candidates:
            writer.writerow(
                [
                    comparison.wave_id,
                    candidate.experiment_id,
                    candidate.phenotype_hash,
                    candidate.eligibility_status.value,
                    candidate.observation_count,
                    candidate.valid_observation_count,
                    candidate.worst_annualized_return_lcb,
                    candidate.median_annualized_return_lcb,
                    candidate.worst_max_drawdown_ucb,
                    candidate.worst_daily_es5_ucb,
                    candidate.worst_net_expectancy_lcb,
                    candidate.median_profit_factor,
                    candidate.median_win_rate,
                    candidate.min_effective_sample_size,
                    candidate.min_trades_per_active_month,
                    candidate.scenario_trade_count_sum,
                    "|".join(candidate.reason_codes),
                    "|".join(candidate.failed_gate_reason_codes),
                    comparison.snapshot_hash,
                    comparison.analysis.analysis_hash,
                    comparison.comparison_hash,
                ]
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare one analyzed GA V2 wave from verified structured evidence"
    )
    parser.add_argument("wave_id", help="Exact V2 wave ID")
    parser.add_argument(
        "--state-path",
        type=Path,
        default=default_state_path(),
        help="V2 orchestration SQLite database",
    )
    parser.add_argument("--json", action="store_true", help="Print canonical JSON")
    parser.add_argument("--csv", type=Path, help="Write candidate evidence as CSV")
    parser.add_argument("--output", type=Path, help="Write the text report to this path")
    args = parser.parse_args(argv)

    try:
        comparison = build_verified_wave_comparison(
            WaveStateStoreV2(args.state_path),
            args.wave_id,
        )
    except Exception as exc:
        print(
            f"ERROR: verified wave comparison unavailable: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    if args.csv is not None:
        _write_csv(args.csv, comparison)
    if args.json:
        print(
            json.dumps(
                {
                    **comparison.model_dump(mode="json"),
                    "comparison_hash": comparison.comparison_hash,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        report = format_verified_comparison(comparison)
        print(report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
