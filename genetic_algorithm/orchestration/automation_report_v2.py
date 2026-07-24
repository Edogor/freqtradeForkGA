"""Compact, durable reports for unattended V2 wave analysis.

Reports deliberately contain decision evidence and summary metrics only.
Strategy source, serialized genomes, individual trades, and worker logs never
enter this derived artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from genetic_algorithm.orchestration.artifact_store_v2 import V2ArtifactStore
from genetic_algorithm.orchestration.wave_analyzer_v2 import WaveAnalysisV2
from genetic_algorithm.orchestration.wave_state_v2 import WaveStateStoreV2


REPORT_SCHEMA_VERSION = "2.0"


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable automation report differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"immutable automation report differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _scenario_summary(record: Any) -> dict[str, Any]:
    metrics = record.metrics
    return {
        "scenario_id": metrics.scenario_id,
        "pair": metrics.pair,
        "role": metrics.role.value,
        "status": metrics.status.value,
        "error_code": metrics.error_code,
        "trade_count": metrics.trade_count,
        "active_months": metrics.active_months,
        "trades_per_active_month": (
            metrics.trade_count / metrics.active_months
            if metrics.active_months
            else 0.0
        ),
        "net_return": metrics.net_return,
        "annualized_net_return_lcb": metrics.annualized_net_return_lcb,
        "max_drawdown": metrics.max_drawdown,
        "max_drawdown_ucb": metrics.max_drawdown_ucb,
        "daily_expected_shortfall_5_ucb": metrics.daily_expected_shortfall_5_ucb,
        "net_expectancy_lcb": metrics.net_expectancy_lcb,
        "profit_factor": metrics.profit_factor,
        "win_rate": metrics.win_rate,
        "effective_sample_size": metrics.effective_sample_size,
        "max_consecutive_losses": metrics.max_consecutive_losses,
        "max_drawdown_duration_days": metrics.max_drawdown_duration_days,
    }


def _attempt_summaries(
    store: WaveStateStoreV2,
    analysis: WaveAnalysisV2,
) -> list[dict[str, Any]]:
    attempts = {
        item.attempt_id: item
        for item in store.list_attempts()
        if item.wave_id == analysis.wave_id
    }
    summaries: list[dict[str, Any]] = []
    for attempt_id in sorted(attempts):
        state = attempts[attempt_id]
        summary: dict[str, Any] = {
            "attempt_id": attempt_id,
            "status": state.status.value,
            "result_status": None,
            "error_code": None,
            "started_at": None,
            "finished_at": None,
            "duration_seconds": None,
            "code_version": None,
            "config_hash": None,
            "data_manifest_hash": None,
            "seeds": [],
            "candidates": [],
        }
        if state.result_path:
            result = V2ArtifactStore(Path(state.result_path).parent).read_verified_result()
            engine_seed_evidence = None
            engine_config_path = Path(state.artifact_root) / "evolution" / "engine_config.yaml"
            if engine_config_path.is_file():
                engine_config = yaml.safe_load(engine_config_path.read_text())
                if isinstance(engine_config, dict):
                    manifest_seed = result.manifest.seeds[0]
                    ga_seed = engine_config.get("genetic_algorithm", {}).get(
                        "random_seed"
                    )
                    island_seeds = [
                        {
                            "name": item.get("name"),
                            "seed": item.get("seed"),
                        }
                        for item in engine_config.get(
                            "generic_island_model", {}
                        ).get("islands", [])
                        if isinstance(item, dict)
                    ]
                    expected_island_seeds = [
                        (manifest_seed + ordinal) % (2**32)
                        for ordinal in range(len(island_seeds))
                    ]
                    engine_seed_evidence = {
                        "manifest_seed": manifest_seed,
                        "ga_seed": ga_seed,
                        "island_seeds": island_seeds,
                        "expected_island_seeds": expected_island_seeds,
                        "contract_matches": (
                            ga_seed == manifest_seed
                            and [item["seed"] for item in island_seeds]
                            == expected_island_seeds
                        ),
                    }
            summary.update(
                {
                    "result_status": result.status.value,
                    "error_code": result.error_code,
                    "started_at": result.started_at.isoformat(),
                    "finished_at": result.finished_at.isoformat(),
                    "duration_seconds": (
                        result.finished_at - result.started_at
                    ).total_seconds(),
                    "code_version": result.manifest.code_version,
                    "config_hash": result.manifest.config_hash,
                    "data_manifest_hash": result.manifest.data_manifest_hash,
                    "seeds": result.manifest.seeds,
                    "engine_seed_evidence": engine_seed_evidence,
                    "candidates": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "phenotype_hash": candidate.phenotype_hash,
                            "status": candidate.status.value,
                            "robust_score": candidate.robust_score,
                            "pareto_rank": candidate.pareto_rank,
                            "gates": [
                                {
                                    "gate_id": gate.gate_id,
                                    "status": gate.status.value,
                                    "passed": gate.passed,
                                    "reason_code": gate.reason_code,
                                }
                                for gate in candidate.gates
                            ],
                            "scenarios": [
                                _scenario_summary(record)
                                for record in candidate.scenarios
                            ],
                        }
                        for candidate in result.candidate_evaluations
                    ],
                }
            )
        summaries.append(summary)
    return summaries


def build_automation_analysis_report(
    store: WaveStateStoreV2,
    *,
    analysis: WaveAnalysisV2,
    root_wave_id: str,
    controller_outcome: str,
    controller_reason_codes: list[str],
) -> dict[str, Any]:
    """Build a sanitized report from hash-verified attempt results."""

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": "SEARCH_ONLY_ANALYSIS",
        "promotion_authorized": False,
        "root_wave_id": root_wave_id,
        "wave_id": analysis.wave_id,
        "analysis_hash": analysis.analysis_hash,
        "analyzed_at": analysis.created_at.isoformat(),
        "controller_outcome": controller_outcome,
        "controller_reason_codes": sorted(set(controller_reason_codes)),
        "planning_allowed": analysis.planning_allowed,
        "has_eligible_candidates": analysis.has_eligible_candidates,
        "analysis_reason_codes": analysis.reason_codes,
        "experiments": [
            item.model_dump(mode="json") for item in analysis.experiments
        ],
        "candidate_analyses": [
            item.model_dump(mode="json") for item in analysis.candidates
        ],
        "attempts": _attempt_summaries(store, analysis),
    }


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.3f}%"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Automation report: {report['wave_id']}",
        "",
        "Search-only evidence. This report never authorizes strategy promotion "
        "or trading.",
        "",
        f"- Root wave: `{report['root_wave_id']}`",
        f"- Analysis hash: `{report['analysis_hash']}`",
        f"- Outcome: `{report['controller_outcome']}`",
        "- Controller reasons: "
        + ", ".join(f"`{item}`" for item in report["controller_reason_codes"]),
        f"- Eligible candidates present: `{report['has_eligible_candidates']}`",
        "",
        "## Candidate scenarios",
        "",
        "| Candidate | Pair | Role | Status | Trades | Return | DD UCB | Win rate | Profit factor |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    rows = 0
    for attempt in report["attempts"]:
        for candidate in attempt["candidates"]:
            for scenario in candidate["scenarios"]:
                rows += 1
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{candidate['candidate_id']}`",
                            scenario["pair"],
                            scenario["role"],
                            scenario["status"],
                            str(scenario["trade_count"]),
                            _percentage(scenario["net_return"]),
                            _percentage(scenario["max_drawdown_ucb"]),
                            _percentage(scenario["win_rate"]),
                            (
                                "—"
                                if scenario["profit_factor"] is None
                                else f"{scenario['profit_factor']:.3f}"
                            ),
                        ]
                    )
                    + " |"
                )
    if not rows:
        lines.append("| — | — | — | — | 0 | — | — | — | — |")
    lines.extend(
        [
            "",
            "## Analysis reasons",
            "",
            *[f"- `{item}`" for item in report["analysis_reason_codes"]],
            "",
        ]
    )
    return "\n".join(lines)


def persist_automation_analysis_report(
    store: WaveStateStoreV2,
    *,
    analysis: WaveAnalysisV2,
    root_wave_id: str,
    automation_root: str | Path,
    controller_outcome: str,
    controller_reason_codes: list[str],
) -> tuple[Path, Path]:
    """Persist immutable JSON/Markdown plus convenient latest copies."""

    report = build_automation_analysis_report(
        store,
        analysis=analysis,
        root_wave_id=root_wave_id,
        controller_outcome=controller_outcome,
        controller_reason_codes=controller_reason_codes,
    )
    json_payload = (
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    markdown_payload = _markdown(report).encode()
    report_hash = hashlib.sha256(json_payload).hexdigest()
    report_dir = Path(automation_root).resolve() / "reports" / analysis.wave_id
    json_path = report_dir / f"{report_hash}.json"
    markdown_path = report_dir / f"{report_hash}.md"
    _write_immutable(json_path, json_payload)
    _write_immutable(markdown_path, markdown_payload)
    latest_dir = Path(automation_root).resolve() / "reports"
    _atomic_replace(latest_dir / "LATEST.json", json_payload)
    _atomic_replace(latest_dir / "LATEST.md", markdown_payload)
    return json_path, markdown_path
