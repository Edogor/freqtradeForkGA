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
from genetic_algorithm.orchestration.wave_materializer_v2 import (
    ChildWaveMaterializationV2,
)
from genetic_algorithm.orchestration.wave_state_v2 import WaveStateStoreV2


REPORT_SCHEMA_VERSION = "2.1"
CAMPAIGN_SUMMARY_SCHEMA_VERSION = "1.1"


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
        temporary.replace(path)
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
            "experiment_id": state.experiment_id,
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
            "artifact_bytes": sum(
                path.stat().st_size
                for path in Path(state.artifact_root).rglob("*")
                if path.is_file()
            ),
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


def _candidate_rank_key(item: dict[str, Any]) -> tuple[float, float, str]:
    return (
        (
            0.0
            if item.get("median_gate_alignment_score") is None
            else float(item["median_gate_alignment_score"])
        ),
        (
            float("-inf")
            if item.get("median_robust_score") is None
            else float(item["median_robust_score"])
        ),
        str(item.get("phenotype_hash", "")),
    )


def _sanitized_candidate_summary(
    candidate: dict[str, Any] | None,
    *,
    arm_type: str | None,
    arm_id: str | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "experiment_id": candidate["experiment_id"],
        "arm_type": arm_type,
        "arm_id": arm_id,
        "phenotype_hash": candidate["phenotype_hash"],
        "eligibility_status": candidate["eligibility_status"],
        "median_gate_alignment_score": candidate.get(
            "median_gate_alignment_score"
        ),
        "median_robust_score": candidate.get("median_robust_score"),
        "min_scenario_net_return": candidate.get("min_scenario_net_return"),
        "profitable_scenario_ratio": candidate.get(
            "profitable_scenario_ratio"
        ),
        "worst_annualized_return_lcb": candidate.get(
            "worst_annualized_return_lcb"
        ),
        "worst_net_expectancy_lcb": candidate.get(
            "worst_net_expectancy_lcb"
        ),
        "worst_max_drawdown_ucb": candidate.get("worst_max_drawdown_ucb"),
        "worst_daily_es5_ucb": candidate.get("worst_daily_es5_ucb"),
        "min_effective_sample_size": candidate.get(
            "min_effective_sample_size"
        ),
        "min_trades_per_active_month": candidate.get(
            "min_trades_per_active_month"
        ),
        "max_drawdown_duration_days": candidate.get(
            "max_drawdown_duration_days"
        ),
        "failed_gate_reason_codes": candidate.get(
            "failed_gate_reason_codes", []
        ),
    }


def _arm_metadata(
    experiment_id: str,
    plan_experiments: dict[str, dict[str, Any]],
    experiment_analyses: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    planned = plan_experiments.get(experiment_id, {})
    analyzed = experiment_analyses.get(experiment_id, {})
    return (
        planned.get("arm_type") or analyzed.get("arm_type"),
        planned.get("arm_id"),
    )


def build_campaign_summary_v2(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one deterministic, sanitized cross-wave campaign rollup."""

    root_wave_ids = {str(item["root_wave_id"]) for item in reports}
    if len(root_wave_ids) > 1:
        raise ValueError("campaign summary cannot mix root wave lineages")
    ordered = sorted(
        reports,
        key=lambda item: (
            str(item["analyzed_at"]),
            str(item["wave_id"]),
            str(item["analysis_hash"]),
        ),
    )
    seen_phenotypes: set[str] = set()
    entries: list[dict[str, Any]] = []
    for report in ordered:
        candidates = report.get("candidate_analyses", [])
        attempts = report.get("attempts", [])
        plan_summary = report.get("plan_summary") or {}
        plan_experiments = {
            str(item["experiment_id"]): item
            for item in plan_summary.get("experiments", [])
        }
        experiment_analyses = {
            str(item["experiment_id"]): item
            for item in report.get("experiments", [])
        }
        experiment_ids = sorted(
            set(experiment_analyses)
            | set(plan_experiments)
            | {str(item["experiment_id"]) for item in candidates}
            | {
                str(item["experiment_id"])
                for item in attempts
                if item.get("experiment_id")
            }
        )

        phenotype_hashes = sorted(
            {
                str(item["phenotype_hash"])
                for item in candidates
                if item.get("phenotype_hash")
            }
        )
        repeated = sorted(set(phenotype_hashes).intersection(seen_phenotypes))
        seen_phenotypes.update(phenotype_hashes)
        best = max(candidates, key=_candidate_rank_key, default=None)
        best_arm_type, best_arm_id = _arm_metadata(
            str(best["experiment_id"]) if best is not None else "",
            plan_experiments,
            experiment_analyses,
        )
        arm_summaries: list[dict[str, Any]] = []
        for experiment_id in experiment_ids:
            experiment_candidates = [
                item
                for item in candidates
                if item.get("experiment_id") == experiment_id
            ]
            experiment_attempts = [
                item
                for item in attempts
                if item.get("experiment_id") == experiment_id
            ]
            arm_best = max(
                experiment_candidates,
                key=_candidate_rank_key,
                default=None,
            )
            arm_type, arm_id = _arm_metadata(
                experiment_id,
                plan_experiments,
                experiment_analyses,
            )
            planned = plan_experiments.get(experiment_id, {})
            arm_summaries.append(
                {
                    "experiment_id": experiment_id,
                    "arm_type": arm_type,
                    "arm_id": arm_id,
                    "planned_seeds": planned.get("seeds", []),
                    "factor_delta": planned.get("factor_delta", {}),
                    "source_mode": planned.get("source_mode"),
                    "parent_experiment_id": planned.get(
                        "parent_experiment_id"
                    ),
                    "parent_config_hash": planned.get("parent_config_hash"),
                    "selected_candidate_key": planned.get(
                        "selected_candidate_key"
                    ),
                    "selected_parent_phenotype_hash": planned.get(
                        "phenotype_hash"
                    ),
                    "attempt_count": len(experiment_attempts),
                    "successful_attempt_count": sum(
                        item.get("result_status") == "SUCCEEDED"
                        for item in experiment_attempts
                    ),
                    "candidate_count": len(experiment_candidates),
                    "promotion_eligible_candidate_count": sum(
                        item.get("eligibility_status") == "ELIGIBLE"
                        for item in experiment_candidates
                    ),
                    "best_candidate": _sanitized_candidate_summary(
                        arm_best,
                        arm_type=arm_type,
                        arm_id=arm_id,
                    ),
                }
            )
        entries.append(
            {
                "wave_id": report["wave_id"],
                "parent_wave_id": plan_summary.get("parent_wave_id"),
                "plan_hash": plan_summary.get("plan_hash"),
                "plan_reason_codes": plan_summary.get(
                    "plan_reason_codes", []
                ),
                "analyzed_at": report["analyzed_at"],
                "controller_outcome": report["controller_outcome"],
                "controller_reason_codes": report["controller_reason_codes"],
                "analysis_reason_codes": report["analysis_reason_codes"],
                "attempt_count": len(attempts),
                "successful_attempt_count": sum(
                    item.get("result_status") == "SUCCEEDED" for item in attempts
                ),
                "duration_seconds": sum(
                    float(item.get("duration_seconds") or 0.0) for item in attempts
                ),
                "artifact_bytes": sum(
                    int(item.get("artifact_bytes") or 0) for item in attempts
                ),
                "seeds": sorted(
                    {
                        int(seed)
                        for item in attempts
                        for seed in item.get("seeds", [])
                    }
                ),
                "candidate_count": len(candidates),
                "promotion_eligible_candidate_count": sum(
                    item.get("eligibility_status") == "ELIGIBLE"
                    for item in candidates
                ),
                "unique_phenotype_count": len(phenotype_hashes),
                "repeated_phenotype_count": len(repeated),
                "repeated_phenotype_hashes": repeated,
                "failed_gate_reason_codes": sorted(
                    {
                        str(reason)
                        for item in candidates
                        for reason in item.get("failed_gate_reason_codes", [])
                    }
                ),
                "selected_parent_phenotype_hashes": sorted(
                    {
                        str(item["phenotype_hash"])
                        for item in plan_experiments.values()
                        if item.get("phenotype_hash")
                    }
                ),
                "arms": arm_summaries,
                "best_candidate": _sanitized_candidate_summary(
                    best,
                    arm_type=best_arm_type,
                    arm_id=best_arm_id,
                ),
            }
        )
    return {
        "schema_version": CAMPAIGN_SUMMARY_SCHEMA_VERSION,
        "report_kind": "SEARCH_ONLY_CAMPAIGN_SUMMARY",
        "promotion_authorized": False,
        "root_wave_id": ordered[0]["root_wave_id"] if ordered else None,
        "wave_count": len(entries),
        "attempt_count": sum(item["attempt_count"] for item in entries),
        "successful_attempt_count": sum(
            item["successful_attempt_count"] for item in entries
        ),
        "duration_seconds": sum(item["duration_seconds"] for item in entries),
        "artifact_bytes": sum(item["artifact_bytes"] for item in entries),
        "unique_phenotype_count": len(seen_phenotypes),
        "repeated_phenotype_observation_count": sum(
            item["repeated_phenotype_count"] for item in entries
        ),
        "waves": entries,
    }


def _campaign_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Automation campaign summary",
        "",
        "Search-only evidence. This summary never authorizes strategy promotion "
        "or trading.",
        "",
        f"- Root wave: `{summary['root_wave_id']}`",
        f"- Waves: `{summary['wave_count']}`",
        f"- Successful attempts: `{summary['successful_attempt_count']}` / "
        f"`{summary['attempt_count']}`",
        f"- Artifact bytes: `{summary['artifact_bytes']}`",
        f"- Unique phenotypes: `{summary['unique_phenotype_count']}`",
        f"- Repeated phenotype observations: "
        f"`{summary['repeated_phenotype_observation_count']}`",
        "",
        "| Wave | Best arm | Outcome | Attempts | Candidates | Eligible | "
        "Gate alignment | Min net return | Return LCB | Expectancy LCB | "
        "DD UCB | ES UCB | Trades/month | DD days | Repeats |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|",
    ]
    for item in summary["waves"]:
        best = item["best_candidate"] or {}

        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['wave_id']}`",
                    str(best.get("arm_id") or best.get("arm_type") or "—"),
                    str(item["controller_outcome"]),
                    str(item["attempt_count"]),
                    str(item["candidate_count"]),
                    str(item["promotion_eligible_candidate_count"]),
                    _summary_metric(best, "median_gate_alignment_score"),
                    _summary_metric(best, "min_scenario_net_return"),
                    _summary_metric(best, "worst_annualized_return_lcb"),
                    _summary_metric(best, "worst_net_expectancy_lcb"),
                    _summary_metric(best, "worst_max_drawdown_ucb"),
                    _summary_metric(best, "worst_daily_es5_ucb"),
                    _summary_metric(best, "min_trades_per_active_month"),
                    _summary_metric(best, "max_drawdown_duration_days"),
                    str(item["repeated_phenotype_count"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _summary_metric(best: dict[str, Any], name: str) -> str:
    raw = best.get(name)
    return "—" if raw is None else f"{float(raw):.4f}"


def _persist_campaign_summary(
    reports_root: Path,
    *,
    root_wave_id: str,
) -> tuple[Path, Path]:
    by_wave: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(reports_root.glob("wave-*/*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("report_kind") != "SEARCH_ONLY_ANALYSIS":
            continue
        if report.get("root_wave_id") != root_wave_id:
            continue
        if report.get("plan_summary") is None:
            report["plan_summary"] = _materialized_plan_summary(
                reports_root.parent,
                str(report["wave_id"]),
            )
        by_wave.setdefault(str(report["wave_id"]), []).append((path, report))
    outcome_priority = {
        "CONTINUE_SEARCH": 1,
        "BLOCKED": 2,
        "STOPPED_LIMIT": 3,
    }
    selected_reports = [
        max(
            items,
            key=lambda pair: (
                outcome_priority.get(
                    str(pair[1].get("controller_outcome")), 0
                ),
                pair[0].name,
            ),
        )[1]
        for _, items in sorted(by_wave.items())
    ]
    summary = build_campaign_summary_v2(selected_reports)
    json_payload = (
        json.dumps(
            summary,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    json_path = reports_root / "CAMPAIGN_SUMMARY.json"
    markdown_path = reports_root / "CAMPAIGN_SUMMARY.md"
    _atomic_replace(json_path, json_payload)
    _atomic_replace(markdown_path, _campaign_markdown(summary).encode())
    return json_path, markdown_path


def _materialized_plan_summary(
    automation_root: Path,
    wave_id: str,
) -> dict[str, Any] | None:
    receipt_path = automation_root / "waves" / wave_id / "materialization.json"
    if not receipt_path.is_file():
        return None
    receipt = ChildWaveMaterializationV2.model_validate_json(
        receipt_path.read_bytes()
    )
    return {
        "parent_wave_id": receipt.plan.parent_wave_id,
        "plan_hash": receipt.plan_hash,
        "plan_reason_codes": receipt.plan.reason_codes,
        "experiments": [
            {
                "experiment_id": item.experiment_id,
                "arm_id": item.arm_id,
                "arm_type": item.arm_type.value,
                "seeds": item.seeds,
                "factor_delta": item.factor_delta,
                "source_mode": item.source.source_mode.value,
                "parent_experiment_id": item.source.parent_experiment_id,
                "parent_config_hash": item.source.parent_config_hash,
                "selected_candidate_key": item.source.selected_candidate_key,
                "phenotype_hash": item.source.phenotype_hash,
            }
            for item in receipt.plan.experiments
        ],
    }


def build_automation_analysis_report(
    store: WaveStateStoreV2,
    *,
    analysis: WaveAnalysisV2,
    root_wave_id: str,
    controller_outcome: str,
    controller_reason_codes: list[str],
    plan_summary: dict[str, Any] | None = None,
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
        "plan_summary": plan_summary,
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
        "| Candidate | Pair | Role | Status | Trades | Return | DD UCB | "
        "Win rate | Profit factor |",
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

    resolved_automation_root = Path(automation_root).resolve()
    report = build_automation_analysis_report(
        store,
        analysis=analysis,
        root_wave_id=root_wave_id,
        controller_outcome=controller_outcome,
        controller_reason_codes=controller_reason_codes,
        plan_summary=_materialized_plan_summary(
            resolved_automation_root,
            analysis.wave_id,
        ),
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
    report_dir = resolved_automation_root / "reports" / analysis.wave_id
    json_path = report_dir / f"{report_hash}.json"
    markdown_path = report_dir / f"{report_hash}.md"
    _write_immutable(json_path, json_payload)
    _write_immutable(markdown_path, markdown_payload)
    latest_dir = resolved_automation_root / "reports"
    _atomic_replace(latest_dir / "LATEST.json", json_payload)
    _atomic_replace(latest_dir / "LATEST.md", markdown_payload)
    _persist_campaign_summary(latest_dir, root_wave_id=root_wave_id)
    return json_path, markdown_path
