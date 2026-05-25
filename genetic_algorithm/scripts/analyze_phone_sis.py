#!/usr/bin/env python3
"""Analyze the phone SIS validation experiment set.

Consumes the sequential phone_sis outputs created by
``run_phone_sis_queue.py`` and writes a compact JSON report with:

* HoF fitness / validation / risk metrics
* indicator distributions
* SIS event counts and top weights
* anti-pattern hit-rate
* rule-based diagnostic findings
* strategy embedding diversity / mode-collapse signals
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from genetic_algorithm.intelligence.anti_pattern import summarise_anti_patterns
from genetic_algorithm.intelligence.diagnostics import diagnose
from genetic_algorithm.intelligence.embedding import embed_population, population_diversity


OUT_BASE = REPO / "genetic_algorithm/output/phone_sis"
DATA_BASE = REPO / "genetic_algorithm/data/phone_sis"
LOG_BASE = REPO / "genetic_algorithm/logs/phone_sis"
PRIORS = REPO / "genetic_algorithm/intelligence/priors/data_driven_enrichment.json"
REPORT = OUT_BASE / "phone_sis_analysis.json"


RUNS = [
    ("P0", "phone_sis_P0_smoke_no_sis", "smoke_no_sis"),
    ("P1", "phone_sis_P1_control_no_sis", "control_no_sis"),
    ("P2", "phone_sis_P2_sis_hardcoded", "sis_hardcoded"),
    ("P3", "phone_sis_P3_sis_data_priors", "sis_data_priors"),
    ("P4", "phone_sis_P4_sis_immigrants_only", "sis_immigrants_only"),
    ("P5", "phone_sis_P5_sis_weights_only", "sis_weights_only"),
]


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(mean(clean), 6) if clean else None


def _safe_max(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(max(clean), 6) if clean else None


def _safe_min(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(min(clean), 6) if clean else None


def _indicator_types(gene: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for ind in gene.get("indicators", []) or []:
        if isinstance(ind, dict):
            name = ind.get("type") or ind.get("name")
        else:
            name = getattr(ind, "type", None) or getattr(ind, "name", None)
        if name:
            out.append(str(name).upper())
    return out


def _load_hof(run_name: str) -> List[Dict[str, Any]]:
    path = DATA_BASE / f"{run_name}_hof/hall_of_fame.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("entries", []) if isinstance(payload, dict) else []


def _load_results(run_name: str) -> Dict[str, Any]:
    files = sorted(glob.glob(str(OUT_BASE / run_name / "results_detailed_*.json")))
    if not files:
        return {}
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def _load_events(run_name: str) -> List[Dict[str, Any]]:
    path = LOG_BASE / f"{run_name}_sis_events.jsonl"
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _load_antipatterns() -> Dict[str, Dict[str, float]]:
    if not PRIORS.exists():
        return {}
    payload = json.loads(PRIORS.read_text(encoding="utf-8"))
    return payload.get("anti_patterns", {}) or {}


def _diagnostic_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(entry.get("metrics", {}) or {})
    gene = entry.get("strategy_gene", {}) or {}
    metrics.setdefault("indicator_count", len(gene.get("indicators", []) or []))
    metrics.setdefault(
        "condition_count",
        len(gene.get("entry_conditions", []) or [])
        + len(gene.get("exit_conditions", []) or [])
        + len(gene.get("short_entry_conditions", []) or [])
        + len(gene.get("short_exit_conditions", []) or []),
    )
    return metrics


def _event_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {
            "events": 0,
            "filtered_total": 0,
            "sis_immigrants_total": 0,
            "mean_evidence_trust": None,
            "top_final_indicator_weights": {},
            "bottom_final_indicator_weights": {},
        }
    final_weights = {}
    for event in reversed(events):
        if event.get("indicator_weights_used"):
            final_weights = event["indicator_weights_used"]
            break
    top = dict(sorted(final_weights.items(), key=lambda kv: kv[1], reverse=True)[:8])
    bottom = dict(sorted(final_weights.items(), key=lambda kv: kv[1])[:8])
    return {
        "events": len(events),
        "filtered_total": int(sum(e.get("n_filtered", 0) or 0 for e in events)),
        "sis_immigrants_total": int(sum(e.get("n_sis_immigrants", 0) or 0 for e in events)),
        "mean_evidence_trust": _safe_mean(e.get("evidence_trust") for e in events),
        "mean_best_fitness_event": _safe_mean(e.get("best_fitness") for e in events),
        "max_best_fitness_event": _safe_max(e.get("best_fitness") for e in events),
        "top_final_indicator_weights": top,
        "bottom_final_indicator_weights": bottom,
    }


def analyze_run(label: str, run_name: str, kind: str, anti_patterns: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    hof = _load_hof(run_name)
    results = _load_results(run_name)
    events = _load_events(run_name)
    genes = [entry.get("strategy_gene", {}) or {} for entry in hof]
    indicators = Counter()
    anti_hits = 0
    anti_pairs = Counter()
    diagnostics = Counter()
    diagnostic_severity = Counter()

    for entry in hof:
        gene = entry.get("strategy_gene", {}) or {}
        inds = _indicator_types(gene)
        indicators.update(inds)
        anti = summarise_anti_patterns(inds, anti_patterns)
        anti_hits += int(anti.get("n_pairs", 0) or 0)
        for pair in anti.get("pairs", []) or []:
            anti_pairs[f"{pair.get('a')}×{pair.get('b')}"] += 1
        report = diagnose(_diagnostic_metrics(entry), anti_pattern_report=anti)
        diagnostics.update(f.code for f in report.findings)
        diagnostic_severity.update(f.severity for f in report.findings)

    embedding = embed_population(genes, project_2d=False)
    diversity = population_diversity(embedding)
    metrics = [entry.get("metrics", {}) or {} for entry in hof]
    fitnesses = [entry.get("fitness") for entry in hof]

    return {
        "label": label,
        "run_name": run_name,
        "kind": kind,
        "hof_entries": len(hof),
        "results_strategy_count": len(results.get("strategies", []) or []),
        "results_summary": results.get("summary", {}),
        "fitness": {
            "best": _safe_max(fitnesses),
            "mean": _safe_mean(fitnesses),
            "min": _safe_min(fitnesses),
        },
        "validation": {
            "best_val_fitness": _safe_max(m.get("val_fitness") for m in metrics),
            "mean_val_fitness": _safe_mean(m.get("val_fitness") for m in metrics),
            "mean_train_val_gap": _safe_mean(m.get("train_val_gap") for m in metrics),
            "mean_pair_generalization_ratio": _safe_mean(
                m.get("pair_generalization_ratio") for m in metrics
            ),
        },
        "risk_return": {
            "mean_profit": _safe_mean(m.get("profit") for m in metrics),
            "best_profit": _safe_max(m.get("profit") for m in metrics),
            "mean_drawdown": _safe_mean(m.get("max_drawdown") for m in metrics),
            "mean_trades": _safe_mean(m.get("num_trades") for m in metrics),
            "mean_win_rate": _safe_mean(m.get("win_rate") for m in metrics),
            "mean_sharpe": _safe_mean(m.get("sharpe_ratio") for m in metrics),
        },
        "indicators": {
            "unique": len(indicators),
            "top": dict(indicators.most_common(12)),
        },
        "sis_events": _event_summary(events),
        "anti_patterns": {
            "hit_count": anti_hits,
            "top_pairs": dict(anti_pairs.most_common(10)),
        },
        "diagnostics": {
            "by_code": dict(diagnostics.most_common()),
            "by_severity": dict(diagnostic_severity.most_common()),
        },
        "embedding": diversity,
    }


def main() -> None:
    anti_patterns = _load_antipatterns()
    runs = [analyze_run(label, run_name, kind, anti_patterns) for label, run_name, kind in RUNS]
    lookup = {r["label"]: r for r in runs}
    comparisons: Dict[str, Any] = {}
    for left, right in [("P2", "P1"), ("P3", "P2"), ("P4", "P1"), ("P5", "P1")]:
        a, b = lookup.get(left), lookup.get(right)
        if not a or not b:
            continue
        comparisons[f"{left}_vs_{right}"] = {
            "best_fitness_delta": (
                None
                if a["fitness"]["best"] is None or b["fitness"]["best"] is None
                else round(a["fitness"]["best"] - b["fitness"]["best"], 6)
            ),
            "hof_entries_delta": a["hof_entries"] - b["hof_entries"],
            "mean_pair_distance_delta": round(
                a["embedding"]["mean_pairwise_distance"]
                - b["embedding"]["mean_pairwise_distance"],
                6,
            ),
            "anti_pattern_hit_delta": (
                a["anti_patterns"]["hit_count"] - b["anti_patterns"]["hit_count"]
            ),
        }

    report = {
        "runs": runs,
        "comparisons": comparisons,
        "notes": [
            "Phone-sized runs validate mechanics only; they are too small for live-trading quality claims.",
            "P3 vs P2 isolates data-driven priors; P4/P5 isolate immigrants vs weights.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(REPORT.relative_to(REPO))
    for run in runs:
        print(
            f"{run['label']} {run['kind']}: hof={run['hof_entries']} "
            f"best={run['fitness']['best']} val={run['validation']['best_val_fitness']} "
            f"events={run['sis_events']['events']} sis_imm={run['sis_events']['sis_immigrants_total']} "
            f"dist={run['embedding']['mean_pairwise_distance']}"
        )


if __name__ == "__main__":
    main()
