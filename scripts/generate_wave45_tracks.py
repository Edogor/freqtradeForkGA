#!/usr/bin/env python3
"""
Generate Wave45 configs and seed artifacts from Wave44 winners.

Outputs:
- genetic_algorithm/data/wave45_ranked_shortlist.csv
- genetic_algorithm/data/wave45_seed_manifest.csv
- genetic_algorithm/data/checkpoints_wave45_seed_top30.json
- genetic_algorithm/config/queue/38..41_wave45_*.yaml
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import yaml

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "genetic_algorithm/data"
QUEUE_DIR = BASE / "genetic_algorithm/config/queue"

RANKED_SHORTLIST_OUT = DATA_DIR / "wave45_ranked_shortlist.csv"
SEED_MANIFEST_OUT = DATA_DIR / "wave45_seed_manifest.csv"
SEED_CHECKPOINT_OUT = DATA_DIR / "checkpoints_wave45_seed_top30.json"

HOF_FILES = {
    "A": DATA_DIR / "hall_of_fame_wave44_A_1h_conservative_5of6/hall_of_fame.json",
    "B": DATA_DIR / "hall_of_fame_wave44_B_1h_balanced_5of6/hall_of_fame.json",
    "C": DATA_DIR / "hall_of_fame_wave44_C_1h_exploratory_5of6/hall_of_fame.json",
    "D": DATA_DIR / "hall_of_fame_wave44_D_1h_stressrobust_5of6/hall_of_fame.json",
}

PAIRS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "PEPE/USDT"]

MIN_TRADES = 350
MAX_DRAWDOWN = 0.22
MIN_PAIR_RATIO = 0.35  # empirical floor from wave44 pair_generalization_ratio range
TOP_N = 30


def zmap(rows, key):
    vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
    mu = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    if sd == 0:
        return [0.0 for _ in rows]
    return [(v - mu) / sd for v in vals]


def _f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def fingerprint(gene: dict) -> str:
    raw = json.dumps(gene, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_wave44_candidates() -> list[dict]:
    rows = []
    for track, p in HOF_FILES.items():
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for e in data.get("entries", []):
            m = e.get("metrics", {}) or {}
            row = {
                "track": track,
                "entry_id": e.get("id", ""),
                "fitness": _f(e.get("fitness")),
                "profit": _f(m.get("profit")),
                "profit_factor": _f(m.get("profit_factor")),
                "sharpe_ratio": _f(m.get("sharpe_ratio")),
                "sortino_ratio": _f(m.get("sortino_ratio")),
                "win_rate": _f(m.get("win_rate")),
                "max_drawdown": _f(m.get("max_drawdown"), 1e9),
                "num_trades": int(_f(m.get("num_trades"), 0)),
                "pair_generalization_ratio": _f(m.get("pair_generalization_ratio"), 0.0),
                "val_profit": _f(m.get("val_profit")),
                "val_fitness": _f(m.get("val_fitness")),
                "generation_found": int(_f(e.get("generation_found"), 0)),
                "strategy_gene": e.get("strategy_gene", {}),
            }
            row["strategy_fingerprint"] = fingerprint(row["strategy_gene"])
            rows.append(row)
    return rows


def gate_candidates(rows: list[dict]) -> list[dict]:
    gated = []
    for r in rows:
        if r["num_trades"] < MIN_TRADES:
            continue
        if r["max_drawdown"] > MAX_DRAWDOWN:
            continue
        if r["pair_generalization_ratio"] < MIN_PAIR_RATIO:
            continue
        if r["val_fitness"] <= 0:
            continue
        gated.append(r)
    return gated


def rank_and_dedupe(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    zp = zmap(rows, "profit")
    zpf = zmap(rows, "profit_factor")
    zsh = zmap(rows, "sharpe_ratio")
    zwr = zmap(rows, "win_rate")
    zdd = zmap(rows, "max_drawdown")
    zpg = zmap(rows, "pair_generalization_ratio")
    zfit = zmap(rows, "fitness")

    for i, r in enumerate(rows):
        r["balanced_score"] = (
            1.00 * zp[i]
            + 0.70 * zpf[i]
            + 0.70 * zsh[i]
            + 0.30 * zwr[i]
            + 0.90 * zpg[i]
            + 0.45 * zfit[i]
            - 1.10 * zdd[i]
        )

    rows.sort(key=lambda x: x["balanced_score"], reverse=True)

    deduped = []
    seen = set()
    for r in rows:
        fp = r["strategy_fingerprint"]
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(r)

    for i, r in enumerate(deduped, start=1):
        r["rank"] = i
    return deduped


def write_ranked_outputs(rows: list[dict]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    shortlist_cols = [
        "rank", "track", "entry_id", "strategy_fingerprint", "balanced_score", "fitness",
        "profit", "profit_factor", "sharpe_ratio", "sortino_ratio", "win_rate",
        "max_drawdown", "num_trades", "pair_generalization_ratio", "generation_found",
    ]
    with RANKED_SHORTLIST_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=shortlist_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in shortlist_cols})

    top = rows[:TOP_N]
    manifest_cols = [
        "rank", "track", "entry_id", "strategy_fingerprint", "balanced_score",
        "fitness", "profit", "max_drawdown", "num_trades", "pair_generalization_ratio",
    ]
    with SEED_MANIFEST_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_cols)
        w.writeheader()
        for r in top:
            w.writerow({k: r.get(k) for k in manifest_cols})

    seed_inds = []
    now = datetime.utcnow().isoformat()
    for i, r in enumerate(top):
        gene = dict(r["strategy_gene"])
        gene["generation"] = 0
        gene["individual_id"] = 9000 + i
        seed_inds.append(
            {
                "id": f"Seed_{i+1}",
                "strategy_gene": gene,
                "fitness": r["fitness"],
                "raw_fitness": r["fitness"],
                "objectives": [],
                "rank": i,
                "crowding_distance": 0.0,
                "metrics": {
                    "source": "wave45_shortlist",
                    "source_track": r["track"],
                    "balanced_score": r["balanced_score"],
                    "profit": r["profit"],
                    "max_drawdown": r["max_drawdown"],
                    "num_trades": r["num_trades"],
                    "pair_generalization_ratio": r["pair_generalization_ratio"],
                },
                "created_at": now,
                "evaluated": True,
                "parent_ids": [],
                "mutations": [],
            }
        )

    with SEED_CHECKPOINT_OUT.open("w") as f:
        json.dump({"population": {"individuals": seed_inds}}, f, indent=2)

    return top


def base_cfg(name, seed, mutation_rate, tournament_size, elite_size, complexity_weight, max_dd, val_pair):
    training = [p for p in PAIRS if p != val_pair]

    return {
        "experiment_name": name,
        "genetic_algorithm": {
            "random_seed": seed,
            "population_size": 48,
            "generations": 32,
            "mutation_rate": mutation_rate,
            "crossover_rate": 0.70,
            "elite_size": elite_size,
            "tournament_size": tournament_size,
            "selection_method": "rank",
            "convergence_patience": 12,
            "adaptive_mutation": True,
            "max_adaptation_factor": 3.0,
            "adaptation_step": 0.18,
            "mutation_cooldown_factor": 0.50,
            "fitness_sharing": True,
            "sharing_radius": 0.34,
            "diversity_threshold": 0.14,
            "allow_self_crossover": False,
            "random_immigrants": 4,
            "mode": "single_objective",
        },
        # Force standard GA path so warm_start loads from seed checkpoint.
        "generic_island_model": {
            "enabled": False,
            "num_islands": 10,
            "population_per_island": 36,
            "parallel_islands": False,
            "specialization": {
                "rotate_seeds": True,
                "indicator_pools": True,
                "indicator_overlap": 0.45,
                "pair_rotation": False,
            },
            "migration": {
                "topology": "fully_connected",
                "interval": 5,
                "count": 2,
                "merge_rounds": True,
                "merge_interval": 10,
            },
        },
        "fitness_weights": {
            "trade_frequency": 0.16,
            "win_rate": 0.15,
            "sharpe_ratio": 0.15,
            "drawdown": 0.16,
            "profit": 0.14,
            "sortino_ratio": 0.10,
            "profit_factor": 0.08,
            "monthly_stability": 0.03,
            "cross_pair": 0.03,
        },
        "fitness_bounds": {
            "profit_min": -20.0,
            "profit_max": 120.0,
            "sharpe_min": -5,
            "sharpe_max": 18.0,
            "sortino_min": -5,
            "sortino_max": 14.0,
            "profit_factor_max": 10,
            "profit_factor_normalization": 3.0,
        },
        "fitness_penalties": {
            "min_trades": MIN_TRADES,
            "min_trades_per_month": 2.5,
            "min_trades_per_month_penalty": 0.08,
            "max_drawdown": max_dd,
            "min_win_rate": 0.45,
            "complexity_weight": complexity_weight,
        },
        "backtesting": {
            "timerange": "20210101-20260328",
            "timeframe": "1h",
            "datadir": "user_data/data/binance",
            "stake_amount": 100,
            "pairs": PAIRS,
            "max_open_trades": 3,
            "fee": 0.001,
            "exchange": "binance",
            "auto_download_data": False,
            "enable_cache": True,
            "timeout": 300,
        },
        "pair_validation": {
            "enabled": True,
            "training_pairs": training,
            "validation_pairs": [val_pair],
            "weight_train": 0.72,
            "weight_val": 0.28,
            "min_val_fitness": 0.02,
            "validate_top_n_only": 12,
        },
        "parallel_evaluation": {"enabled": False, "num_workers": 1},
        "walk_forward": {"enabled": False},
        "holdout_validation": {"enabled": False},
        "holdout_monitoring": {"enabled": False},
        "monte_carlo": {"enabled": False},
        "deflated_sharpe": {"enabled": True, "min_trades": 30},
        "strategy_constraints": {
            "min_trades": MIN_TRADES,
            "max_drawdown": max_dd,
            "min_win_rate": 0.45,
            "timeframes": ["1h"],
            "stoploss_range": [-0.13, -0.06],
            "roi_range": [0.005, 0.03],
            "max_open_trades_range": [1, 3],
            "max_indicators": 4,
        },
        "warm_start": {
            "enabled": True,
            "source_type": "population",
            "source_checkpoint": str(SEED_CHECKPOINT_OUT.relative_to(BASE)),
            "top_n": TOP_N,
        },
        "storage": {
            "database": f"genetic_algorithm/data/{name}.db",
            "strategy_dir": f"user_data/strategies/{name}",
            "checkpoint_dir": f"genetic_algorithm/data/checkpoints_{name}",
            "checkpoint_interval": 5,
            "keep_history": True,
        },
        "hall_of_fame": {
            "enabled": True,
            "max_size": 30,
            "inject_count": 6,
            "min_fitness": 0.10,
            "directory": f"genetic_algorithm/data/hall_of_fame_{name}",
        },
        "terminal_monitor": {"enabled": True, "default_mode": "simple"},
        "logging": {"level": "INFO", "file": f"genetic_algorithm/logs/{name}.log", "console": True},
    }


def write_yaml(path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# {cfg['experiment_name']}\n")
        f.write("# Auto-generated by scripts/generate_wave45_tracks.py\n\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def main():
    raw = load_wave44_candidates()
    gated = gate_candidates(raw)
    ranked = rank_and_dedupe(gated)

    if len(ranked) < 10:
        raise RuntimeError(
            f"Only {len(ranked)} ranked candidates after gates; cannot safely seed Wave45."
        )

    top = write_ranked_outputs(ranked)

    track_defs = [
        ("38_wave45_A_1h_conservative_5of6.yaml", "wave45_A_1h_conservative_5of6", 4501, 0.15, 4, 5, 0.014, 0.18, "XRP/USDT"),
        ("39_wave45_B_1h_balanced_5of6.yaml", "wave45_B_1h_balanced_5of6", 4502, 0.19, 5, 4, 0.010, 0.20, "SOL/USDT"),
        ("40_wave45_C_1h_exploratory_5of6.yaml", "wave45_C_1h_exploratory_5of6", 4503, 0.27, 4, 3, 0.008, 0.22, "PEPE/USDT"),
        ("41_wave45_D_1h_stressrobust_5of6.yaml", "wave45_D_1h_stressrobust_5of6", 4504, 0.21, 6, 4, 0.016, 0.16, "BNB/USDT"),
    ]

    written = []
    for fname, exp, seed, mut, tourn, elite, cpx, dd, valp in track_defs:
        cfg = base_cfg(exp, seed, mut, tourn, elite, cpx, dd, valp)
        out = QUEUE_DIR / fname
        write_yaml(out, cfg)
        written.append(out)

    print(f"Loaded Wave44 HOF candidates: {len(raw)}")
    print(f"After hard gates: {len(gated)}")
    print(f"Ranked unique shortlist: {len(ranked)}")
    print(f"Wrote shortlist: {RANKED_SHORTLIST_OUT}")
    print(f"Wrote seed manifest: {SEED_MANIFEST_OUT}")
    print(f"Wrote seed checkpoint: {SEED_CHECKPOINT_OUT}")
    print("Top 10 Wave45 seeds:")
    for r in top[:10]:
        print(
            f"  rank={r['rank']:>2} track={r['track']} score={r['balanced_score']:.3f} "
            f"fit={r['fitness']:.3f} profit={r['profit']:.2f} dd={r['max_drawdown']:.3f} "
            f"trades={r['num_trades']} pgr={r['pair_generalization_ratio']:.3f}"
        )
    print("\nQueue files written:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
