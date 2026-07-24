#!/usr/bin/env python3
"""
Generate Wave44 multi-search configs from latest backtest matrix.

Goals:
- Start from current best candidates (no scratch)
- No walk-forward/holdout
- Multi-pair validation focus (strict global bias)
- Emit 4 track YAML files for queue daemon
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import yaml

BASE = Path(__file__).resolve().parents[1]
MATRIX_DIR = BASE / "user_data/backtest_results/matrix_20260715"
STRAT_CSV = MATRIX_DIR / "matrix_strategy_summary.csv"
PAIR_CSV = MATRIX_DIR / "matrix_pair_summary.csv"
MANIFEST_OUT = BASE / "genetic_algorithm/data/wave44_seed_manifest.csv"
QUEUE_DIR = BASE / "genetic_algorithm/config/queue"

PAIRS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "PEPE/USDT"]


def zscore_map(rows, key):
    vals = [float(r[key]) for r in rows]
    mu = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    if sd == 0:
        return {r["strategy"]: 0.0 for r in rows}
    return {r["strategy"]: (float(r[key]) - mu) / sd for r in rows}


def load_strategy_rows():
    rows = []
    with STRAT_CSV.open() as f:
        for r in csv.DictReader(f):
            if int(float(r["trades"])) <= 0:
                continue
            rows.append(r)
    return rows


def pair_pass_counts():
    counts = {}
    with PAIR_CSV.open() as f:
        for r in csv.DictReader(f):
            s = r["strategy"]
            p = r["pair"]
            prof = float(r["profit_total_abs"])
            counts.setdefault(s, {"profitable": 0, "pairs": set()})
            counts[s]["pairs"].add(p)
            if prof > 0:
                counts[s]["profitable"] += 1
    return {k: v["profitable"] for k, v in counts.items()}


def build_seed_manifest(rows):
    pass_counts = pair_pass_counts()

    z_profit = zscore_map(rows, "profit_total_abs")
    z_pf = zscore_map(rows, "profit_factor")
    z_sh = zscore_map(rows, "sharpe")
    z_dd = zscore_map(rows, "max_drawdown_account_pct")
    z_wr = zscore_map(rows, "winrate")

    enriched = []
    for r in rows:
        s = r["strategy"]
        pair_pass = pass_counts.get(s, 0)
        # Strict gate: must pass >= 5 of 6 pairs
        if pair_pass < 5:
            continue

        score = (
            1.00 * z_profit[s]
            + 0.70 * z_pf[s]
            + 0.70 * z_sh[s]
            + 0.30 * z_wr[s]
            - 1.10 * z_dd[s]
        )

        enriched.append(
            {
                "strategy": s,
                "cohort": r["cohort"],
                "timeframe": r["timeframe"],
                "trades": int(float(r["trades"])),
                "winrate": float(r["winrate"]),
                "profit_total_abs": float(r["profit_total_abs"]),
                "profit_factor": float(r["profit_factor"]),
                "sharpe": float(r["sharpe"]),
                "max_drawdown_account_pct": float(r["max_drawdown_account_pct"]),
                "pair_pass_count": pair_pass,
                "balanced_score": score,
            }
        )

    enriched.sort(key=lambda x: x["balanced_score"], reverse=True)
    top = enriched[:30]

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0].keys()))
        w.writeheader()
        w.writerows(top)

    return top


def base_cfg(name, seed, mutation_rate, tournament_size, elite_size, complexity_weight, max_dd, val_pair):
    training = [p for p in PAIRS if p != val_pair]

    cfg = {
        "experiment_name": name,
        "genetic_algorithm": {
            "random_seed": seed,
            "population_size": 36,
            "generations": 28,
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
        "generic_island_model": {
            "enabled": True,
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
        # Balanced objective (profit + risk + stability + pair behavior)
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
            "min_trades": 350,
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
        # Multi-pair validation only (no walk-forward/holdout)
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
            "min_trades": 350,
            "max_drawdown": max_dd,
            "min_win_rate": 0.45,
            "timeframes": ["1h"],
            "stoploss_range": [-0.13, -0.06],
            "roi_range": [0.005, 0.03],
            "max_open_trades_range": [1, 3],
            "max_indicators": 4,
        },
        # Warm-start from best robust historical checkpoint
        "warm_start": {
            "enabled": True,
            "source_type": "population",
            "source_checkpoint": "genetic_algorithm/data/checkpoints_wave42_A_seed_42_robustness/island_checkpoint_gen9_20260526_084343.json",
            "top_n": 30,
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
        "logging": {
            "level": "INFO",
            "file": f"genetic_algorithm/logs/{name}.log",
            "console": True,
        },
    }
    return cfg


def write_yaml(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# {cfg['experiment_name']}\n")
        f.write("# Auto-generated by scripts/generate_wave44_tracks.py\n\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def main():
    rows = load_strategy_rows()
    top30 = build_seed_manifest(rows)

    if len(top30) < 10:
        raise RuntimeError(
            f"Only {len(top30)} seed candidates passed the strict 5/6 rule. "
            "Check matrix CSVs before launching wave44."
        )

    track_defs = [
        # Conservative robust
        ("34_wave44_A_1h_conservative_5of6.yaml", "wave44_A_1h_conservative_5of6", 4401, 0.16, 4, 5, 0.014, 0.18, "XRP/USDT"),
        # Balanced core
        ("35_wave44_B_1h_balanced_5of6.yaml", "wave44_B_1h_balanced_5of6", 4402, 0.20, 5, 4, 0.010, 0.20, "SOL/USDT"),
        # Exploratory diversity
        ("36_wave44_C_1h_exploratory_5of6.yaml", "wave44_C_1h_exploratory_5of6", 4403, 0.28, 4, 3, 0.008, 0.22, "PEPE/USDT"),
        # Stress-robust
        ("37_wave44_D_1h_stressrobust_5of6.yaml", "wave44_D_1h_stressrobust_5of6", 4404, 0.22, 6, 4, 0.016, 0.16, "BNB/USDT"),
    ]

    written = []
    for fname, exp, seed, mut, tourn, elite, cpx, dd, valp in track_defs:
        cfg = base_cfg(exp, seed, mut, tourn, elite, cpx, dd, valp)
        out = QUEUE_DIR / fname
        write_yaml(out, cfg)
        written.append(out)

    print(f"Seed manifest: {MANIFEST_OUT}")
    print(f"Seed candidates (strict 5/6 profitable pairs): {len(top30)}")
    print("Top 10 seeds by balanced score:")
    for row in top30[:10]:
        print(f"  {row['strategy']:<30} score={row['balanced_score']:.3f} pass={row['pair_pass_count']}/6 trades={row['trades']}")

    print("\nQueue files written:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
