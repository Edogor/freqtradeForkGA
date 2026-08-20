"""
Wave 39-43 Generator (25 configs, 5 per wave).
Based on W38-D template (known-working) + W28-A1 success components.

Design philosophy:
- W39 = PROVEN PATH (replicate what works)
- W40 = MORE EXPLORATION (mutation, MTF, sortino, indicators)
- W41 = EVEN MORE EXPLORATION (NSGA, regime, extreme islands, hybrid TF, more pairs)
- W42 = ROBUSTNESS (5x seed sweep of best config = reproducibility test)
- W43 = NOVEL IDEAS (self_crossover, aggressive adaptive, overlap, tournament, no_elitism)
"""
import copy, os, yaml, sys

OUT_DIR = "genetic_algorithm/config/queue"

# ---------------- Indicator pools ----------------

# Indicators that produced 0% or near-0% Tier-1 strategies historically
INDICATOR_KILLERS = {
    "ADX", "STOCH", "CMF", "WILLR", "PSAR", "KAMA",
    "RSI", "MACD", "SUPERTREND", "CDL_DARKCLOUD",
}

# Indicators that produced highest T1 yield
TIER1_WINNERS = [
    "ROC", "DONCHIAN", "SMA", "AROON", "CCI", "VWAP",
    "OBV", "TEMA", "BBANDS", "CDL_3BLACKCROWS",
    "CDL_DOJI", "EMA", "ICHIMOKU", "SAR",
]

# Full broad pool (excludes killers)
FULL_POOL = TIER1_WINNERS + [
    "MFI", "ATR", "STOCHF", "CDL_3WHITESOLDIERS",
]

# ---------------- Base template (mirrors W38-D but with W28-A1 success genes) ----------------

def base_config(name, timeframe="1h"):
    """Build base config with proven defaults."""
    is_1h = timeframe == "1h"
    cfg = {
        "experiment_name": name,
        "genetic_algorithm": {
            "random_seed": None,
            "population_size": 30,
            "generations": 35 if is_1h else 25,
            "mutation_rate": 0.18,
            "crossover_rate": 0.70,
            "elite_size": 3,
            "tournament_size": 4,
            "selection_method": "rank",
            "convergence_patience": 20,
            "adaptive_mutation": True,
            "max_adaptation_factor": 2.5,
            "adaptation_step": 0.12,
            "mutation_cooldown_factor": 0.5,
            # W28-A1 KEY DIFFERENTIATOR: fitness sharing enabled with broad radius
            "fitness_sharing": True,
            "sharing_radius": 0.30,
            "diversity_threshold": 0.12,
            "allow_self_crossover": False,
            "random_immigrants": 3,
            "mode": "single_objective",
        },
        "generic_island_model": {
            "enabled": True,
            "num_islands": 10 if is_1h else 8,
            "population_per_island": 30,
            "parallel_islands": False,  # NO WORKERS, per user pref
            "specialization": {
                "rotate_seeds": True,
                "indicator_pools": True,
                "indicator_overlap": 0.40,  # W28-A1 used 0.4
                "pair_rotation": False,
            },
            "migration": {
                # W28-A1 used fully_connected + merge_rounds — RESTORE THIS
                "topology": "fully_connected",
                "interval": 5,
                "count": 2,
                "merge_rounds": True,
                "merge_interval": 10,
            },
        },
        # W28-A1 fitness weights (favoring win_rate + trade_frequency over raw profit)
        "fitness_weights": {
            "trade_frequency": 0.21,
            "win_rate": 0.17,
            "sharpe_ratio": 0.15,
            "drawdown": 0.13,
            "profit": 0.10,
            "sortino_ratio": 0.09,
            "profit_factor": 0.07,
            "monthly_stability": 0.04,
            "cross_pair": 0.04,
        },
        "fitness_bounds": {
            "profit_min": -20.0, "profit_max": 100.0,
            "sharpe_min": -5, "sharpe_max": 15.0,
            "sortino_min": -5, "sortino_max": 12.0,
            "profit_factor_max": 10,
            "profit_factor_normalization": 3.0,
        },
        "trade_frequency_thresholds": {
            "auto_scale_by_timeframe": True,
            "shape": "gaussian",
        },
        "fitness_penalties": {
            "min_trades": 50 if is_1h else 20,
            "min_trades_per_month": 1.5 if is_1h else 1.0,
            "min_trades_per_month_penalty": 0.05,
            "max_drawdown": 0.20,
            "min_win_rate": 0.40,
            "complexity_weight": 0.01,
        },
        "backtesting": {
            "timerange": "20210101-20260328",
            "timeframe": timeframe,
            "datadir": "user_data/data/binance",
            "stake_amount": 100,
            # W28-A1 used 6 pairs incl. PEPE
            "pairs": [
                "BTC/USDT", "ETH/USDT", "BNB/USDT",
                "SOL/USDT", "XRP/USDT", "PEPE/USDT",
            ],
            "max_open_trades": 3,
            "fee": 0.001,
            "exchange": "binance",
            "auto_download_data": False,
            "enable_cache": True,
            "timeout": 300,
        },
        # Pair-split validation (no WF, no holdout, per user pref)
        "pair_validation": {
            "enabled": True,
            "training_pairs": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "PEPE/USDT"],
            "validation_pairs": ["SOL/USDT", "XRP/USDT"],
            "weight_train": 0.6, "weight_val": 0.4,
            "min_val_fitness": 0.0,
            "validate_top_n_only": 10,
        },
        "parallel_evaluation": {"enabled": False, "num_workers": 1},
        "walk_forward": {"enabled": False},
        "holdout_validation": {"enabled": False},
        "holdout_monitoring": {"enabled": False},
        "monte_carlo": {"enabled": False},
        "deflated_sharpe": {
            "enabled": True,
            "min_trades": 30 if is_1h else 12,
        },
        "strategy_constraints": {
            "min_trades": 50 if is_1h else 20,
            "max_drawdown": 0.20,
            "min_win_rate": 0.40,
            "timeframes": [timeframe],
            "stoploss_range": [-0.13, -0.07] if is_1h else [-0.15, -0.04],
            "roi_range": [0.005, 0.03] if is_1h else [0.015, 0.10],
            "max_open_trades_range": [1, 4],
            "max_indicators": 4,
        },
        "surrogate_model": {
            "enabled": True,
            "min_samples": 40,
            "use_after_gen": 4,
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
            "max_size": 25,
            "inject_count": 2,
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


# ---------------- WAVE 39 — PROVEN PATH ----------------

def wave39_A():
    """1h replication of wave28_A1 success — full 6 pairs incl PEPE, 10×30 islands."""
    c = base_config("wave39_A_1h_w28a1_replication", "1h")
    # Already incorporates W28-A1 settings via base. Just bump pop slightly.
    c["genetic_algorithm"]["population_size"] = 30
    c["generic_island_model"]["num_islands"] = 10
    return c

def wave39_B():
    """1h time-holdout: train on 2021-2024-01 only. Manual holdout test afterwards."""
    c = base_config("wave39_B_1h_time_holdout_2021_2024", "1h")
    c["backtesting"]["timerange"] = "20210101-20240101"
    # tighter min_trades since shorter timerange
    c["strategy_constraints"]["min_trades"] = 40
    c["fitness_penalties"]["min_trades"] = 40
    return c

def wave39_C():
    """4h replication of W28-A1 style (extends successful W38-D into proven framework)."""
    c = base_config("wave39_C_4h_w28a1_style", "4h")
    c["generic_island_model"]["num_islands"] = 8
    c["genetic_algorithm"]["generations"] = 30
    return c

def wave39_D():
    """1h with strictly filtered indicator pool — only Tier-1 winners."""
    c = base_config("wave39_D_1h_filtered_winners", "1h")
    c["strategy_constraints"]["allowed_indicators"] = TIER1_WINNERS[:]
    c["indicators"] = {"available": TIER1_WINNERS[:]}
    return c

def wave39_E():
    """1h cost-stress: realistic live-trading fees + slippage."""
    c = base_config("wave39_E_1h_cost_stress", "1h")
    c["backtesting"]["fee"] = 0.0015  # 0.15% per side (realistic taker)
    c["backtesting"]["slippage_pct"] = 0.001  # 0.1% slippage estimate
    # Compensate: require higher profit per trade
    c["strategy_constraints"]["roi_range"] = [0.008, 0.035]
    return c

# ---------------- WAVE 40 — MORE EXPLORATION ----------------

def wave40_A():
    """1h HIGH mutation (0.35) + wide sharing_radius (0.50) → escape local optima."""
    c = base_config("wave40_A_1h_high_mutation_diversity", "1h")
    c["genetic_algorithm"]["mutation_rate"] = 0.35
    c["genetic_algorithm"]["sharing_radius"] = 0.50
    c["genetic_algorithm"]["random_immigrants"] = 5
    c["genetic_algorithm"]["diversity_threshold"] = 0.20
    return c

def wave40_B():
    """4h sortino-dominant fitness — penalize downside vol harder."""
    c = base_config("wave40_B_4h_sortino_dominant", "4h")
    c["fitness_weights"] = {
        "sortino_ratio": 0.30,
        "drawdown": 0.20,
        "win_rate": 0.15,
        "profit": 0.12,
        "trade_frequency": 0.10,
        "sharpe_ratio": 0.07,
        "profit_factor": 0.03,
        "monthly_stability": 0.03,
    }
    return c

def wave40_C():
    """1h with 4h informative_timeframe (multi-timeframe context, first MTF try)."""
    c = base_config("wave40_C_1h_with_4h_informative", "1h")
    c["backtesting"]["informative_timeframes"] = ["4h"]
    c["strategy_constraints"]["informative_indicators"] = True
    return c

def wave40_D():
    """4h with broad indicator pool — let GA explore new combos with longer TF."""
    c = base_config("wave40_D_4h_broad_pool", "4h")
    c["indicators"] = {"available": FULL_POOL[:]}
    c["strategy_constraints"]["max_indicators"] = 5
    return c

def wave40_E():
    """1h short-enabled — first GA run with can_short=True."""
    c = base_config("wave40_E_1h_short_enabled", "1h")
    c["backtesting"]["can_short"] = True
    c["strategy_constraints"]["can_short"] = True
    return c

# ---------------- WAVE 41 — EVEN MORE EXPLORATION ----------------

def wave41_A():
    """1h NSGA-II multi-objective: profit + sharpe + drawdown as 3 objectives."""
    c = base_config("wave41_A_1h_nsga2_multi_obj", "1h")
    c["genetic_algorithm"]["mode"] = "multi_objective"
    c["genetic_algorithm"]["nsga2"] = True
    c["genetic_algorithm"]["objectives"] = ["profit", "sharpe_ratio", "drawdown"]
    return c

def wave41_B():
    """4h regime-aware island specialists (bull/bear/sideways)."""
    c = base_config("wave41_B_4h_regime_specialists", "4h")
    c["generic_island_model"]["specialization"]["regime_specialization"] = True
    c["generic_island_model"]["specialization"]["regimes"] = ["bull", "bear", "sideways"]
    return c

def wave41_C():
    """1h EXTREME islands: 20 islands × 20 pop = 400 per gen, max diversity."""
    c = base_config("wave41_C_1h_extreme_islands", "1h")
    c["generic_island_model"]["num_islands"] = 20
    c["generic_island_model"]["population_per_island"] = 20
    c["genetic_algorithm"]["population_size"] = 20
    c["genetic_algorithm"]["generations"] = 30  # cap runtime
    return c

def wave41_D():
    """Hybrid timeframe per island — some islands 1h, some 4h. Cross-pollination."""
    c = base_config("wave41_D_hybrid_tf_islands", "1h")
    c["strategy_constraints"]["timeframes"] = ["1h", "4h"]
    c["generic_island_model"]["specialization"]["timeframe_specialization"] = True
    c["generic_island_model"]["specialization"]["island_timeframes"] = [
        "1h", "1h", "1h", "1h", "1h", "4h", "4h", "4h", "4h", "4h"
    ]
    return c

def wave41_E():
    """1h with 8 pairs (add DOGE, AVAX, MATIC) — more diversification."""
    c = base_config("wave41_E_1h_8pairs", "1h")
    c["backtesting"]["pairs"] = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "XRP/USDT", "PEPE/USDT", "DOGE/USDT", "AVAX/USDT",
    ]
    c["pair_validation"]["training_pairs"] = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "PEPE/USDT", "DOGE/USDT"
    ]
    c["pair_validation"]["validation_pairs"] = ["SOL/USDT", "XRP/USDT", "AVAX/USDT"]
    return c

# ---------------- WAVE 42 — ROBUSTNESS / SEED SWEEP ----------------

def wave42_seed(letter, seed):
    """Same as W39-A but with fixed seed — to test reproducibility."""
    c = base_config(f"wave42_{letter}_seed_{seed}_robustness", "1h")
    c["genetic_algorithm"]["random_seed"] = seed
    return c

def wave42_A(): return wave42_seed("A", 42)
def wave42_B(): return wave42_seed("B", 1337)
def wave42_C(): return wave42_seed("C", 271828)
def wave42_D(): return wave42_seed("D", 31415)
def wave42_E(): return wave42_seed("E", 99991)

# ---------------- WAVE 43 — NOVEL IDEAS ----------------

def wave43_A():
    """1h with self_crossover allowed + larger random_immigrants pool."""
    c = base_config("wave43_A_1h_self_crossover", "1h")
    c["genetic_algorithm"]["allow_self_crossover"] = True
    c["genetic_algorithm"]["random_immigrants"] = 6
    return c

def wave43_B():
    """1h aggressive adaptive mutation (factor 4) — strong response to stagnation."""
    c = base_config("wave43_B_1h_aggressive_adaptive", "1h")
    c["genetic_algorithm"]["max_adaptation_factor"] = 4.0
    c["genetic_algorithm"]["adaptation_step"] = 0.25
    c["genetic_algorithm"]["convergence_patience"] = 8  # trigger fast
    return c

def wave43_C():
    """1h EXTREME tournament size (8) — strong selection pressure."""
    c = base_config("wave43_C_1h_strong_selection", "1h")
    c["genetic_algorithm"]["tournament_size"] = 8
    c["genetic_algorithm"]["elite_size"] = 6
    return c

def wave43_D():
    """1h VERY high indicator_overlap (0.80) — heavy gene sharing across islands."""
    c = base_config("wave43_D_1h_high_overlap", "1h")
    c["generic_island_model"]["specialization"]["indicator_overlap"] = 0.80
    return c

def wave43_E():
    """1h penalty-heavy: strict PGR + min_win_rate + min_trades 150."""
    c = base_config("wave43_E_1h_strict_penalties", "1h")
    c["fitness_penalties"]["min_trades"] = 150
    c["fitness_penalties"]["min_win_rate"] = 0.50
    c["fitness_penalties"]["max_drawdown"] = 0.12
    c["strategy_constraints"]["min_trades"] = 150
    c["strategy_constraints"]["min_win_rate"] = 0.50
    c["strategy_constraints"]["max_drawdown"] = 0.12
    # Reward pair generalization more heavily
    c["pair_validation"]["weight_train"] = 0.45
    c["pair_validation"]["weight_val"] = 0.55
    c["pair_validation"]["min_val_fitness"] = 0.25
    return c

# ---------------- Build all ----------------

WAVES = {
    # Wave 39 — PROVEN PATH
    "09_wave39_A_1h_w28a1_replication.yaml":    wave39_A,
    "10_wave39_B_1h_time_holdout.yaml":         wave39_B,
    "11_wave39_C_4h_w28a1_style.yaml":          wave39_C,
    "12_wave39_D_1h_filtered_winners.yaml":     wave39_D,
    "13_wave39_E_1h_cost_stress.yaml":          wave39_E,
    # Wave 40 — MORE EXPLORATION
    "14_wave40_A_1h_high_mutation.yaml":        wave40_A,
    "15_wave40_B_4h_sortino_dominant.yaml":     wave40_B,
    "16_wave40_C_1h_with_4h_informative.yaml":  wave40_C,
    "17_wave40_D_4h_broad_pool.yaml":           wave40_D,
    "18_wave40_E_1h_short_enabled.yaml":        wave40_E,
    # Wave 41 — EVEN MORE EXPLORATION
    "19_wave41_A_1h_nsga2.yaml":                wave41_A,
    "20_wave41_B_4h_regime_specialists.yaml":   wave41_B,
    "21_wave41_C_1h_extreme_islands.yaml":      wave41_C,
    "22_wave41_D_hybrid_tf_islands.yaml":       wave41_D,
    "23_wave41_E_1h_8pairs.yaml":               wave41_E,
    # Wave 42 — ROBUSTNESS (seed sweep)
    "24_wave42_A_seed_42.yaml":                 wave42_A,
    "25_wave42_B_seed_1337.yaml":               wave42_B,
    "26_wave42_C_seed_271828.yaml":             wave42_C,
    "27_wave42_D_seed_31415.yaml":              wave42_D,
    "28_wave42_E_seed_99991.yaml":              wave42_E,
    # Wave 43 — NOVEL IDEAS
    "29_wave43_A_1h_self_crossover.yaml":       wave43_A,
    "30_wave43_B_1h_aggressive_adaptive.yaml":  wave43_B,
    "31_wave43_C_1h_strong_selection.yaml":     wave43_C,
    "32_wave43_D_1h_high_overlap.yaml":         wave43_D,
    "33_wave43_E_1h_strict_penalties.yaml":     wave43_E,
}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for fname, fn in WAVES.items():
        cfg = fn()
        path = os.path.join(OUT_DIR, fname)
        header = f"# {cfg['experiment_name']}\n# Auto-generated by gen_waves.py\n\n"
        with open(path, "w") as f:
            f.write(header)
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        written.append(path)
        print(f"  wrote {path}")
    print(f"\nTotal: {len(written)} configs written.")

if __name__ == "__main__":
    main()
