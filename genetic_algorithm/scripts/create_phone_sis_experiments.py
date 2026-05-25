#!/usr/bin/env python3
"""Create phone-safe SIS validation experiment configs.

The generated configs are intentionally tiny and sequential-run friendly:
no LLM, no walk-forward/holdout, no parallel workers, small island model,
short timerange, isolated phone_sis output/log/data paths.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO / "genetic_algorithm/config/sis_comparison/v2/ctrl_no_sis.yaml"
OUT_DIR = REPO / "genetic_algorithm/config/phone_sis"
PRIORS = "genetic_algorithm/intelligence/priors/data_driven_enrichment.json"
CORPUS = "genetic_algorithm/data/strategy_corpus_clustered.parquet"
MODELS = "genetic_algorithm/ml/models"


def deep_update(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def phone_common(name: str, *, seed: int = 42, smoke: bool = False) -> Dict[str, Any]:
    population = 4 if smoke else 8
    generations = 2 if smoke else 6
    island_pop = 4
    return {
        "experiment_name": name,
        "genetic_algorithm": {
            "random_seed": seed,
            "population_size": population,
            "generations": generations,
            "mutation_rate": 0.20,
            "crossover_rate": 0.70,
            "elite_size": 1,
            "tournament_size": 2,
            "selection_method": "tournament",
            "convergence_patience": 4,
            "adaptive_mutation": True,
            "fitness_sharing": False,
            "allow_self_crossover": False,
            "random_immigrants": 1,
            "mode": "single_objective",
        },
        "surrogate": {"enabled": False},
        "generic_island_model": {
            "enabled": not smoke,
            "num_islands": 2,
            "population_per_island": island_pop,
            "generations": generations,
            "parallel_islands": False,
            "specialization": {
                "rotate_seeds": True,
                "indicator_pools": True,
                "indicator_overlap": 0.5,
                "pair_rotation": False,
            },
            "migration": {
                "topology": "ring",
                "interval": 2,
                "count": 1,
                "merge_rounds": False,
            },
        },
        "fitness_penalties": {
            "min_trades": 3,
            "min_trades_per_month": 1,
            "min_trades_per_month_penalty": 0.02,
            "max_drawdown": 0.50,
            "min_win_rate": 0.20,
            "complexity_weight": 0.005,
        },
        "backtesting": {
            "timerange": "20250101-20250301",
            "stake_amount": 100,
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "max_open_trades": 3,
            "fee": 0.001,
            "exchange": "binance",
            "auto_download_data": True,
            "enable_cache": True,
            "timeout": 120,
            "timeframes": ["15m"],
        },
        "walk_forward": {"enabled": False},
        "holdout_validation": {"enabled": False},
        "holdout_monitoring": {"enabled": False},
        "monte_carlo": {"enabled": False},
        "pair_validation": {
            "enabled": not smoke,
            "training_pairs": ["BTC/USDT"],
            "validation_pairs": ["ETH/USDT"],
            "weight_train": 0.70,
            "weight_val": 0.30,
            "min_val_fitness": -1.0,
            "validate_top_n_only": 2,
        },
        "strategy_constraints": {
            "min_trades": 3,
            "max_drawdown": 0.50,
            "min_win_rate": 0.20,
            "timeframes": ["15m"],
            "stoploss_range": [-0.12, -0.02],
            "roi_range": [0.005, 0.06],
            "max_open_trades_range": [1, 3],
            "trailing_stop_positive_range": [0.005, 0.03],
            "trailing_stop_offset_addition_range": [0.005, 0.03],
        },
        "indicators": {
            "available": [
                "RSI",
                "MACD",
                "BBANDS",
                "EMA",
                "SMA",
                "ADX",
                "ATR",
                "CCI",
                "STOCH",
                "DONCHIAN",
                "ROC",
                "AROON",
                "MFI",
                "WILLR",
                "TEMA",
                "KAMA",
                "PSAR",
            ],
            "min_per_strategy": 2,
            "max_per_strategy": 4,
        },
        "storage": {
            "database": f"genetic_algorithm/data/phone_sis/{name}.db",
            "strategy_dir": f"user_data/strategies/phone_sis/{name}",
            "checkpoint_dir": f"genetic_algorithm/data/phone_sis/{name}_checkpoints",
            "checkpoint_interval": 2,
            "keep_history": True,
            "min_disk_gb": 1.0,
            "min_disk_gb_runtime": 1.0,
            "max_cache_disk_mb": 1000,
        },
        "logging": {
            "level": "INFO",
            "file": f"genetic_algorithm/logs/phone_sis/{name}.log",
            "console": True,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "visualization": {"enabled": False},
        "terminal_monitor": {"enabled": False},
        "hall_of_fame": {
            "enabled": True,
            "max_size": 10,
            "inject_count": 1,
            "min_fitness": -1.0,
            "directory": f"genetic_algorithm/data/phone_sis/{name}_hof",
        },
        "advanced": {
            "parallel_evaluation": False,
            "max_workers": 1,
            "enable_dry_run": False,
            "enable_ml_fitness": False,
            "llm": {"enabled": False},
        },
        "parallel_evaluation": {
            "enabled": False,
            "num_workers": 1,
            "worker_log_level": "WARNING",
            "backtest_timeout": 120,
        },
        "output": {
            "dir": f"genetic_algorithm/output/phone_sis/{name}",
            "save_stats": True,
            "stats_file": "evolution_stats.json",
            "save_snapshots": True,
            "snapshot_interval": 1,
        },
    }


def sis_block(
    name: str,
    *,
    enabled: bool,
    priors_file: Optional[str] = None,
    seed_filtering: bool = True,
    immigrants: bool = True,
    indicator_weights: bool = True,
    operator_weights: bool = True,
    synergy_weights: bool = True,
) -> Dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    block: Dict[str, Any] = {
        "enabled": True,
        "corpus_path": CORPUS,
        "models_dir": MODELS,
        "filter_fraction": 0.15 if seed_filtering else 0.0,
        "immigrants_per_gen": 1 if immigrants else 0,
        "indicator_weight_scale": 1.5,
        "operator_weight_scale": 1.2,
        "log_file": f"genetic_algorithm/logs/phone_sis/{name}_sis_events.jsonl",
        "auto_rebuild": False,
        "hooks": {
            "seed_filtering": seed_filtering,
            "immigrants": immigrants,
            "indicator_weights": indicator_weights,
            "operator_weights": operator_weights,
            "synergy_weights": synergy_weights,
        },
        "anti_pattern": {
            "scale": 0.5,
            "floor": 0.6,
        },
    }
    if priors_file:
        block["priors_file"] = priors_file
    return block


def make_config(base: Dict[str, Any], name: str, sis: Dict[str, Any], *, smoke: bool = False) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    deep_update(cfg, phone_common(name, smoke=smoke))
    cfg["sis"] = sis
    return cfg


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with BASE_CONFIG.open("r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    experiments = [
        ("00_P0_smoke_no_sis.yaml", make_config(base, "phone_sis_P0_smoke_no_sis", sis_block("phone_sis_P0_smoke_no_sis", enabled=False), smoke=True)),
        ("01_P1_control_no_sis.yaml", make_config(base, "phone_sis_P1_control_no_sis", sis_block("phone_sis_P1_control_no_sis", enabled=False))),
        ("02_P2_sis_hardcoded.yaml", make_config(base, "phone_sis_P2_sis_hardcoded", sis_block("phone_sis_P2_sis_hardcoded", enabled=True))),
        ("03_P3_sis_data_priors.yaml", make_config(base, "phone_sis_P3_sis_data_priors", sis_block("phone_sis_P3_sis_data_priors", enabled=True, priors_file=PRIORS))),
        ("04_P4_sis_immigrants_only.yaml", make_config(base, "phone_sis_P4_sis_immigrants_only", sis_block("phone_sis_P4_sis_immigrants_only", enabled=True, seed_filtering=False, immigrants=True, indicator_weights=False, operator_weights=False, synergy_weights=False))),
        ("05_P5_sis_weights_only.yaml", make_config(base, "phone_sis_P5_sis_weights_only", sis_block("phone_sis_P5_sis_weights_only", enabled=True, seed_filtering=False, immigrants=False, indicator_weights=True, operator_weights=True, synergy_weights=True))),
    ]

    for filename, cfg in experiments:
        path = OUT_DIR / filename
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, width=100)
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
