"""
GA Config Validator

Validates GA configuration at startup to catch misconfigurations early,
before spending time on evolution that would fail or produce bad results.

Returns a list of errors (critical, must fix) and warnings (suboptimal, but runs).
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def validate_ga_config(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Validate a GA configuration dictionary.
    
    Args:
        config: Full GA config dict (top-level keys: genetic_algorithm, backtesting, etc.)
        
    Returns:
        Tuple of (errors, warnings). Errors are fatal; warnings are informational.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # --- genetic_algorithm section ---
    ga = config.get('genetic_algorithm')
    if not ga:
        errors.append("Missing 'genetic_algorithm' section")
    else:
        _validate_int_range(ga, 'population_size', 2, 10000, errors)
        _validate_int_range(ga, 'generations', 1, 100000, errors)
        _validate_float_range(ga, 'mutation_rate', 0, 1, errors)
        _validate_float_range(ga, 'crossover_rate', 0, 1, errors)
        
        ps = ga.get('population_size', 0)
        es = ga.get('elite_size', 0)
        if isinstance(ps, int) and isinstance(es, int) and es >= ps:
            errors.append(f"elite_size ({es}) must be < population_size ({ps})")
        
        ts = ga.get('tournament_size', 3)
        if isinstance(ts, int) and isinstance(ps, int) and ts > ps:
            errors.append(f"tournament_size ({ts}) must be <= population_size ({ps})")
        
        mode = ga.get('mode', 'single_objective')
        if mode not in ('single_objective', 'nsga2'):
            errors.append(f"mode must be 'single_objective' or 'nsga2', got '{mode}'")
    
    # --- backtesting section ---
    bt = config.get('backtesting')
    if not bt:
        errors.append("Missing 'backtesting' section")
    else:
        pairs = bt.get('pairs', [])
        if not pairs:
            errors.append("backtesting.pairs must be a non-empty list")
        
        timerange = bt.get('timerange', '')
        if not timerange:
            warnings.append("backtesting.timerange is empty — will use all available data")
        elif '-' not in timerange:
            errors.append(f"backtesting.timerange format should be 'YYYYMMDD-YYYYMMDD', got '{timerange}'")
        
        datadir = bt.get('datadir', '')
        if not datadir:
            warnings.append("backtesting.datadir not set — will use FreqTrade default")
        
        fee = bt.get('fee', 0.001)
        if isinstance(fee, (int, float)) and fee > 0.05:
            warnings.append(f"backtesting.fee={fee} seems high (>5%). Typical range: 0.0005-0.002")
        
        fee_noise = bt.get('fee_noise_std', 0)
        if isinstance(fee_noise, (int, float)) and fee_noise > 0.01:
            warnings.append(f"backtesting.fee_noise_std={fee_noise} is large. Typical: 0.0001-0.0005")
    
    # --- fitness_weights section ---
    fw = config.get('fitness_weights', {})
    if fw:
        total = sum(v for v in fw.values() if isinstance(v, (int, float)))
        if abs(total - 1.0) > 0.10:
            warnings.append(f"fitness_weights sum to {total:.3f} (expected ~1.0, will be auto-normalized)")
        for key, val in fw.items():
            if isinstance(val, (int, float)) and val < 0:
                errors.append(f"fitness_weights.{key}={val} is negative (must be >= 0)")

        # Validate fitness weight keys against known valid keys
        from genetic_algorithm.evaluation.fitness import (
            VALID_FITNESS_WEIGHT_KEYS, FITNESS_WEIGHT_KEY_ALIASES,
        )
        for key in fw:
            if key in FITNESS_WEIGHT_KEY_ALIASES:
                canonical = FITNESS_WEIGHT_KEY_ALIASES[key]
                warnings.append(
                    f"fitness_weights.{key} is deprecated — use '{canonical}' instead "
                    f"(auto-resolved at runtime)")
            elif key not in VALID_FITNESS_WEIGHT_KEYS:
                errors.append(
                    f"fitness_weights.{key} is not a recognized key and will be ignored. "
                    f"Valid keys: {', '.join(sorted(VALID_FITNESS_WEIGHT_KEYS))}")
    
    # --- walk_forward section ---
    wf = config.get('walk_forward', {})
    if wf.get('enabled'):
        td = wf.get('train_days')
        vd = wf.get('validation_days')
        if td is not None and isinstance(td, (int, float)) and td < 7:
            errors.append(f"walk_forward.train_days={td} is too short (minimum 7)")
        if vd is not None and isinstance(vd, (int, float)) and vd < 1:
            errors.append(f"walk_forward.validation_days={vd} is too short (minimum 1)")
    
    # --- strategy_constraints section ---
    sc = config.get('strategy_constraints', {})
    sl = sc.get('stoploss_range', [-0.20, -0.05])
    if isinstance(sl, list) and len(sl) == 2:
        if sl[0] > sl[1]:
            errors.append(f"strategy_constraints.stoploss_range[0] ({sl[0]}) must be <= [1] ({sl[1]})")
        if sl[1] > 0:
            warnings.append(f"strategy_constraints.stoploss_range upper bound ({sl[1]}) is positive. "
                          "Stoploss should normally be negative")
    
    # --- monte_carlo section ---
    mc = config.get('monte_carlo', {})
    if mc.get('enabled'):
        nperms = mc.get('num_permutations', 100)
        if isinstance(nperms, int) and nperms > 1000:
            warnings.append(f"monte_carlo.num_permutations={nperms} — this will slow down evaluation")
    
    # --- indicators section ---
    ind = config.get('indicators', {})
    max_ind = ind.get('max_per_strategy', 6)
    min_ind = ind.get('min_per_strategy', 2)
    if isinstance(max_ind, int) and isinstance(min_ind, int) and min_ind > max_ind:
        errors.append(f"indicators.min_per_strategy ({min_ind}) > max_per_strategy ({max_ind})")
    
    # --- pair_validation section ---
    pv = config.get('pair_validation', {})
    if pv.get('enabled', False):
        tp = pv.get('training_pairs', [])
        vp = pv.get('validation_pairs', [])
        if not isinstance(tp, list) or not tp:
            errors.append("pair_validation.training_pairs must be a non-empty list")
        if not isinstance(vp, list) or not vp:
            errors.append("pair_validation.validation_pairs must be a non-empty list")
        wt = pv.get('weight_train', 0.6)
        wv = pv.get('weight_val', 0.4)
        if isinstance(wt, (int, float)) and isinstance(wv, (int, float)):
            if abs((wt + wv) - 1.0) > 0.05:
                warnings.append(f"pair_validation weights sum to {wt + wv:.2f} (expected ~1.0)")
    
    # --- Anti-pattern warnings (validated across 79+ experiments) ---
    _check_experiment_anti_patterns(config, warnings)
    
    return errors, warnings


def _check_experiment_anti_patterns(config: Dict[str, Any], warnings: List[str]) -> None:
    """
    Check for configuration anti-patterns validated across 79+ GA experiments.
    
    These are soft warnings — advanced users may intentionally test edge cases,
    but these patterns have been shown to consistently produce poor results.
    """
    ga = config.get('genetic_algorithm', {})
    bt = config.get('backtesting', {})
    mc = config.get('monte_carlo', {})
    adv = config.get('advanced', {})
    island_cfg = ga.get('island_model', {})
    llm_cfg = adv.get('llm', {})

    pop_size = ga.get('population_size', 0)
    elite_size = ga.get('elite_size', 0)
    selection = ga.get('selection_method', 'tournament')
    crossover_method = ga.get('crossover_method', 'uniform')
    patience = ga.get('early_stopping', {}).get('patience', 999)
    island_enabled = island_cfg.get('enabled', False)
    llm_enabled = llm_cfg.get('enabled', False)
    pairs = bt.get('pairs', [])

    # AP-1: Standard GA population > 15 leads to overfitting (E19: 59-65% degradation)
    if not island_enabled and isinstance(pop_size, int) and pop_size > 15:
        warnings.append(
            f"[ANTI-PATTERN] population_size={pop_size} > 15: "
            f"standard GA runs with pop > 15 show 59-65% holdout degradation "
            f"(validated in E19, E30). Recommended: 10-15."
        )

    # AP-2: Island model population > 6 leads to extreme overfitting (E24: 62-100% degradation)
    if island_enabled:
        island_pop = island_cfg.get('population_per_island', pop_size)
        if isinstance(island_pop, int) and island_pop > 6:
            warnings.append(
                f"[ANTI-PATTERN] island population_per_island={island_pop} > 6: "
                f"island model with pop > 6 shows 62-100% holdout degradation "
                f"(validated in E24). Sacred limit: 6."
            )

    # AP-3: LLM + rank selection is harmful (E44: loses 0.2065 vs tournament)
    if llm_enabled and selection == 'rank':
        warnings.append(
            "[ANTI-PATTERN] LLM enabled with rank selection: "
            "this combination loses ~0.2065 fitness vs tournament selection "
            "(validated in E44). Use tournament selection with LLM."
        )

    # AP-4: Component crossover + island model causes 33-53% degradation (E18)
    if island_enabled and crossover_method == 'component':
        warnings.append(
            "[ANTI-PATTERN] island_model + component crossover: "
            "shows 33-53% holdout degradation (validated in E18). "
            "Use uniform crossover with island model."
        )

    # AP-5: Low patience + low elite causes premature stopping (E45: stopped at gen 8)
    if isinstance(patience, int) and isinstance(elite_size, int):
        if patience <= 4 and elite_size < 3:
            warnings.append(
                f"[ANTI-PATTERN] patience={patience} with elite_size={elite_size}: "
                f"low patience + low elite causes premature early stopping "
                f"(E45 stopped at gen 8). Use patience >= 6 with elite < 3, "
                f"or elite >= 3 with patience >= 4."
            )

    # AP-6: Monte Carlo permutations > 15 causes early stopping (E26)
    if mc.get('enabled', False):
        num_perms = mc.get('num_permutations', 100)
        if isinstance(num_perms, int) and num_perms > 15:
            warnings.append(
                f"[ANTI-PATTERN] monte_carlo.num_permutations={num_perms} > 15: "
                f"high MC permutations slow evaluation and trigger early stopping "
                f"(validated in E26). Recommended: 10-15."
            )

    # AP-7: 3+ trading pairs fails to generalize (E32: 0 SAFE / 5 WARNING)
    if isinstance(pairs, list) and len(pairs) > 2:
        warnings.append(
            f"[ANTI-PATTERN] {len(pairs)} trading pairs: "
            f"3+ pairs consistently fail to generalize "
            f"(E32: 0 SAFE / 5 WARNING vs E21 with 2 pairs: all SAFE). "
            f"Recommended: 1-2 pairs."
        )

    # AP-8: NSGA-II with fitness_sharing enabled (NSGA-II has its own diversity mechanism)
    if ga.get('mode') == 'nsga2' and ga.get('fitness_sharing', False):
        warnings.append(
            "[ANTI-PATTERN] NSGA-II with fitness_sharing enabled: "
            "NSGA-II uses crowding distance for diversity — fitness_sharing "
            "is redundant and may interfere. Set fitness_sharing: false."
        )

    # AP-9: Island model + walk-forward (incompatible, WF silently disabled)
    wf = config.get('walk_forward', {})
    if island_enabled and wf.get('enabled', False):
        warnings.append(
            "[ANTI-PATTERN] island_model + walk_forward both enabled: "
            "these are incompatible (island splits by regime, WF splits temporally). "
            "Walk-forward will be silently disabled. Choose one."
        )

    # AP-10: Island model + Monte Carlo / CPCV (incompatible)
    if island_enabled and mc.get('enabled', False):
        warnings.append(
            "[ANTI-PATTERN] island_model + monte_carlo both enabled: "
            "Monte Carlo requires full temporal data but island model uses "
            "regime segments. MC will be silently ignored."
        )


def _validate_int_range(section: Dict, key: str, min_val: int, max_val: int, 
                        errors: List[str]) -> None:
    """Validate integer config value is in range."""
    val = section.get(key)
    if val is None:
        errors.append(f"Missing required key: {key}")
    elif not isinstance(val, int):
        errors.append(f"{key} must be an integer, got {type(val).__name__}")
    elif val < min_val or val > max_val:
        errors.append(f"{key}={val} must be between {min_val} and {max_val}")


def _validate_float_range(section: Dict, key: str, min_val: float, max_val: float,
                          errors: List[str]) -> None:
    """Validate float config value is in range."""
    val = section.get(key)
    if val is None:
        return  # Optional
    if not isinstance(val, (int, float)):
        errors.append(f"{key} must be a number, got {type(val).__name__}")
    elif val < min_val or val > max_val:
        errors.append(f"{key}={val} must be between {min_val} and {max_val}")


def validate_and_log(config: Dict[str, Any]) -> bool:
    """
    Validate config and log results.
    
    Returns:
        True if valid (no errors), False if invalid.
    """
    errors, warnings = validate_ga_config(config)
    
    for w in warnings:
        logger.warning(f"[CONFIG] Warning: {w}")
    
    if errors:
        for e in errors:
            logger.error(f"[CONFIG] Error: {e}")
        return False
    
    logger.info(f"[CONFIG] Validation passed ({len(warnings)} warnings)")
    return True


def preflight_check(config: Dict[str, Any], data_root: str = None) -> Tuple[List[str], List[str]]:
    """
    Pre-launch preflight check for experiment configs.
    
    Validates:
      - No incompatible feature combinations (island+WF, island+MC, NSGA-II+fitness_sharing)
      - Backtest timeout adequate for timerange length
      - Data files exist for all configured pairs/timeframes
      - Pair-split validation pairs don't overlap with training pairs
    
    Args:
        config: Full GA config dict
        data_root: Optional path to data directory (auto-detected if None)
    
    Returns:
        Tuple of (errors, warnings)
    """
    from pathlib import Path

    errors: List[str] = []
    warnings: List[str] = []

    # --- Incompatible feature combos ---
    island_cfg = config.get('island_model', {})
    generic_island_cfg = config.get('generic_island_model', {})
    island_enabled = island_cfg.get('enabled', False) or generic_island_cfg.get('enabled', False)
    wf_enabled = config.get('walk_forward', {}).get('enabled', False)
    mc_enabled = config.get('monte_carlo', {}).get('enabled', False)

    if island_enabled and wf_enabled:
        errors.append("Island model and walk-forward are incompatible (regime segmentation conflicts "
                       "with temporal WF splits). Disable one of them.")
    if island_enabled and mc_enabled:
        warnings.append("Monte Carlo is silently ignored when island model is active. "
                         "Set monte_carlo.enabled: false to avoid confusion.")

    ga = config.get('genetic_algorithm', {})
    if ga.get('mode') == 'nsga2' and ga.get('fitness_sharing', False):
        errors.append("NSGA-II + fitness_sharing is incompatible (crowding distance conflicts "
                       "with fitness sharing). Disable fitness_sharing for NSGA-II.")

    # --- Island elite size check ---
    if island_enabled:
        island_pop = (island_cfg.get('islands', [{}])[0].get('population_size', 6)
                      if island_cfg.get('enabled') else generic_island_cfg.get('population_per_island', 6))
        island_elite = ga.get('elite_size', 1)
        if island_elite > 1 and island_pop <= 6:
            warnings.append(f"Island elite_size={island_elite} with pop={island_pop} reduces effective "
                           f"diversity. Recommended: elite_size=1 for island pop <= 6 (AP-1).")

    # --- Backtest timeout check ---
    bt = config.get('backtesting', {})
    timeout = bt.get('timeout', 120)
    timerange = bt.get('timerange', '')
    if timerange and '-' in timerange:
        try:
            start_str, end_str = timerange.split('-')
            from datetime import datetime
            start = datetime.strptime(start_str, '%Y%m%d')
            end = datetime.strptime(end_str, '%Y%m%d')
            days = (end - start).days
            if days > 365 and timeout < 120:
                warnings.append(f"Backtest timeout={timeout}s may be too low for {days}-day timerange. "
                               f"Recommended: timeout >= 120 for >1yr data.")
            if days > 1095 and timeout < 180:
                warnings.append(f"Backtest timeout={timeout}s may be too low for {days}-day timerange. "
                               f"Recommended: timeout >= 180 for >3yr data.")
        except (ValueError, IndexError):
            pass

    # --- Data availability check ---
    pairs = bt.get('pairs', [])
    exchange = bt.get('exchange', 'binance')
    if data_root:
        data_dir = Path(data_root)
    else:
        data_dir = Path(__file__).parent.parent.parent / 'user_data' / 'data' / exchange

    if pairs and data_dir.exists():
        for pair in pairs:
            pair_file_base = pair.replace('/', '_')
            # Check for common data formats (feather, json)
            has_data = any(data_dir.glob(f"{pair_file_base}-*"))
            if not has_data:
                errors.append(f"No data files found for {pair} in {data_dir}. "
                             f"Download with: freqtrade download-data --pairs {pair}")
    elif pairs and not data_dir.exists():
        warnings.append(f"Data directory {data_dir} does not exist. Verify data is available.")

    # --- Pair-split validation check ---
    pv = config.get('pair_validation', {})
    if pv.get('enabled', False):
        train_pairs = set(pv.get('training_pairs', []))
        val_pairs = set(pv.get('validation_pairs', []))
        overlap = train_pairs & val_pairs
        if overlap:
            errors.append(f"pair_validation: training and validation pairs overlap: {overlap}. "
                         f"They must be disjoint for valid overfitting measurement.")
        if not train_pairs:
            errors.append("pair_validation.training_pairs is empty.")
        if not val_pairs:
            errors.append("pair_validation.validation_pairs is empty.")

    # =========================================================================
    # Anti-pattern checks (AP-1 through AP-20, from GA_FIXES_AND_IMPROVEMENTS.md)
    # Each check is backed by experimental evidence from 150+ experiments.
    # =========================================================================
    ga = config.get('genetic_algorithm', {})
    ps = ga.get('population_size', 15)
    es = ga.get('elite_size', 3)
    mr = ga.get('mutation_rate', 0.15)
    sel = ga.get('selection_method', 'tournament')
    cx = ga.get('crossover_method', 'uniform')
    patience = ga.get('convergence_patience', ga.get('patience', 6))
    mode = ga.get('mode', 'single_objective')

    llm_cfg = config.get('llm', config.get('llm_guidance', {}))
    llm_enabled = llm_cfg.get('enabled', False)
    mc_cfg = config.get('monte_carlo', {})
    sc = config.get('strategy_constraints', {})
    parsimony = config.get('parsimony', {})
    strict_constraints = (parsimony.get('enabled', False)
                          or sc.get('max_indicators', 99) <= 4)
    wf_cfg = config.get('walk_forward', {})
    short_cfg = config.get('short_selling', {})

    # AP-7: NSGA-II is broken (produces near-zero fitness 0.0008)
    if mode == 'nsga2':
        warnings.append(
            "[AP-7] NSGA-II mode produces degenerate near-zero fitness (E2=0.0008). "
            "The Pareto objective formulation needs fixing before NSGA-II is usable. "
            "Use mode='single_objective' instead.")

    # AP-8/AP-13: Population > 15 in standard GA causes overfitting
    if not island_enabled and isinstance(ps, int) and ps > 15:
        warnings.append(
            f"[AP-8] population_size={ps} exceeds proven safe ceiling of 15 for standard GA. "
            f"E19 (pop=20)=59-65% holdout degradation, E30 (pop=18)=all WARNING. "
            f"Recommended: population_size <= 15.")

    # AP-6: Island population > 6 causes severe overfitting
    if island_enabled:
        gim = config.get('generic_island_model', {})
        island_pop = gim.get('population_per_island', 6)
        if isinstance(island_pop, int) and island_pop > 6:
            warnings.append(
                f"[AP-6] Island population_per_island={island_pop} exceeds safe limit of 6. "
                f"E24 (pop=8)=62-100% holdout degradation (EXTREME OVERFIT). "
                f"Recommended: population_per_island <= 6.")

    # AP-3/AP-9: Component crossover harmful with island model or strict constraints
    if cx == 'component':
        if island_enabled:
            warnings.append(
                "[AP-3] crossover_method='component' on island model causes overfitting. "
                "E18: 0S/4W/1O, 33-53% degradation. Use crossover_method='uniform'.")
        if strict_constraints:
            warnings.append(
                "[AP-9] crossover_method='component' with strict parsimony/constraints causes "
                "overfitting. E20: 52-61% degradation. Use crossover_method='uniform'.")

    # AP-17: patience >= 8 with elite < 3 causes holdout degradation
    if isinstance(patience, int) and patience >= 8 and isinstance(es, int) and es < 3:
        warnings.append(
            f"[AP-17] convergence_patience={patience} with elite_size={es} risks premature "
            f"holdout degradation. E50 (patience=8, elite=2)=55.6% degradation. "
            f"Recommended: elite_size >= 3 when patience >= 8.")

    # AP-14: patience <= 4 with elite == 2 causes premature early stop
    if isinstance(patience, int) and patience <= 4 and isinstance(es, int) and es == 2:
        warnings.append(
            f"[AP-14] convergence_patience={patience} with elite_size={es} risks premature "
            f"early stop. E45 stopped at Gen 8 with 37.9% degradation. "
            f"Use patience >= 6 with elite_size=2, or increase elite to 3.")

    # AP-15: LLM + rank selection = negative synergy
    if llm_enabled and sel == 'rank':
        warnings.append(
            "[AP-15] LLM guidance with rank selection produces negative synergy. "
            "E44 (rank+LLM)=0.4285 vs E43 (tournament+LLM)=0.6350. "
            "Use selection_method='tournament' with LLM.")

    # AP-5: LLM + high mutation = all WARNING
    if llm_enabled and isinstance(mr, (int, float)) and mr > 0.20:
        warnings.append(
            f"[AP-5] LLM guidance with mutation_rate={mr:.2f} (>0.20) is too aggressive. "
            f"E12 (LLM+mut=0.40)=0S/5W/0O, all WARNING. "
            f"Recommended: mutation_rate <= 0.20 with LLM.")

    # AP-20: mutation >= 0.18 + tournament + elite <= 2 causes holdout early stop
    if (isinstance(mr, (int, float)) and mr >= 0.18
            and sel == 'tournament'
            and isinstance(es, int) and es <= 2):
        warnings.append(
            f"[AP-20] mutation_rate={mr:.2f} + tournament + elite_size={es} triggers "
            f"holdout early stop. E77 stopped at Gen 8, only 4/5 SAFE. "
            f"Use elite_size >= 3 with mutation >= 0.18.")

    # AP-4: Walk-forward train_days > 150 degrades results
    if wf_cfg.get('enabled'):
        td = wf_cfg.get('train_days', 120)
        if isinstance(td, (int, float)) and td > 150:
            warnings.append(
                f"[AP-4] walk_forward.train_days={td} exceeds optimal range. "
                f"E13 (180d)=score 0.503 vs E7 (120d)=score 0.136. "
                f"Recommended: train_days=120 (sweet spot).")

    # AP-11: Monte Carlo permutations > 20 causes premature convergence
    if mc_cfg.get('enabled'):
        nperms = mc_cfg.get('num_permutations', 15)
        if isinstance(nperms, int) and nperms > 20:
            warnings.append(
                f"[AP-11] monte_carlo.num_permutations={nperms} (>20) causes premature "
                f"convergence. E26 (MC=30) early stopped Gen 8, E8 (MC=15) ran full 12 gens. "
                f"Recommended: num_permutations <= 15.")

    # AP-12: More than 2 pairs in standard GA degrades generalization
    bt_pairs = config.get('backtesting', {}).get('pairs', [])
    if not island_enabled and len(bt_pairs) > 2:
        # Only warn for standard GA without pair_validation (pair-split handles >2 intentionally)
        pv_check = config.get('pair_validation', {})
        if not pv_check.get('enabled', False):
            warnings.append(
                f"[AP-12] {len(bt_pairs)} pairs in standard GA (no pair-split) degrades "
                f"generalization. E32 (3 pairs)=0S/5W, all MC=0.0 OVERFIT. "
                f"Use pair_validation for >2 pairs, or stick to 2 pairs.")

    # AP-10: Short selling with small population doubles search space
    if short_cfg.get('enabled', False) and isinstance(ps, int) and ps < 25:
        warnings.append(
            f"[AP-10] short_selling enabled with population_size={ps} (<25). "
            f"Shorts double the search space. E25: only 1 HoF from 180 strategies. "
            f"Recommended: population_size >= 25 or generations >= 20 for shorts.")

    return errors, warnings
