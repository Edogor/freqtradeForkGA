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
    
    return errors, warnings


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

    return errors, warnings
