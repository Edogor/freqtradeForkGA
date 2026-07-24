"""Compatibility facade for the canonical GA configuration contract.

Profitability hypotheses do not belong in a startup validator.  The runtime
contract blocks only mechanical impossibilities and internally contradictory
settings.  Tuning claims must be tested through paired, out-of-sample V2
experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from genetic_algorithm.config.invariants import validate_runtime_invariants


logger = logging.getLogger(__name__)


def validate_ga_config(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return versioned mechanical errors and factual semantic warnings."""

    return validate_runtime_invariants(config)


def validate_and_log(config: dict[str, Any]) -> bool:
    """Run the same canonical validation used by every executable entry point."""

    from genetic_algorithm.config.schema import validate_config

    errors, warnings = validate_config(config)
    for warning in warnings:
        logger.warning("[CONFIG] Warning: %s", warning)
    for error in errors:
        logger.error("[CONFIG] Error: %s", error)
    if errors:
        return False
    logger.info("[CONFIG] Validation passed (%d warnings)", len(warnings))
    return True


def preflight_check(
    config: dict[str, Any],
    data_root: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Validate runtime invariants and optional local data availability.

    The data check is intentionally contextual and therefore remains outside
    the pure config contract.  It does not infer optimal population sizes,
    pair counts, timeouts, or search parameters.
    """

    errors, warnings = validate_runtime_invariants(config)
    backtesting = config.get("backtesting", {})
    pairs = backtesting.get("pairs", [])
    exchange = backtesting.get("exchange", "binance")
    directory = (
        Path(data_root)
        if data_root is not None
        else Path(__file__).resolve().parents[2] / "user_data" / "data" / str(exchange)
    )
    if pairs and directory.exists():
        for pair in pairs:
            pair_file_base = pair.replace("/", "_").replace(":", "_")
            if not any(directory.glob(f"{pair_file_base}-*")):
                errors.append(f"no data files found for {pair} in {directory}")
    elif pairs:
        warnings.append(f"data directory {directory} does not exist")
    return sorted(set(errors)), sorted(set(warnings))
