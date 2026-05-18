"""Backward-compatibility shim — real code moved to genetic_algorithm.engine.operators.mutation."""
from genetic_algorithm.engine.operators.mutation import *  # noqa: F401,F403
from genetic_algorithm.engine.operators.mutation import (  # noqa: F401
    _create_random_condition,
    _create_random_indicator,
    _mutate_indicator_params,
    _mutate_condition_threshold,
    _mutate_regime_gene,
    _THRESHOLD_CLAMPS,
    _VALID_REGIME_MODES,
    _VALID_REGIMES,
)
