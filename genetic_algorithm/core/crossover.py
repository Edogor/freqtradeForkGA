"""Backward-compatibility shim — real code moved to genetic_algorithm.engine.operators.crossover."""
from genetic_algorithm.engine.operators.crossover import *  # noqa: F401,F403
from genetic_algorithm.engine.operators.crossover import (  # noqa: F401
    _enforce_min_entry_conditions,
    _fix_invalid_operators,
)
