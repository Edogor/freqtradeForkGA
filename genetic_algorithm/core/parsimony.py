"""Backward-compatibility shim — real code moved to genetic_algorithm.advanced.parsimony."""
from genetic_algorithm.advanced.parsimony import *  # noqa: F401,F403
from genetic_algorithm.advanced.parsimony import (  # noqa: F401
    _build_removal_candidates,
    _apply_removal,
)
