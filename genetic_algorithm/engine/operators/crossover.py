"""
Crossover Operators

Implements various crossover strategies for combining two parent
strategies to create offspring.
"""

import copy
import random
from itertools import zip_longest
from typing import Tuple

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.core.mutation import clamp_condition_thresholds
from genetic_algorithm.strategies.operator_registry import (
    is_valid_operator, resolve_indicator_type, get_standard_operators,
    CDL_POSITIVE_ONLY_TYPES, CDL_NEGATIVE_ONLY_TYPES,
)


def _deduplicate_indicators(gene: StrategyGene) -> None:
    """Remove duplicate indicators keeping the first.
    
    For multi-output indicators that use fixed column names (MACD, STOCH,
    BBANDS, etc.), dedup by (type, timeframe) to prevent column overwrites
    when two instances have different parameters.
    
    For single-output indicators whose column names include the period
    (RSI, EMA, SMA, etc.), dedup by (type, params, timeframe) to allow
    multiple instances with different periods.
    """
    # Indicators whose generated columns do NOT include parameters in the name,
    # so two instances with different params would overwrite each other.
    _FIXED_COLUMN_TYPES = frozenset({
        'MACD', 'STOCH', 'BBANDS', 'SUPERTREND', 'ICHIMOKU', 'DONCHIAN',
        'AROON', 'OBV', 'CMF', 'VROC', 'VWAP', 'PSAR',
    })
    seen = set()
    deduped = []
    for ind in gene.indicators:
        tf = ind.timeframe or ''
        if ind.type in _FIXED_COLUMN_TYPES:
            key = (ind.type, tf)
        else:
            key = (ind.type, str(sorted(ind.parameters.items())) if ind.parameters else '', tf)
        if key not in seen:
            seen.add(key)
            deduped.append(ind)
    gene.indicators = deduped


def _fix_invalid_operators(gene: StrategyGene) -> None:
    """Fix conditions whose operator is invalid for their indicator type.
    
    After crossover or mutation, a condition may reference an operator that
    came from a different indicator context.  Replace with a random valid
    operator, or REMOVE the condition entirely if the indicator shouldn't
    be used for that signal direction (e.g. CDL_DOJI as exit, CDL_EVENINGSTAR
    as entry).
    """
    def _resolve(cond):
        ind_type = resolve_indicator_type(cond.indicator)
        for ind in gene.indicators:
            if (ind.instance_id and ind.instance_id == cond.indicator) or ind.type == ind_type:
                return ind.type
        return ind_type

    # Remove exit conditions for positive-only CDL patterns (CDL_DOJI, CDL_HAMMER, etc.)
    # These patterns can only signal "pattern detected" (>0), never "pattern absent" (<0).
    # They are not meaningful as exit signals.
    gene.exit_conditions = [
        cond for cond in gene.exit_conditions
        if _resolve(cond) not in CDL_POSITIVE_ONLY_TYPES
    ]
    # Remove entry conditions for negative-only CDL patterns (CDL_EVENINGSTAR, etc.)
    gene.entry_conditions = [
        cond for cond in gene.entry_conditions
        if _resolve(cond) not in CDL_NEGATIVE_ONLY_TYPES
    ]

    # Short directions invert the candlestick semantics of long directions:
    # bearish-only patterns can enter shorts, while bullish-only patterns can
    # exit them.
    gene.short_entry_conditions = [
        cond for cond in gene.short_entry_conditions
        if _resolve(cond) not in CDL_POSITIVE_ONLY_TYPES
    ]
    gene.short_exit_conditions = [
        cond for cond in gene.short_exit_conditions
        if _resolve(cond) not in CDL_NEGATIVE_ONLY_TYPES
    ]

    # Fix operators on remaining conditions
    for cond in (
        gene.entry_conditions
        + gene.exit_conditions
        + gene.short_entry_conditions
        + gene.short_exit_conditions
    ):
        ind_type = _resolve(cond)
        if not is_valid_operator(ind_type, cond.operator):
            valid_ops = get_standard_operators(ind_type)
            if valid_ops:
                cond.operator = random.choice(valid_ops)
            # else: unknown indicator, leave as-is (will be caught later)


def _enforce_max_indicators(gene: StrategyGene, config: dict) -> None:
    """Trim indicators to max_per_strategy, removing lowest-weight first.
    
    After trimming, removes any conditions that reference indicators
    no longer present (orphaned conditions).
    """
    max_indicators = (config or {}).get('indicators', {}).get('max_per_strategy', 6)
    if len(gene.indicators) <= max_indicators:
        return
    
    # Sort by weight descending (higher weight = more important = keep)
    # Indicators without an explicit weight default to 1.0
    gene.indicators.sort(key=lambda ind: getattr(ind, 'weight', 1.0), reverse=True)
    gene.indicators = gene.indicators[:max_indicators]
    
    # Build set of remaining indicator references.
    # Conditions may reference indicators by instance_id ('RSI_0') or by bare
    # type ('RSI'), so we must match against both to avoid orphaning valid
    # conditions when instance_id is None or when conditions predate instance_id
    # assignment.
    remaining_refs = set()
    for ind in gene.indicators:
        if ind.instance_id:
            remaining_refs.add(ind.instance_id)
        remaining_refs.add(ind.type)
    
    # Remove orphaned conditions (reference indicators we just trimmed)
    gene.entry_conditions = [c for c in gene.entry_conditions if c.indicator in remaining_refs]
    gene.exit_conditions = [c for c in gene.exit_conditions if c.indicator in remaining_refs]
    gene.short_entry_conditions = [c for c in gene.short_entry_conditions if c.indicator in remaining_refs]
    gene.short_exit_conditions = [c for c in gene.short_exit_conditions if c.indicator in remaining_refs]
    
def _deduplicate_conditions(conditions: list) -> list:
    """Remove exact-duplicate conditions and prune subsumed pairs.
    
    Two conditions on the same indicator with the same operator where one
    threshold is strictly tighter than the other: keep only the tighter one.
    E.g. 'vroc < -117' AND 'vroc < -200' → keep 'vroc < -200'.
    """
    # 1. Exact dedup
    seen = set()
    unique = []
    for c in conditions:
        key = (
            c.indicator,
            c.operator,
            round(c.threshold, 6),
            c.logic,
            round(c.threshold_upper, 6),
            c.lookback,
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    # 2. Subsumption pruning for '<' / '>' operators
    #    Only safe when ALL conditions in a group share AND logic.
    #    Under OR logic, the *looser* condition dominates, which inverts
    #    the selection — skipping subsumption avoids silent signal loss.
    result = []
    by_ind_op = {}  # (indicator, operator) → list of conditions
    for c in unique:
        if c.operator in ('<', 'less_than'):
            by_ind_op.setdefault((c.indicator, '<'), []).append(c)
        elif c.operator in ('>', 'greater_than'):
            by_ind_op.setdefault((c.indicator, '>'), []).append(c)
        else:
            result.append(c)  # keep non-comparable operators as-is
    
    for (ind, op), conds in by_ind_op.items():
        # If any condition uses OR logic, subsumption is unsafe — keep all
        if any(getattr(c, 'logic', 'AND') == 'OR' for c in conds):
            result.extend(conds)
            continue

        if op == '<':
            # For AND logic: x < A AND x < B → keep min(A, B)
            keeper = min(conds, key=lambda c: c.threshold)
        else:
            # For AND logic: x > A AND x > B → keep max(A, B)
            keeper = max(conds, key=lambda c: c.threshold)
        result.append(keeper)
    
    return result


def _enforce_min_entry_conditions(gene: StrategyGene, config: dict) -> None:
    """Ensure a strategy gene has at least min_entry_conditions entry conditions
    AND at least min_exit_conditions exit conditions.
    
    If the gene has fewer, generates additional random conditions from
    the available indicators.
    """
    indicator_config = (config or {}).get('indicators', {})
    # Without a config, preserve the historical no-op for already valid genes
    # while still restoring StrategyGene's structural entry invariant if a
    # crossover legitimately pruned the group to zero.
    min_entry = indicator_config.get('min_entry_conditions', 2 if config else 1)
    min_exit = indicator_config.get('min_exit_conditions', 1 if config else 0)
    
    # Enforce entry conditions
    if len(gene.entry_conditions) < min_entry:
        _top_up_conditions(gene, min_entry - len(gene.entry_conditions), 
                          is_entry=True, indicator_config=indicator_config)
    
    # Last-resort fallback: if entry conditions are STILL empty after top-up,
    # reference a real retained indicator.  The previous synthetic ``volume``
    # reference was itself an orphan and disappeared during canonicalization.
    if not gene.entry_conditions:
        from genetic_algorithm.core.strategy_gene import ConditionGene
        import logging
        logging.getLogger(__name__).warning(
            "[ENFORCE] Entry conditions still empty after top-up — "
            "injecting retained-indicator fallback"
        )
        if gene.indicators:
            fallback = next(
                (ind for ind in gene.indicators if not ind.type.startswith('CDL_')),
                gene.indicators[0],
            )
            valid_ops = get_standard_operators(fallback.type)
            gene.entry_conditions.append(ConditionGene(
                indicator=fallback.instance_id or fallback.type,
                operator=valid_ops[0] if valid_ops else '>',
                threshold=0.0,
                logic='AND',
            ))
    
    # Enforce exit conditions
    if len(gene.exit_conditions) < min_exit:
        _top_up_conditions(gene, min_exit - len(gene.exit_conditions),
                          is_entry=False, indicator_config=indicator_config)


def _top_up_conditions(gene: StrategyGene, needed: int, is_entry: bool, 
                       indicator_config: dict) -> None:
    """Add random conditions to meet the minimum requirement."""
    from genetic_algorithm.core.mutation import _create_random_condition
    available_indicators = gene.indicators
    if not available_indicators:
        return
    
    conditions = gene.entry_conditions if is_entry else gene.exit_conditions
    attempts = 0
    added = 0
    # Filter out CDL-only indicators which can't produce standard conditions
    usable_indicators = [ind for ind in available_indicators
                         if not ind.type.startswith('CDL_')]
    if not usable_indicators:
        usable_indicators = available_indicators  # Fall back to all if none are usable
    while added < needed and attempts < 10:
        ind = random.choice(usable_indicators)
        ind_type = ind.type  # _create_random_condition expects the type (e.g. 'RSI')
        ind_ref = ind.instance_id or ind_type  # conditions reference instance_id
        try:
            new_cond = _create_random_condition(ind_type, is_entry, indicator_config)
            if new_cond:
                # Update the condition to reference instance_id, not bare type
                new_cond.indicator = ind_ref
                existing_keys = {(c.indicator, c.operator, str(c.threshold)) for c in conditions}
                new_key = (new_cond.indicator, new_cond.operator, str(new_cond.threshold))
                if new_key not in existing_keys:
                    conditions.append(new_cond)
                    added += 1
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to create random condition for {ind_ref}: {e}")
        attempts += 1


def _finalize_crossover_gene(gene: StrategyGene, config: dict) -> None:
    """Restore the canonical StrategyGene contract after recombination.

    Crossover combines indicator and condition lists independently.  Every
    operation which removes or reorders indicators must therefore happen
    before the final orphan pruning and minimum-condition top-up.  The old
    pipeline topped conditions up first and then removed indicators, which
    could leave references such as ``MACD_1`` after fixed-column indicator
    deduplication had retained only ``MACD_0``.
    """
    # Reconnect references inherited from both parents before structural
    # cleanup, then perform every operation that can remove an indicator.
    gene.assign_instance_ids()
    _deduplicate_indicators(gene)
    _enforce_max_indicators(gene, config)

    # Indicator trimming/reordering may have changed canonical IDs.  Remap
    # surviving references, then genuinely remove all eliminated references;
    # prune_orphaned_conditions is intentionally allowed to reach zero here.
    gene.assign_instance_ids()
    gene.prune_orphaned_conditions()

    _fix_invalid_operators(gene)
    for attr in (
        'entry_conditions',
        'exit_conditions',
        'short_entry_conditions',
        'short_exit_conditions',
    ):
        conditions = _deduplicate_conditions(getattr(gene, attr))
        clamp_condition_thresholds(conditions)
        setattr(gene, attr, conditions)

    # Directional CDL filtering and condition deduplication can reduce a list
    # below its configured minimum, so top up only after those lossy steps.
    gene.prune_orphaned_conditions()
    _enforce_min_entry_conditions(gene, config)

    # Top-up conditions start with an existing instance ID, but this final
    # canonicalization also deduplicates a coincidental identical condition.
    gene.assign_instance_ids()
    gene.prune_orphaned_conditions()


def single_point_crossover(parent1: Individual, parent2: Individual, 
                          generation: int, ind_id: int,
                          config: dict = None) -> Tuple[Individual, Individual]:
    """
    Single-point crossover.
    
    Split both parents at a random point and swap the second parts.
    
    Note: After crossover, `prune_orphaned_conditions()` is called to
    remove conditions that reference indicators not present in the child.
    This keeps the indicator set clean rather than adding random indicators.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        config: Optional config dict for indicator setup
        
    Returns:
        Tuple of two offspring individuals
    """
    # Create copies of parent genes
    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    
    # Crossover indicators
    min_ind_len = min(len(parent1.strategy_gene.indicators), len(parent2.strategy_gene.indicators))
    if min_ind_len >= 2:
        point = random.randint(1, min_ind_len - 1)
        child1_gene.indicators = ([copy.deepcopy(ind) for ind in parent1.strategy_gene.indicators[:point]] + 
                                 [copy.deepcopy(ind) for ind in parent2.strategy_gene.indicators[point:]])
        child2_gene.indicators = ([copy.deepcopy(ind) for ind in parent2.strategy_gene.indicators[:point]] + 
                                 [copy.deepcopy(ind) for ind in parent1.strategy_gene.indicators[point:]])
    
    # Crossover entry conditions
    min_entry_len = min(len(parent1.strategy_gene.entry_conditions), len(parent2.strategy_gene.entry_conditions))
    if min_entry_len >= 2:
        point = random.randint(1, min_entry_len - 1)
        child1_gene.entry_conditions = ([copy.deepcopy(cond) for cond in parent1.strategy_gene.entry_conditions[:point]] + 
                                       [copy.deepcopy(cond) for cond in parent2.strategy_gene.entry_conditions[point:]])
        child2_gene.entry_conditions = ([copy.deepcopy(cond) for cond in parent2.strategy_gene.entry_conditions[:point]] + 
                                       [copy.deepcopy(cond) for cond in parent1.strategy_gene.entry_conditions[point:]])
    
    # Crossover exit conditions
    min_exit_len = min(len(parent1.strategy_gene.exit_conditions), len(parent2.strategy_gene.exit_conditions))
    if min_exit_len >= 2:
        point = random.randint(1, min_exit_len - 1)
        child1_gene.exit_conditions = ([copy.deepcopy(cond) for cond in parent1.strategy_gene.exit_conditions[:point]] + 
                                      [copy.deepcopy(cond) for cond in parent2.strategy_gene.exit_conditions[point:]])
        child2_gene.exit_conditions = ([copy.deepcopy(cond) for cond in parent2.strategy_gene.exit_conditions[:point]] + 
                                      [copy.deepcopy(cond) for cond in parent1.strategy_gene.exit_conditions[point:]])
    
    # Randomly inherit scalar parameters
    for attr in ['timeframe', 'stoploss']:
        if random.random() < 0.5:
            setattr(child1_gene, attr, getattr(parent2.strategy_gene, attr))
            setattr(child2_gene, attr, getattr(parent1.strategy_gene, attr))
    
    # Swap trailing stop parameters as a unit
    if random.random() < 0.5:
        for attr in ['trailing_stop', 'trailing_stop_positive', 'trailing_stop_positive_offset']:
            setattr(child1_gene, attr, getattr(parent2.strategy_gene, attr))
            setattr(child2_gene, attr, getattr(parent1.strategy_gene, attr))
    
    # Swap informative_timeframes along with timeframe for consistency
    if random.random() < 0.5:
        child1_gene.informative_timeframes = list(parent2.strategy_gene.informative_timeframes)
        child2_gene.informative_timeframes = list(parent1.strategy_gene.informative_timeframes)
    
    if random.random() < 0.5:
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    # Restore the complete condition/indicator contract after recombination.
    for g in (child1_gene, child2_gene):
        _finalize_crossover_gene(g, config)
    
    return (Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
            Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]))


def _uniform_crossover_lists(parent1_list, parent2_list, swap_prob):
    """Helper for uniform crossover on lists."""
    child1_list, child2_list = [], []
    max_len = max(len(parent1_list), len(parent2_list))
    
    for i in range(max_len):
        if random.random() < swap_prob:
            if i < len(parent2_list):
                child1_list.append(copy.deepcopy(parent2_list[i]))
            if i < len(parent1_list):
                child2_list.append(copy.deepcopy(parent1_list[i]))
        else:
            if i < len(parent1_list):
                child1_list.append(copy.deepcopy(parent1_list[i]))
            if i < len(parent2_list):
                child2_list.append(copy.deepcopy(parent2_list[i]))
    
    return child1_list, child2_list


def uniform_crossover(parent1: Individual, parent2: Individual,
                     generation: int, ind_id: int,
                     swap_prob: float = 0.5,
                     config: dict = None) -> Tuple[Individual, Individual]:
    """
    Uniform crossover.
    
    Each component is randomly inherited from either parent.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual  
        generation: Generation number for offspring
        ind_id: Starting individual ID
        swap_prob: Probability of swapping each component
        
    Returns:
        Tuple of two offspring individuals
    """
    # Create copies of parent genes
    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    
    # Uniform crossover for indicators
    child1_indicators, child2_indicators = _uniform_crossover_lists(
        parent1.strategy_gene.indicators, parent2.strategy_gene.indicators, swap_prob)
    
    # Ensure at least one indicator in each child
    child1_gene.indicators = child1_indicators if child1_indicators else [parent1.strategy_gene.indicators[0]]
    child2_gene.indicators = child2_indicators if child2_indicators else [parent2.strategy_gene.indicators[0]]
    
    # Uniform crossover for entry conditions
    child1_entry, child2_entry = _uniform_crossover_lists(
        parent1.strategy_gene.entry_conditions, parent2.strategy_gene.entry_conditions, swap_prob)
    
    # Ensure at least one entry condition in each child
    child1_gene.entry_conditions = child1_entry if child1_entry else [parent1.strategy_gene.entry_conditions[0]]
    child2_gene.entry_conditions = child2_entry if child2_entry else [parent2.strategy_gene.entry_conditions[0]]
    
    # Uniform crossover for exit conditions
    child1_gene.exit_conditions, child2_gene.exit_conditions = _uniform_crossover_lists(
        parent1.strategy_gene.exit_conditions, parent2.strategy_gene.exit_conditions, swap_prob)
    
    # Uniform crossover for independent short conditions (if either parent has them)
    if parent1.strategy_gene.short_entry_conditions or parent2.strategy_gene.short_entry_conditions:
        child1_gene.short_entry_conditions, child2_gene.short_entry_conditions = _uniform_crossover_lists(
            parent1.strategy_gene.short_entry_conditions, parent2.strategy_gene.short_entry_conditions, swap_prob)
        child1_gene.short_exit_conditions, child2_gene.short_exit_conditions = _uniform_crossover_lists(
            parent1.strategy_gene.short_exit_conditions, parent2.strategy_gene.short_exit_conditions, swap_prob)
    
    # Randomly inherit scalar parameters
    for attr in ['timeframe', 'stoploss']:
        if random.random() < swap_prob:
            val1 = getattr(parent2.strategy_gene, attr)
            val2 = getattr(parent1.strategy_gene, attr)
            setattr(child1_gene, attr, val1)
            setattr(child2_gene, attr, val2)
    
    if random.random() < swap_prob:
        child1_gene.informative_timeframes = list(parent2.strategy_gene.informative_timeframes)
        child2_gene.informative_timeframes = list(parent1.strategy_gene.informative_timeframes)
    
    if random.random() < swap_prob:
        child1_gene.trailing_stop = parent2.strategy_gene.trailing_stop
        child1_gene.trailing_stop_positive = parent2.strategy_gene.trailing_stop_positive
        child1_gene.trailing_stop_positive_offset = parent2.strategy_gene.trailing_stop_positive_offset
        child2_gene.trailing_stop = parent1.strategy_gene.trailing_stop
        child2_gene.trailing_stop_positive = parent1.strategy_gene.trailing_stop_positive
        child2_gene.trailing_stop_positive_offset = parent1.strategy_gene.trailing_stop_positive_offset
    
    # Swap regime specialization fields (Phase 1B)
    if random.random() < swap_prob:
        child1_gene.preferred_regime = parent2.strategy_gene.preferred_regime
        child1_gene.regime_mode = parent2.strategy_gene.regime_mode
        child2_gene.preferred_regime = parent1.strategy_gene.preferred_regime
        child2_gene.regime_mode = parent1.strategy_gene.regime_mode
    
    if random.random() < swap_prob:
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    # Restore the complete condition/indicator contract after recombination.
    for g in (child1_gene, child2_gene):
        _finalize_crossover_gene(g, config)
    
    return (Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
            Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]))


def component_crossover(parent1: Individual, parent2: Individual,
                       generation: int, ind_id: int,
                       config: dict = None) -> Tuple[Individual, Individual]:
    """
    Component-based crossover.
    
    Exchange entire indicator sets or rule sets between parents.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        
    Returns:
        Tuple of two offspring individuals
    """
    # Create copies of parent genes
    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    
    # Swap components based on random decisions
    swaps = {
        'indicators': random.random() < 0.5,
        'entry': random.random() < 0.5,
        'exit': random.random() < 0.5,
        'risk': random.random() < 0.5,
    }
    
    if swaps['indicators']:
        child1_gene.indicators, child2_gene.indicators = [copy.deepcopy(ind) for ind in parent2.strategy_gene.indicators], [copy.deepcopy(ind) for ind in parent1.strategy_gene.indicators]
        # Swap informative_timeframes with indicators for consistency
        child1_gene.informative_timeframes = list(parent2.strategy_gene.informative_timeframes)
        child2_gene.informative_timeframes = list(parent1.strategy_gene.informative_timeframes)
    
    if swaps['entry']:
        child1_gene.entry_conditions, child2_gene.entry_conditions = [copy.deepcopy(cond) for cond in parent2.strategy_gene.entry_conditions], [copy.deepcopy(cond) for cond in parent1.strategy_gene.entry_conditions]
    
    if swaps['exit']:
        child1_gene.exit_conditions, child2_gene.exit_conditions = [copy.deepcopy(cond) for cond in parent2.strategy_gene.exit_conditions], [copy.deepcopy(cond) for cond in parent1.strategy_gene.exit_conditions]
    
    if swaps['risk']:
        # Swap all risk parameters (stoploss, ROI, trailing stop + params)
        child1_gene.stoploss = parent2.strategy_gene.stoploss
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child1_gene.trailing_stop = parent2.strategy_gene.trailing_stop
        child1_gene.trailing_stop_positive = parent2.strategy_gene.trailing_stop_positive
        child1_gene.trailing_stop_positive_offset = parent2.strategy_gene.trailing_stop_positive_offset
        # Include regime specialization in risk swap (Phase 1B)
        child1_gene.preferred_regime = parent2.strategy_gene.preferred_regime
        child1_gene.regime_mode = parent2.strategy_gene.regime_mode
        
        child2_gene.stoploss = parent1.strategy_gene.stoploss
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
        child2_gene.trailing_stop = parent1.strategy_gene.trailing_stop
        child2_gene.trailing_stop_positive = parent1.strategy_gene.trailing_stop_positive
        child2_gene.trailing_stop_positive_offset = parent1.strategy_gene.trailing_stop_positive_offset
        child2_gene.preferred_regime = parent1.strategy_gene.preferred_regime
        child2_gene.regime_mode = parent1.strategy_gene.regime_mode
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    # Restore the complete condition/indicator contract after recombination.
    for g in (child1_gene, child2_gene):
        _finalize_crossover_gene(g, config)
    
    return (Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
            Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]))


def cross_niche_union_crossover(
    parent1: Individual,
    parent2: Individual,
    generation: int,
    ind_id: int,
    config: dict | None = None,
) -> Tuple[Individual, Individual]:
    """Create bridge children that retain structural material from both parents.

    Normal point crossover cannot exchange a one-indicator parent.  This
    operator is deliberately generic: it unions genes and conditions without
    prescribing an entry/exit rule, then lets the normal crossover cleanup and
    later mutation decide the executable strategy.
    """

    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    child1_gene.indicators = _interleaved_indicator_union(
        parent1.strategy_gene.indicators,
        parent2.strategy_gene.indicators,
        config,
    )
    child2_gene.indicators = _interleaved_indicator_union(
        parent2.strategy_gene.indicators,
        parent1.strategy_gene.indicators,
        config,
    )
    for attr in (
        "entry_conditions",
        "exit_conditions",
        "short_entry_conditions",
        "short_exit_conditions",
    ):
        setattr(
            child1_gene,
            attr,
            _interleaved_condition_union(
                getattr(parent1.strategy_gene, attr),
                getattr(parent2.strategy_gene, attr),
            ),
        )
        setattr(
            child2_gene,
            attr,
            _interleaved_condition_union(
                getattr(parent2.strategy_gene, attr),
                getattr(parent1.strategy_gene, attr),
            ),
        )

    # Risk/execution scalars remain ordinary heritable material rather than a
    # bridge-specific trading rule.
    if random.random() < 0.5:
        for attr in ("stoploss", "minimal_roi", "trailing_stop", "trailing_stop_positive", "trailing_stop_positive_offset"):
            left, right = getattr(parent1.strategy_gene, attr), getattr(parent2.strategy_gene, attr)
            setattr(child1_gene, attr, copy.deepcopy(right))
            setattr(child2_gene, attr, copy.deepcopy(left))
    for offset, gene in enumerate((child1_gene, child2_gene)):
        gene.generation = generation
        gene.individual_id = ind_id + offset
        _finalize_crossover_gene(gene, config or {})
    return (
        Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
        Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]),
    )


def _interleaved_indicator_union(first, second, config: dict | None):
    """Union indicators while reserving a slot for both parent sources."""

    maximum = int((config or {}).get("indicators", {}).get("max_per_strategy", 5))
    maximum = max(2, maximum)
    result = []
    seen = set()
    for left, right in zip_longest(first, second):
        for indicator in (left, right):
            if indicator is None:
                continue
            key = (
                indicator.type,
                tuple(sorted((indicator.parameters or {}).items())),
                indicator.timeframe,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(copy.deepcopy(indicator))
            if len(result) >= maximum:
                return result
    return result


def _interleaved_condition_union(first, second):
    result = []
    for left, right in zip_longest(first, second):
        if left is not None:
            result.append(copy.deepcopy(left))
        if right is not None:
            result.append(copy.deepcopy(right))
    return result


def crossover(parent1: Individual, parent2: Individual,
             generation: int, ind_id: int,
             method: str = 'single_point',
             config: dict = None,
             **kwargs) -> Tuple[Individual, Individual]:
    """
    Perform crossover using specified method.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        method: Crossover method ('single_point', 'uniform', 'component')
        config: Configuration dictionary with indicator settings
        **kwargs: Additional arguments for crossover method
        
    Returns:
        Tuple of two offspring individuals
    """
    crossover_methods = {
        'single_point': single_point_crossover,
        'uniform': uniform_crossover,
        'component': component_crossover,
        'cross_niche_union': cross_niche_union_crossover,
    }
    
    if method not in crossover_methods:
        raise ValueError(f"Unknown crossover method: {method}")
    
    # Pass config to crossover methods
    kwargs['config'] = config
    return crossover_methods[method](parent1, parent2, generation, ind_id, **kwargs)
