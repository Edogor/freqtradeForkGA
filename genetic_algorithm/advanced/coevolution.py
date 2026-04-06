"""
Phase 4.3 — Multi-Population Cooperative Coevolution

Maintains **three** separate sub-populations that evolve independently:
  1. Entry population  — entry conditions + their indicator subset
  2. Exit population   — exit conditions + their indicator subset
  3. Risk population   — stoploss, ROI table, trailing stop, max open trades

A complete strategy is assembled by randomly sampling one individual from
each sub-population and composing a full StrategyGene.  Each component's
fitness is the **average fitness of all composed strategies that include it**
(credit assignment via collaborative evaluation).

This forces modular evolution: components must work well with many partners,
preventing overfitting to a single entry/exit/risk combination.

Config:
    coevolution:
        enabled: true
        sub_population_size: 10      # Per sub-population
        generations: 12
        collaborators_per_eval: 3    # How many random partners to average
        elite_size: 2
        mutation_rate: 0.20
        crossover_rate: 0.70

Usage:
    from genetic_algorithm.core.coevolution import CoevolutionEngine
    engine = CoevolutionEngine(config, strategy_generator, fitness_evaluator)
    best_strategies = engine.run()
"""

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component genes — lightweight wrappers around sub-parts of StrategyGene
# ---------------------------------------------------------------------------

@dataclass
class EntryComponent:
    """Entry conditions + their supporting indicators."""
    indicators: list = field(default_factory=list)     # List[IndicatorGene]
    conditions: list = field(default_factory=list)      # List[ConditionGene]
    fitness_accumulator: float = 0.0
    eval_count: int = 0

    @property
    def fitness(self) -> Optional[float]:
        return self.fitness_accumulator / self.eval_count if self.eval_count else None

    def reset_fitness(self):
        self.fitness_accumulator = 0.0
        self.eval_count = 0


@dataclass
class ExitComponent:
    """Exit conditions + their supporting indicators."""
    indicators: list = field(default_factory=list)
    conditions: list = field(default_factory=list)
    fitness_accumulator: float = 0.0
    eval_count: int = 0

    @property
    def fitness(self) -> Optional[float]:
        return self.fitness_accumulator / self.eval_count if self.eval_count else None

    def reset_fitness(self):
        self.fitness_accumulator = 0.0
        self.eval_count = 0


@dataclass
class RiskComponent:
    """Risk parameters: stoploss, ROI, trailing, max open trades."""
    stoploss: float = -0.05
    minimal_roi: Dict[str, float] = field(default_factory=lambda: {"0": 0.04, "30": 0.02, "60": 0.01})
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: Optional[float] = None
    max_open_trades: int = 3
    fitness_accumulator: float = 0.0
    eval_count: int = 0

    @property
    def fitness(self) -> Optional[float]:
        return self.fitness_accumulator / self.eval_count if self.eval_count else None

    def reset_fitness(self):
        self.fitness_accumulator = 0.0
        self.eval_count = 0


# ---------------------------------------------------------------------------
# Decomposition / Composition helpers
# ---------------------------------------------------------------------------

def decompose_gene(gene) -> Tuple[EntryComponent, ExitComponent, RiskComponent]:
    """Split a StrategyGene into three coevolution components."""
    # Figure out which indicators are used by entry vs exit conditions
    entry_indicator_ids = set()
    for cond in gene.entry_conditions:
        ind_id = getattr(cond, 'indicator_id', None) or getattr(cond, 'indicator_instance_id', None)
        if ind_id is not None:
            entry_indicator_ids.add(ind_id)
        sec_id = getattr(cond, 'secondary_indicator_id', None)
        if sec_id is not None:
            entry_indicator_ids.add(sec_id)

    exit_indicator_ids = set()
    for cond in gene.exit_conditions:
        ind_id = getattr(cond, 'indicator_id', None) or getattr(cond, 'indicator_instance_id', None)
        if ind_id is not None:
            exit_indicator_ids.add(ind_id)
        sec_id = getattr(cond, 'secondary_indicator_id', None)
        if sec_id is not None:
            exit_indicator_ids.add(sec_id)

    # Assign indicators
    entry_indicators = []
    exit_indicators = []
    for ind in gene.indicators:
        iid = getattr(ind, 'instance_id', None)
        if iid in entry_indicator_ids:
            entry_indicators.append(copy.deepcopy(ind))
        if iid in exit_indicator_ids:
            exit_indicators.append(copy.deepcopy(ind))
        # Shared indicators go to both

    entry = EntryComponent(
        indicators=entry_indicators,
        conditions=copy.deepcopy(gene.entry_conditions),
    )
    exit_ = ExitComponent(
        indicators=exit_indicators,
        conditions=copy.deepcopy(gene.exit_conditions),
    )
    risk = RiskComponent(
        stoploss=gene.stoploss,
        minimal_roi=copy.deepcopy(gene.minimal_roi),
        trailing_stop=gene.trailing_stop,
        trailing_stop_positive=getattr(gene, 'trailing_stop_positive', None),
        trailing_stop_positive_offset=getattr(gene, 'trailing_stop_positive_offset', None),
        max_open_trades=gene.max_open_trades,
    )

    return entry, exit_, risk


def compose_gene(entry: EntryComponent, exit_: ExitComponent, risk: RiskComponent,
                 template_gene) -> 'StrategyGene':
    """Combine three components back into a full StrategyGene.

    Uses *template_gene* for boilerplate fields (timeframe, generation, etc.).
    """
    gene = template_gene.copy()

    # Merge indicators (de-duplicate by instance_id)
    seen = set()
    merged = []
    for ind in entry.indicators + exit_.indicators:
        iid = getattr(ind, 'instance_id', id(ind))
        if iid not in seen:
            merged.append(copy.deepcopy(ind))
            seen.add(iid)
    gene.indicators = merged

    gene.entry_conditions = copy.deepcopy(entry.conditions)
    gene.exit_conditions = copy.deepcopy(exit_.conditions)
    gene.stoploss = risk.stoploss
    gene.minimal_roi = copy.deepcopy(risk.minimal_roi)
    gene.trailing_stop = risk.trailing_stop
    gene.trailing_stop_positive = risk.trailing_stop_positive
    gene.trailing_stop_positive_offset = risk.trailing_stop_positive_offset
    gene.max_open_trades = risk.max_open_trades

    return gene


# ---------------------------------------------------------------------------
# Sub-population operators
# ---------------------------------------------------------------------------

def _tournament_select(pool: list, k: int = 3):
    """Tournament selection on component list using .fitness property."""
    candidates = random.sample(pool, min(k, len(pool)))
    return max(candidates, key=lambda c: c.fitness if c.fitness is not None else -1e9)


def _crossover_components(p1, p2, component_type):
    """Uniform crossover between two components of the same type."""
    child = copy.deepcopy(p1)
    if component_type in ('entry', 'exit'):
        # Swap conditions with 50% probability each
        all_conds = list(p1.conditions) + list(p2.conditions)
        random.shuffle(all_conds)
        half = max(1, len(all_conds) // 2)
        child.conditions = copy.deepcopy(all_conds[:half])
        # Collect required indicators
        needed_ids = set()
        for c in child.conditions:
            cid = getattr(c, 'indicator_id', None) or getattr(c, 'indicator_instance_id', None)
            if cid is not None:
                needed_ids.add(cid)
            sid = getattr(c, 'secondary_indicator_id', None)
            if sid is not None:
                needed_ids.add(sid)
        all_inds = {getattr(i, 'instance_id', None): i for i in p1.indicators + p2.indicators}
        child.indicators = [copy.deepcopy(all_inds[iid]) for iid in needed_ids if iid in all_inds]
    elif component_type == 'risk':
        # Uniform blend of numeric risk params
        if random.random() < 0.5:
            child.stoploss = p2.stoploss
        if random.random() < 0.5:
            child.minimal_roi = copy.deepcopy(p2.minimal_roi)
        if random.random() < 0.5:
            child.trailing_stop = p2.trailing_stop
            child.trailing_stop_positive = p2.trailing_stop_positive
            child.trailing_stop_positive_offset = p2.trailing_stop_positive_offset
        if random.random() < 0.5:
            child.max_open_trades = p2.max_open_trades

    child.reset_fitness()
    return child


def _mutate_component(comp, component_type, rate: float = 0.20):
    """Simple mutation for a component."""
    if random.random() > rate:
        return comp

    if component_type == 'risk':
        r = random.random()
        if r < 0.3:
            comp.stoploss = max(-0.15, min(-0.01, comp.stoploss + random.gauss(0, 0.01)))
        elif r < 0.6:
            # Perturb one ROI value
            if comp.minimal_roi:
                key = random.choice(list(comp.minimal_roi.keys()))
                comp.minimal_roi[key] = max(0.001, comp.minimal_roi[key] + random.gauss(0, 0.005))
        elif r < 0.8:
            comp.trailing_stop = not comp.trailing_stop
        else:
            comp.max_open_trades = max(1, min(8, comp.max_open_trades + random.choice([-1, 0, 1])))

    # Entry/exit mutation is handled by re-using the main GA's mutate
    # on the composed gene — so we only do lightweight tweaks here
    comp.reset_fitness()
    return comp


# ---------------------------------------------------------------------------
# Coevolution Engine
# ---------------------------------------------------------------------------

class CoevolutionEngine:
    """Runs cooperative coevolution with three sub-populations."""

    def __init__(self, config: Dict[str, Any],
                 strategy_generator=None,
                 fitness_evaluator=None):
        co_cfg = config.get('coevolution', {})
        self.enabled = co_cfg.get('enabled', False)
        self.sub_pop_size = co_cfg.get('sub_population_size', 10)
        self.generations = co_cfg.get('generations', 12)
        self.collaborators = co_cfg.get('collaborators_per_eval', 3)
        self.elite_size = co_cfg.get('elite_size', 2)
        self.mutation_rate = co_cfg.get('mutation_rate', 0.20)
        self.crossover_rate = co_cfg.get('crossover_rate', 0.70)

        self.strategy_generator = strategy_generator
        self.fitness_evaluator = fitness_evaluator
        self.config = config

        self.entry_pop: List[EntryComponent] = []
        self.exit_pop: List[ExitComponent] = []
        self.risk_pop: List[RiskComponent] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self):
        """Create initial sub-populations by decomposing random strategies."""
        if self.strategy_generator is None:
            raise RuntimeError("CoevolutionEngine requires a strategy_generator")

        self.entry_pop, self.exit_pop, self.risk_pop = [], [], []

        for i in range(self.sub_pop_size):
            gene = self.strategy_generator.generate_random_strategy(
                generation=0, individual_id=i)
            entry, exit_, risk = decompose_gene(gene)
            self.entry_pop.append(entry)
            self.exit_pop.append(exit_)
            self.risk_pop.append(risk)

        logger.info(f"[COEVOLUTION] Initialised 3 sub-populations × {self.sub_pop_size}")

    # ------------------------------------------------------------------
    # Evaluation with credit assignment
    # ------------------------------------------------------------------

    def _evaluate_population(self, template_gene):
        """Evaluate all components by composing random collaborator triples."""
        # Reset all fitnesses
        for c in self.entry_pop + self.exit_pop + self.risk_pop:
            c.reset_fitness()

        # For each entry component, pair it with random exit & risk collaborators
        for entry in self.entry_pop:
            for _ in range(self.collaborators):
                exit_ = random.choice(self.exit_pop)
                risk = random.choice(self.risk_pop)
                fitness = self._eval_triple(entry, exit_, risk, template_gene)
                if fitness is not None:
                    entry.fitness_accumulator += fitness
                    entry.eval_count += 1
                    exit_.fitness_accumulator += fitness
                    exit_.eval_count += 1
                    risk.fitness_accumulator += fitness
                    risk.eval_count += 1

        # Also evaluate each exit and risk with random partners (balanced)
        for exit_ in self.exit_pop:
            for _ in range(max(1, self.collaborators - 1)):
                entry = random.choice(self.entry_pop)
                risk = random.choice(self.risk_pop)
                fitness = self._eval_triple(entry, exit_, risk, template_gene)
                if fitness is not None:
                    entry.fitness_accumulator += fitness
                    entry.eval_count += 1
                    exit_.fitness_accumulator += fitness
                    exit_.eval_count += 1
                    risk.fitness_accumulator += fitness
                    risk.eval_count += 1

    def _eval_triple(self, entry, exit_, risk, template_gene) -> Optional[float]:
        """Compose and evaluate a single strategy from three components."""
        try:
            gene = compose_gene(entry, exit_, risk, template_gene)
            fitness, _ = self.fitness_evaluator.evaluate(gene)
            return fitness
        except Exception as e:
            logger.debug(f"[COEVOLUTION] Eval failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Evolution operators per sub-population
    # ------------------------------------------------------------------

    def _evolve_subpop(self, pop: list, component_type: str) -> list:
        """One generation of evolution for a single sub-population."""
        # Sort by fitness
        pop.sort(key=lambda c: c.fitness if c.fitness is not None else -1e9, reverse=True)

        # Elitism
        next_gen = [copy.deepcopy(c) for c in pop[:self.elite_size]]

        # Fill remaining slots
        while len(next_gen) < self.sub_pop_size:
            p1 = _tournament_select(pop)
            if random.random() < self.crossover_rate:
                p2 = _tournament_select(pop)
                child = _crossover_components(p1, p2, component_type)
            else:
                child = copy.deepcopy(p1)
            child = _mutate_component(child, component_type, self.mutation_rate)
            next_gen.append(child)

        return next_gen[:self.sub_pop_size]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, template_gene=None) -> List:
        """Run the full coevolution loop.

        Args:
            template_gene: A StrategyGene to use as boilerplate for composed
                           strategies (timeframe, generation, etc.).

        Returns:
            List of top composed StrategyGene objects.
        """
        if not self.enabled:
            return []

        if not self.entry_pop:
            self.initialize()

        # Use a freshly generated gene as template if none provided
        if template_gene is None and self.strategy_generator:
            template_gene = self.strategy_generator.generate_random_strategy(0, 0)

        logger.info(f"[COEVOLUTION] Starting {self.generations}-generation coevolution "
                     f"({self.sub_pop_size} × 3 sub-populations)")

        best_fitness = -1e9
        for gen in range(self.generations):
            # Evaluate
            self._evaluate_population(template_gene)

            # Log progress
            entry_best = max((c.fitness for c in self.entry_pop if c.fitness), default=0)
            exit_best = max((c.fitness for c in self.exit_pop if c.fitness), default=0)
            risk_best = max((c.fitness for c in self.risk_pop if c.fitness), default=0)
            gen_best = max(entry_best, exit_best, risk_best)
            if gen_best > best_fitness:
                best_fitness = gen_best

            logger.info(f"[COEVOLUTION] Gen {gen + 1}/{self.generations} — "
                         f"Entry={entry_best:.4f} Exit={exit_best:.4f} Risk={risk_best:.4f}")

            # Evolve each sub-population independently
            if gen < self.generations - 1:
                self.entry_pop = self._evolve_subpop(self.entry_pop, 'entry')
                self.exit_pop = self._evolve_subpop(self.exit_pop, 'exit')
                self.risk_pop = self._evolve_subpop(self.risk_pop, 'risk')

        # Compose final best strategies
        return self._compose_best(template_gene, n=5)

    def _compose_best(self, template_gene, n: int = 5) -> list:
        """Compose top strategies from the best of each sub-population."""
        # Sort each sub-pop by fitness
        self.entry_pop.sort(key=lambda c: c.fitness if c.fitness else -1e9, reverse=True)
        self.exit_pop.sort(key=lambda c: c.fitness if c.fitness else -1e9, reverse=True)
        self.risk_pop.sort(key=lambda c: c.fitness if c.fitness else -1e9, reverse=True)

        results = []
        for i in range(min(n, self.sub_pop_size)):
            # Pair top-i entry with top-i exit and top-i risk
            entry = self.entry_pop[min(i, len(self.entry_pop) - 1)]
            exit_ = self.exit_pop[min(i, len(self.exit_pop) - 1)]
            risk = self.risk_pop[min(i, len(self.risk_pop) - 1)]
            gene = compose_gene(entry, exit_, risk, template_gene)
            gene.generation = self.generations
            gene.individual_id = i
            results.append(gene)

        logger.info(f"[COEVOLUTION] Composed {len(results)} final strategies")
        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'entry_pop_size': len(self.entry_pop),
            'exit_pop_size': len(self.exit_pop),
            'risk_pop_size': len(self.risk_pop),
        }
