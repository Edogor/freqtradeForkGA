"""
Individual Class

Represents a single individual in the population.
Wraps a StrategyGene with fitness and metadata.
"""

from dataclasses import dataclass, field
import copy
import math
from typing import Optional, Dict, Any, List
from datetime import datetime

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.panel_contract import validate_panel_identity


FITNESS_EVIDENCE_UNMEASURED = "UNMEASURED"
FITNESS_EVIDENCE_BACKTEST_VALID = "BACKTEST_VALID"
FITNESS_EVIDENCE_BACKTEST_FAILED = "BACKTEST_FAILED"
FITNESS_EVIDENCE_SURROGATE = "SURROGATE_ESTIMATE"
FITNESS_EVIDENCE_LEGACY_UNKNOWN = "LEGACY_UNKNOWN"
VALID_FITNESS_EVIDENCE = {
    FITNESS_EVIDENCE_UNMEASURED,
    FITNESS_EVIDENCE_BACKTEST_VALID,
    FITNESS_EVIDENCE_BACKTEST_FAILED,
    FITNESS_EVIDENCE_SURROGATE,
    FITNESS_EVIDENCE_LEGACY_UNKNOWN,
}


def infer_fitness_evidence(
    *,
    metrics: Dict[str, Any] | None,
    evaluated: bool,
    fitness: Optional[float],
    legacy_unknown: bool = False,
) -> str:
    """Infer provenance only for legacy payloads that predate the contract."""
    values = metrics or {}
    explicit = values.get("fitness_evidence")
    if explicit in VALID_FITNESS_EVIDENCE:
        return explicit
    if values.get("surrogate_evaluated") or "surrogate_fitness" in values:
        return FITNESS_EVIDENCE_SURROGATE
    if values.get("error"):
        return FITNESS_EVIDENCE_BACKTEST_FAILED
    if evaluated and fitness is not None:
        try:
            if math.isfinite(float(fitness)):
                if legacy_unknown and not any(
                    key in values
                    for key in ("profit", "num_trades", "max_drawdown", "sharpe_ratio")
                ):
                    return FITNESS_EVIDENCE_LEGACY_UNKNOWN
                return FITNESS_EVIDENCE_BACKTEST_VALID
        except (TypeError, ValueError, OverflowError):
            pass
    return FITNESS_EVIDENCE_LEGACY_UNKNOWN if legacy_unknown else FITNESS_EVIDENCE_UNMEASURED


@dataclass
class Individual:
    """
    An individual in the genetic algorithm population.
    
    Wraps a StrategyGene with fitness score and metadata for tracking
    performance and evolution history.
    
    Supports both single-objective (fitness) and multi-objective (objectives) modes.
    """
    
    strategy_gene: StrategyGene
    fitness: Optional[float] = None  # This is the shared_fitness used for selection
    raw_fitness: Optional[float] = None  # Original fitness before fitness sharing
    
    # Multi-objective support (NSGA-II)
    objectives: Optional[List[float]] = None  # Vector of objective values (e.g., [profit, -drawdown, sharpe])
    rank: int = 0  # Pareto front rank (1 = best front, 2 = second front, etc.)
    crowding_distance: float = 0.0  # Crowding distance for diversity preservation
    
    # Performance metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    evaluated: bool = False
    fitness_evidence: Optional[str] = None
    fitness_panel_id: Optional[str] = None
    fitness_panel_role: Optional[str] = None
    
    # Evolution history
    parent_ids: list = field(default_factory=list)
    mutations: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.fitness_evidence is None:
            self.fitness_evidence = infer_fitness_evidence(
                metrics=self.metrics,
                evaluated=self.evaluated,
                fitness=self.raw_fitness if self.raw_fitness is not None else self.fitness,
            )
        if self.fitness_evidence not in VALID_FITNESS_EVIDENCE:
            raise ValueError(f"Unknown fitness evidence: {self.fitness_evidence!r}")
        if self.metrics is None:
            self.metrics = {}
        self.metrics["fitness_evidence"] = self.fitness_evidence
        if self.fitness_panel_id is None:
            self.fitness_panel_id = self.metrics.get("fitness_panel_id")
        if self.fitness_panel_role is None:
            self.fitness_panel_role = self.metrics.get("fitness_panel_role")
        validate_panel_identity(self.fitness_panel_id, self.fitness_panel_role)
        if self.fitness_panel_id is not None:
            self.metrics["fitness_panel_id"] = self.fitness_panel_id
        if self.fitness_panel_role is not None:
            self.metrics["fitness_panel_role"] = self.fitness_panel_role
    
    def __lt__(self, other: 'Individual') -> bool:
        """Enable sorting by fitness (higher is better)."""
        if self.fitness is None and other.fitness is None:
            return False
        if self.fitness is None:
            return True  # Unevaluated individuals go last
        if other.fitness is None:
            return False
        return self.fitness < other.fitness
    
    def __eq__(self, other: 'Individual') -> bool:
        """Check equality based on fitness."""
        if self.fitness is None or other.fitness is None:
            return False
        return abs(self.fitness - other.fitness) < 1e-6
    
    @property
    def id(self) -> str:
        """Unique identifier for this individual."""
        return f"Gen{self.strategy_gene.generation}_Ind{self.strategy_gene.individual_id}"
    
    @property
    def has_measured_fitness(self) -> bool:
        """Whether fitness comes from a successful real backtest."""
        value = self.raw_fitness if self.raw_fitness is not None else self.fitness
        if (
            not self.evaluated
            or self.fitness_evidence != FITNESS_EVIDENCE_BACKTEST_VALID
            or value is None
        ):
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    def has_comparable_fitness(self, panel_id: str) -> bool:
        """Whether a measured score belongs to the required exact panel."""
        return self.has_measured_fitness and self.fitness_panel_id == panel_id

    def assign_fitness_panel(self, panel_id: str, role: str) -> None:
        """Bind real-backtest evidence to its deterministic comparison context."""
        if self.fitness_evidence not in {
            FITNESS_EVIDENCE_BACKTEST_VALID,
            FITNESS_EVIDENCE_BACKTEST_FAILED,
        }:
            raise ValueError("Only real backtests can be assigned an evaluation panel")
        validate_panel_identity(panel_id, role)
        self.fitness_panel_id = panel_id
        self.fitness_panel_role = role
        self.metrics["fitness_panel_id"] = panel_id
        self.metrics["fitness_panel_role"] = role

    @property
    def surrogate_evaluated(self) -> bool:
        """Compatibility view for historical callers."""
        return self.fitness_evidence == FITNESS_EVIDENCE_SURROGATE

    @surrogate_evaluated.setter
    def surrogate_evaluated(self, value: bool) -> None:
        if value:
            self.fitness_evidence = FITNESS_EVIDENCE_SURROGATE
        elif self.fitness_evidence == FITNESS_EVIDENCE_SURROGATE:
            self.fitness_evidence = FITNESS_EVIDENCE_UNMEASURED
        if isinstance(self.metrics, dict):
            self.metrics["fitness_evidence"] = self.fitness_evidence

    def set_fitness(
        self,
        fitness: float,
        metrics: Dict[str, Any],
        *,
        evidence: Optional[str] = None,
    ):
        """
        Set fitness and metrics for this individual.
        
        Args:
            fitness: Overall fitness score (raw fitness before sharing)
            metrics: Dictionary of performance metrics
        """
        if evidence is None:
            evidence = (
                FITNESS_EVIDENCE_BACKTEST_FAILED
                if metrics.get("error")
                else FITNESS_EVIDENCE_BACKTEST_VALID
            )
        if evidence not in {
            FITNESS_EVIDENCE_BACKTEST_VALID,
            FITNESS_EVIDENCE_BACKTEST_FAILED,
        }:
            raise ValueError("set_fitness accepts only real backtest evidence")
        panel_id = metrics.get("fitness_panel_id")
        panel_role = metrics.get("fitness_panel_role")
        validate_panel_identity(panel_id, panel_role)
        self.raw_fitness = fitness
        self.fitness = fitness  # Initially same as raw_fitness, may be adjusted by fitness sharing
        # Preserve metadata keys set before evaluation
        _PRESERVE_KEYS = ('origin', 'llm_provider', 'island_name')
        preserved = {k: v for k, v in self.metrics.items() if k in _PRESERVE_KEYS}
        self.metrics = metrics
        self.metrics.update(preserved)
        self.fitness_evidence = evidence
        self.metrics["fitness_evidence"] = evidence
        self.fitness_panel_id = panel_id
        self.fitness_panel_role = panel_role
        self.metrics.pop("surrogate_evaluated", None)
        self.metrics.pop("surrogate_fitness", None)
        self.metrics.pop("surrogate_mutated", None)
        self.evaluated = True

    def set_surrogate_fitness(
        self,
        fitness: float,
        *,
        predicted_fitness: float,
        surrogate_mutated: bool,
        validation_r2: Optional[float],
    ) -> None:
        """Record a search-only estimate without presenting it as a backtest."""
        if not math.isfinite(float(fitness)) or not math.isfinite(
            float(predicted_fitness)
        ):
            raise ValueError("Surrogate fitness and prediction must be finite")
        preserved = {
            key: value
            for key, value in self.metrics.items()
            if key in {"origin", "llm_provider", "island_name"}
        }
        self.raw_fitness = fitness
        self.fitness = fitness
        self.objectives = None
        self.rank = 0
        self.crowding_distance = 0.0
        self.evaluated = True
        self.fitness_evidence = FITNESS_EVIDENCE_SURROGATE
        self.fitness_panel_id = None
        self.fitness_panel_role = None
        self.metrics = {
            **preserved,
            "fitness_evidence": FITNESS_EVIDENCE_SURROGATE,
            "surrogate_evaluated": True,
            "surrogate_fitness": predicted_fitness,
            "surrogate_mutated": surrogate_mutated,
            "surrogate_validation_r2": validation_r2,
        }

    def adopt_unevaluated_genome(self, source: 'Individual') -> None:
        """Replace this population slot with a mutated genome, clearing stale evidence."""
        preserved = {
            key: value
            for key, value in self.metrics.items()
            if key in {"origin", "llm_provider", "island_name"}
        }
        self.strategy_gene = source.strategy_gene
        self.parent_ids = copy.deepcopy(source.parent_ids)
        self.mutations = copy.deepcopy(source.mutations)
        self.fitness = None
        self.raw_fitness = None
        self.objectives = None
        self.rank = 0
        self.crowding_distance = 0.0
        self.evaluated = False
        self.fitness_evidence = FITNESS_EVIDENCE_UNMEASURED
        self.fitness_panel_id = None
        self.fitness_panel_role = None
        self.metrics = {
            **preserved,
            "fitness_evidence": FITNESS_EVIDENCE_UNMEASURED,
        }
    
    def set_shared_fitness(self, shared_fitness: float):
        """
        Set shared fitness (after fitness sharing applied).
        
        Args:
            shared_fitness: Fitness after diversity-based sharing adjustment
        """
        self.fitness = shared_fitness
    
    def set_objectives(self, objectives: List[float], metrics: Dict[str, float]):
        """
        Set objectives for multi-objective optimization (NSGA-II).
        
        Does NOT overwrite .fitness / .raw_fitness — those are set by
        set_fitness() with the proper weighted scalar and are used by
        holdout degradation, reporting, and convergence tracking.
        NSGA-II selection uses .rank and .crowding_distance instead.
        
        Args:
            objectives: List of objective values (all to be maximized)
            metrics: Dictionary of performance metrics
        """
        self.objectives = objectives
        # Preserve metadata keys set before evaluation
        _PRESERVE_KEYS = ('origin', 'llm_provider', 'island_name')
        preserved = {k: v for k, v in self.metrics.items() if k in _PRESERVE_KEYS}
        self.metrics = metrics
        self.metrics.update(preserved)
        self.fitness_evidence = (
            FITNESS_EVIDENCE_BACKTEST_FAILED
            if metrics.get("error")
            else FITNESS_EVIDENCE_BACKTEST_VALID
        )
        self.metrics["fitness_evidence"] = self.fitness_evidence
        self.evaluated = True
        # Set scalar fitness only if it wasn't already set by set_fitness().
        # This handles the edge case where set_objectives() is called without
        # a prior set_fitness() (e.g., deserialization).
        if self.fitness is None and objectives:
            self.fitness = objectives[0]
            self.raw_fitness = objectives[0]
    
    def nsga2_compare(self, other: 'Individual') -> int:
        """
        NSGA-II comparison: prefer lower rank, then higher crowding distance.
        
        Returns:
            1 if self is better, -1 if other is better, 0 if equal
        """
        # Lower rank is better
        if self.rank < other.rank:
            return 1
        if self.rank > other.rank:
            return -1
        # Same rank: higher crowding distance is better (more diverse)
        if self.crowding_distance > other.crowding_distance:
            return 1
        if self.crowding_distance < other.crowding_distance:
            return -1
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert individual to dictionary for storage."""
        return {
            'id': self.id,
            'strategy_gene': self.strategy_gene.to_dict(),
            'fitness': self.fitness,
            'raw_fitness': self.raw_fitness,
            'objectives': copy.deepcopy(self.objectives),
            'rank': self.rank,
            'crowding_distance': self.crowding_distance,
            'metrics': copy.deepcopy(self.metrics),
            'created_at': self.created_at.isoformat(),
            'evaluated': self.evaluated,
            'fitness_evidence': self.fitness_evidence,
            'fitness_panel_id': self.fitness_panel_id,
            'fitness_panel_role': self.fitness_panel_role,
            'parent_ids': copy.deepcopy(self.parent_ids),
            'mutations': copy.deepcopy(self.mutations),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """Create individual from dictionary."""
        source_gene = copy.deepcopy(data['strategy_gene'])
        strategy_gene = StrategyGene.from_dict(source_gene)
        gene_migrated = strategy_gene.to_dict() != source_gene

        # Never attach an old evaluation to a phenotype-affecting migration.
        # Canonical checkpoints restore exactly; legacy/noncanonical payloads
        # remain loadable but are forced through evaluation again.
        fitness = None if gene_migrated else data.get('fitness')
        raw_fitness = (
            None
            if gene_migrated
            else data.get('raw_fitness', data.get('fitness'))
        )
        objectives = None if gene_migrated else copy.deepcopy(data.get('objectives'))
        metrics = {} if gene_migrated else copy.deepcopy(data.get('metrics', {}))
        fitness_evidence = (
            FITNESS_EVIDENCE_UNMEASURED
            if gene_migrated
            else data.get('fitness_evidence')
        )
        
        individual = cls(
            strategy_gene=strategy_gene,
            fitness=fitness,
            raw_fitness=raw_fitness,
            objectives=objectives,
            rank=0 if gene_migrated else data.get('rank', 0),
            crowding_distance=0.0 if gene_migrated else data.get('crowding_distance', 0.0),
            metrics=metrics,
            evaluated=False if gene_migrated else data.get('evaluated', False),
            fitness_evidence=fitness_evidence,
            fitness_panel_id=(
                None if gene_migrated else data.get('fitness_panel_id')
            ),
            fitness_panel_role=(
                None if gene_migrated else data.get('fitness_panel_role')
            ),
            parent_ids=copy.deepcopy(data.get('parent_ids', [])),
            mutations=copy.deepcopy(data.get('mutations', [])),
        )
        
        if 'created_at' in data:
            individual.created_at = datetime.fromisoformat(data['created_at'])
        
        return individual
