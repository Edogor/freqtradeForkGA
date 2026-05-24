"""Tests for cooperative coevolution (T3.4).

Covers decompose/compose round-trip, mutation/crossover invariants,
sub-population evolution, balanced credit assignment, and the new
``evaluate_composed_for_hof`` helper used by the runner.
"""

import os
from unittest.mock import MagicMock

import pytest
import yaml

from genetic_algorithm.advanced.coevolution import (
    CoevolutionEngine,
    EntryComponent,
    ExitComponent,
    RiskComponent,
    compose_gene,
    decompose_gene,
    _crossover_components,
    _mutate_component,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gene(seed: int = 0):
    """Build a minimal real gene object (not MagicMock) compatible with
    decompose/compose plumbing.  Uses SimpleNamespace so attribute
    arithmetic in mutate works without surprises.
    """
    import copy as _copy
    from types import SimpleNamespace

    # Indicators with instance_ids that match condition references
    def _ind(iid, name):
        return SimpleNamespace(instance_id=iid, name=name, type=name)

    indicators = [_ind(1, "RSI"), _ind(2, "MACD"), _ind(3, "ADX")]
    entry_cond = SimpleNamespace(
        indicator_id=1, secondary_indicator_id=None, op="<", value=30 + seed
    )
    exit_cond = SimpleNamespace(
        indicator_id=3, secondary_indicator_id=None, op=">", value=25 + seed
    )

    gene = SimpleNamespace(
        indicators=indicators,
        entry_conditions=[entry_cond],
        exit_conditions=[exit_cond],
        stoploss=-0.05 - 0.01 * seed,
        trailing_stop=True,
        trailing_stop_positive=0.02,
        trailing_stop_positive_offset=0.03,
        max_open_trades=3,
        minimal_roi={"0": 0.10, "30": 0.05, "60": 0.0},
        timeframe="1h",
        generation=0,
        individual_id=seed,
    )

    def _copy_fn(self=gene):
        return SimpleNamespace(**_copy.deepcopy(self.__dict__))

    gene.copy = _copy_fn
    gene.to_dict = lambda self=gene: {"seed": self.individual_id}
    return gene


def _make_generator():
    """Mock strategy generator returning predictable genes."""
    gen = MagicMock()
    gen.generate_random_strategy.side_effect = lambda generation=0, individual_id=0: _make_gene(
        seed=individual_id
    )
    return gen


def _make_evaluator(scores=None):
    """Mock fitness_evaluator.evaluate(gene) -> (fitness, metrics).

    Optionally drive scores deterministically.
    """
    evaluator = MagicMock()
    if scores is None:
        evaluator.evaluate.return_value = (0.5, {"profit": 1.0})
    else:
        it = iter(scores)
        evaluator.evaluate.side_effect = lambda gene: (next(it), {"profit": 1.0})
    return evaluator


def _base_config(enabled=True):
    return {
        "coevolution": {
            "enabled": enabled,
            "sub_population_size": 4,
            "generations": 3,
            "collaborators_per_eval": 2,
            "elite_size": 1,
            "crossover_rate": 0.5,
            "mutation_rate": 0.2,
        }
    }


# ---------------------------------------------------------------------------
# Decompose / compose round-trip
# ---------------------------------------------------------------------------


class TestDecomposeCompose:
    def test_decompose_produces_three_components(self):
        gene = _make_gene()
        entry, exit_, risk = decompose_gene(gene)
        assert isinstance(entry, EntryComponent)
        assert isinstance(exit_, ExitComponent)
        assert isinstance(risk, RiskComponent)

    def test_round_trip_preserves_entry_fields(self):
        gene = _make_gene(seed=7)
        entry, exit_, risk = decompose_gene(gene)
        composed = compose_gene(entry, exit_, risk, template_gene=gene)
        # Entry conditions are preserved exactly
        assert len(composed.entry_conditions) == len(gene.entry_conditions)
        assert composed.entry_conditions[0].value == gene.entry_conditions[0].value

    def test_round_trip_preserves_risk_fields(self):
        gene = _make_gene(seed=3)
        entry, exit_, risk = decompose_gene(gene)
        composed = compose_gene(entry, exit_, risk, template_gene=gene)
        assert composed.stoploss == gene.stoploss
        assert composed.minimal_roi == gene.minimal_roi


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class TestComponentOperators:
    def test_crossover_returns_same_component_type(self):
        g1, g2 = _make_gene(1), _make_gene(2)
        e1, _, _ = decompose_gene(g1)
        e2, _, _ = decompose_gene(g2)
        child = _crossover_components(e1, e2, "entry")
        assert isinstance(child, EntryComponent)

    def test_mutate_returns_same_component_type(self):
        g = _make_gene()
        _, _, r = decompose_gene(g)
        mutated = _mutate_component(r, "risk", rate=1.0)
        assert isinstance(mutated, RiskComponent)


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    def test_disabled_run_returns_empty(self):
        eng = CoevolutionEngine(
            _base_config(enabled=False),
            strategy_generator=_make_generator(),
            fitness_evaluator=_make_evaluator(),
        )
        assert eng.run() == []

    def test_initialize_creates_balanced_pops(self):
        eng = CoevolutionEngine(
            _base_config(),
            strategy_generator=_make_generator(),
            fitness_evaluator=_make_evaluator(),
        )
        eng.initialize()
        assert len(eng.entry_pop) == 4
        assert len(eng.exit_pop) == 4
        assert len(eng.risk_pop) == 4

    def test_run_produces_composed_genes(self):
        eng = CoevolutionEngine(
            _base_config(),
            strategy_generator=_make_generator(),
            fitness_evaluator=_make_evaluator(),
        )
        result = eng.run()
        assert len(result) > 0
        assert len(result) <= 5


# ---------------------------------------------------------------------------
# Credit assignment — T3.4 risk-pop loop fix
# ---------------------------------------------------------------------------


class TestCreditAssignment:
    def test_every_risk_component_evaluated(self):
        """Every risk component must receive at least one direct evaluation,
        not only ride along on entry/exit evaluations.
        """
        eng = CoevolutionEngine(
            _base_config(),
            strategy_generator=_make_generator(),
            fitness_evaluator=_make_evaluator(),
        )
        eng.initialize()
        eng._evaluate_population(template_gene=_make_gene())
        for risk in eng.risk_pop:
            assert risk.eval_count > 0, "Risk component never evaluated"

    def test_every_exit_component_evaluated(self):
        eng = CoevolutionEngine(
            _base_config(),
            strategy_generator=_make_generator(),
            fitness_evaluator=_make_evaluator(),
        )
        eng.initialize()
        eng._evaluate_population(template_gene=_make_gene())
        for exit_ in eng.exit_pop:
            assert exit_.eval_count > 0


# ---------------------------------------------------------------------------
# HoF integration (T3.4)
# ---------------------------------------------------------------------------


class TestHofInjection:
    def test_evaluate_composed_returns_individuals(self):
        eng = CoevolutionEngine(
            _base_config(),
            strategy_generator=_make_generator(),
            fitness_evaluator=_make_evaluator(scores=[0.8, 0.7, 0.6, 0.5, 0.4] * 50),
        )
        eng.run()
        individuals = eng.evaluate_composed_for_hof(
            template_gene=_make_gene(), n=3
        )
        from genetic_algorithm.genome.individual import Individual

        assert len(individuals) <= 3
        assert all(isinstance(i, Individual) for i in individuals)
        assert all(i.evaluated for i in individuals)
        assert all(i.fitness is not None for i in individuals)

    def test_evaluate_composed_skips_failed_evaluations(self):
        evaluator = MagicMock()
        evaluator.evaluate.return_value = (None, {})
        eng = CoevolutionEngine(
            _base_config(),
            strategy_generator=_make_generator(),
            fitness_evaluator=evaluator,
        )
        eng.run()  # populates _pop with prior evaluator before swap
        # Re-evaluate composed: every call returns None → no individuals
        individuals = eng.evaluate_composed_for_hof(
            template_gene=_make_gene(), n=3
        )
        assert individuals == []

    def test_individuals_have_ids(self):
        eng = CoevolutionEngine(
            _base_config(),
            strategy_generator=_make_generator(),
            fitness_evaluator=_make_evaluator(),
        )
        eng.run()
        individuals = eng.evaluate_composed_for_hof(
            template_gene=_make_gene(), n=3
        )
        # Individual.id is auto-derived; we just want every entry to
        # expose a value so HoF dedup logic doesn't crash.
        ids = [getattr(i, "id", None) for i in individuals]
        assert all(i is not None for i in ids)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


class TestTemplate:
    def test_template_loads(self):
        here = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(here, "config", "templates", "coevolution_pilot.yaml")
        assert os.path.exists(path), f"Missing template: {path}"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["coevolution"]["enabled"] is True
        assert cfg["coevolution"]["inject_into_hof"] is True
        assert cfg["coevolution"]["sub_population_size"] >= 2
