"""Tests for T2.4 — NSGA-II template + Pareto-front HoF defaults."""
from __future__ import annotations

import os

import pytest


TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "templates", "nsga2.yaml"
)


@pytest.fixture
def template_cfg():
    """Parse the nsga2.yaml template and return the resolved config dict."""
    yaml = pytest.importorskip("yaml")
    with open(TEMPLATE_PATH, "r") as fh:
        return yaml.safe_load(fh)


def test_template_file_exists():
    assert os.path.isfile(TEMPLATE_PATH), f"Missing template: {TEMPLATE_PATH}"


def test_template_mode_is_nsga2(template_cfg):
    assert template_cfg.get("genetic_algorithm", {}).get("mode") == "nsga2"


def test_template_defines_three_objectives(template_cfg):
    objs = template_cfg.get("nsga2", {}).get("objectives", [])
    assert len(objs) == 3
    names = [o["name"] for o in objs]
    assert "profit" in names
    assert "max_drawdown" in names
    assert "sharpe_ratio" in names


def test_template_objectives_have_required_fields(template_cfg):
    for obj in template_cfg["nsga2"]["objectives"]:
        assert "name" in obj
        assert obj.get("type") in {"maximize", "minimize", "goldilocks"}


def test_template_pareto_archive_enabled(template_cfg):
    arch = template_cfg.get("pareto_archive", {})
    assert arch.get("enabled") is True
    assert arch.get("max_size", 0) > 0


def test_template_cost_profile_is_realistic(template_cfg):
    bt = template_cfg.get("backtesting", {})
    assert bt.get("cost_profile") == "realistic"


def test_template_fitness_weights_sum_close_to_one(template_cfg):
    fw = template_cfg.get("fitness_weights", {})
    total = sum(v for v in fw.values() if isinstance(v, (int, float)))
    # Auto-normalised at runtime, but template should be close.
    assert 0.95 <= total <= 1.05, f"weights sum to {total}"


def test_template_disables_walk_forward_and_holdout(template_cfg):
    """User preference: pair-split validation only, no WF / holdout."""
    assert not template_cfg.get("walk_forward", {}).get("enabled", False)
    assert not template_cfg.get("holdout", {}).get("enabled", False)


def test_template_pair_validation_enabled(template_cfg):
    pv = template_cfg.get("pair_validation", {})
    assert pv.get("enabled") is True


def test_template_t2_quality_blocks_present(template_cfg):
    """T2.5/T2.6 knobs must be documented (default off, but visible)."""
    fitness_block = template_cfg.get("fitness", {})
    assert "use_bootstrap_ci" in fitness_block
    assert "ic_penalty" in fitness_block


def test_pareto_front_hof_documented():
    """Read the template header and confirm HoF docs are present."""
    with open(TEMPLATE_PATH, "r") as fh:
        text = fh.read()
    assert "Pareto front" in text
    assert "Hall-of-Fame" in text or "HoF" in text
