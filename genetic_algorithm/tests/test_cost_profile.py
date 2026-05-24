"""Tests for T2.7 — Realistic cost profiles."""
from __future__ import annotations

import pytest

from genetic_algorithm.evaluation import cost_profile as cp


def test_default_profile_is_realistic():
    p = cp.get_profile(cp.DEFAULT_PROFILE)
    assert p.name == "realistic"
    assert p.fee > 0
    assert p.slippage_pct > 0


def test_known_profiles_present_and_ordered():
    opt = cp.get_profile("optimistic")
    real = cp.get_profile("realistic")
    stress = cp.get_profile("stress")
    # Costs should monotonically rise from optimistic → realistic → stress
    assert opt.fee < real.fee < stress.fee
    assert opt.slippage_pct < real.slippage_pct < stress.slippage_pct


def test_unknown_profile_falls_back_to_default(caplog):
    p = cp.get_profile("imaginary")
    assert p.name == cp.DEFAULT_PROFILE


def test_empty_name_falls_back():
    p = cp.get_profile("")
    assert p.name == cp.DEFAULT_PROFILE


def test_get_profile_is_case_insensitive_and_trims():
    assert cp.get_profile("  STRESS  ").name == "stress"


def test_resolve_costs_uses_profile_when_no_explicit():
    cfg = {"backtesting": {"cost_profile": "stress"}}
    out = cp.resolve_costs(cfg)
    assert out["profile_name"] == "stress"
    assert out["fee"] == cp.PROFILES["stress"].fee
    assert out["slippage_pct"] == cp.PROFILES["stress"].slippage_pct
    assert out["from_explicit_fee"] is False
    assert out["from_explicit_slippage"] is False


def test_resolve_costs_explicit_overrides_profile():
    cfg = {"backtesting": {"cost_profile": "stress", "fee": 0.0001,
                           "slippage_pct": 0.0005}}
    out = cp.resolve_costs(cfg)
    assert out["fee"] == pytest.approx(0.0001)
    assert out["slippage_pct"] == pytest.approx(0.0005)
    assert out["from_explicit_fee"] is True
    assert out["from_explicit_slippage"] is True
    # Profile name still reported for traceability.
    assert out["profile_name"] == "stress"


def test_resolve_costs_top_level_profile_field():
    cfg = {"cost_profile": "optimistic", "backtesting": {}}
    out = cp.resolve_costs(cfg)
    assert out["profile_name"] == "optimistic"


def test_resolve_costs_default_when_nothing_specified():
    out = cp.resolve_costs({"backtesting": {}})
    assert out["profile_name"] == cp.DEFAULT_PROFILE


def test_apply_profile_to_config_mutates():
    cfg = {"backtesting": {}}
    resolved = cp.apply_profile_to_config(cfg, profile_name="stress")
    assert cfg["backtesting"]["fee"] == cp.PROFILES["stress"].fee
    assert cfg["backtesting"]["slippage_pct"] == cp.PROFILES["stress"].slippage_pct
    assert cfg["backtesting"]["cost_profile"] == "stress"
    assert resolved["profile_name"] == "stress"


def test_apply_profile_preserves_explicit_fee_by_default():
    cfg = {"backtesting": {"fee": 0.00005}}
    cp.apply_profile_to_config(cfg, profile_name="stress")
    # Explicit value preserved.
    assert cfg["backtesting"]["fee"] == pytest.approx(0.00005)
    # But slippage gets filled from the profile (it wasn't explicit).
    assert cfg["backtesting"]["slippage_pct"] == cp.PROFILES["stress"].slippage_pct


def test_apply_profile_overwrite_explicit_flag():
    cfg = {"backtesting": {"fee": 0.00005, "slippage_pct": 0.0}}
    cp.apply_profile_to_config(cfg, profile_name="realistic", overwrite_explicit=True)
    assert cfg["backtesting"]["fee"] == cp.PROFILES["realistic"].fee
    assert cfg["backtesting"]["slippage_pct"] == cp.PROFILES["realistic"].slippage_pct


def test_list_profiles_contains_all_three():
    lst = cp.list_profiles()
    assert set(lst) == {"optimistic", "realistic", "stress"}
    assert lst["realistic"]["fee"] > 0
