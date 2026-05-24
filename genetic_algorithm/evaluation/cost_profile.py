"""T2.7 — Realistic transaction-cost profiles.

Centralises the cost assumptions used throughout the GA so we never
accidentally evolve strategies that only work at unrealistic
``fee=0.001`` levels.

Three named profiles
--------------------
* ``optimistic`` — close to maker fees on Binance/Bybit, low slippage.
  Useful for sanity checks, **not** for ranking decisions.
* ``realistic`` *(default)* — current real-world taker fee plus a
  conservative slippage allowance.  This is what live trading actually
  sees.
* ``stress``    — pessimistic regime (wide spreads, volatile execution).
  Used by ``hof stress`` to filter for cost-robust strategies.

The numbers below are calibrated against Bybit/Binance taker schedules
as of 2026Q2.  They can be overridden per-config with explicit
``backtesting.fee`` / ``backtesting.slippage_pct`` values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "realistic"


@dataclass(frozen=True)
class CostProfile:
    """Concrete cost assumptions for a single backtest pass."""

    name: str
    fee: float
    slippage_pct: float
    # Optional knobs reserved for future use (T2.7 hardening, T6 live bridge).
    funding_rate_bps_per_8h: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "fee": self.fee,
            "slippage_pct": self.slippage_pct,
            "funding_rate_bps_per_8h": self.funding_rate_bps_per_8h,
            "notes": self.notes,
        }


# Calibrated profile catalog.  Bybit/Binance perp taker is currently
# ~0.055% but realistic effective cost incl. spread + queueing is
# closer to 0.12%.  Stress models a regime shift / illiquid alt pair.
PROFILES: Dict[str, CostProfile] = {
    "optimistic": CostProfile(
        name="optimistic",
        fee=0.0006,        # ~maker fee
        slippage_pct=0.0002,
        notes="Maker fees, low slippage. Use for sanity checks only.",
    ),
    "realistic": CostProfile(
        name="realistic",
        fee=0.0012,        # taker fee blended w/ spread
        slippage_pct=0.0008,
        notes="Bybit/Binance taker + conservative slippage. Default for ranking.",
    ),
    "stress": CostProfile(
        name="stress",
        fee=0.0020,
        slippage_pct=0.0020,
        notes="Pessimistic regime: wide spreads, volatile execution.",
    ),
}


def get_profile(name: str) -> CostProfile:
    """Return the named profile.  Falls back to the default with a warning."""
    if not name:
        return PROFILES[DEFAULT_PROFILE]
    key = name.strip().lower()
    if key not in PROFILES:
        logger.warning(
            f"[COST] Unknown cost_profile {name!r}; falling back to {DEFAULT_PROFILE!r}. "
            f"Valid: {sorted(PROFILES)}"
        )
        return PROFILES[DEFAULT_PROFILE]
    return PROFILES[key]


def resolve_costs(config: Dict[str, Any]) -> Dict[str, float]:
    """Return the ``(fee, slippage_pct)`` pair to use for this config.

    Resolution order (first non-None wins):
      1. Explicit ``backtesting.fee`` / ``backtesting.slippage_pct``
      2. Named ``backtesting.cost_profile`` (or top-level ``cost_profile``)
      3. The default ``realistic`` profile

    The function returns a small dict so callers can also surface the
    resolved profile name for telemetry.
    """
    bt = config.get("backtesting") or {}
    explicit_fee = bt.get("fee")
    explicit_slip = bt.get("slippage_pct")

    profile_name = bt.get("cost_profile") or config.get("cost_profile") or DEFAULT_PROFILE
    profile = get_profile(profile_name)

    fee = float(explicit_fee) if explicit_fee is not None else profile.fee
    slip = float(explicit_slip) if explicit_slip is not None else profile.slippage_pct
    return {
        "fee": fee,
        "slippage_pct": slip,
        "profile_name": profile.name,
        "from_explicit_fee": explicit_fee is not None,
        "from_explicit_slippage": explicit_slip is not None,
    }


def apply_profile_to_config(config: Dict[str, Any],
                            profile_name: Optional[str] = None,
                            *, overwrite_explicit: bool = False) -> Dict[str, float]:
    """Mutate ``config['backtesting']`` so fee/slippage reflect the profile.

    If ``profile_name`` is given it overrides any config-level profile.
    By default we *never* clobber an explicit ``fee`` / ``slippage_pct``
    the user has set in the config; set ``overwrite_explicit=True`` to
    force the profile values everywhere.  Returns the resolved values
    for logging/telemetry.
    """
    bt = config.setdefault("backtesting", {})
    if profile_name:
        bt["cost_profile"] = profile_name
    if overwrite_explicit:
        # Drop any explicit values so the profile wins.
        bt.pop("fee", None)
        bt.pop("slippage_pct", None)
    resolved = resolve_costs(config)
    if overwrite_explicit or not resolved["from_explicit_fee"]:
        bt["fee"] = resolved["fee"]
    if overwrite_explicit or not resolved["from_explicit_slippage"]:
        bt["slippage_pct"] = resolved["slippage_pct"]
    return resolved


def list_profiles() -> Dict[str, Dict[str, Any]]:
    """Return a name → dict mapping for CLI display."""
    return {name: p.to_dict() for name, p in PROFILES.items()}
