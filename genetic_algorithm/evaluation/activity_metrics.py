"""Small, shared activity metrics used by search and strict replay."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any


def _utc_month(value: Any) -> str | None:
    """Return ``YYYY-MM`` for a date-like close timestamp."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value.strftime("%Y-%m")
    elif isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        seconds = (
            float(value) / 1000.0
            if abs(float(value)) >= 1e11
            else float(value)
        )
        try:
            parsed = datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        raw = str(value).strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                return date.fromisoformat(raw[:10]).strftime("%Y-%m")
            except ValueError:
                return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC)
    return parsed.strftime("%Y-%m")


def count_active_trade_months(
    trades: Sequence[Mapping[str, Any]] | None,
    *,
    daily_profit_abs: Sequence[Sequence[Any]] | None = None,
) -> int:
    """Count UTC months containing at least one closed trade.

    The trade ledger is the semantic source of activity.  ``daily_profit_abs``
    remains a compatibility fallback for old cached results whose compact
    trade records did not retain close timestamps.
    """

    months: set[str] = set()
    for trade in trades or ():
        if not isinstance(trade, Mapping):
            continue
        close_value = trade.get("close_date")
        if close_value is None:
            close_value = trade.get("close_date_utc")
        if close_value is None:
            close_value = trade.get("close_timestamp")
        month = _utc_month(close_value)
        if month is not None:
            months.add(month)
    if months:
        return len(months)

    # Compatibility only: realized daily PnL usually identifies close days,
    # but can hide activity when same-day trades cancel exactly.
    for row in daily_profit_abs or ():
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        day, pnl = row
        try:
            active = math.isfinite(float(pnl)) and float(pnl) != 0.0
        except (TypeError, ValueError):
            continue
        if active:
            month = _utc_month(day)
            if month is not None:
                months.add(month)
    return len(months)
