"""Canonical calendar-aligned equity and downside metrics for GA V2.

Freqtrade's strategy report exposes absolute realised PnL grouped by close day,
plus starting/final wallet balances.  That is enough to build a capital-weighted
*realised-close* equity curve without summing trade-return percentages.  It is
not a mark-to-market curve: adverse movement inside open trades is absent.  The
method is therefore persisted explicitly and must not silently satisfy a V2
mark-to-market promotion gate.

All returns and drawdowns in this module are decimal ratios.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import fmean, stdev
from typing import Any


REALIZED_CLOSE_EQUITY = "REALIZED_CLOSE"
MARK_TO_MARKET_EQUITY = "MARK_TO_MARKET"
RISK_METRIC_CONTRACT_VERSION = "calendar-effective-v1"
CALENDAR_DAYS_PER_YEAR = 365
MONTHS_PER_YEAR = 12
DEFAULT_ANNUAL_RISK_FREE_RATE = 0.0
RISK_RATIO_EPSILON = 1e-15


class EquityDataError(ValueError):
    """Raised when raw equity inputs are inconsistent or unusable."""


@dataclass(frozen=True)
class DailyEquitySeriesV2:
    """One complete calendar series derived from absolute daily wallet PnL."""

    dates: tuple[date, ...]
    daily_profit_abs: tuple[float, ...]
    daily_net_returns: tuple[float, ...]
    equity_curve: tuple[float, ...]
    starting_balance: float
    final_balance: float
    equity_method: str = REALIZED_CLOSE_EQUITY

    def __post_init__(self) -> None:
        count = len(self.dates)
        if len(self.daily_profit_abs) != count or len(self.daily_net_returns) != count:
            raise EquityDataError("dates, daily PnL, and daily returns must have equal length")
        if len(self.equity_curve) != count + 1:
            raise EquityDataError(
                "equity_curve must include initial balance plus one point per day"
            )


@dataclass(frozen=True)
class EquityRiskMetricsV2:
    """Point estimates computed only from one calendar-aligned equity series."""

    risk_metric_contract_version: str
    periods_per_year: int
    annual_risk_free_rate: float
    periodic_risk_free_rate: float
    observation_days: int
    total_net_return: float
    annualized_net_return: float | None
    daily_return_mean: float | None
    daily_return_volatility: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown: float
    max_drawdown_duration_days: int
    time_under_water_ratio: float
    ulcer_index: float
    daily_expected_shortfall_5: float | None
    calmar_ratio: float | None

    @property
    def has_defined_risk_ratios(self) -> bool:
        """Whether both volatility ratios are measured instead of undefined."""
        return self.sharpe_ratio is not None and self.sortino_ratio is not None


@dataclass(frozen=True)
class PeriodicRiskRatiosV2:
    """Sharpe/Sortino components under one explicit periodic-return contract."""

    risk_metric_contract_version: str
    observation_count: int
    periods_per_year: int
    annual_risk_free_rate: float
    periodic_risk_free_rate: float
    periodic_return_mean: float | None
    periodic_excess_return_mean: float | None
    periodic_return_volatility: float | None
    periodic_downside_deviation: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None


@dataclass(frozen=True)
class BootstrapRiskBoundsV2:
    """Deterministic moving-block-bootstrap bounds for shadow gates."""

    samples: int
    block_length_days: int
    confidence: float
    seed: int
    annualized_net_return_lcb: float
    max_drawdown_ucb: float
    daily_expected_shortfall_5_ucb: float


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise EquityDataError(f"unsupported date value: {value!r}")
    text = value.strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise EquityDataError(f"invalid ISO date: {value!r}") from exc


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise EquityDataError(f"{field_name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EquityDataError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise EquityDataError(f"{field_name} must be finite")
    return number


def _effective_periodic_rate(annual_rate: object, periods_per_year: int) -> float:
    """Convert one effective annual decimal rate into an effective period rate."""

    if (
        not isinstance(periods_per_year, int)
        or isinstance(periods_per_year, bool)
        or periods_per_year < 1
    ):
        raise EquityDataError("periods_per_year must be a positive integer")
    annual = _finite_float(annual_rate, "annual_risk_free_rate")
    if annual <= -1.0:
        raise EquityDataError("annual_risk_free_rate must be greater than -1")
    return math.expm1(math.log1p(annual) / periods_per_year)


def calculate_periodic_risk_ratios(
    periodic_returns: Sequence[float],
    *,
    periods_per_year: int,
    annual_risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE,
) -> PeriodicRiskRatiosV2:
    """Calculate canonical annualized Sharpe and Sortino ratios.

    Inputs are decimal net returns at one complete, equally spaced frequency.
    The risk-free rate is an effective annual decimal rate and is converted
    geometrically. Sharpe uses sample volatility (``ddof=1``). Sortino uses a
    full-sample lower-partial-moment denominator around that periodic
    risk-free target; observations above the target contribute zero. Ratios
    that have fewer than two observations or no measurable denominator are
    ``None`` instead of a favorable sentinel.
    """

    returns = tuple(
        _finite_float(value, f"periodic_returns[{index}]")
        for index, value in enumerate(periodic_returns)
    )
    if any(value < -1.0 for value in returns):
        raise EquityDataError("periodic returns below -100% are invalid")

    periodic_rf = _effective_periodic_rate(annual_risk_free_rate, periods_per_year)
    excess = tuple(value - periodic_rf for value in returns)
    mean_return = fmean(returns) if returns else None
    mean_excess = fmean(excess) if excess else None
    volatility = stdev(returns) if len(returns) >= 2 else None
    downside_deviation = (
        math.sqrt(fmean(min(value, 0.0) ** 2 for value in excess)) if excess else None
    )

    scale = math.sqrt(periods_per_year)
    sharpe = None
    if mean_excess is not None and volatility is not None and volatility > RISK_RATIO_EPSILON:
        sharpe = mean_excess / volatility * scale

    sortino = None
    if (
        mean_excess is not None
        and downside_deviation is not None
        and downside_deviation > RISK_RATIO_EPSILON
    ):
        sortino = mean_excess / downside_deviation * scale

    return PeriodicRiskRatiosV2(
        risk_metric_contract_version=RISK_METRIC_CONTRACT_VERSION,
        observation_count=len(returns),
        periods_per_year=periods_per_year,
        annual_risk_free_rate=_finite_float(
            annual_risk_free_rate, "annual_risk_free_rate"
        ),
        periodic_risk_free_rate=periodic_rf,
        periodic_return_mean=mean_return,
        periodic_excess_return_mean=mean_excess,
        periodic_return_volatility=volatility,
        periodic_downside_deviation=downside_deviation,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
    )


def _validated_equity_curve(series: DailyEquitySeriesV2) -> tuple[float, ...]:
    if not series.dates:
        raise EquityDataError("daily equity series must contain at least one calendar day")
    if series.equity_method not in {REALIZED_CLOSE_EQUITY, MARK_TO_MARKET_EQUITY}:
        raise EquityDataError(f"unsupported equity_method: {series.equity_method!r}")

    starting = _finite_float(series.starting_balance, "starting_balance")
    final = _finite_float(series.final_balance, "final_balance")
    if starting <= 0:
        raise EquityDataError("starting_balance must be positive")
    if final < 0:
        raise EquityDataError("final_balance must not be negative")

    for index, day in enumerate(series.dates):
        if not isinstance(day, date):
            raise EquityDataError(f"dates[{index}] must be a date")
        if index and day != series.dates[index - 1] + timedelta(days=1):
            raise EquityDataError("daily equity dates must be consecutive calendar days")

    equity = tuple(
        _finite_float(value, f"equity_curve[{index}]")
        for index, value in enumerate(series.equity_curve)
    )
    if not math.isclose(equity[0], starting, rel_tol=1e-12, abs_tol=1e-12):
        raise EquityDataError("equity_curve does not start at starting_balance")
    if not math.isclose(equity[-1], final, rel_tol=1e-12, abs_tol=1e-12):
        raise EquityDataError("equity_curve does not end at final_balance")
    return equity


def _validate_daily_equity_series(series: DailyEquitySeriesV2) -> None:
    """Reject internally inconsistent point-metric inputs before scoring."""

    equity = _validated_equity_curve(series)
    for index, (pnl_value, return_value) in enumerate(
        zip(series.daily_profit_abs, series.daily_net_returns, strict=True)
    ):
        previous = equity[index]
        current = equity[index + 1]
        pnl = _finite_float(pnl_value, f"daily_profit_abs[{index}]")
        daily_return = _finite_float(return_value, f"daily_net_returns[{index}]")
        if previous <= 0:
            raise EquityDataError("wallet equity is non-positive before period end")
        if current < 0:
            raise EquityDataError("wallet equity fell below zero")
        measured_pnl = current - previous
        tolerance = max(1e-12, abs(current) * 1e-10)
        if not math.isclose(pnl, measured_pnl, rel_tol=1e-10, abs_tol=tolerance):
            raise EquityDataError(f"daily PnL does not reconcile at index {index}")
        measured_return = measured_pnl / previous
        if not math.isclose(
            daily_return,
            measured_return,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise EquityDataError(f"daily return does not reconcile at index {index}")


def build_realized_close_equity(
    *,
    period_start: date | datetime | str,
    period_end: date | datetime | str,
    starting_balance: float,
    daily_profit_abs: Mapping[date | datetime | str, float]
    | Iterable[tuple[date | datetime | str, float]],
    expected_final_balance: float | None = None,
) -> DailyEquitySeriesV2:
    """Build a zero-filled calendar equity curve from absolute realised PnL.

    Daily return is ``day_profit_abs / prior_day_equity``.  Duplicate date rows
    are summed, inactive days are explicit zero returns, and PnL outside the
    declared period is rejected.  A non-positive wallet makes subsequent return
    ratios undefined and therefore fails closed.
    """

    start = _as_date(period_start)
    end = _as_date(period_end)
    if end < start:
        raise EquityDataError("period_end must not precede period_start")

    initial = _finite_float(starting_balance, "starting_balance")
    if initial <= 0:
        raise EquityDataError("starting_balance must be positive")

    items = daily_profit_abs.items() if isinstance(daily_profit_abs, Mapping) else daily_profit_abs
    pnl_by_day: dict[date, float] = {}
    for raw_day, raw_pnl in items:
        day = _as_date(raw_day)
        if day < start or day > end:
            raise EquityDataError(f"daily PnL date {day.isoformat()} is outside declared period")
        pnl = _finite_float(raw_pnl, f"daily_profit_abs[{day.isoformat()}]")
        pnl_by_day[day] = pnl_by_day.get(day, 0.0) + pnl

    day_count = (end - start).days + 1
    dates = tuple(start + timedelta(days=offset) for offset in range(day_count))
    profits: list[float] = []
    returns: list[float] = []
    equity: list[float] = [initial]

    for day in dates:
        previous = equity[-1]
        if previous <= 0:
            raise EquityDataError("wallet equity is non-positive before period end")
        pnl = pnl_by_day.get(day, 0.0)
        current = previous + pnl
        daily_return = pnl / previous
        if not math.isfinite(current) or not math.isfinite(daily_return):
            raise EquityDataError("daily equity calculation produced a non-finite value")
        if current < 0:
            raise EquityDataError("wallet equity fell below zero")
        profits.append(pnl)
        returns.append(daily_return)
        equity.append(current)

    final = equity[-1]
    if expected_final_balance is not None:
        expected = _finite_float(expected_final_balance, "expected_final_balance")
        # Freqtrade rounds every absolute daily PnL value to 10 decimals.
        # Permit only that accumulated rounding envelope.
        tolerance = max(1e-7, abs(expected) * 1e-8, len(dates) * 1e-10)
        if not math.isclose(final, expected, rel_tol=1e-8, abs_tol=tolerance):
            raise EquityDataError(
                f"daily PnL closes at {final:.12g}, expected final balance {expected:.12g}"
            )

    return DailyEquitySeriesV2(
        dates=dates,
        daily_profit_abs=tuple(profits),
        daily_net_returns=tuple(returns),
        equity_curve=tuple(equity),
        starting_balance=initial,
        final_balance=final,
    )


def build_equity_from_freqtrade_stats(stats: Mapping[str, object]) -> DailyEquitySeriesV2:
    """Extract and validate the realised-close series from strategy stats."""

    required = ("backtest_start", "backtest_end", "starting_balance")
    missing = [field for field in required if stats.get(field) is None]
    if missing:
        raise EquityDataError(f"Freqtrade stats missing: {', '.join(missing)}")

    daily_profit = stats.get("daily_profit")
    if daily_profit is None:
        if int(stats.get("total_trades", 0) or 0) == 0:
            daily_profit = []
        else:
            raise EquityDataError("Freqtrade stats missing daily_profit for a traded strategy")
    if not isinstance(daily_profit, (list, tuple, dict)):
        raise EquityDataError("Freqtrade daily_profit must be a mapping or sequence of pairs")

    return build_realized_close_equity(
        period_start=stats["backtest_start"],
        period_end=stats["backtest_end"],
        starting_balance=stats["starting_balance"],
        daily_profit_abs=daily_profit,
        expected_final_balance=stats.get("final_balance"),
    )


def _trade_records(trades: object) -> list[Mapping[str, Any]]:
    if hasattr(trades, "to_dict"):
        records = trades.to_dict("records")
    elif isinstance(trades, (list, tuple)):
        records = list(trades)
    else:
        raise EquityDataError("trades must be a DataFrame or sequence of mappings")
    if not all(isinstance(row, Mapping) for row in records):
        raise EquityDataError("every trade must be a mapping")
    return records


def _utc_timestamp(value: object, field_name: str) -> Any:
    """Parse one timestamp lazily so this metrics module stays pandas-optional."""

    try:
        import pandas as pd

        timestamp = pd.to_datetime(value, utc=True, errors="raise")
    except Exception as exc:
        raise EquityDataError(f"{field_name} must be a valid timestamp") from exc
    if getattr(timestamp, "ndim", 0) != 0:
        raise EquityDataError(f"{field_name} must be a scalar timestamp")
    return timestamp


def _order_fill_timestamp(value: object, field_name: str) -> Any:
    """Parse Freqtrade's millisecond order timestamp without unit guessing."""

    try:
        import pandas as pd

        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise ValueError
            timestamp = pd.to_datetime(value, unit="ms", utc=True, errors="raise")
        else:
            timestamp = pd.to_datetime(value, utc=True, errors="raise")
    except Exception as exc:
        raise EquityDataError(f"{field_name} must be a valid timestamp") from exc
    if getattr(timestamp, "ndim", 0) != 0:
        raise EquityDataError(f"{field_name} must be a scalar timestamp")
    return timestamp


def _daily_pair_closes(ohlcv_by_pair: Mapping[object, object]) -> dict[str, dict[date, float]]:
    """Reduce candle frames to one verified end-of-day close per pair."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a project dependency
        raise EquityDataError("pandas is required for mark-to-market equity") from exc

    result: dict[str, dict[date, float]] = {}
    for raw_pair, frame in ohlcv_by_pair.items():
        pair = raw_pair[0] if isinstance(raw_pair, tuple) and raw_pair else raw_pair
        pair = str(pair)
        if not hasattr(frame, "columns"):
            raise EquityDataError(f"OHLCV for {pair} is not a DataFrame")
        if "close" not in frame.columns:
            raise EquityDataError(f"OHLCV for {pair} has no close column")

        if "date" in frame.columns:
            raw_dates = frame["date"]
        else:
            raw_dates = frame.index
        timestamps = pd.to_datetime(raw_dates, utc=True, errors="coerce")
        closes = pd.to_numeric(frame["close"], errors="coerce")
        close_series = pd.Series(closes.to_numpy(), index=timestamps).dropna().sort_index()
        if close_series.empty:
            raise EquityDataError(f"OHLCV for {pair} has no finite dated closes")
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in close_series):
            raise EquityDataError(f"OHLCV for {pair} contains invalid close prices")
        daily = close_series.groupby(close_series.index.date).last()
        result[pair] = {day: float(value) for day, value in daily.items()}
    return result


def _position_profit(
    *,
    amount: float,
    open_rate: float,
    close_rate: float,
    fee_open: float,
    fee_close: float,
    is_short: bool,
) -> float:
    """Mirror Freqtrade's absolute, fee-adjusted position PnL.

    Leverage is intentionally absent: Freqtrade already applies leverage when it
    derives the filled base amount. Multiplying the absolute PnL again would
    double-count leverage.
    """

    open_value = amount * open_rate * (1.0 - fee_open if is_short else 1.0 + fee_open)
    close_value = amount * close_rate * (1.0 + fee_close if is_short else 1.0 - fee_close)
    return open_value - close_value if is_short else close_value - open_value


def _trade_ledger_pnl(
    trade: Mapping[str, Any],
    *,
    cutoff: Any,
    mark: float | None,
    trade_index: int,
) -> tuple[float, float]:
    """Replay filled orders up to ``cutoff`` and return (PnL, open amount)."""

    amount = 0.0
    average_entry = 0.0
    realized = 0.0
    tolerance = 1e-10

    for order_index, order in enumerate(trade["orders"]):
        if order["filled_at"] >= cutoff:
            break
        fill_amount = order["amount"]
        fill_price = order["price"]
        if order["is_entry"]:
            new_amount = amount + fill_amount
            average_entry = (
                (amount * average_entry + fill_amount * fill_price) / new_amount
            )
            amount = new_amount
            continue

        if fill_amount > amount + tolerance:
            raise EquityDataError(
                f"trade {trade_index} order {order_index} exits more than the open position"
            )
        exit_amount = min(fill_amount, amount)
        realized += _position_profit(
            amount=exit_amount,
            open_rate=average_entry,
            close_rate=fill_price,
            fee_open=trade["fee_open"],
            fee_close=trade["fee_close"],
            is_short=trade["is_short"],
        )
        amount -= exit_amount
        if amount <= tolerance:
            amount = 0.0
            average_entry = 0.0

    realized += sum(
        event["amount"]
        for event in trade["funding_events"]
        if event["occurred_at"] < cutoff
    )

    if amount > 0.0:
        if mark is None:
            raise EquityDataError(
                f"missing mark for open {trade['pair']} position before {cutoff.isoformat()}"
            )
        realized += _position_profit(
            amount=amount,
            open_rate=average_entry,
            close_rate=mark,
            fee_open=trade["fee_open"],
            fee_close=trade["fee_close"],
            is_short=trade["is_short"],
        )
    return realized, amount


def _normalize_trade_orders(
    orders: object,
    *,
    trade_index: int,
    start_ts: Any,
    end_ts: Any,
) -> list[dict[str, Any]]:
    if not isinstance(orders, (list, tuple)) or len(orders) < 2:
        raise EquityDataError("trade must contain filled entry and exit order provenance")
    if not all(isinstance(order, Mapping) for order in orders):
        raise EquityDataError("trade orders must be mappings with entry/exit provenance")

    normalized: list[dict[str, Any]] = []
    previous_fill = None
    required = ("ft_is_entry", "amount", "safe_price", "order_filled_timestamp")
    for order_index, order in enumerate(orders):
        missing = [field for field in required if order.get(field) is None]
        if missing:
            raise EquityDataError(
                f"trade {trade_index} order {order_index} missing: {', '.join(missing)}"
            )
        filled_at = _order_fill_timestamp(
            order["order_filled_timestamp"],
            f"trade[{trade_index}].orders[{order_index}].order_filled_timestamp",
        )
        if not start_ts <= filled_at < end_ts:
            raise EquityDataError(
                f"trade {trade_index} order {order_index} lies outside the declared period"
            )
        if previous_fill is not None and filled_at < previous_fill:
            raise EquityDataError(f"trade {trade_index} orders are not chronological")
        previous_fill = filled_at
        fill_amount = _finite_float(
            order["amount"], f"trade[{trade_index}].orders[{order_index}].amount"
        )
        fill_price = _finite_float(
            order["safe_price"], f"trade[{trade_index}].orders[{order_index}].safe_price"
        )
        if fill_amount <= 0 or fill_price <= 0:
            raise EquityDataError(
                f"trade {trade_index} order {order_index} has non-positive amount or price"
            )
        normalized.append(
            {
                "is_entry": bool(order["ft_is_entry"]),
                "amount": fill_amount,
                "price": fill_price,
                "filled_at": filled_at,
            }
        )
    if not normalized[0]["is_entry"]:
        raise EquityDataError(f"trade {trade_index} starts with an exit order")
    return normalized


def _normalize_funding_events(
    raw_events: object,
    *,
    aggregate_funding: float,
    trade_index: int,
    first_fill: Any,
    final_fill: Any,
) -> list[dict[str, Any]]:
    if raw_events is None:
        if math.isclose(aggregate_funding, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return []
        raise EquityDataError(
            "funding-bearing trade has no timestamped funding ledger; "
            "aggregate funding cannot produce daily equity"
        )
    if not isinstance(raw_events, (list, tuple)):
        raise EquityDataError(f"trade {trade_index} funding_fee_events must be a sequence")

    normalized: list[dict[str, Any]] = []
    previous_time = None
    for event_index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            raise EquityDataError(
                f"trade {trade_index} funding event {event_index} must be a mapping"
            )
        missing = [field for field in ("timestamp", "amount") if event.get(field) is None]
        if missing:
            raise EquityDataError(
                f"trade {trade_index} funding event {event_index} missing: {', '.join(missing)}"
            )
        occurred_at = _order_fill_timestamp(
            event["timestamp"],
            f"trade[{trade_index}].funding_fee_events[{event_index}].timestamp",
        )
        amount = _finite_float(
            event["amount"],
            f"trade[{trade_index}].funding_fee_events[{event_index}].amount",
        )
        if not first_fill <= occurred_at <= final_fill:
            raise EquityDataError(
                f"trade {trade_index} funding event {event_index} lies outside its filled position"
            )
        if previous_time is not None and occurred_at < previous_time:
            raise EquityDataError(f"trade {trade_index} funding events are not chronological")
        previous_time = occurred_at
        if math.isclose(amount, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise EquityDataError(
                f"trade {trade_index} funding event {event_index} has zero amount"
            )
        normalized.append({"occurred_at": occurred_at, "amount": amount})

    event_total = sum(event["amount"] for event in normalized)
    tolerance = max(1e-10, abs(aggregate_funding) * 1e-9, len(normalized) * 1e-12)
    if not math.isclose(
        event_total,
        aggregate_funding,
        rel_tol=1e-9,
        abs_tol=tolerance,
    ):
        raise EquityDataError(
            f"trade {trade_index} funding ledger sums to {event_total:.12g}, "
            f"expected funding_fees {aggregate_funding:.12g}"
        )
    return normalized


def _normalize_trade_ledger(
    row: Mapping[str, Any],
    *,
    trade_index: int,
    start_ts: Any,
    end_ts: Any,
) -> dict[str, Any]:
    required = {
        "pair", "stake_amount", "max_stake_amount", "amount", "open_date",
        "close_date", "open_rate", "close_rate", "fee_open", "fee_close",
        "profit_abs", "leverage", "is_short", "is_open", "funding_fees", "orders",
    }
    missing = sorted(field for field in required if field not in row)
    if missing:
        raise EquityDataError(f"trade {trade_index} missing: {', '.join(missing)}")
    if bool(row["is_open"]):
        raise EquityDataError(
            "open trades at period end require terminal wallet and position evidence"
        )

    leverage = _finite_float(row["leverage"], f"trade[{trade_index}].leverage")
    funding = _finite_float(row["funding_fees"], f"trade[{trade_index}].funding_fees")
    if leverage <= 0:
        raise EquityDataError(f"trade {trade_index} has non-positive leverage")

    stake = _finite_float(row["stake_amount"], f"trade[{trade_index}].stake_amount")
    max_stake = _finite_float(
        row["max_stake_amount"], f"trade[{trade_index}].max_stake_amount"
    )
    if stake < 0 or max_stake <= 0:
        raise EquityDataError(f"trade {trade_index} has invalid stake evidence")

    numeric = {
        field: _finite_float(row[field], f"trade[{trade_index}].{field}")
        for field in ("amount", "open_rate", "close_rate", "fee_open", "fee_close", "profit_abs")
    }
    if numeric["amount"] < 0 or numeric["open_rate"] <= 0 or numeric["close_rate"] <= 0:
        raise EquityDataError(f"trade {trade_index} has invalid terminal amount or rate")
    if numeric["fee_open"] < 0 or numeric["fee_close"] < 0:
        raise EquityDataError(f"trade {trade_index} has a negative fee")

    opened = _utc_timestamp(row["open_date"], f"trade[{trade_index}].open_date")
    closed = _utc_timestamp(row["close_date"], f"trade[{trade_index}].close_date")
    if not start_ts <= opened < end_ts or not start_ts <= closed < end_ts:
        raise EquityDataError(f"trade {trade_index} lies outside the declared period")
    if closed < opened:
        raise EquityDataError(f"trade {trade_index} closes before it opens")

    orders = _normalize_trade_orders(
        row["orders"],
        trade_index=trade_index,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    return {
        "pair": str(row["pair"]),
        "opened": opened,
        "closed": closed,
        "leverage": leverage,
        "is_short": bool(row["is_short"]),
        "orders": orders,
        "funding_events": _normalize_funding_events(
            row.get("funding_fee_events"),
            aggregate_funding=funding,
            trade_index=trade_index,
            first_fill=orders[0]["filled_at"],
            final_fill=orders[-1]["filled_at"],
        ),
        **numeric,
    }


def _reconcile_trade_ledgers(trades: Sequence[Mapping[str, Any]], end_ts: Any) -> None:
    for index, trade in enumerate(trades):
        replayed, open_amount = _trade_ledger_pnl(
            trade,
            cutoff=end_ts,
            mark=None,
            trade_index=index,
        )
        if open_amount > 0.0:
            raise EquityDataError(f"trade {index} order ledger does not fully close the position")
        tolerance = max(1e-7, abs(trade["profit_abs"]) * 1e-8, len(trade["orders"]) * 1e-8)
        if not math.isclose(
            replayed,
            trade["profit_abs"],
            rel_tol=1e-8,
            abs_tol=tolerance,
        ):
            raise EquityDataError(
                f"trade {index} order ledger yields {replayed:.12g}, "
                f"expected profit_abs {trade['profit_abs']:.12g}"
            )


def _account_pnl_at_cutoff(
    trades: Sequence[Mapping[str, Any]],
    *,
    cutoff: Any,
    day: date,
    closes_by_pair: Mapping[str, Mapping[date, float]],
) -> float:
    account_pnl = 0.0
    for index, trade in enumerate(trades):
        if trade["orders"][0]["filled_at"] >= cutoff:
            continue
        final_fill = trade["orders"][-1]["filled_at"]
        mark = None
        if final_fill >= cutoff:
            pair_closes = closes_by_pair.get(trade["pair"])
            if pair_closes is None:
                raise EquityDataError(f"missing OHLCV for traded pair {trade['pair']}")
            mark = pair_closes.get(day)
            if mark is None:
                raise EquityDataError(
                    f"missing end-of-day close for open {trade['pair']} trade "
                    f"on {day.isoformat()}"
                )
        trade_pnl, _ = _trade_ledger_pnl(
            trade,
            cutoff=cutoff,
            mark=mark,
            trade_index=index,
        )
        account_pnl += trade_pnl
    return account_pnl


def build_mark_to_market_equity(
    *,
    period_start: date | datetime | str,
    period_end: date | datetime | str,
    starting_balance: float,
    trades: object,
    ohlcv_by_pair: Mapping[object, object],
    expected_final_balance: float | None = None,
) -> DailyEquitySeriesV2:
    """Build end-of-day account equity by replaying Freqtrade order fills.

    Account equity is initial wallet plus realised PnL and fee-adjusted
    unrealised PnL for every position at each UTC day boundary. The order ledger
    supports long/short, leverage (encoded in filled amount), multiple entries,
    and partial exits. Every closed trade is independently reconciled against
    Freqtrade's final ``profit_abs`` before the portfolio series is accepted.

    Timestamped funding cashflows are applied on their actual UTC event days and
    reconciled against Freqtrade's aggregate funding total. Open trades and
    aggregate-only funding remain fail-closed: neither can produce a complete
    daily return path without inventing terminal or temporal evidence.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a project dependency
        raise EquityDataError("pandas is required for mark-to-market equity") from exc

    start = _as_date(period_start)
    end = _as_date(period_end)
    if end < start:
        raise EquityDataError("period_end must not precede period_start")
    initial = _finite_float(starting_balance, "starting_balance")
    if initial <= 0:
        raise EquityDataError("starting_balance must be positive")
    if not isinstance(ohlcv_by_pair, Mapping):
        raise EquityDataError("ohlcv_by_pair must be a mapping")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    normalized = [
        _normalize_trade_ledger(
            row,
            trade_index=index,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        for index, row in enumerate(_trade_records(trades))
    ]

    closes_by_pair = _daily_pair_closes(ohlcv_by_pair) if normalized else {}
    dates = tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))
    equity: list[float] = [initial]
    changes: list[float] = []
    returns: list[float] = []

    # Catch wrong fill semantics before an apparently plausible portfolio
    # curve can reach a promotion gate.
    _reconcile_trade_ledgers(normalized, end_ts)

    for day in dates:
        cutoff = pd.Timestamp(day + timedelta(days=1), tz="UTC")
        account_pnl = _account_pnl_at_cutoff(
            normalized,
            cutoff=cutoff,
            day=day,
            closes_by_pair=closes_by_pair,
        )

        current = initial + account_pnl
        previous = equity[-1]
        if previous <= 0 or current < 0 or not math.isfinite(current):
            raise EquityDataError("mark-to-market wallet equity became invalid")
        change = current - previous
        daily_return = change / previous
        changes.append(change)
        returns.append(daily_return)
        equity.append(current)

    final = equity[-1]
    if expected_final_balance is not None:
        expected = _finite_float(expected_final_balance, "expected_final_balance")
        tolerance = max(1e-7, abs(expected) * 1e-8, len(normalized) * 1e-10)
        if not math.isclose(final, expected, rel_tol=1e-8, abs_tol=tolerance):
            raise EquityDataError(
                f"mark-to-market closes at {final:.12g}, expected final balance {expected:.12g}"
            )

    return DailyEquitySeriesV2(
        dates=dates,
        daily_profit_abs=tuple(changes),
        daily_net_returns=tuple(returns),
        equity_curve=tuple(equity),
        starting_balance=initial,
        final_balance=final,
        equity_method=MARK_TO_MARKET_EQUITY,
    )


def _expected_shortfall_loss(returns: Sequence[float], alpha: float = 0.05) -> float | None:
    if not returns:
        return None
    tail_count = max(1, math.ceil(len(returns) * alpha))
    tail_mean = fmean(sorted(returns)[:tail_count])
    return max(0.0, -tail_mean)


def _annualized_return(
    returns: Sequence[float],
    *,
    periods_per_year: int = CALENDAR_DAYS_PER_YEAR,
) -> float | None:
    if not returns:
        return None
    if (
        not isinstance(periods_per_year, int)
        or isinstance(periods_per_year, bool)
        or periods_per_year < 1
    ):
        raise EquityDataError("periods_per_year must be a positive integer")

    finite_returns = tuple(
        _finite_float(value, f"returns[{index}]") for index, value in enumerate(returns)
    )
    if any(value < -1.0 for value in finite_returns):
        raise EquityDataError("returns below -100% are invalid")
    if any(value == -1.0 for value in finite_returns):
        return -1.0

    annual_log_growth = (
        fmean(math.log1p(value) for value in finite_returns) * periods_per_year
    )
    try:
        annualized = math.expm1(annual_log_growth)
    except OverflowError as exc:
        raise EquityDataError("annualized return is not finite") from exc
    if not math.isfinite(annualized):
        raise EquityDataError("annualized return is not finite")
    return annualized


def _drawdown_statistics(equity: Sequence[float]) -> tuple[float, int, float, float]:
    if len(equity) < 2:
        return 0.0, 0, 0.0, 0.0

    peak = equity[0]
    drawdowns: list[float] = []
    current_duration = 0
    max_duration = 0
    underwater_days = 0

    for value in equity[1:]:
        peak = max(peak, value)
        drawdown = (peak - value) / peak if peak > 0 else 1.0
        drawdowns.append(drawdown)
        if drawdown > 1e-15:
            underwater_days += 1
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    max_drawdown = max(drawdowns, default=0.0)
    time_under_water = underwater_days / len(drawdowns) if drawdowns else 0.0
    ulcer_index = math.sqrt(fmean(value * value for value in drawdowns)) if drawdowns else 0.0
    return max_drawdown, max_duration, time_under_water, ulcer_index


def calculate_equity_risk_metrics(
    series: DailyEquitySeriesV2,
    *,
    annual_risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE,
) -> EquityRiskMetricsV2:
    """Calculate point metrics from the canonical daily series.

    Undefined ratios are returned as ``None`` rather than a favourable or
    punitive sentinel.  In particular, zero variance/downside never becomes a
    magic Sharpe/Sortino score.
    """

    _validate_daily_equity_series(series)
    returns = series.daily_net_returns
    count = len(returns)
    total_return = series.final_balance / series.starting_balance - 1.0
    annualized = _annualized_return(returns, periods_per_year=CALENDAR_DAYS_PER_YEAR)
    ratios = calculate_periodic_risk_ratios(
        returns,
        periods_per_year=CALENDAR_DAYS_PER_YEAR,
        annual_risk_free_rate=annual_risk_free_rate,
    )

    max_dd, dd_duration, time_under_water, ulcer = _drawdown_statistics(series.equity_curve)
    calmar = None
    if annualized is not None and max_dd > RISK_RATIO_EPSILON:
        calmar = annualized / max_dd

    return EquityRiskMetricsV2(
        risk_metric_contract_version=RISK_METRIC_CONTRACT_VERSION,
        periods_per_year=CALENDAR_DAYS_PER_YEAR,
        annual_risk_free_rate=ratios.annual_risk_free_rate,
        periodic_risk_free_rate=ratios.periodic_risk_free_rate,
        observation_days=count,
        total_net_return=total_return,
        annualized_net_return=annualized,
        daily_return_mean=ratios.periodic_return_mean,
        daily_return_volatility=ratios.periodic_return_volatility,
        sharpe_ratio=ratios.sharpe_ratio,
        sortino_ratio=ratios.sortino_ratio,
        max_drawdown=max_dd,
        max_drawdown_duration_days=dd_duration,
        time_under_water_ratio=time_under_water,
        ulcer_index=ulcer,
        daily_expected_shortfall_5=_expected_shortfall_loss(returns),
        calmar_ratio=calmar,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise EquityDataError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def moving_block_bootstrap_bounds(
    daily_returns: Sequence[float],
    *,
    samples: int = 1000,
    block_length_days: int = 10,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapRiskBoundsV2:
    """Estimate adverse metric bounds while retaining short-range dependence."""

    returns = tuple(_finite_float(value, "daily_return") for value in daily_returns)
    if samples < 100:
        raise EquityDataError("bootstrap samples must be at least 100")
    if not 0.5 < confidence < 1.0:
        raise EquityDataError("confidence must be between 0.5 and 1.0")
    if block_length_days < 1:
        raise EquityDataError("block_length_days must be positive")
    if len(returns) < max(30, block_length_days * 2):
        raise EquityDataError("at least max(30, 2 * block_length_days) daily returns are required")
    if any(value <= -1.0 for value in returns):
        raise EquityDataError("daily returns at or below -100% cannot be bootstrapped")

    rng = random.Random(seed)  # noqa: S311 - reproducible bootstrap, not cryptography
    block = min(block_length_days, len(returns))
    max_start = len(returns) - block
    annual_returns: list[float] = []
    max_drawdowns: list[float] = []
    expected_shortfalls: list[float] = []

    for _ in range(samples):
        sampled: list[float] = []
        while len(sampled) < len(returns):
            start = rng.randint(0, max_start)
            sampled.extend(returns[start : start + block])
        sampled = sampled[: len(returns)]

        annualized = _annualized_return(sampled)
        annual_returns.append(annualized if annualized is not None else -1.0)

        equity = [1.0]
        for value in sampled:
            equity.append(equity[-1] * (1.0 + value))
        max_drawdowns.append(_drawdown_statistics(equity)[0])
        expected_shortfalls.append(_expected_shortfall_loss(sampled) or 0.0)

    tail = 1.0 - confidence
    return BootstrapRiskBoundsV2(
        samples=samples,
        block_length_days=block_length_days,
        confidence=confidence,
        seed=seed,
        annualized_net_return_lcb=_quantile(annual_returns, tail),
        max_drawdown_ucb=_quantile(max_drawdowns, confidence),
        daily_expected_shortfall_5_ucb=_quantile(expected_shortfalls, confidence),
    )
