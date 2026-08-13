"""Production scoring contract for the hardcore six-pair search.

``raw-multipair-score-v3`` deliberately consumes point estimates from six
independent pair backtests.  It does not consume confidence bounds, effective
sample sizes, Sharpe/Sortino ratios, gate results, or any temporal validation
output.  Cross-pair validation is the only generalisation signal in this
contract.

All return-like inputs are decimal ratios (``0.10`` means ten percent).  The
panel end date is inclusive and activity is measured over its calendar-month
buckets.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from statistics import median
from typing import Any


RAW_MULTIPAIR_SCORE_VERSION = "raw-multipair-score-v3"

HARDCORE_DEVELOPMENT_PAIRS = (
    "BTC/USDT",
    "SOL/USDT",
    "XRP/USDT",
)
HARDCORE_VALIDATION_PAIRS = (
    "BNB/USDT",
    "ETH/USDT",
    "PEPE/USDT",
)
HARDCORE_TIMEFRAMES = frozenset({"15m", "1h"})
HARDCORE_PERIOD_START = date(2023, 5, 9)
HARDCORE_PERIOD_END = date(2026, 3, 26)
HARDCORE_FEE_RATE = 0.001
HARDCORE_SLIPPAGE_RATE = 0.0005

DEVELOPMENT_GROUP_WEIGHT = 0.4
VALIDATION_GROUP_WEIGHT = 0.6
WORST_PAIR_WEIGHT = 0.6
GROUP_MEDIAN_WEIGHT = 0.4

# Profit is measured as simple net-return velocity over the immutable panel.
# The former v2 scale used ten percent *total* return, which was almost fully
# saturated by a strategy earning only a few percent per year.  Three percent
# per week is an intentionally ambitious soft reference, not a qualification
# gate: tanh remains continuous above and below it.
TARGET_WEEKLY_NET_RETURN = 0.03
EXPECTANCY_SCALE = 0.005
PROFIT_FACTOR_CAP = 3.0
PROFIT_FACTOR_SCALE = 0.5
PROFIT_FACTOR_FULL_CREDIT_TRADES = 30
# Roughly one thousand trades per pair over this 35-month panel is the desired
# high-frequency daytrading region.  It is deliberately a soft saturation
# point: candidates below it remain valid and receive a smooth gradient.
ACTIVITY_TARGET_TRADES_PER_PAIR = 1000.0
ACTIVITY_TARGET_ACTIVE_MONTH_RATIO = 0.90
HOLDING_MEDIAN_TARGET_HOURS: Mapping[str, float] = {
    "15m": 8.0,
    "1h": 12.0,
}
HOLDING_P90_TARGET_HOURS: Mapping[str, float] = {
    "15m": 24.0,
    "1h": 36.0,
}
MAX_DRAWDOWN_SCALE = 0.20
DRAWDOWN_DURATION_SCALE_DAYS = 120.0
LOSS_STREAK_SCALE = 10.0
OVERTRADING_FREE_TRADES_PER_MONTH = 60.0
OVERTRADING_SCALE_TRADES_PER_MONTH = 60.0

COMPONENT_NAMES = (
    "return_score",
    "expectancy_score",
    "profit_factor_score",
    "activity_score",
    "holding_score",
    "drawdown_risk",
    "drawdown_duration_risk",
    "loss_streak_risk",
    "overtrading_risk",
)

COMPONENT_WEIGHTS: Mapping[str, float] = {
    # Net profit velocity and pairwise activity dominate.  Expectancy and PF
    # remain useful gradients, but cannot make a handful of lucky trades look
    # like a production-quality high-frequency strategy.
    "return_score": 0.40,
    "expectancy_score": 0.10,
    "profit_factor_score": 0.05,
    "activity_score": 0.35,
    "holding_score": 0.05,
    "drawdown_risk": -0.12,
    "drawdown_duration_risk": -0.07,
    "loss_streak_risk": -0.04,
    "overtrading_risk": -0.02,
}

_BENEFIT_COMPONENTS = frozenset(COMPONENT_NAMES[:5])


class RawMultiPairStatus(StrEnum):
    """Candidate-level result of the scoring contract."""

    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class RawMultiPairPanel:
    """Identity of the one supported six-pair production panel."""

    timeframe: str
    development_pairs: tuple[str, ...] = HARDCORE_DEVELOPMENT_PAIRS
    validation_pairs: tuple[str, ...] = HARDCORE_VALIDATION_PAIRS
    period_start: date = HARDCORE_PERIOD_START
    period_end: date = HARDCORE_PERIOD_END
    fee_rate: float = HARDCORE_FEE_RATE
    slippage_rate: float = HARDCORE_SLIPPAGE_RATE
    data_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        if self.timeframe not in HARDCORE_TIMEFRAMES:
            raise ValueError("raw-multipair-score-v3 supports only 15m and 1h timeframes")

        development = _canonical_pairs(self.development_pairs, "development_pairs")
        validation = _canonical_pairs(self.validation_pairs, "validation_pairs")
        if set(development) != set(HARDCORE_DEVELOPMENT_PAIRS):
            raise ValueError("development_pairs do not match the hardcore panel")
        if set(validation) != set(HARDCORE_VALIDATION_PAIRS):
            raise ValueError("validation_pairs do not match the hardcore panel")
        if set(development) & set(validation):
            raise ValueError("development and validation pairs must be disjoint")
        object.__setattr__(self, "development_pairs", development)
        object.__setattr__(self, "validation_pairs", validation)

        if isinstance(self.period_start, datetime) or not isinstance(self.period_start, date):
            raise TypeError("period_start must be a date")
        if isinstance(self.period_end, datetime) or not isinstance(self.period_end, date):
            raise TypeError("period_end must be a date")
        if self.period_start != HARDCORE_PERIOD_START or self.period_end != HARDCORE_PERIOD_END:
            raise ValueError("dates do not match the immutable hardcore panel")

        fee = _finite_number(self.fee_rate, "fee_rate")
        slippage = _finite_number(self.slippage_rate, "slippage_rate")
        if not math.isclose(fee, HARDCORE_FEE_RATE, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("fee_rate must be 0.001")
        if not math.isclose(slippage, HARDCORE_SLIPPAGE_RATE, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("slippage_rate must be 0.0005")

        if self.data_manifest_hash is not None:
            value = self.data_manifest_hash
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("data_manifest_hash must be a lowercase SHA-256 digest")

    @property
    def calendar_months(self) -> int:
        """Number of inclusive UTC calendar-month buckets in the panel."""

        return (
            (self.period_end.year - self.period_start.year) * 12
            + self.period_end.month
            - self.period_start.month
            + 1
        )

    @property
    def calendar_weeks(self) -> float:
        """Inclusive panel duration expressed as seven-day weeks."""

        return ((self.period_end - self.period_start).days + 1) / 7.0

    @property
    def panel_id(self) -> str:
        """Stable identity including the timeframe and scoring semantics."""

        payload = json.dumps(self._identity_descriptor(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"raw_multipair_panel_v3_{digest}"

    def _identity_descriptor(self) -> dict[str, Any]:
        return {
            "score_version": RAW_MULTIPAIR_SCORE_VERSION,
            "timeframe": self.timeframe,
            "development_pairs": list(self.development_pairs),
            "validation_pairs": list(self.validation_pairs),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "data_manifest_hash": self.data_manifest_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe persisted panel contract."""

        return {
            "panel_id": self.panel_id,
            **self._identity_descriptor(),
            "calendar_months": self.calendar_months,
            "data_identity_verified": self.data_manifest_hash is not None,
        }


@dataclass(frozen=True)
class PairScenario:
    """Raw point estimates from one independent pair backtest.

    Trade-dependent fields may be absent only when ``trade_count`` is zero.
    A zero-trade scenario is still valid evidence and receives the worst edge
    and activity components.
    """

    pair: str
    timeframe: str
    success: bool
    trade_count: int
    active_months: int
    period_start: date | datetime | None = None
    period_end: date | datetime | None = None
    net_return: float | None = None
    net_expectancy: float | None = None
    profit_factor: float | None = None
    profit_factor_censored: bool | None = None
    median_holding_hours: float | None = None
    p90_holding_hours: float | None = None
    max_drawdown: float | None = None
    max_drawdown_duration_days: float | None = None
    max_consecutive_losses: int | None = None
    technical_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "timeframe": self.timeframe,
            "success": self.success,
            "trade_count": self.trade_count,
            "active_months": self.active_months,
            "period_start": (
                self.period_start.isoformat()
                if self.period_start is not None
                else None
            ),
            "period_end": (
                self.period_end.isoformat()
                if self.period_end is not None
                else None
            ),
            "net_return": self.net_return,
            "net_expectancy": self.net_expectancy,
            "profit_factor": self.profit_factor,
            "profit_factor_censored": self.profit_factor_censored,
            "median_holding_hours": self.median_holding_hours,
            "p90_holding_hours": self.p90_holding_hours,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "max_consecutive_losses": self.max_consecutive_losses,
            "technical_error": self.technical_error,
        }


@dataclass(frozen=True)
class ComponentVector:
    """The nine bounded components used by the public formula."""

    return_score: float
    expectancy_score: float
    profit_factor_score: float
    activity_score: float
    holding_score: float
    drawdown_risk: float
    drawdown_duration_risk: float
    loss_streak_risk: float
    overtrading_risk: float

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in COMPONENT_NAMES}


@dataclass(frozen=True)
class PairScoreComponents:
    """Auditable normalized values for one pair."""

    pair: str
    group: str
    components: ComponentVector
    effective_profit_factor: float
    trades_per_calendar_month: float
    active_month_ratio: float
    zero_trades: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "group": self.group,
            "components": self.components.to_dict(),
            "effective_profit_factor": self.effective_profit_factor,
            "trades_per_calendar_month": self.trades_per_calendar_month,
            "active_month_ratio": self.active_month_ratio,
            "zero_trades": self.zero_trades,
        }


@dataclass(frozen=True)
class RawMultiPairScore:
    """Candidate score or a fail-closed invalid result."""

    score_version: str
    panel_id: str
    timeframe: str
    status: RawMultiPairStatus
    score: float | None
    reason_code: str
    reason_detail: str | None
    pair_components: tuple[PairScoreComponents, ...] = ()
    aggregate_components: ComponentVector | None = None

    @property
    def is_valid(self) -> bool:
        return self.status is RawMultiPairStatus.VALID

    def pair_component_map(self) -> dict[str, PairScoreComponents]:
        return {item.pair: item for item in self.pair_components}

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe output suitable for outcomes and archives."""

        return {
            "score_version": self.score_version,
            "panel_id": self.panel_id,
            "timeframe": self.timeframe,
            "status": self.status.value,
            "score": self.score,
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "pair_components": {item.pair: item.to_dict() for item in self.pair_components},
            "aggregate_components": (
                self.aggregate_components.to_dict()
                if self.aggregate_components is not None
                else None
            ),
        }


class _EvidenceError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


def score_raw_multipair(
    panel: RawMultiPairPanel,
    scenarios: Iterable[PairScenario],
) -> RawMultiPairScore:
    """Score exactly one independent scenario for every panel pair.

    Evidence failures are represented as ``INVALID`` results so an evolution
    worker cannot accidentally turn missing measurements into a numerical
    fitness.  Invalid results never contain a partial score.
    """

    try:
        supplied = tuple(scenarios)
    except TypeError as exc:
        return _invalid(panel, "INVALID_EVIDENCE_TYPE", str(exc))

    if len(supplied) != 6:
        return _invalid(
            panel,
            "PAIR_SET_MISMATCH",
            f"expected six scenarios, received {len(supplied)}",
        )
    if any(not isinstance(item, PairScenario) for item in supplied):
        return _invalid(
            panel,
            "INVALID_EVIDENCE_TYPE",
            "every scenario must be a PairScenario",
        )

    by_pair: dict[str, PairScenario] = {}
    for scenario in supplied:
        if scenario.pair in by_pair:
            return _invalid(
                panel,
                "DUPLICATE_PAIR",
                f"duplicate scenario for {scenario.pair}",
            )
        by_pair[scenario.pair] = scenario

    expected_pairs = set(panel.development_pairs) | set(panel.validation_pairs)
    if set(by_pair) != expected_pairs:
        missing = sorted(expected_pairs - set(by_pair))
        unexpected = sorted(set(by_pair) - expected_pairs)
        return _invalid(
            panel,
            "PAIR_SET_MISMATCH",
            f"missing={missing}, unexpected={unexpected}",
        )

    components: list[PairScoreComponents] = []
    try:
        for pair in sorted(expected_pairs):
            scenario = by_pair[pair]
            if scenario.timeframe != panel.timeframe:
                raise _EvidenceError(
                    "TIMEFRAME_MISMATCH",
                    f"{pair} has timeframe {scenario.timeframe!r}, expected {panel.timeframe!r}",
                )
            group = "development" if pair in panel.development_pairs else "validation"
            components.append(_score_pair(panel, scenario, group))
    except _EvidenceError as exc:
        return _invalid(panel, exc.reason_code, exc.detail)

    pair_map = {item.pair: item.components for item in components}
    aggregate_values: dict[str, float] = {}
    for name in COMPONENT_NAMES:
        development_values = [getattr(pair_map[pair], name) for pair in panel.development_pairs]
        validation_values = [getattr(pair_map[pair], name) for pair in panel.validation_pairs]
        group_aggregator = _benefit_group if name in _BENEFIT_COMPONENTS else _risk_group
        aggregate_values[name] = DEVELOPMENT_GROUP_WEIGHT * group_aggregator(
            development_values
        ) + VALIDATION_GROUP_WEIGHT * group_aggregator(validation_values)

    aggregate = ComponentVector(**aggregate_values)
    score = 100.0 * sum(
        COMPONENT_WEIGHTS[name] * aggregate_values[name] for name in COMPONENT_NAMES
    )
    if not math.isfinite(score):  # Defensive: normalized components are finite.
        return _invalid(panel, "NONFINITE_SCORE", "score is not finite")

    return RawMultiPairScore(
        score_version=RAW_MULTIPAIR_SCORE_VERSION,
        panel_id=panel.panel_id,
        timeframe=panel.timeframe,
        status=RawMultiPairStatus.VALID,
        score=score,
        reason_code="SCORED",
        reason_detail=None,
        pair_components=tuple(components),
        aggregate_components=aggregate,
    )


def material_improvement_threshold(previous_score: float) -> float:
    """Return the absolute improvement required over ``previous_score``."""

    previous = _finite_number(previous_score, "previous_score")
    return max(0.25, 0.005 * max(1.0, abs(previous)))


def is_material_improvement(
    previous_score: float | None,
    candidate_score: float,
) -> bool:
    """Whether a candidate advances a lane's incumbent.

    A missing incumbent accepts the first finite score, including a negative
    one.  This is what permits useful negative champions to seed later runs.
    """

    candidate = _finite_number(candidate_score, "candidate_score")
    if previous_score is None:
        return True
    previous = _finite_number(previous_score, "previous_score")
    return candidate - previous >= material_improvement_threshold(previous)


def _score_pair(
    panel: RawMultiPairPanel,
    scenario: PairScenario,
    group: str,
) -> PairScoreComponents:
    if not isinstance(scenario.success, bool):
        raise _EvidenceError("INVALID_METRIC", f"{scenario.pair}: success must be boolean")
    if not scenario.success or scenario.technical_error is not None:
        detail = scenario.technical_error or "backtest reported success=false"
        raise _EvidenceError("TECHNICAL_ERROR", f"{scenario.pair}: {detail}")

    trade_count = _nonnegative_integer(scenario.trade_count, f"{scenario.pair}.trade_count")
    active_months = _nonnegative_integer(scenario.active_months, f"{scenario.pair}.active_months")
    if active_months > panel.calendar_months:
        raise _EvidenceError(
            "INVALID_METRIC",
            f"{scenario.pair}.active_months exceeds panel calendar months",
        )

    if trade_count == 0:
        if active_months != 0:
            raise _EvidenceError(
                "INVALID_METRIC",
                f"{scenario.pair}: zero trades require zero active months",
            )
        _validate_optional_zero_trade_metrics(scenario)
        return PairScoreComponents(
            pair=scenario.pair,
            group=group,
            components=ComponentVector(
                return_score=-1.0,
                expectancy_score=-1.0,
                profit_factor_score=-1.0,
                activity_score=-1.0,
                holding_score=0.0,
                drawdown_risk=0.0,
                drawdown_duration_risk=0.0,
                loss_streak_risk=0.0,
                overtrading_risk=0.0,
            ),
            effective_profit_factor=0.0,
            trades_per_calendar_month=0.0,
            active_month_ratio=0.0,
            zero_trades=True,
        )

    if active_months == 0:
        raise _EvidenceError(
            "INVALID_METRIC",
            f"{scenario.pair}: positive trade count requires active months",
        )

    net_return = _required_metric(scenario.pair, "net_return", scenario.net_return)
    net_expectancy = _required_metric(scenario.pair, "net_expectancy", scenario.net_expectancy)
    profit_factor = _required_metric(scenario.pair, "profit_factor", scenario.profit_factor)
    median_holding = _required_metric(
        scenario.pair, "median_holding_hours", scenario.median_holding_hours
    )
    p90_holding = _required_metric(scenario.pair, "p90_holding_hours", scenario.p90_holding_hours)
    max_drawdown = _required_metric(scenario.pair, "max_drawdown", scenario.max_drawdown)
    drawdown_duration = _required_metric(
        scenario.pair,
        "max_drawdown_duration_days",
        scenario.max_drawdown_duration_days,
    )
    loss_streak = _required_integer(
        scenario.pair,
        "max_consecutive_losses",
        scenario.max_consecutive_losses,
    )

    nonnegative = {
        "profit_factor": profit_factor,
        "median_holding_hours": median_holding,
        "p90_holding_hours": p90_holding,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_days": drawdown_duration,
    }
    for name, value in nonnegative.items():
        if value < 0.0:
            raise _EvidenceError("INVALID_METRIC", f"{scenario.pair}.{name} must be non-negative")
    if p90_holding < median_holding:
        raise _EvidenceError(
            "INVALID_METRIC",
            f"{scenario.pair}.p90_holding_hours is below its median",
        )
    if loss_streak < 0 or loss_streak > trade_count:
        raise _EvidenceError(
            "INVALID_METRIC",
            f"{scenario.pair}.max_consecutive_losses is outside [0, trade_count]",
        )
    if not isinstance(scenario.profit_factor_censored, bool):
        raise _EvidenceError(
            "MISSING_METRIC",
            f"{scenario.pair}.profit_factor_censored is required",
        )

    effective_pf = _effective_profit_factor(
        profit_factor,
        censored=scenario.profit_factor_censored,
        trade_count=trade_count,
    )
    trades_per_month = trade_count / panel.calendar_months
    active_month_ratio = active_months / panel.calendar_months
    trade_rate_progress = min(1.0, trade_count / ACTIVITY_TARGET_TRADES_PER_PAIR)
    month_coverage_progress = min(
        1.0,
        active_month_ratio / ACTIVITY_TARGET_ACTIVE_MONTH_RATIO,
    )
    # Geometric blending prevents a burst of trades in a few months from
    # hiding long inactive stretches.  Activity is a smooth shortfall in
    # [-1, 0]: reaching the target removes the penalty but never grants a
    # positive score capable of compensating for losses.
    activity_progress = math.sqrt(trade_rate_progress * month_coverage_progress)
    activity = activity_progress - 1.0
    holding_progress = 0.5 * _threshold_reward(
        median_holding, HOLDING_MEDIAN_TARGET_HOURS[panel.timeframe]
    ) + 0.5 * _threshold_reward(
        p90_holding, HOLDING_P90_TARGET_HOURS[panel.timeframe]
    )
    # Like activity, holding time is a shortfall penalty rather than a free
    # positive reward.  A frequent break-even strategy therefore remains
    # neutral instead of scoring positively merely for closing quickly.
    holding = holding_progress - 1.0
    overtrading_excess = max(0.0, trades_per_month - OVERTRADING_FREE_TRADES_PER_MONTH)

    vector = ComponentVector(
        return_score=math.tanh(
            (net_return / panel.calendar_weeks) / TARGET_WEEKLY_NET_RETURN
        ),
        expectancy_score=math.tanh(net_expectancy / EXPECTANCY_SCALE),
        profit_factor_score=math.tanh((effective_pf - 1.0) / PROFIT_FACTOR_SCALE),
        activity_score=activity,
        holding_score=holding,
        drawdown_risk=math.tanh(max_drawdown / MAX_DRAWDOWN_SCALE),
        drawdown_duration_risk=math.tanh(drawdown_duration / DRAWDOWN_DURATION_SCALE_DAYS),
        loss_streak_risk=math.tanh(loss_streak / LOSS_STREAK_SCALE),
        overtrading_risk=math.tanh(overtrading_excess / OVERTRADING_SCALE_TRADES_PER_MONTH),
    )
    return PairScoreComponents(
        pair=scenario.pair,
        group=group,
        components=vector,
        effective_profit_factor=effective_pf,
        trades_per_calendar_month=trades_per_month,
        active_month_ratio=active_month_ratio,
        zero_trades=False,
    )


def _effective_profit_factor(
    value: float,
    *,
    censored: bool,
    trade_count: int,
) -> float:
    capped = min(PROFIT_FACTOR_CAP, value)
    if capped <= 1.0:
        return capped
    # Apply evidence damping to every profitable PF, not only the censored
    # no-loss case.  A finite PF based on nine trades is still thin evidence.
    evidence = min(1.0, trade_count / PROFIT_FACTOR_FULL_CREDIT_TRADES)
    return 1.0 + (capped - 1.0) * evidence


def _threshold_reward(value: float, target: float) -> float:
    if value <= target:
        return 1.0
    return target / value


def _benefit_group(values: list[float]) -> float:
    return WORST_PAIR_WEIGHT * min(values) + GROUP_MEDIAN_WEIGHT * median(values)


def _risk_group(values: list[float]) -> float:
    return WORST_PAIR_WEIGHT * max(values) + GROUP_MEDIAN_WEIGHT * median(values)


def _canonical_pairs(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list")
    if len(values) != 3 or any(not isinstance(pair, str) or not pair for pair in values):
        raise ValueError(f"{name} must contain exactly three non-empty pair names")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate pairs")
    return tuple(sorted(values))


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required_metric(pair: str, name: str, value: Any) -> float:
    if value is None:
        raise _EvidenceError("MISSING_METRIC", f"{pair}.{name} is required")
    try:
        return _finite_number(value, f"{pair}.{name}")
    except ValueError as exc:
        reason = "NONFINITE_METRIC" if "finite" in str(exc) else "INVALID_METRIC"
        raise _EvidenceError(reason, str(exc)) from exc


def _required_integer(pair: str, name: str, value: Any) -> int:
    if value is None:
        raise _EvidenceError("MISSING_METRIC", f"{pair}.{name} is required")
    return _nonnegative_integer(value, f"{pair}.{name}")


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _EvidenceError("INVALID_METRIC", f"{name} must be an integer")
    if value < 0:
        raise _EvidenceError("INVALID_METRIC", f"{name} must be non-negative")
    return value


def _validate_optional_zero_trade_metrics(scenario: PairScenario) -> None:
    numeric_names = (
        "net_return",
        "net_expectancy",
        "profit_factor",
        "median_holding_hours",
        "p90_holding_hours",
        "max_drawdown",
        "max_drawdown_duration_days",
    )
    for name in numeric_names:
        value = getattr(scenario, name)
        if value is None:
            continue
        try:
            number = _finite_number(value, f"{scenario.pair}.{name}")
        except ValueError as exc:
            reason = "NONFINITE_METRIC" if "finite" in str(exc) else "INVALID_METRIC"
            raise _EvidenceError(reason, str(exc)) from exc
        if number != 0.0:
            raise _EvidenceError(
                "INVALID_METRIC",
                f"{scenario.pair}.{name} must be zero or absent when there are no trades",
            )
    if scenario.max_consecutive_losses not in (None, 0):
        raise _EvidenceError(
            "INVALID_METRIC",
            f"{scenario.pair}.max_consecutive_losses must be zero or absent",
        )
    if scenario.profit_factor_censored not in (None, False):
        raise _EvidenceError(
            "INVALID_METRIC",
            f"{scenario.pair}.profit_factor_censored cannot be true without trades",
        )


def _invalid(
    panel: RawMultiPairPanel,
    reason_code: str,
    detail: str,
) -> RawMultiPairScore:
    return RawMultiPairScore(
        score_version=RAW_MULTIPAIR_SCORE_VERSION,
        panel_id=panel.panel_id,
        timeframe=panel.timeframe,
        status=RawMultiPairStatus.INVALID,
        score=None,
        reason_code=reason_code,
        reason_detail=detail,
    )
