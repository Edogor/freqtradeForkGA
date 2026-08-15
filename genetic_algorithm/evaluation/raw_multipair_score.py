"""Production scoring contract for the hardcore six-pair search.

``raw-multipair-score-v4`` deliberately consumes point estimates from six
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


RAW_MULTIPAIR_SCORE_VERSION = "raw-multipair-score-v4"

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

@dataclass(frozen=True)
class RawMultiPairPolicyV4:
    """Hashable continuous policy; no field is a qualification gate."""

    annual_return_scale: float = 0.20
    expectancy_scale: float = 0.005
    profit_factor_cap: float = 3.0
    profit_factor_scale: float = 0.5
    profit_factor_full_credit_trades: int = 30
    activity_target_trades_per_month_15m: float = 17.0
    activity_target_trades_per_month_1h: float = 10.0
    activity_target_active_month_ratio: float = 0.75
    holding_median_target_hours_15m: float = 8.0
    holding_median_target_hours_1h: float = 12.0
    holding_p90_target_hours_15m: float = 24.0
    holding_p90_target_hours_1h: float = 36.0
    max_drawdown_scale: float = 0.20
    loss_streak_scale: float = 10.0
    overtrading_free_trades_per_month: float = 60.0
    overtrading_scale_trades_per_month: float = 60.0
    development_group_weight: float = 0.40
    validation_group_weight: float = 0.60
    worst_pair_weight: float = 0.60
    group_median_weight: float = 0.40
    edge_return_weight: float = 0.50
    edge_expectancy_weight: float = 0.30
    edge_profit_factor_weight: float = 0.20
    score_edge_weight: float = 0.60
    score_productive_frequency_weight: float = 0.30
    score_holding_weight: float = 0.02
    score_drawdown_weight: float = -0.040
    score_drawdown_duration_weight: float = -0.020
    score_loss_streak_weight: float = -0.015
    score_overtrading_weight: float = -0.005

    def __post_init__(self) -> None:
        values = self.to_dict()
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values.values()
        ):
            raise ValueError("raw multipair policy values must be finite numbers")
        positive = (
            "annual_return_scale",
            "expectancy_scale",
            "profit_factor_cap",
            "profit_factor_scale",
            "profit_factor_full_credit_trades",
            "activity_target_trades_per_month_15m",
            "activity_target_trades_per_month_1h",
            "activity_target_active_month_ratio",
            "holding_median_target_hours_15m",
            "holding_median_target_hours_1h",
            "holding_p90_target_hours_15m",
            "holding_p90_target_hours_1h",
            "max_drawdown_scale",
            "loss_streak_scale",
            "overtrading_scale_trades_per_month",
        )
        if any(float(values[name]) <= 0.0 for name in positive):
            raise ValueError("raw multipair policy scales and targets must be positive")
        if not 0.0 < self.activity_target_active_month_ratio <= 1.0:
            raise ValueError("activity month ratio must be in (0, 1]")
        if self.overtrading_free_trades_per_month < 0.0:
            raise ValueError("overtrading free rate must be non-negative")
        if any(
            value < 0.0
            for value in (
                self.development_group_weight,
                self.validation_group_weight,
                self.worst_pair_weight,
                self.group_median_weight,
            )
        ):
            raise ValueError("raw multipair aggregation weights must be non-negative")
        if not math.isclose(
            self.development_group_weight + self.validation_group_weight,
            1.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.worst_pair_weight + self.group_median_weight,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError("raw multipair aggregation weights must sum to one")
        if not math.isclose(
            self.edge_return_weight
            + self.edge_expectancy_weight
            + self.edge_profit_factor_weight,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError("edge weights must sum to one")
        if not math.isclose(
            abs(self.score_drawdown_weight)
            + abs(self.score_drawdown_duration_weight)
            + abs(self.score_loss_streak_weight)
            + abs(self.score_overtrading_weight),
            0.08,
            abs_tol=1e-12,
        ):
            raise ValueError("v4 risk weights must total eight percent")
        if any(
            value > 0.0
            for value in (
                self.score_drawdown_weight,
                self.score_drawdown_duration_weight,
                self.score_loss_streak_weight,
                self.score_overtrading_weight,
            )
        ):
            raise ValueError("risk score weights must be non-positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> RawMultiPairPolicyV4:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("raw multipair policy must be a mapping")
        known = set(cls.__dataclass_fields__)
        unexpected = set(value) - known
        if unexpected:
            raise ValueError(f"unknown raw multipair policy fields: {sorted(unexpected)}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def policy_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def activity_target(self, timeframe: str) -> float:
        return (
            self.activity_target_trades_per_month_15m
            if timeframe == "15m"
            else self.activity_target_trades_per_month_1h
        )

    def holding_targets(self, timeframe: str) -> tuple[float, float]:
        if timeframe == "15m":
            return (
                self.holding_median_target_hours_15m,
                self.holding_p90_target_hours_15m,
            )
        return (
            self.holding_median_target_hours_1h,
            self.holding_p90_target_hours_1h,
        )

COMPONENT_NAMES = (
    "return_score",
    "expectancy_score",
    "profit_factor_score",
    "activity_score",
    "edge_score",
    "productive_frequency_score",
    "holding_score",
    "drawdown_risk",
    "drawdown_duration_risk",
    "loss_streak_risk",
    "overtrading_risk",
)

_BENEFIT_COMPONENTS = frozenset(COMPONENT_NAMES[:7])


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
    policy: RawMultiPairPolicyV4 = RawMultiPairPolicyV4()

    def __post_init__(self) -> None:  # noqa: C901 - one fail-closed panel contract
        if self.timeframe not in HARDCORE_TIMEFRAMES:
            raise ValueError("raw-multipair-score-v4 supports only 15m and 1h timeframes")
        if not isinstance(self.policy, RawMultiPairPolicyV4):
            raise TypeError("policy must be a RawMultiPairPolicyV4")

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
    def calendar_years(self) -> float:
        return ((self.period_end - self.period_start).days + 1) / 365.2425

    @property
    def calendar_days(self) -> int:
        return (self.period_end - self.period_start).days + 1

    @property
    def panel_id(self) -> str:
        """Stable identity including the timeframe and scoring semantics."""

        payload = json.dumps(self._identity_descriptor(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"raw_multipair_panel_v4_{digest}"

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
            "policy_hash": self.policy.policy_hash,
            "policy": self.policy.to_dict(),
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
    """The eleven bounded components used by the public formula."""

    return_score: float
    expectancy_score: float
    profit_factor_score: float
    activity_score: float
    edge_score: float
    productive_frequency_score: float
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
    policy_hash: str
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
            "policy_hash": self.policy_hash,
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
        aggregate_values[name] = panel.policy.development_group_weight * group_aggregator(
            development_values,
            policy=panel.policy,
        ) + panel.policy.validation_group_weight * group_aggregator(
            validation_values,
            policy=panel.policy,
        )

    aggregate = ComponentVector(**aggregate_values)
    policy = panel.policy
    score = 100.0 * (
        policy.score_edge_weight * aggregate_values["edge_score"]
        + policy.score_productive_frequency_weight
        * aggregate_values["productive_frequency_score"]
        + policy.score_holding_weight * aggregate_values["holding_score"]
        + policy.score_drawdown_weight * aggregate_values["drawdown_risk"]
        + policy.score_drawdown_duration_weight
        * aggregate_values["drawdown_duration_risk"]
        + policy.score_loss_streak_weight * aggregate_values["loss_streak_risk"]
        + policy.score_overtrading_weight * aggregate_values["overtrading_risk"]
    )
    if not math.isfinite(score):  # Defensive: normalized components are finite.
        return _invalid(panel, "NONFINITE_SCORE", "score is not finite")

    return RawMultiPairScore(
        score_version=RAW_MULTIPAIR_SCORE_VERSION,
        policy_hash=panel.policy.policy_hash,
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
                activity_score=0.0,
                edge_score=-1.0,
                productive_frequency_score=0.0,
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

    policy = panel.policy
    effective_pf = _effective_profit_factor(
        profit_factor,
        censored=scenario.profit_factor_censored,
        trade_count=trade_count,
        policy=policy,
    )
    trades_per_month = trade_count / panel.calendar_months
    active_month_ratio = active_months / panel.calendar_months
    trade_rate_progress = _soft_saturating_progress(
        trades_per_month / policy.activity_target(panel.timeframe)
    )
    month_coverage_progress = _soft_saturating_progress(
        active_month_ratio / policy.activity_target_active_month_ratio
    )
    # Geometric blending prevents a burst of trades in a few months from
    # hiding long inactive stretches.  The smoothstep has zero slope where it
    # reaches saturation, avoiding a selection cliff at the 17/10 references.
    # Activity itself never grants edge: only Q*A enters productive frequency.
    activity = math.sqrt(trade_rate_progress * month_coverage_progress)
    holding_median_target, holding_p90_target = policy.holding_targets(
        panel.timeframe
    )
    holding_progress = 0.5 * _threshold_reward(
        median_holding, holding_median_target
    ) + 0.5 * _threshold_reward(
        p90_holding, holding_p90_target
    )
    # Like activity, holding time is a shortfall penalty rather than a free
    # positive reward.  A frequent break-even strategy therefore remains
    # neutral instead of scoring positively merely for closing quickly.
    holding = holding_progress - 1.0
    overtrading_excess = max(
        0.0,
        trades_per_month - policy.overtrading_free_trades_per_month,
    )

    return_score = math.tanh(
        (net_return / panel.calendar_years) / policy.annual_return_scale
    )
    expectancy_score = math.tanh(net_expectancy / policy.expectancy_scale)
    profit_factor_score = math.tanh(
        (effective_pf - 1.0) / policy.profit_factor_scale
    )
    edge_score = (
        policy.edge_return_weight * return_score
        + policy.edge_expectancy_weight * expectancy_score
        + policy.edge_profit_factor_weight * profit_factor_score
    )
    productive_frequency_score = edge_score * activity

    vector = ComponentVector(
        return_score=return_score,
        expectancy_score=expectancy_score,
        profit_factor_score=profit_factor_score,
        activity_score=activity,
        edge_score=edge_score,
        productive_frequency_score=productive_frequency_score,
        holding_score=holding,
        drawdown_risk=math.tanh(max_drawdown / policy.max_drawdown_scale),
        drawdown_duration_risk=min(1.0, drawdown_duration / panel.calendar_days),
        loss_streak_risk=math.tanh(loss_streak / policy.loss_streak_scale),
        overtrading_risk=math.tanh(
            overtrading_excess / policy.overtrading_scale_trades_per_month
        ),
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
    policy: RawMultiPairPolicyV4,
) -> float:
    capped = min(policy.profit_factor_cap, value)
    if capped <= 1.0:
        return capped
    # Apply evidence damping to every profitable PF, not only the censored
    # no-loss case.  A finite PF based on nine trades is still thin evidence.
    evidence = min(1.0, trade_count / policy.profit_factor_full_credit_trades)
    return 1.0 + (capped - 1.0) * evidence


def _threshold_reward(value: float, target: float) -> float:
    if value <= target:
        return 1.0
    return target / value


def _soft_saturating_progress(ratio: float) -> float:
    """C1-continuous progress from zero to a softly saturated reference."""

    bounded = max(0.0, min(1.0, ratio))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def _benefit_group(
    values: list[float], *, policy: RawMultiPairPolicyV4
) -> float:
    return (
        policy.worst_pair_weight * min(values)
        + policy.group_median_weight * median(values)
    )


def _risk_group(values: list[float], *, policy: RawMultiPairPolicyV4) -> float:
    return (
        policy.worst_pair_weight * max(values)
        + policy.group_median_weight * median(values)
    )


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
        policy_hash=panel.policy.policy_hash,
        panel_id=panel.panel_id,
        timeframe=panel.timeframe,
        status=RawMultiPairStatus.INVALID,
        score=None,
        reason_code=reason_code,
        reason_detail=detail,
    )
