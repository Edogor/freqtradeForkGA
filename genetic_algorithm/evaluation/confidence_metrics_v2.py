"""Conservative, deterministic confidence estimates for ordered trade returns."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean
from typing import Any

from genetic_algorithm.evaluation.equity_metrics_v2 import EquityDataError, _quantile


EXPECTANCY_CONTRACT_VERSION = "net-expectancy-clustered-v1"


class InsufficientTradeEvidenceError(EquityDataError):
    """Raised when valid trade evidence is too sparse for a confidence claim."""


@dataclass(frozen=True)
class ExpectancyConfidenceV2:
    """Mean trade return and its dependence-aware lower confidence bound."""

    mean: float
    lower_confidence_bound: float
    effective_sample_size: float
    observations: int
    samples: int
    block_length_trades: int
    confidence: float
    seed: int


@dataclass(frozen=True)
class ClusteredExpectancyConfidenceV2:
    """Trade- and committed-capital expectancy with clustered uncertainty."""

    expectancy_contract_version: str
    quote_currency: str
    observations: int
    pair_count: int
    temporal_clusters: int
    cluster_days: int
    samples: int
    block_length_clusters: int
    confidence: float
    seed: int
    total_net_profit_abs: float
    total_committed_capital: float
    mean_net_profit_abs_per_trade: float
    mean_trade_return: float
    mean_trade_return_lcb: float
    return_on_committed_capital: float
    return_on_committed_capital_lcb: float
    effective_sample_size: float
    serial_effective_sample_size: float
    capital_effective_sample_size: float
    temporal_effective_sample_size: float
    effective_pair_count: float
    pair_capital_hhi: float
    max_trade_capital_share: float
    max_cluster_capital_share: float


@dataclass(frozen=True)
class _TradeObservation:
    pair: str
    quote_currency: str
    close_timestamp: datetime
    net_profit_abs: float
    committed_capital: float
    net_return: float


def _finite_returns(values: Sequence[float]) -> tuple[float, ...]:
    try:
        returns = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise EquityDataError("trade returns must be numeric") from exc
    if not returns or any(not math.isfinite(value) for value in returns):
        raise EquityDataError("trade returns must be a non-empty finite sequence")
    return returns


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise EquityDataError(f"{field_name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EquityDataError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise EquityDataError(f"{field_name} must be finite")
    return number


def _timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, (str, datetime)):
        raise EquityDataError(f"{field_name} must be an ISO timestamp")
    try:
        timestamp = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        )
    except ValueError as exc:
        raise EquityDataError(f"{field_name} must be an ISO timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise EquityDataError(f"{field_name} must include a timezone")
    return timestamp.astimezone(UTC)


def _quote_currency(pair: str) -> str:
    if "/" not in pair:
        raise EquityDataError(f"trade pair has no quote currency: {pair!r}")
    quote = pair.rsplit("/", 1)[1].split(":")[-1].strip()
    if not quote:
        raise EquityDataError(f"trade pair has no quote currency: {pair!r}")
    return quote


def _trade_observation(trade: Mapping[str, Any], index: int) -> _TradeObservation:
    prefix = f"trades[{index}]"
    pair = trade.get("pair")
    if not isinstance(pair, str) or not pair.strip():
        raise EquityDataError(f"{prefix}.pair must be non-empty")
    pair = pair.strip()

    open_timestamp = _timestamp(trade.get("open_date"), f"{prefix}.open_date")
    close_timestamp = _timestamp(trade.get("close_date"), f"{prefix}.close_date")
    if close_timestamp < open_timestamp:
        raise EquityDataError(f"{prefix}.close_date precedes open_date")
    if not isinstance(trade.get("is_open"), bool) or trade["is_open"]:
        raise EquityDataError(f"{prefix} must be a closed trade")
    if not isinstance(trade.get("is_short"), bool):
        raise EquityDataError(f"{prefix}.is_short must be boolean")

    max_stake = _finite_number(trade.get("max_stake_amount"), f"{prefix}.max_stake_amount")
    fee_open = _finite_number(trade.get("fee_open"), f"{prefix}.fee_open")
    leverage = _finite_number(trade.get("leverage"), f"{prefix}.leverage")
    profit_abs = _finite_number(trade.get("profit_abs"), f"{prefix}.profit_abs")
    reported_return = _finite_number(trade.get("profit_ratio"), f"{prefix}.profit_ratio")
    if max_stake <= 0:
        raise EquityDataError(f"{prefix}.max_stake_amount must be positive")
    if not 0.0 <= fee_open < 1.0:
        raise EquityDataError(f"{prefix}.fee_open must be in [0, 1)")
    if leverage <= 0:
        raise EquityDataError(f"{prefix}.leverage must be positive")

    fee_factor = 1.0 - fee_open if trade["is_short"] else 1.0 + fee_open
    committed_capital = max_stake * fee_factor
    measured_return = profit_abs / committed_capital
    tolerance = max(1e-8, abs(reported_return) * 1e-6)
    if not math.isclose(measured_return, reported_return, rel_tol=1e-6, abs_tol=tolerance):
        raise EquityDataError(
            f"{prefix}.profit_ratio does not reconcile with net profit and committed capital"
        )

    funding_fees = trade.get("funding_fees")
    if funding_fees is not None:
        _finite_number(funding_fees, f"{prefix}.funding_fees")

    return _TradeObservation(
        pair=pair,
        quote_currency=_quote_currency(pair),
        close_timestamp=close_timestamp,
        net_profit_abs=profit_abs,
        committed_capital=committed_capital,
        net_return=measured_return,
    )


def _kish_effective_sample_size(weights: Sequence[float]) -> float:
    total = sum(weights)
    squared = sum(value * value for value in weights)
    if total <= 0 or squared <= 0:
        raise EquityDataError("positive capital weights are required")
    return total * total / squared


def _cluster_key(timestamp: datetime, cluster_days: int) -> int:
    return timestamp.date().toordinal() // cluster_days


def effective_sample_size(values: Sequence[float]) -> float:
    """Estimate N_eff from the initial positive autocorrelation sequence.

    Positive serial dependence reduces evidence; negative correlations are not
    allowed to inflate the result above the actual trade count.  Near-constant
    samples fail closed because they cannot support a dependence estimate.
    """

    returns = _finite_returns(values)
    count = len(returns)
    if count < 2:
        raise InsufficientTradeEvidenceError(
            "at least two trade returns are required for N_eff"
        )
    mean = fmean(returns)
    centered = [value - mean for value in returns]
    denominator = sum(value * value for value in centered)
    if denominator <= 1e-24:
        raise InsufficientTradeEvidenceError(
            "near-constant trade returns cannot establish N_eff"
        )

    positive_sum = 0.0
    # Initial-positive-sequence truncation avoids unstable long-lag noise.
    for lag in range(1, count):
        rho = sum(centered[i] * centered[i + lag] for i in range(count - lag)) / denominator
        if rho <= 0:
            break
        positive_sum += rho
    estimate = count / (1.0 + 2.0 * positive_sum)
    return min(float(count), max(1.0, estimate))


def clustered_trade_expectancy_lcb(  # noqa: C901
    trades: Sequence[Mapping[str, Any]],
    *,
    samples: int = 1000,
    cluster_days: int = 1,
    block_length_clusters: int = 5,
    confidence: float = 0.95,
    seed: int = 42,
) -> ClusteredExpectancyConfidenceV2:
    """Estimate net expectancy while retaining capital and time clustering.

    ``profit_abs`` is Freqtrade's after-fee/funding quote-currency PnL.
    Committed capital is maximum margin/stake plus the entry-fee convention
    used by Freqtrade's reported ``profit_ratio``. Trades closing in the same
    fixed UTC bucket remain together during the moving-block bootstrap, so
    contemporaneous cross-pair observations are never resampled as independent.
    """

    if samples < 100:
        raise EquityDataError("bootstrap samples must be at least 100")
    if (
        not isinstance(cluster_days, int)
        or isinstance(cluster_days, bool)
        or cluster_days < 1
    ):
        raise EquityDataError("cluster_days must be a positive integer")
    if (
        not isinstance(block_length_clusters, int)
        or isinstance(block_length_clusters, bool)
        or block_length_clusters < 1
    ):
        raise EquityDataError("block_length_clusters must be a positive integer")
    if not 0.5 < confidence < 1.0:
        raise EquityDataError("confidence must be between 0.5 and 1.0")
    if not isinstance(trades, Sequence) or isinstance(trades, (str, bytes)):
        raise EquityDataError("trades must be a sequence of mappings")

    observations: list[_TradeObservation] = []
    for index, trade in enumerate(trades):
        if not isinstance(trade, Mapping):
            raise EquityDataError(f"trades[{index}] must be a mapping")
        observations.append(_trade_observation(trade, index))
    observations.sort(
        key=lambda item: (
            item.close_timestamp,
            item.pair,
            item.net_profit_abs,
            item.committed_capital,
        )
    )
    if len(observations) < 20:
        raise InsufficientTradeEvidenceError(
            "at least 20 reconciled trades are required"
        )

    quote_currencies = {item.quote_currency for item in observations}
    if len(quote_currencies) != 1:
        raise EquityDataError("trade profits use multiple quote currencies")
    quote_currency = next(iter(quote_currencies))

    by_cluster: dict[int, list[_TradeObservation]] = defaultdict(list)
    for observation in observations:
        by_cluster[_cluster_key(observation.close_timestamp, cluster_days)].append(
            observation
        )
    cluster_records = [by_cluster[key] for key in sorted(by_cluster)]
    minimum_clusters = max(10, block_length_clusters * 2)
    if len(cluster_records) < minimum_clusters:
        raise InsufficientTradeEvidenceError(
            f"at least {minimum_clusters} temporal trade clusters are required"
        )

    trade_capitals = [item.committed_capital for item in observations]
    trade_returns = [item.net_return for item in observations]
    cluster_capitals = [
        sum(item.committed_capital for item in records) for records in cluster_records
    ]
    cluster_returns = [
        sum(item.net_profit_abs for item in records) / capital
        for records, capital in zip(cluster_records, cluster_capitals, strict=True)
    ]
    serial_n_eff = effective_sample_size(cluster_returns)
    capital_n_eff = _kish_effective_sample_size(trade_capitals)
    temporal_n_eff = _kish_effective_sample_size(cluster_capitals)
    combined_n_eff = min(serial_n_eff, capital_n_eff, temporal_n_eff)

    total_capital = sum(trade_capitals)
    total_profit = sum(item.net_profit_abs for item in observations)
    pair_capital: dict[str, float] = defaultdict(float)
    for item in observations:
        pair_capital[item.pair] += item.committed_capital
    pair_n_eff = _kish_effective_sample_size(tuple(pair_capital.values()))
    pair_hhi = sum((capital / total_capital) ** 2 for capital in pair_capital.values())

    block = min(block_length_clusters, len(cluster_records))
    max_start = len(cluster_records) - block
    rng = random.Random(seed)  # noqa: S311 - deterministic bootstrap, not cryptography
    trade_means: list[float] = []
    capital_returns: list[float] = []
    for _ in range(samples):
        sampled_clusters: list[list[_TradeObservation]] = []
        while len(sampled_clusters) < len(cluster_records):
            start = rng.randint(0, max_start)
            sampled_clusters.extend(cluster_records[start : start + block])
        sampled = [
            item
            for records in sampled_clusters[: len(cluster_records)]
            for item in records
        ]
        sampled_capital = sum(item.committed_capital for item in sampled)
        trade_means.append(fmean(item.net_return for item in sampled))
        capital_returns.append(
            sum(item.net_profit_abs for item in sampled) / sampled_capital
        )

    return ClusteredExpectancyConfidenceV2(
        expectancy_contract_version=EXPECTANCY_CONTRACT_VERSION,
        quote_currency=quote_currency,
        observations=len(observations),
        pair_count=len(pair_capital),
        temporal_clusters=len(cluster_records),
        cluster_days=cluster_days,
        samples=samples,
        block_length_clusters=block_length_clusters,
        confidence=confidence,
        seed=seed,
        total_net_profit_abs=total_profit,
        total_committed_capital=total_capital,
        mean_net_profit_abs_per_trade=total_profit / len(observations),
        mean_trade_return=fmean(trade_returns),
        mean_trade_return_lcb=_quantile(trade_means, 1.0 - confidence),
        return_on_committed_capital=total_profit / total_capital,
        return_on_committed_capital_lcb=_quantile(
            capital_returns,
            1.0 - confidence,
        ),
        effective_sample_size=combined_n_eff,
        serial_effective_sample_size=serial_n_eff,
        capital_effective_sample_size=capital_n_eff,
        temporal_effective_sample_size=temporal_n_eff,
        effective_pair_count=pair_n_eff,
        pair_capital_hhi=pair_hhi,
        max_trade_capital_share=max(trade_capitals) / total_capital,
        max_cluster_capital_share=max(cluster_capitals) / total_capital,
    )


def moving_block_expectancy_lcb(
    trade_returns: Sequence[float],
    *,
    samples: int = 1000,
    block_length_trades: int = 5,
    confidence: float = 0.95,
    seed: int = 42,
) -> ExpectancyConfidenceV2:
    """Bootstrap mean expectancy while retaining local trade-order dependence."""

    returns = _finite_returns(trade_returns)
    if samples < 100:
        raise EquityDataError("bootstrap samples must be at least 100")
    if not 0.5 < confidence < 1.0:
        raise EquityDataError("confidence must be between 0.5 and 1.0")
    if block_length_trades < 1:
        raise EquityDataError("block_length_trades must be positive")
    if len(returns) < max(20, block_length_trades * 2):
        raise InsufficientTradeEvidenceError(
            "at least max(20, 2 * block length) trades are required"
        )

    n_eff = effective_sample_size(returns)
    block = min(block_length_trades, len(returns))
    max_start = len(returns) - block
    rng = random.Random(seed)  # noqa: S311 - deterministic bootstrap, not cryptography
    means: list[float] = []
    for _ in range(samples):
        sampled: list[float] = []
        while len(sampled) < len(returns):
            start = rng.randint(0, max_start)
            sampled.extend(returns[start : start + block])
        means.append(fmean(sampled[: len(returns)]))

    return ExpectancyConfidenceV2(
        mean=fmean(returns),
        lower_confidence_bound=_quantile(means, 1.0 - confidence),
        effective_sample_size=n_eff,
        observations=len(returns),
        samples=samples,
        block_length_trades=block_length_trades,
        confidence=confidence,
        seed=seed,
    )
