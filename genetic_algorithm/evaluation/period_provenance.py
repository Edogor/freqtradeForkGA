"""Fail-closed validation for measured backtest period provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any


MISSING_PERIOD_EVIDENCE = "MISSING_PERIOD_EVIDENCE"
PERIOD_COVERAGE_MISMATCH = "PERIOD_COVERAGE_MISMATCH"


@dataclass(frozen=True)
class ExactPeriodEvidence:
    """Result of comparing a measured period with an immutable declaration."""

    period_start: date | None
    period_end: date | None
    measured_start: datetime | None = None
    measured_end: datetime | None = None
    error_code: str | None = None
    error_detail: str | None = None

    @property
    def valid(self) -> bool:
        return self.error_code is None


@dataclass(frozen=True)
class _ParsedBoundary:
    day: date
    timestamp: datetime | None


def _as_boundary(value: Any) -> _ParsedBoundary | None:
    if isinstance(value, datetime):
        timestamp = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return _ParsedBoundary(timestamp.date(), timestamp)
    if isinstance(value, date):
        return _ParsedBoundary(value, None)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return _ParsedBoundary(date.fromisoformat(text), None)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        timestamp = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
        return _ParsedBoundary(timestamp.date(), timestamp)


def _timeframe_delta(timeframe: str) -> timedelta:
    if timeframe == "15m":
        return timedelta(minutes=15)
    if timeframe == "1h":
        return timedelta(hours=1)
    if timeframe == "4h":
        return timedelta(hours=4)
    raise ValueError(f"unsupported exact-panel timeframe: {timeframe!r}")


def validate_exact_period_evidence(
    *,
    expected_start: date,
    expected_end: date,
    observed_start: Any,
    observed_end: Any,
    evidence_label: str,
    timeframe: str | None = None,
) -> ExactPeriodEvidence:
    """Require parseable measured dates that exactly match the declaration.

    A hardcore timeframe additionally requires exact UTC candle boundaries.
    Date-only values are insufficient for that proof because they cannot
    distinguish a complete first/last day from warmup-trimmed data.
    """

    parsed_start = _as_boundary(observed_start)
    parsed_end = _as_boundary(observed_end)
    if parsed_start is None or parsed_end is None:
        return ExactPeriodEvidence(
            period_start=parsed_start.day if parsed_start is not None else None,
            period_end=parsed_end.day if parsed_end is not None else None,
            error_code=MISSING_PERIOD_EVIDENCE,
            error_detail=(
                f"{evidence_label} has no parseable measured start/end period"
            ),
        )
    expected = (expected_start, expected_end)
    measured = (parsed_start.day, parsed_end.day)
    if measured != expected:
        return ExactPeriodEvidence(
            period_start=parsed_start.day,
            period_end=parsed_end.day,
            measured_start=parsed_start.timestamp,
            measured_end=parsed_end.timestamp,
            error_code=PERIOD_COVERAGE_MISMATCH,
            error_detail=(
                f"{evidence_label} measured={measured}, expected={expected}"
            ),
        )
    if timeframe is not None:
        if parsed_start.timestamp is None or parsed_end.timestamp is None:
            return ExactPeriodEvidence(
                period_start=parsed_start.day,
                period_end=parsed_end.day,
                measured_start=parsed_start.timestamp,
                measured_end=parsed_end.timestamp,
                error_code=MISSING_PERIOD_EVIDENCE,
                error_detail=(
                    f"{evidence_label} requires exact {timeframe} UTC "
                    "boundary timestamps"
                ),
            )
        candle_delta = _timeframe_delta(timeframe)
        expected_start_at = datetime.combine(expected_start, time.min, tzinfo=UTC)
        expected_end_at = (
            datetime.combine(expected_end + timedelta(days=1), time.min, tzinfo=UTC)
            - candle_delta
        )
        observed = (parsed_start.timestamp, parsed_end.timestamp)
        expected_timestamps = (expected_start_at, expected_end_at)
        if observed != expected_timestamps:
            return ExactPeriodEvidence(
                period_start=parsed_start.day,
                period_end=parsed_end.day,
                measured_start=parsed_start.timestamp,
                measured_end=parsed_end.timestamp,
                error_code=PERIOD_COVERAGE_MISMATCH,
                error_detail=(
                    f"{evidence_label} measured={observed}, "
                    f"expected={expected_timestamps}"
                ),
            )
    return ExactPeriodEvidence(
        period_start=parsed_start.day,
        period_end=parsed_end.day,
        measured_start=parsed_start.timestamp,
        measured_end=parsed_end.timestamp,
    )


def validate_exact_period_timestamps_v5(
    *,
    expected_start: datetime,
    expected_end: datetime,
    observed_start: Any,
    observed_end: Any,
    evidence_label: str,
    timeframe: str,
) -> ExactPeriodEvidence:
    """V5 exact-candle provenance with non-midnight 4h panel support.

    V4 intentionally accepted date declarations and derived its first/last
    candle.  V5 binds timestamps directly because the honest 4h PEPE panel
    starts at 04:00 UTC after its required warmup history.
    """

    if expected_start.tzinfo is None or expected_end.tzinfo is None:
        raise ValueError("expected V5 period timestamps must be timezone aware")
    _timeframe_delta(timeframe)  # Reject unsupported lanes consistently.
    start = _as_boundary(observed_start)
    end = _as_boundary(observed_end)
    if start is None or end is None or start.timestamp is None or end.timestamp is None:
        return ExactPeriodEvidence(
            period_start=start.day if start is not None else None,
            period_end=end.day if end is not None else None,
            measured_start=start.timestamp if start is not None else None,
            measured_end=end.timestamp if end is not None else None,
            error_code=MISSING_PERIOD_EVIDENCE,
            error_detail=f"{evidence_label} requires exact {timeframe} UTC timestamps",
        )
    expected = (expected_start.astimezone(UTC), expected_end.astimezone(UTC))
    observed = (start.timestamp, end.timestamp)
    if observed != expected:
        return ExactPeriodEvidence(
            period_start=start.day,
            period_end=end.day,
            measured_start=start.timestamp,
            measured_end=end.timestamp,
            error_code=PERIOD_COVERAGE_MISMATCH,
            error_detail=f"{evidence_label} measured={observed}, expected={expected}",
        )
    return ExactPeriodEvidence(
        period_start=start.day,
        period_end=end.day,
        measured_start=start.timestamp,
        measured_end=end.timestamp,
    )
