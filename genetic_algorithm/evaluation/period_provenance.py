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
