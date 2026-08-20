from datetime import UTC, datetime

from genetic_algorithm.evaluation.period_provenance import (
    PERIOD_COVERAGE_MISMATCH,
    validate_exact_period_timestamps_v5,
)
from genetic_algorithm.evaluation.raw_multipair_score_v5 import V5_PANEL_BOUNDS


def test_4h_v5_requires_the_honest_post_warmup_timestamp():
    start, end = V5_PANEL_BOUNDS["4h"]
    valid = validate_exact_period_timestamps_v5(
        expected_start=start,
        expected_end=end,
        observed_start=start,
        observed_end=end,
        evidence_label="PEPE/USDT",
        timeframe="4h",
    )
    assert valid.valid

    shifted = validate_exact_period_timestamps_v5(
        expected_start=start,
        expected_end=end,
        observed_start=datetime(2023, 5, 18, 8, tzinfo=UTC),
        observed_end=end,
        evidence_label="PEPE/USDT",
        timeframe="4h",
    )
    assert shifted.error_code == PERIOD_COVERAGE_MISMATCH
