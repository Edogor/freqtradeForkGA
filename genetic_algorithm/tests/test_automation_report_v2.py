"""Sanitized, deterministic campaign report regressions."""

import pytest

from genetic_algorithm.orchestration.automation_report_v2 import (
    build_campaign_summary_v2,
)


def _report(wave: str, when: str, phenotype: str, alignment: float):
    return {
        "root_wave_id": "wave-root",
        "wave_id": wave,
        "analysis_hash": wave.removeprefix("wave-").ljust(64, "0"),
        "analyzed_at": when,
        "controller_outcome": "CONTINUE_SEARCH",
        "controller_reason_codes": ["PLAN_COMPLETE"],
        "analysis_reason_codes": ["NO_ELIGIBLE_CANDIDATES"],
        "candidate_analyses": [
            {
                "experiment_id": f"experiment-{wave}",
                "phenotype_hash": phenotype,
                "eligibility_status": "INELIGIBLE",
                "median_gate_alignment_score": alignment,
                "median_robust_score": -0.1,
                "min_scenario_net_return": 0.01,
                "worst_max_drawdown_ucb": 0.2,
                "min_trades_per_active_month": 3.0,
                "max_drawdown_duration_days": 200,
                "failed_gate_reason_codes": ["TRADE_RATE_TOO_LOW"],
            }
        ],
        "attempts": [
            {
                "result_status": "SUCCEEDED",
                "duration_seconds": 120,
                "artifact_bytes": 1024,
                "seeds": [1],
            }
        ],
    }


def test_campaign_summary_is_order_independent_and_warns_on_repeated_phenotype():
    first = _report("wave-a", "2026-07-26T00:00:00Z", "a" * 64, 0.7)
    second = _report("wave-b", "2026-07-26T01:00:00Z", "a" * 64, 0.8)

    forward = build_campaign_summary_v2([first, second])
    reversed_input = build_campaign_summary_v2([second, first])

    assert forward == reversed_input
    assert forward["wave_count"] == 2
    assert forward["attempt_count"] == 2
    assert forward["successful_attempt_count"] == 2
    assert forward["duration_seconds"] == 240
    assert forward["artifact_bytes"] == 2048
    assert forward["unique_phenotype_count"] == 1
    assert forward["repeated_phenotype_observation_count"] == 1
    assert forward["waves"][1]["repeated_phenotype_hashes"] == ["a" * 64]
    assert (
        forward["waves"][1]["best_candidate"]["median_gate_alignment_score"]
        == 0.8
    )


def test_campaign_summary_contains_no_strategy_or_trade_payloads():
    summary = build_campaign_summary_v2(
        [_report("wave-a", "2026-07-26T00:00:00Z", "a" * 64, 0.7)]
    )

    serialized = str(summary)
    assert "strategy_code" not in serialized
    assert "genome" not in serialized
    assert "trades" not in summary["waves"][0]["best_candidate"]


def test_campaign_summary_rejects_mixed_root_lineages():
    first = _report("wave-a", "2026-07-26T00:00:00Z", "a" * 64, 0.7)
    second = _report("wave-b", "2026-07-26T01:00:00Z", "b" * 64, 0.8)
    second["root_wave_id"] = "another-root"

    with pytest.raises(ValueError, match="cannot mix root wave lineages"):
        build_campaign_summary_v2([first, second])
