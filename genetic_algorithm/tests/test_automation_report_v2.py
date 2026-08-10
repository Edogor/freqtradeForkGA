"""Sanitized, deterministic campaign report regressions."""

import pytest

from genetic_algorithm.orchestration.automation_report_v2 import (
    _candidate_gate_distances,
    _campaign_markdown,
    _engine_seed_evidence,
    _markdown,
    _planner_selected_input_parents,
    _signed_threshold_margin,
    build_campaign_summary_v2,
)
from genetic_algorithm.orchestration.evolution_worker_v2 import derive_search_seed


def _report(
    wave: str,
    when: str,
    phenotype: str,
    alignment: float,
    *,
    parent_wave: str = "parent-wave",
    parent_experiment: str = "parent-experiment",
    parent_phenotype: str = "p" * 64,
):
    experiment_id = f"experiment-{wave}"
    return {
        "root_wave_id": "wave-root",
        "wave_id": wave,
        "analysis_hash": wave.removeprefix("wave-").ljust(64, "0"),
        "analyzed_at": when,
        "controller_outcome": "CONTINUE_SEARCH",
        "controller_reason_codes": ["PLAN_COMPLETE"],
        "analysis_reason_codes": ["NO_ELIGIBLE_CANDIDATES"],
        "experiments": [
            {
                "experiment_id": experiment_id,
                "arm_type": "EXPLORE",
            }
        ],
        "plan_summary": {
            "parent_wave_id": parent_wave,
            "plan_hash": "f" * 64,
            "plan_reason_codes": ["PLAN_COMPLETE"],
            "experiments": [
                {
                    "experiment_id": experiment_id,
                    "arm_id": "explore",
                    "arm_type": "EXPLORE",
                    "seeds": [1],
                    "factor_delta": {"mutation_rate": 0.05},
                    "source_mode": "SELECTED_CANDIDATE",
                    "parent_experiment_id": parent_experiment,
                    "parent_config_hash": "c" * 64,
                    "selected_candidate_key": "candidate-key",
                    "phenotype_hash": parent_phenotype,
                }
            ],
        },
        "candidate_analyses": [
            {
                "experiment_id": experiment_id,
                "phenotype_hash": phenotype,
                "eligibility_status": "INELIGIBLE",
                "median_gate_alignment_score": alignment,
                "median_robust_score": -0.1,
                "min_scenario_net_return": 0.01,
                "profitable_scenario_ratio": 1.0,
                "worst_annualized_return_lcb": -0.02,
                "worst_net_expectancy_lcb": -0.001,
                "worst_max_drawdown_ucb": 0.2,
                "worst_daily_es5_ucb": 0.02,
                "min_effective_sample_size": 40,
                "min_trades_per_active_month": 3.0,
                "max_drawdown_duration_days": 200,
                "failed_gate_reason_codes": ["TRADE_RATE_TOO_LOW"],
                "gate_distances": [
                    {
                        "gate_id": "TRADES_PER_ACTIVE_MONTH",
                        "observation_count": 1,
                        "passed_observation_count": 0,
                        "contract_consistent": True,
                        "operator": ">=",
                        "threshold": 5.0,
                        "observed_min": 3.0,
                        "observed_max": 3.0,
                        "worst_signed_threshold_margin": -2.0,
                        "max_threshold_shortfall": 2.0,
                        "reason_codes": ["TRADE_RATE_TOO_LOW"],
                    }
                ],
            }
        ],
        "attempts": [
            {
                "experiment_id": experiment_id,
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
    assert forward["schema_version"] == "1.4"
    assert forward["wave_count"] == 2
    assert forward["attempt_count"] == 2
    assert forward["successful_attempt_count"] == 2
    assert forward["duration_seconds"] == 240
    assert forward["artifact_bytes"] == 2048
    assert forward["unique_phenotype_count"] == 1
    assert forward["repeated_phenotype_observation_count"] == 1
    assert forward["waves"][1]["repeated_phenotype_hashes"] == ["a" * 64]
    assert (
        forward["waves"][1]["diagnostic_top_candidate"]["median_gate_alignment_score"]
        == 0.8
    )
    assert "best_candidate" not in forward["waves"][1]
    assert (
        forward["waves"][1]["diagnostic_candidate_ranking"]["planner_selection"]
        is False
    )
    assert forward["waves"][1]["diagnostic_top_candidate"]["planner_selected"] is False
    assert forward["waves"][1]["diagnostic_top_candidate"]["arm_id"] == "explore"
    assert forward["waves"][1]["diagnostic_top_candidate"]["arm_type"] == "EXPLORE"
    assert forward["waves"][1]["arms"][0]["candidate_count"] == 1
    assert forward["waves"][1]["arms"][0]["successful_attempt_count"] == 1
    assert forward["waves"][1]["parent_wave_id"] == "parent-wave"
    assert forward["waves"][1]["plan_hash"] == "f" * 64
    assert forward["waves"][1]["arms"][0]["planned_seeds"] == [1]
    assert forward["waves"][1]["arms"][0]["factor_delta"] == {"mutation_rate": 0.05}
    assert forward["waves"][1]["arms"][0]["selected_parent_phenotype_hash"] == "p" * 64
    assert forward["waves"][1]["selected_parent_phenotype_hashes"] == ["p" * 64]
    assert (
        forward["waves"][1]["diagnostic_top_candidate"]["worst_net_expectancy_lcb"]
        == -0.001
    )
    assert (
        forward["waves"][1]["diagnostic_top_candidate"]["gate_distances"][0][
            "worst_signed_threshold_margin"
        ]
        == -2.0
    )
    assert (
        forward["waves"][1]["planner_selected_input_parents"][0]["selection_role"]
        == "PLANNER_SELECTED_INPUT_PARENT"
    )


def test_campaign_summary_contains_no_strategy_or_trade_payloads():
    summary = build_campaign_summary_v2(
        [_report("wave-a", "2026-07-26T00:00:00Z", "a" * 64, 0.7)]
    )

    serialized = str(summary)
    assert "strategy_code" not in serialized
    assert "genome" not in serialized
    assert "trades" not in summary["waves"][0]["diagnostic_top_candidate"]


def test_engine_seed_evidence_uses_salted_search_seed_contract():
    manifest_seed = 8001
    salt = 3
    expected_search_seed = derive_search_seed(manifest_seed, salt)
    assert expected_search_seed != manifest_seed
    engine_config = {
        "genetic_algorithm": {
            "random_seed": expected_search_seed,
            "search_seed_salt": salt,
        },
        "generic_island_model": {
            "islands": [
                {"name": "momentum", "seed": expected_search_seed},
                {"name": "trend", "seed": (expected_search_seed + 1) % (2**32)},
            ]
        },
    }

    evidence = _engine_seed_evidence(manifest_seed, engine_config)

    assert evidence["manifest_seed"] == manifest_seed
    assert evidence["search_seed_salt"] == salt
    assert evidence["expected_ga_seed"] == expected_search_seed
    assert evidence["expected_island_seeds"] == [
        expected_search_seed,
        (expected_search_seed + 1) % (2**32),
    ]
    assert evidence["contract_matches"] is True

    engine_config["generic_island_model"]["islands"][1]["seed"] = manifest_seed + 1
    assert _engine_seed_evidence(manifest_seed, engine_config)["contract_matches"] is False


def test_campaign_summary_rejects_mixed_root_lineages():
    first = _report("wave-a", "2026-07-26T00:00:00Z", "a" * 64, 0.7)
    second = _report("wave-b", "2026-07-26T01:00:00Z", "b" * 64, 0.8)
    second["root_wave_id"] = "another-root"

    with pytest.raises(ValueError, match="cannot mix root wave lineages"):
        build_campaign_summary_v2([first, second])


def test_campaign_summary_maps_actual_planner_selection_to_source_wave():
    source = _report(
        "wave-a",
        "2026-07-26T00:00:00Z",
        "a" * 64,
        0.7,
    )
    child = _report(
        "wave-b",
        "2026-07-26T01:00:00Z",
        "b" * 64,
        0.8,
        parent_wave="wave-a",
        parent_experiment="experiment-wave-a",
        parent_phenotype="a" * 64,
    )

    summary = build_campaign_summary_v2([child, source])

    event = next(
        item
        for item in summary["planner_selection_events"]
        if item["source_wave_id"] == "wave-a"
    )
    assert event["child_wave_id"] == "wave-b"
    assert event["selection_role"] == "PLANNER_SELECTED_PARENT_FOR_CHILD"
    assert event["source_report_available"] is True
    assert event["selected_candidate"]["planner_selected"] is True
    assert event["selected_candidate"]["reporting_role"] == "PLANNER_SELECTED_PARENT"
    assert event["selected_candidate"]["phenotype_hash"] == "a" * 64
    assert summary["waves"][0]["planner_selected_for_child_waves"][0] == event


def test_gate_distance_is_direction_aware_and_aggregated():
    assert _signed_threshold_margin(-0.02, 0.0, ">") == -0.02
    assert _signed_threshold_margin(0.18, 0.25, "<=") == pytest.approx(0.07)
    assert _signed_threshold_margin(7.0, 5.0, ">=") == 2.0
    assert _signed_threshold_margin(5.0, 5.0, "==") == 0.0

    attempts = [
        {
            "experiment_id": "experiment-a",
            "candidates": [
                {
                    "phenotype_hash": "a" * 64,
                    "gates": [
                        {
                            "gate_id": "TRADE_RATE",
                            "status": "VALID",
                            "passed": False,
                            "observed": 3.0,
                            "threshold": 5.0,
                            "operator": ">=",
                            "signed_threshold_margin": -2.0,
                            "threshold_shortfall": 2.0,
                            "reason_code": "TRADE_RATE_TOO_LOW",
                        }
                    ],
                }
            ],
        },
        {
            "experiment_id": "experiment-a",
            "candidates": [
                {
                    "phenotype_hash": "a" * 64,
                    "gates": [
                        {
                            "gate_id": "TRADE_RATE",
                            "status": "VALID",
                            "passed": False,
                            "observed": 4.0,
                            "threshold": 5.0,
                            "operator": ">=",
                            "signed_threshold_margin": -1.0,
                            "threshold_shortfall": 1.0,
                            "reason_code": "TRADE_RATE_TOO_LOW",
                        }
                    ],
                }
            ],
        },
    ]

    summary = _candidate_gate_distances(attempts)[("experiment-a", "a" * 64)][0]
    assert summary["contract_consistent"] is True
    assert summary["observation_count"] == 2
    assert summary["passed_observation_count"] == 0
    assert summary["observed_min"] == 3.0
    assert summary["observed_max"] == 4.0
    assert summary["worst_signed_threshold_margin"] == -2.0
    assert summary["max_threshold_shortfall"] == 2.0


def test_selected_parent_records_group_consuming_arms():
    plan_summary = {
        "parent_wave_id": "wave-parent",
        "experiments": [
            {
                "experiment_id": "experiment-replication",
                "arm_id": "replication",
                "arm_type": "REPLICATION",
                "source_mode": "SELECTED_CANDIDATES",
                "parent_experiment_id": "experiment-parent",
                "parent_config_hash": "c" * 64,
                "selected_candidate_key": "candidate-key",
                "phenotype_hash": "p" * 64,
            },
            {
                "experiment_id": "experiment-explore",
                "arm_id": "explore",
                "arm_type": "EXPLORE",
                "source_mode": "SELECTED_CANDIDATES",
                "parent_experiment_id": "experiment-parent",
                "parent_config_hash": "c" * 64,
                "selected_candidate_key": "candidate-key",
                "phenotype_hash": "p" * 64,
            },
            {
                "experiment_id": "experiment-control",
                "arm_id": "control",
                "arm_type": "CONTROL",
                "source_mode": "BASELINE_CONTROL",
                "parent_experiment_id": "experiment-baseline",
                "parent_config_hash": "b" * 64,
                "selected_candidate_key": None,
                "phenotype_hash": None,
            },
        ],
    }

    selected = _planner_selected_input_parents(plan_summary)

    assert len(selected) == 1
    assert selected[0]["selected_from_wave_id"] == "wave-parent"
    assert [item["arm_id"] for item in selected[0]["used_by_arms"]] == [
        "explore",
        "replication",
    ]


def test_markdown_explicitly_separates_diagnostic_and_planner_selection():
    source = _report(
        "wave-a",
        "2026-07-26T00:00:00Z",
        "a" * 64,
        0.7,
    )
    child = _report(
        "wave-b",
        "2026-07-26T01:00:00Z",
        "b" * 64,
        0.8,
        parent_wave="wave-a",
        parent_experiment="experiment-wave-a",
        parent_phenotype="a" * 64,
    )
    campaign_markdown = _campaign_markdown(build_campaign_summary_v2([source, child]))

    assert "Diagnostic top arm" in campaign_markdown
    assert "Actual planner-selected parents" in campaign_markdown
    assert "It is not the planner-selected parent" in campaign_markdown
    assert "TRADES_PER_ACTIVE_MONTH -2.0000" in campaign_markdown
    assert "| Best arm |" not in campaign_markdown

    child.update(
        {
            "has_eligible_candidates": False,
            "attempts": [],
            "planner_selected_input_parents": _planner_selected_input_parents(
                child["plan_summary"]
            ),
            "diagnostic_candidate_ranking": {
                "purpose": "REPORTING_ONLY",
                "planner_selection": False,
                "basis": "GATE_ALIGNMENT_THEN_ROBUST_SCORE",
            },
        }
    )
    wave_markdown = _markdown(child)

    assert "Actual planner-selected input parents" in wave_markdown
    assert "Diagnostic candidate ranking" in wave_markdown
    assert "not a planner selection" in wave_markdown
    assert "TRADES_PER_ACTIVE_MONTH -2.0000" in wave_markdown
