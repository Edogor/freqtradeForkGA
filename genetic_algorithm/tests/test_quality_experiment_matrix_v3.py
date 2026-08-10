from pathlib import Path

from genetic_algorithm.orchestration.quality_experiment_matrix_v3 import (
    ARM_PRESETS,
    DIAGNOSTIC_SEEDS,
    ArmRunOutcomeV3,
    choose_winner,
    diagnostic_specs,
    full_specs,
    resolved_run_config,
)


def test_matrix_has_30_paired_diagnostics_and_10_fresh_confirmations():
    diagnostic = diagnostic_specs()
    full = full_specs({"1h": "A4", "15m": "A3"})

    assert len(diagnostic) == 30
    assert len(full) == 40
    assert len({item.run_id for item in full}) == 40
    for timeframe in ("1h", "15m"):
        for seed in DIAGNOSTIC_SEEDS:
            assert {
                item.arm_id
                for item in diagnostic
                if item.timeframe == timeframe and item.paired_seed == seed
            } == set(ARM_PRESETS)


def test_timeframe_materialization_updates_every_scenario():
    spec = next(item for item in diagnostic_specs() if item.arm_id == "A4")
    payload = spec.__class__(**{**spec.__dict__, "timeframe": "15m"})
    config = resolved_run_config(payload)

    assert config["backtesting"]["timeframe"] == "15m"
    assert config["strategy_constraints"]["timeframes"] == ["15m"]
    assert all(
        item["timeframe"] == "15m"
        for section in ("promotion_v2", "qualification_v3")
        for item in config[section]["required_scenarios"]
    )


def test_winner_prefers_qualification_yield_before_raw_score():
    rows = []
    for arm in ARM_PRESETS:
        for seed in DIAGNOSTIC_SEEDS:
            rows.append(
                ArmRunOutcomeV3(
                    timeframe="1h",
                    arm_id=arm,
                    paired_seed=seed,
                    completed=True,
                    v3_pass_candidate_count=1 if arm == "A3" else 0,
                    max_feasibility_stage=4 if arm == "A3" else 3,
                    best_robust_score=0.01 if arm == "A3" else 100.0,
                    behavior_duplicate_fraction=0.2,
                )
            )

    assert choose_winner(rows, "1h") == "A3"
