"""Bridge tests from hardcore decisions into the immutable V2 worker."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from genetic_algorithm.evaluation.raw_multipair_score import RawMultiPairPanel
from genetic_algorithm.config.schema import load_config
from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration import evolution_worker_v2
from genetic_algorithm.orchestration.hardcore_backend_v1 import (
    _classify_failure,
    _engine_stop_reason,
    _holding_hours,
    _live_engine_progress,
    _live_pair_metrics,
    _materialized_config,
    _scenario_from_record,
    _strict_replay_top_n,
)
from genetic_algorithm.orchestration.hardcore_campaign_v1 import (
    CampaignLane,
    EvolutionStopReason,
    EvolutionRunRequestV1,
    FailureClass,
    HardcoreCampaignError,
    SearchRecipe,
    build_island_blueprints,
)


NOW = datetime(2026, 8, 12, tzinfo=UTC)


def test_live_progress_uses_raw_score_and_checkpoint_plateau(tmp_path):
    evolution = tmp_path / "evolution"
    checkpoints = evolution / "checkpoints"
    checkpoints.mkdir(parents=True)
    rows = [
        {
            "generation": 4,
            "best_fitness": 99.0,
            "best_raw_fitness": -7.0,
        },
        {
            "generation": 5,
            "best_fitness": 12.0,
            "best_raw_fitness": 8.5,
        },
    ]
    (evolution / "generation_trace_v2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (checkpoints / "island_checkpoint_gen5_test.json").write_text(
        json.dumps(
            {"common_panel_replay": {"no_improvement_checks": 3}}
        ),
        encoding="utf-8",
    )

    generation, score, plateau = _live_engine_progress(tmp_path)

    assert generation == 6
    assert score == 8.5
    assert plateau == 3


def test_live_pair_metrics_are_read_from_checkpoint(tmp_path):
    target = tmp_path / "evolution/checkpoints"
    target.mkdir(parents=True)
    pairs = ("BTC/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "ETH/USDT", "PEPE/USDT")
    raw_pair = {
        "success": True,
        "trade_count": 35,
        "active_months": 20,
        "net_return": 0.1,
        "net_expectancy": 0.005,
        "profit_factor": 1.5,
        "profit_factor_censored": False,
        "median_holding_hours": 2.0,
        "p90_holding_hours": 8.0,
        "max_drawdown": 0.05,
        "max_drawdown_duration_days": 20.0,
        "max_consecutive_losses": 3,
    }
    component = {
        "return_score": 0.5,
        "expectancy_score": 0.5,
        "profit_factor_score": 0.5,
        "activity_score": 0.5,
        "holding_score": 1.0,
        "drawdown_risk": 0.2,
        "drawdown_duration_risk": 0.2,
        "loss_streak_risk": 0.2,
        "overtrading_risk": 0.0,
    }
    checkpoint = {
        "island_populations": {
            "mixed-1": {
                "individuals": [
                    {
                        "evaluated": True,
                        "raw_fitness": 12.5,
                        "metrics": {
                            "raw_multipair_status": "VALID",
                            "raw_pair_metrics": {
                                pair: dict(raw_pair) for pair in pairs
                            },
                            "raw_multipair_result": {
                                "timeframe": "15m",
                                "pair_components": {
                                    pair: {"components": dict(component)}
                                    for pair in pairs
                                },
                            },
                        },
                    }
                ]
            }
        }
    }
    (target / "island_checkpoint_gen5_test.json").write_text(
        json.dumps(checkpoint),
        encoding="utf-8",
    )

    metrics = _live_pair_metrics(tmp_path)

    assert [item.pair for item in metrics] == list(pairs)
    assert all(item.trade_count > 0 for item in metrics)


def _request(repo_root: Path, *, recipe: SearchRecipe) -> EvolutionRunRequestV1:
    path = (
        repo_root
        / "genetic_algorithm/config/presets/hardcore_multipair_canary_15m_v1.yaml"
    )
    import hashlib

    return EvolutionRunRequestV1(
        campaign_id="backend-test",
        run_id=f"backend-test-{recipe.value}",
        run_sequence=1,
        lane=CampaignLane.FIFTEEN_MINUTES,
        recipe=recipe,
        recipe_parameters={
            SearchRecipe.BALANCED: {
                "mutation_rate": 0.20,
                "crossover_rate": 0.75,
                "indicator_overlap": 0.35,
            },
            SearchRecipe.EXPLORE: {
                "mutation_rate": 0.30,
                "crossover_rate": 0.70,
                "indicator_overlap": 0.20,
            },
            SearchRecipe.RECOMBINE: {
                "mutation_rate": 0.25,
                "crossover_rate": 0.85,
                "indicator_overlap": 0.50,
            },
        }[recipe],
        attempt_ordinal=0,
        created_at=NOW,
        campaign_deadline=NOW.replace(day=19),
        config_path=str(path.resolve()),
        config_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        resolved_config_sha256=canonical_config_hash(load_config(path)),
        search_seed=1234,
        panel_id=RawMultiPairPanel(timeframe="15m").panel_id,
        islands=build_island_blueprints(
            lane=CampaignLane.FIFTEEN_MINUTES,
            search_seed=1234,
            archive=[],
            recipe=recipe,
            population_size=3,
            generations=1,
        ),
    )


def test_backend_materializes_canary_shape_and_recipe_specific_pools():
    repo_root = Path(__file__).resolve().parents[2]
    balanced = _materialized_config(_request(repo_root, recipe=SearchRecipe.BALANCED))
    explore = _materialized_config(_request(repo_root, recipe=SearchRecipe.EXPLORE))

    assert balanced["genetic_algorithm"]["population_size"] == 3
    assert balanced["generic_island_model"]["generations"] == 1
    assert {
        (item["population_size"], item["generations"])
        for item in balanced["generic_island_model"]["islands"]
    } == {(3, 1)}
    assert balanced["genetic_algorithm"]["mutation_rate"] == 0.20
    assert explore["genetic_algorithm"]["mutation_rate"] == 0.30
    assert [item["indicator_pool"] for item in balanced["generic_island_model"]["islands"]] != [
        item["indicator_pool"] for item in explore["generic_island_model"]["islands"]
    ]
    assert _strict_replay_top_n(balanced) == 2


def test_both_canary_lane_worker_specs_resolve_two_strict_candidates():
    repo_root = Path(__file__).resolve().parents[2]
    for lane, filename in (
        (CampaignLane.FIFTEEN_MINUTES, "hardcore_multipair_canary_15m_v1.yaml"),
        (CampaignLane.ONE_HOUR, "hardcore_multipair_canary_1h_v1.yaml"),
    ):
        request = _request(repo_root, recipe=SearchRecipe.BALANCED)
        path = repo_root / "genetic_algorithm/config/presets" / filename
        import hashlib

        request = request.model_copy(
            update={
                "lane": lane,
                "config_path": str(path.resolve()),
                "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "resolved_config_sha256": canonical_config_hash(load_config(path)),
                "panel_id": RawMultiPairPanel(timeframe=lane.value).panel_id,
                "islands": build_island_blueprints(
                    lane=lane,
                    search_seed=1234,
                    archive=[],
                    recipe=SearchRecipe.BALANCED,
                    population_size=3,
                    generations=1,
                ),
            }
        )

        assert _strict_replay_top_n(_materialized_config(request)) == 2


@pytest.mark.parametrize("bad_value", [True, 0, 13, "2", None])
def test_strict_replay_top_n_is_fail_closed(bad_value):
    with pytest.raises(HardcoreCampaignError, match="output.top_n"):
        _strict_replay_top_n({"output": {"top_n": bad_value}})


def test_raw_reconstruction_accepts_only_versioned_point_record():
    metrics = SimpleNamespace(
        pair="BTC/USDT",
        timeframe="15m",
        period_start="2023-05-09",
        period_end="2026-03-26",
        measured_period_start="2023-05-09T00:00:00+00:00",
        measured_period_end="2026-03-26T23:45:00+00:00",
        status="INCONCLUSIVE",
        success=True,
        no_trades=False,
        error_code="RAW_POINT_METRICS_ONLY",
        error_detail=None,
        trade_count=2,
        active_months=1,
        net_return=0.01,
        net_expectancy=0.002,
        net_expectancy_on_committed_capital=None,
        profit_factor=1.2,
        profit_factor_censored=False,
        max_drawdown=0.02,
        max_drawdown_duration_days=3.0,
        max_consecutive_losses=1,
    )
    record = SimpleNamespace(
        metrics=metrics,
        trades=[
            {"trade_duration": 60.0},
            {"trade_duration": 180.0},
        ],
    )

    scenario = _scenario_from_record(record, timeframe=CampaignLane.FIFTEEN_MINUTES)

    assert scenario.success is True
    assert scenario.technical_error is None
    assert scenario.net_expectancy == 0.002
    assert scenario.median_holding_hours == 2.0
    assert scenario.p90_holding_hours == pytest.approx(2.8)


def test_strict_replay_does_not_tolerate_legacy_inconclusive_gate_code():
    metrics = SimpleNamespace(
        pair="BTC/USDT",
        timeframe="15m",
        period_start="2023-05-09",
        period_end="2026-03-26",
        measured_period_start="2023-05-09T00:00:00+00:00",
        measured_period_end="2026-03-26T23:45:00+00:00",
        status="INCONCLUSIVE",
        success=True,
        no_trades=False,
        error_code="EXPECTANCY_BOOTSTRAP_INSUFFICIENT",
        error_detail=None,
        trade_count=2,
        active_months=1,
    )

    scenario = _scenario_from_record(
        SimpleNamespace(metrics=metrics, trades=[]),
        timeframe=CampaignLane.FIFTEEN_MINUTES,
    )

    assert scenario.success is False
    assert scenario.technical_error == "EXPECTANCY_BOOTSTRAP_INSUFFICIENT"


def test_zero_trade_replay_remains_valid_raw_evidence_even_if_v2_is_inconclusive():
    metrics = SimpleNamespace(
        pair="BTC/USDT",
        timeframe="15m",
        period_start="2023-05-09T00:00:00+00:00",
        period_end="2026-03-26T23:45:00+00:00",
        measured_period_start="2023-05-09T00:00:00+00:00",
        measured_period_end="2026-03-26T23:45:00+00:00",
        status="INCONCLUSIVE",
        success=True,
        no_trades=True,
        error_code="NO_TRADES",
        error_detail=None,
        trade_count=0,
        active_months=0,
    )
    scenario = _scenario_from_record(
        SimpleNamespace(metrics=metrics, trades=[]),
        timeframe=CampaignLane.FIFTEEN_MINUTES,
    )

    assert scenario.success is True
    assert scenario.trade_count == 0
    assert scenario.technical_error is None
    assert scenario.period_start.isoformat() == "2023-05-09T00:00:00+00:00"
    assert scenario.period_end.isoformat() == "2026-03-26T23:45:00+00:00"


def test_zero_trade_replay_with_mismatched_period_is_technical_data_failure():
    metrics = SimpleNamespace(
        pair="BTC/USDT",
        timeframe="15m",
        period_start="2023-05-09",
        period_end="2026-03-26",
        measured_period_start="2023-05-09T15:00:00+00:00",
        measured_period_end="2026-03-26T23:45:00+00:00",
        status="INCONCLUSIVE",
        success=True,
        no_trades=True,
        error_code="NO_TRADES",
        error_detail=None,
        trade_count=0,
        active_months=0,
    )

    scenario = _scenario_from_record(
        SimpleNamespace(metrics=metrics, trades=[]),
        timeframe=CampaignLane.FIFTEEN_MINUTES,
    )

    assert scenario.success is False
    assert scenario.technical_error.startswith("PERIOD_COVERAGE_MISMATCH")
    assert _classify_failure(scenario.technical_error, None) == (
        FailureClass.DETERMINISTIC_DATA
    )


@pytest.mark.parametrize(
    "error_code",
    [
        "MISSING_EQUITY_EVIDENCE",
        "MISSING_PERIOD_EVIDENCE",
        "PERIOD_COVERAGE_MISMATCH",
    ],
)
def test_strict_replay_rejects_data_evidence_codes_deterministically(error_code):
    metrics = SimpleNamespace(
        pair="BTC/USDT",
        timeframe="15m",
        period_start="2023-05-09",
        period_end="2026-03-26",
        measured_period_start="2023-05-09T00:00:00+00:00",
        measured_period_end="2026-03-26T23:45:00+00:00",
        status="INVALID",
        success=True,
        no_trades=False,
        error_code=error_code,
        error_detail="measured data evidence is invalid",
        trade_count=2,
        active_months=1,
    )

    scenario = _scenario_from_record(
        SimpleNamespace(metrics=metrics, trades=[]),
        timeframe=CampaignLane.FIFTEEN_MINUTES,
    )

    assert scenario.success is False
    assert scenario.technical_error.startswith(error_code)
    assert _classify_failure(error_code, None) == FailureClass.DETERMINISTIC_DATA


def test_technical_zero_trade_record_is_not_converted_to_valid_evidence():
    metrics = SimpleNamespace(
        pair="BTC/USDT",
        timeframe="15m",
        period_start="2023-05-09",
        period_end="2026-03-26",
        measured_period_start="2023-05-09T00:00:00+00:00",
        measured_period_end="2026-03-26T23:45:00+00:00",
        status="INVALID",
        success=False,
        no_trades=True,
        error_code="BACKTEST_EXCEPTION",
        error_detail=None,
        trade_count=0,
        active_months=0,
        net_return=None,
        net_expectancy=None,
        profit_factor=None,
        profit_factor_censored=None,
        max_drawdown=None,
        max_drawdown_duration_days=None,
        max_consecutive_losses=None,
    )

    scenario = _scenario_from_record(
        SimpleNamespace(metrics=metrics, trades=[]),
        timeframe=CampaignLane.FIFTEEN_MINUTES,
    )

    assert scenario.success is False
    assert scenario.technical_error == "BACKTEST_EXCEPTION"


def test_unknown_engine_reason_fails_closed():
    assert _engine_stop_reason({"reason": "UNRECOGNIZED"}) == (
        EvolutionStopReason.TECHNICAL_INVALID
    )


def test_holding_fallback_uses_open_and_close_timestamps():
    measured = _holding_hours(
        [
            {
                "open_date": "2026-01-01T00:00:00+00:00",
                "close_date": "2026-01-01T06:00:00+00:00",
            },
            {
                "open_timestamp": 1_767_225_600_000,
                "close_timestamp": 1_767_268_800_000,
            },
        ]
    )
    assert measured == pytest.approx((9.0, 11.4))


def test_hardcore_worker_routes_parent_seeds_only_to_archive_islands(
    tmp_path: Path, monkeypatch
):
    calls = {"archive": None, "strict": None}

    class FakeAlgorithm:
        def __init__(self, config_path, **_kwargs):
            self.island_configs = [SimpleNamespace(name="one")]

        def set_archive_initial_seeds(self, seeds):
            calls["archive"] = list(seeds)

        def set_strict_initial_seeds(self, seeds):
            calls["strict"] = list(seeds)

        def evolve(self):
            return {"one": []}

    monkeypatch.setattr(
        "genetic_algorithm.core.generic_island_model.GenericIslandModelEvolution",
        FakeAlgorithm,
    )
    monkeypatch.setattr(
        "genetic_algorithm.engine.island_results.extract_island_finalists",
        lambda *_args, **_kwargs: SimpleNamespace(finalists=[]),
    )
    path = tmp_path / "engine.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "safety_profile": {"name": "hardcore_multipair_v1"},
                "generic_island_model": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    seeds = [SimpleNamespace(name="champion")]

    evolution_worker_v2._run_evolution_engine(path, seeds)

    assert calls["archive"] == seeds
    assert calls["strict"] is None
