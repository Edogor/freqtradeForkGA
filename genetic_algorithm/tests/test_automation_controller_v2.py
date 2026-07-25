"""Contract tests for guarded, restartable unattended wave search."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from freqtrade.misc import pair_to_filename
from genetic_algorithm import cli
from genetic_algorithm.orchestration import automation_controller_v2
from genetic_algorithm.config.schema import load_config
from genetic_algorithm.orchestration.automation_controller_v2 import (
    AutomationBootstrapIntentV2,
    AutomationControllerV2,
    bootstrap_automation_wave,
    build_automation_preflight,
    default_automation_policy,
    render_systemd_user_unit,
)
from genetic_algorithm.orchestration.attempt_state_v2 import (
    AttemptLifecycleStatus,
    WorkerKind,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
)
from genetic_algorithm.orchestration.wave_state_v2 import (
    WaveLifecycleStatus,
    WaveStateStoreV2,
)


NOW = datetime(2026, 7, 24, 9, 0, tzinfo=UTC)


class _SchedulerMustNotRun:
    def run_once(self, *, observed_at=None):  # pragma: no cover - failure path
        raise AssertionError("scheduler ran after an external stop guard")

    def close(self, *, wait=True):
        return None


def test_real_preset_preflight_proves_pair_split_data_and_resources(
):
    repo_root = Path(__file__).resolve().parents[2]
    automation_root = (
        repo_root / "genetic_algorithm/data/v2/preflight-test-empty"
    )
    assert not automation_root.exists()
    report = build_automation_preflight(
        repo_root / "genetic_algorithm/config/presets/automation_island_v2.yaml",
        automation_root=automation_root,
        repo_root=repo_root,
        checked_at=NOW,
    )

    assert report.ready is True
    assert report.reason_codes == ["PREFLIGHT_READY"]
    assert report.virtualenv_active is True
    assert report.evolution_pairs == ["BTC/USDT", "SOL/USDT"]
    assert report.validation_pairs == ["BNB/USDT", "ETH/USDT"]
    assert report.disk_free_bytes > report.disk_reserve_bytes
    assert report.memory_available_bytes > report.memory_reserve_bytes
    policy = default_automation_policy(
        load_config(
            repo_root
            / "genetic_algorithm/config/presets/automation_island_v2.yaml"
        ),
        automation_root=automation_root,
    )
    assert policy.root_seeds == [4001]
    assert policy.max_waves == 1
    assert report.config_hash
    assert load_config(
        repo_root / "genetic_algorithm/config/presets/automation_island_v2.yaml"
    )["pair_validation"]["evaluation_mode"] == "independent_pairs"


def test_preflight_blocks_an_existing_campaign_with_different_contract(
    tmp_path: Path,
    monkeypatch,
):
    repo_root = Path(__file__).resolve().parents[2]
    automation_root = tmp_path / "automation"
    automation_root.mkdir()
    intent = AutomationBootstrapIntentV2(
        created_at=NOW,
        config_hash="0" * 64,
        automation_policy_hash="1" * 64,
        seeds=[3001],
        wave_id="wave-root-old",
        experiment_id="experiment-root-old",
    )
    (automation_root / "bootstrap_intent.json").write_text(
        intent.model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        automation_controller_v2.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=100 * 1024**3),
    )
    monkeypatch.setattr(
        automation_controller_v2,
        "_available_memory_bytes",
        lambda: 16 * 1024**3,
    )

    report = build_automation_preflight(
        repo_root / "genetic_algorithm/config/presets/automation_island_v2.yaml",
        automation_root=automation_root,
        repo_root=repo_root,
        checked_at=NOW,
    )

    assert report.ready is False
    assert report.reason_codes == [
        "BOOTSTRAP_INTENT_CONFIG_MISMATCH",
        "BOOTSTRAP_INTENT_POLICY_MISMATCH",
        "BOOTSTRAP_INTENT_SEED_MISMATCH",
    ]


def test_systemd_unit_restarts_crashes_but_not_guarded_stop(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    unit = render_systemd_user_unit(
        repo_root=repo_root,
        config_path=repo_root
        / "genetic_algorithm/config/presets/automation_island_v2.yaml",
        state_path=tmp_path / "state.sqlite3",
        automation_root=tmp_path / "automation",
    )

    assert f"WorkingDirectory={repo_root}" in unit
    assert f'WorkingDirectory="{repo_root}"' not in unit
    assert f'"{repo_root / ".venv/bin/python"}"' in unit
    assert "automation start" in unit
    assert "Restart=on-failure" in unit
    assert "SuccessExitStatus=2" in unit
    assert "RestartPreventExitStatus=2" in unit
    assert "KillMode=control-group" in unit


def test_two_wave_canary_changes_only_campaign_identity_and_budget():
    repo_root = Path(__file__).resolve().parents[2]
    baseline = load_config(
        repo_root
        / "genetic_algorithm/config/presets/automation_island_v2.yaml"
    )
    canary = load_config(
        repo_root
        / "genetic_algorithm/config/presets/automation_island_canary_v2.yaml"
    )

    assert canary["automation_controller"] == {
        "root_seeds": [5002],
        "max_waves": 2,
    }
    baseline_without_campaign = {
        key: value
        for key, value in baseline.items()
        if key != "automation_controller"
    }
    canary_without_campaign = {
        key: value
        for key, value in canary.items()
        if key != "automation_controller"
    }
    assert canary_without_campaign == baseline_without_campaign


def _bootstrap(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "genetic_algorithm/config/presets/automation_island_v2.yaml"
    config = load_config(config_path)
    automation_root = tmp_path / "automation"
    policy = default_automation_policy(config, automation_root=automation_root)
    store = WaveStateStoreV2(tmp_path / "state.sqlite3")
    receipt = bootstrap_automation_wave(
        config_path,
        store=store,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
        created_at=NOW,
    )
    return repo_root, automation_root, policy, store, receipt


def test_bootstrap_is_idempotent_and_queues_only_generic_island(tmp_path: Path):
    repo_root, automation_root, policy, store, first = _bootstrap(tmp_path)
    second = bootstrap_automation_wave(
        repo_root / "genetic_algorithm/config/presets/automation_island_v2.yaml",
        store=store,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
        created_at=NOW + timedelta(days=1),
    )

    assert second == first
    wave = store.get_wave(first.intent.wave_id)
    assert wave.status == WaveLifecycleStatus.COLLECTING
    attempts = [
        item for item in store.list_attempts() if item.wave_id == wave.wave_id
    ]
    assert len(attempts) == 1
    assert attempts[0].status == AttemptLifecycleStatus.QUEUED
    assert (
        attempts[0].worker_binding.worker_kind
        == WorkerKind.GENERIC_ISLAND_EVOLUTION
    )
    assert (automation_root / "bootstrap_intent.json").is_file()
    assert (automation_root / "bootstrap_receipt.json").is_file()


def test_kill_switch_prevents_scheduler_launch_and_preserves_queue(tmp_path: Path):
    repo_root, automation_root, policy, store, receipt = _bootstrap(tmp_path)
    Path(policy.kill_switch_path).write_text("stop\n", encoding="utf-8")
    controller = AutomationControllerV2(
        store=store,
        root_wave_id=receipt.intent.wave_id,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
        scheduler=_SchedulerMustNotRun(),
    )

    tick = controller.run_once(observed_at=NOW + timedelta(minutes=1))

    assert tick.outcome == "STOPPED_KILL_SWITCH"
    assert tick.reason_codes == ["KILL_SWITCH_PRESENT"]
    assert store.list_attempts()[0].status == AttemptLifecycleStatus.QUEUED


def test_single_wave_diagnostic_budget_blocks_child_plan(tmp_path: Path):
    repo_root, automation_root, policy, store, receipt = _bootstrap(tmp_path)
    controller = AutomationControllerV2(
        store=store,
        root_wave_id=receipt.intent.wave_id,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
        scheduler=_SchedulerMustNotRun(),
    )

    reasons = controller._guard_plan(
        [store.get_wave(receipt.intent.wave_id)],
        SimpleNamespace(experiments=[]),
    )

    assert reasons == ["MAX_WAVES_REACHED"]
    assert len(store.list_waves()) == 1


def test_runtime_limit_prevents_scheduler_launch_after_restart(tmp_path: Path):
    repo_root, automation_root, policy, store, receipt = _bootstrap(tmp_path)
    controller = AutomationControllerV2(
        store=store,
        root_wave_id=receipt.intent.wave_id,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
        scheduler=_SchedulerMustNotRun(),
    )

    tick = controller.run_once(
        observed_at=NOW + timedelta(seconds=policy.max_runtime_seconds + 1)
    )

    assert tick.outcome == "STOPPED_LIMIT"
    assert tick.reason_codes == ["MAX_RUNTIME_REACHED"]
    assert store.list_attempts()[0].status == AttemptLifecycleStatus.QUEUED


def test_disk_limit_prevents_scheduler_launch(tmp_path: Path):
    repo_root, automation_root, policy, store, receipt = _bootstrap(tmp_path)
    policy = policy.model_copy(update={"max_artifact_bytes": 1})
    controller = AutomationControllerV2(
        store=store,
        root_wave_id=receipt.intent.wave_id,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
        scheduler=_SchedulerMustNotRun(),
    )

    tick = controller.run_once(observed_at=NOW + timedelta(minutes=1))

    assert tick.outcome == "STOPPED_LIMIT"
    assert tick.reason_codes == ["MAX_ARTIFACT_BYTES_REACHED"]
    assert store.list_attempts()[0].status == AttemptLifecycleStatus.QUEUED


def test_free_disk_reserve_prevents_scheduler_launch(tmp_path: Path):
    repo_root, automation_root, policy, store, receipt = _bootstrap(tmp_path)
    policy = policy.model_copy(
        update={
            "max_artifact_bytes": 10**15,
            "min_free_disk_bytes": 10**15,
        }
    )
    controller = AutomationControllerV2(
        store=store,
        root_wave_id=receipt.intent.wave_id,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
        scheduler=_SchedulerMustNotRun(),
    )

    tick = controller.run_once(observed_at=NOW + timedelta(minutes=1))

    assert tick.outcome == "STOPPED_LIMIT"
    assert tick.reason_codes == ["MIN_FREE_DISK_RESERVE_REACHED"]


def test_memory_reserve_defers_claim_without_stopping_lineage(
    tmp_path: Path,
    monkeypatch,
):
    repo_root, automation_root, policy, store, receipt = _bootstrap(tmp_path)
    policy = policy.model_copy(
        update={
            "min_free_disk_bytes": 1,
            "min_available_memory_bytes": 2 * 1024**3,
        }
    )
    monkeypatch.setattr(
        automation_controller_v2,
        "_available_memory_bytes",
        lambda: 1024**3,
    )
    controller = AutomationControllerV2(
        store=store,
        root_wave_id=receipt.intent.wave_id,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
        scheduler=_SchedulerMustNotRun(),
    )

    tick = controller.run_once(observed_at=NOW + timedelta(minutes=1))

    assert tick.outcome == "WAITING"
    assert tick.reason_codes == ["MEMORY_RESERVE_ACTIVE"]
    assert store.list_attempts()[0].status == AttemptLifecycleStatus.QUEUED


def test_three_consecutive_fully_failed_waves_block_further_recovery(
    tmp_path: Path,
    monkeypatch,
):
    repo_root, automation_root, policy, store, receipt = _bootstrap(tmp_path)
    controller = AutomationControllerV2(
        store=store,
        root_wave_id=receipt.intent.wave_id,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
        scheduler=_SchedulerMustNotRun(),
    )
    monkeypatch.setattr(controller, "_artifact_bytes", lambda: 0)
    monkeypatch.setattr(
        controller,
        "_attempts",
        lambda wave_id: [
            SimpleNamespace(status=AttemptLifecycleStatus.FAILED)
        ],
    )

    reasons = controller._guard_plan(
        [
            SimpleNamespace(wave_id="wave-1"),
            SimpleNamespace(wave_id="wave-2"),
            SimpleNamespace(wave_id="wave-3"),
        ],
        SimpleNamespace(experiments=[]),
    )

    assert "MAX_CONSECUTIVE_FAILED_WAVES_REACHED" in reasons


def test_cli_stop_then_start_bootstraps_without_launching_work(tmp_path: Path):
    automation_root = tmp_path / "cli-automation"
    state_path = tmp_path / "cli-state.sqlite3"
    unit_path = tmp_path / "ga-automation.service"

    assert (
        cli.main(
            [
                "automation",
                "service-unit",
                "automation_island_v2",
                "--output",
                str(unit_path),
                "--automation-root",
                str(automation_root),
                "--state-db",
                str(state_path),
            ]
        )
        == 0
    )
    assert "RestartPreventExitStatus=2" in unit_path.read_text(encoding="utf-8")
    assert (
        cli.main(
            [
                "automation",
                "stop",
                "--automation-root",
                str(automation_root),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "automation",
                "start",
                "automation_island_v2",
                "--once",
                "--automation-root",
                str(automation_root),
                "--state-db",
                str(state_path),
            ]
        )
        == 2
    )
    assert (
        cli.main(
            [
                "automation",
                "status",
                "--automation-root",
                str(automation_root),
                "--state-db",
                str(state_path),
            ]
        )
        == 0
    )


def test_real_mini_controller_executes_two_waves_across_restart(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    suffix = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:10].upper()
    train_pair = f"UNITTESTAUTOMATIONTRAIN{suffix}/BTC"
    validation_pair = f"UNITTESTAUTOMATIONVAL{suffix}/BTC"
    data_root = repo_root / "tests/testdata"
    data_paths = [
        data_root / f"{pair_to_filename(pair)}-1h.feather"
        for pair in (train_pair, validation_pair)
    ]
    candles = pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01", periods=240, freq="1h", tz="UTC"
            ),
            "open": [100.0] * 240,
            "high": [101.0] * 240,
            "low": [99.0] * 240,
            "close": [100.5] * 240,
            "volume": [10.0] * 240,
        }
    )
    for path in data_paths:
        candles.to_feather(path)

    try:
        config = load_config(
            repo_root / "genetic_algorithm/config/presets/safe_v2.yaml"
        )
        config["safety_profile"]["name"] = "automation_island_v2"
        config["backtesting"].update(
            {
                "pairs": [train_pair, validation_pair],
                "timerange": "20240101-20240111",
                "timeframe": "1h",
                "dataformat_ohlcv": "feather",
                "auto_download_data": False,
                "enable_cache": False,
                "dynamic_slippage": False,
                "spread_pct": 0.0,
                "funding_rate": 0.0,
            }
        )
        config["genetic_algorithm"].update(
            {
                "max_runtime_minutes": 10,
                "population_size": 4,
                "generations": 1,
                "elite_size": 1,
                "tournament_size": 2,
                "random_immigrants": 1,
            }
        )
        # This E2E specifically proves child-wave materialization; the shipped
        # diagnostic preset is intentionally bounded to its root wave.
        config["automation_controller"].update(
            {"root_seeds": [2001], "max_waves": 2}
        )
        config["parallel_evaluation"].update({"enabled": False, "num_workers": 1})
        config["output"]["top_n"] = 1
        config["generic_island_model"].update(
            {
                "enabled": True,
                "num_islands": 2,
                "population_per_island": 4,
                "generations": 1,
                "parallel_islands": False,
                "specialization": {
                    "rotate_seeds": True,
                    "indicator_pools": False,
                    "indicator_overlap": 0.5,
                    "pair_rotation": False,
                    "pair_subset_size": 1,
                },
                "migration": {
                    "topology": "ring",
                    "interval": 1,
                    "count": 1,
                    "merge_rounds": False,
                    "merge_interval": 2,
                    "tournament_size": 2,
                },
                "walk_forward": {"enabled": False},
                "external_migration": {"enabled": False, "directory": ""},
                "islands": [
                    {
                        "name": "a",
                        "population_size": 4,
                        "generations": 1,
                        "seed": 101,
                        "indicator_pool": ["RSI", "EMA"],
                        "pairs": [train_pair, validation_pair],
                        "walk_forward_enabled": False,
                    },
                    {
                        "name": "b",
                        "population_size": 4,
                        "generations": 1,
                        "seed": 102,
                        "indicator_pool": ["MACD", "BBANDS"],
                        "pairs": [train_pair, validation_pair],
                        "walk_forward_enabled": False,
                    },
                ],
            }
        )
        config["pair_validation"] = {
            "enabled": True,
            "evaluation_mode": "independent_pairs",
            "training_pairs": [train_pair],
            "validation_pairs": [validation_pair],
            "weight_train": 0.6,
            "weight_val": 0.4,
            "min_val_fitness": 0.0,
            "validate_top_n_only": 0,
            "worst_pair_weight": 0.5,
            "min_profitable_pair_ratio": 0.0,
            "profitable_pair_penalty_floor": 0.1,
            "max_pair_loss_pct": 0.0,
            "worst_pair_loss_penalty_floor": 0.1,
        }
        promotion = ShadowGatePolicyV2(
            policy_version="automation-controller-mini-v2",
            required_scenarios=[
                ScenarioRequirementV2(
                    scenario_id="train",
                    pair=train_pair,
                    timeframe="1h",
                    role="TRAIN",
                    period_start="2024-01-01",
                    period_end="2024-01-02",
                    cost_multiplier=1.0,
                ),
                ScenarioRequirementV2(
                    scenario_id="pair-validation",
                    pair=validation_pair,
                    timeframe="1h",
                    role="PAIR_VALIDATION",
                    period_start="2024-01-01",
                    period_end="2024-01-02",
                    cost_multiplier=1.0,
                ),
            ],
            min_effective_sample_size=1,
            min_active_months=1,
            min_trades_per_active_month=0,
            require_final_test=False,
        )
        config["promotion_v2"] = {
            "enabled": True,
            **promotion.model_dump(mode="json"),
        }
        config_path = tmp_path / "mini-automation.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=True))
        automation_root = tmp_path / "real-automation"
        policy = default_automation_policy(config, automation_root=automation_root)
        policy = policy.model_copy(
            update={"min_free_disk_bytes": 100 * 1024**2}
        )
        store = WaveStateStoreV2(tmp_path / "real-state.sqlite3")
        receipt = bootstrap_automation_wave(
            config_path,
            store=store,
            policy=policy,
            automation_root=automation_root,
            repo_root=repo_root,
            created_at=datetime.now(UTC),
        )

        with AutomationControllerV2(
            store=store,
            root_wave_id=receipt.intent.wave_id,
            policy=policy,
            automation_root=automation_root,
            repo_root=repo_root,
        ) as first_controller:
            first_tick = first_controller.run_once()
            assert first_tick.launched_attempt_ids

        # A fresh controller must recover the persisted terminal root evidence
        # and continue the exact lineage without a duplicate bootstrap/run.
        with AutomationControllerV2(
            store=store,
            root_wave_id=receipt.intent.wave_id,
            policy=policy,
            automation_root=automation_root,
            repo_root=repo_root,
        ) as restarted_controller:
            deadline = time.monotonic() + 30
            while len(store.list_waves()) < 2 and time.monotonic() < deadline:
                restarted_controller.run_once()
                time.sleep(0.05)

        waves = store.list_waves()
        assert len(waves) == 2
        assert waves[0].status == WaveLifecycleStatus.QUEUED
        assert waves[1].parent_wave_id == waves[0].wave_id
        assert waves[1].status == WaveLifecycleStatus.DRAFT
        child_attempts = [
            item for item in store.list_attempts() if item.wave_id == waves[1].wave_id
        ]
        assert child_attempts
        assert all(
            item.status == AttemptLifecycleStatus.QUEUED for item in child_attempts
        )
        assert all(
            item.worker_binding.worker_kind == WorkerKind.GENERIC_ISLAND_EVOLUTION
            for item in child_attempts
        )

        # The systemd entry point replays bootstrap on every restart. An
        # already queued root must remain immutable while the child resumes.
        restarted_receipt = bootstrap_automation_wave(
            config_path,
            store=store,
            policy=policy,
            automation_root=automation_root,
            repo_root=repo_root,
            created_at=datetime.now(UTC),
        )
        assert restarted_receipt == receipt
        assert (
            store.get_wave(receipt.intent.wave_id).status
            == WaveLifecycleStatus.QUEUED
        )

        with AutomationControllerV2(
            store=store,
            root_wave_id=receipt.intent.wave_id,
            policy=policy,
            automation_root=automation_root,
            repo_root=repo_root,
        ) as child_controller:
            deadline = time.monotonic() + 30
            final_tick = None
            while time.monotonic() < deadline:
                final_tick = child_controller.run_once()
                if final_tick.outcome == "STOPPED_LIMIT":
                    break
                time.sleep(0.05)

        assert final_tick is not None
        assert final_tick.outcome == "STOPPED_LIMIT"
        assert final_tick.reason_codes == ["MAX_WAVES_REACHED"]
        waves = store.list_waves()
        assert len(waves) == 2
        assert waves[0].status == WaveLifecycleStatus.QUEUED
        assert waves[1].status == WaveLifecycleStatus.BLOCKED
        child_attempts = [
            item for item in store.list_attempts() if item.wave_id == waves[1].wave_id
        ]
        assert all(
            item.status == AttemptLifecycleStatus.SUCCEEDED
            for item in child_attempts
        )
        report_path = automation_root / "reports" / "LATEST.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["root_wave_id"] == receipt.intent.wave_id
        assert report["wave_id"] == waves[1].wave_id
        assert report["controller_outcome"] == "STOPPED_LIMIT"
        assert report["controller_reason_codes"] == ["MAX_WAVES_REACHED"]
        assert report["attempts"][0]["candidates"]
        assert report["attempts"][0]["engine_seed_evidence"]["contract_matches"]

        def keys(value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield key
                    yield from keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from keys(nested)

        assert "strategy_code" not in set(keys(report))
        assert "trades" not in set(keys(report))
    finally:
        for path in data_paths:
            path.unlink(missing_ok=True)
