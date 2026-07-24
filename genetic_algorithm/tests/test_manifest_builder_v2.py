"""Tests for code provenance capture and coherent attempt-manifest assembly."""

from __future__ import annotations

import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from genetic_algorithm.config.schema import load_config
from genetic_algorithm.orchestration.code_manifest_v2 import capture_code_manifest
from genetic_algorithm.orchestration.manifest_builder_v2 import build_attempt_manifest_bundle
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
)


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _repository(root: Path) -> None:
    _git(root, "init", "-q")
    (root / "engine.py").write_text("VALUE = 1\n")
    _git(root, "add", "engine.py")
    _git(
        root,
        "-c",
        "user.name=GA Test",
        "-c",
        "user.email=ga-test@example.invalid",
        "commit",
        "-q",
        "-m",
        "initial",
    )


def _policy() -> ShadowGatePolicyV2:
    return ShadowGatePolicyV2(
        policy_version="manifest-builder-test",
        required_scenarios=[
            ScenarioRequirementV2(
                scenario_id="final-btc",
                pair="BTC/USDT",
                timeframe="1h",
                role="FINAL_TEST",
                period_start=date(2025, 1, 1),
                period_end=date(2025, 1, 2),
                cost_multiplier=1.0,
            )
        ],
    )


def test_code_manifest_detects_tracked_and_untracked_source_changes(tmp_path: Path):
    _repository(tmp_path)
    clean = capture_code_manifest(tmp_path)

    assert clean.dirty is False
    assert clean.dirty_patch_hash is None

    (tmp_path / "engine.py").write_text("VALUE = 2\n")
    (tmp_path / "new_rule.py").write_text("RULE = True\n")
    dirty = capture_code_manifest(tmp_path)

    assert dirty.dirty is True
    assert len(dirty.dirty_patch_hash or "") == 64
    assert [item.relative_path for item in dirty.untracked_source_files] == ["new_rule.py"]
    assert capture_code_manifest(tmp_path) == dirty

    (tmp_path / "attempts").mkdir()
    (tmp_path / "attempts" / "result.json").write_text('{"status": "SUCCEEDED"}\n')
    assert capture_code_manifest(tmp_path) == dirty

    generated = tmp_path / "genetic_algorithm" / "data" / "checkpoints"
    generated.mkdir(parents=True)
    (generated / "generated_strategy.py").write_text("SHOULD_NOT_AFFECT_ENGINE = True\n")
    assert capture_code_manifest(tmp_path) == dirty

    (tmp_path / "new_rule.py").write_text("RULE = False\n")
    changed_again = capture_code_manifest(tmp_path)
    assert changed_again.dirty_patch_hash != dirty.dirty_patch_hash


def test_attempt_builder_binds_config_code_and_data_hashes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repository(repo)
    policy = _policy()
    config = load_config("genetic_algorithm/config/presets/safe_v2.yaml")
    config["backtesting"].update(
        {
            "exchange": "binance",
            "dataformat_ohlcv": "feather",
            "pairs": ["BTC/USDT"],
            "timerange": "20240101-20250101",
            "timeframe": "1h",
        }
    )
    config["promotion_v2"] = {
        "enabled": True,
        **policy.model_dump(mode="json"),
    }
    code_manifest = capture_code_manifest(repo)
    data_root = tmp_path / "ohlcv"
    data_root.mkdir()
    close = [100.0 + index for index in range(48)]
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC"),
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [10.0] * 48,
        }
    ).to_feather(data_root / "BTC_USDT-1h.feather")
    artifact_root = tmp_path / "attempts" / "attempt-001"

    bundle = build_attempt_manifest_bundle(
        attempt_id="attempt-001",
        wave_id="wave-001",
        experiment_id="control-001",
        created_at=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        resolved_config=config,
        policy=policy,
        fitness_policy_version="fitness-v2-test",
        seeds=[42],
        worker_count=1,
        artifact_root=artifact_root,
        repo_root=repo,
        data_root=data_root,
    )

    assert bundle.manifest.code_version == code_manifest.code_version
    assert bundle.manifest.dirty_patch_hash is None
    assert bundle.manifest.data_manifest_hash == bundle.data_manifest.manifest_hash
    assert bundle.manifest.split_manifest_hash == bundle.split_manifest.split_hash
    assert bundle.manifest.resolved_config_path == str(
        artifact_root.resolve() / "resolved_config.yaml"
    )


def test_attempt_builder_rejects_invalid_v2_config_before_hashing(tmp_path: Path):
    policy = _policy()
    config = load_config("genetic_algorithm/config/presets/safe_v2.yaml")
    config["promotion_v2"] = {
        "enabled": True,
        **policy.model_dump(mode="json"),
    }
    config["genetic_algorithm"]["mutaton_rate"] = 0.2

    with pytest.raises(ValueError, match=r"genetic_algorithm\.mutaton_rate"):
        build_attempt_manifest_bundle(
            attempt_id="attempt-invalid-config",
            wave_id="wave-invalid-config",
            experiment_id="experiment-invalid-config",
            created_at=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
            resolved_config=config,
            policy=policy,
            fitness_policy_version="fitness-v2-test",
            seeds=[42],
            worker_count=1,
            artifact_root=tmp_path / "attempts" / "attempt-invalid-config",
            repo_root=tmp_path / "unreached-repo",
            data_root=tmp_path / "unreached-data",
        )
