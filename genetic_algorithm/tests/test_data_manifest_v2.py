"""Tests for exact, content-addressed OHLCV replay manifests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from genetic_algorithm.orchestration.data_manifest_v2 import (
    DataManifestError,
    build_data_manifest,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
)


def _policy() -> ShadowGatePolicyV2:
    return ShadowGatePolicyV2(
        policy_version="data-manifest-test",
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


def _config() -> dict:
    return {
        "backtesting": {
            "exchange": "binance",
            "dataformat_ohlcv": "feather",
        }
    }


def _candles() -> pd.DataFrame:
    close = [100.0 + index for index in range(48)]
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC"),
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [10.0] * 48,
        }
    )


def _write(frame: pd.DataFrame, root: Path) -> None:
    frame.to_feather(root / "BTC_USDT-1h.feather")


def test_manifest_is_deterministic_and_proves_exact_coverage(tmp_path: Path):
    _write(_candles(), tmp_path)

    first = build_data_manifest(_config(), _policy(), data_root=tmp_path)
    second = build_data_manifest(_config(), _policy(), data_root=tmp_path)

    assert first == second
    assert len(first.manifest_hash) == 64
    assert first.files[0].candle_count == 48
    assert first.scenario_coverage[0].expected_candles == 48
    assert first.scenario_coverage[0].actual_candles == 48


def test_manifest_hash_changes_when_one_candle_changes(tmp_path: Path):
    frame = _candles()
    _write(frame, tmp_path)
    before = build_data_manifest(_config(), _policy(), data_root=tmp_path)

    frame.loc[12, "close"] += 0.25
    _write(frame, tmp_path)
    after = build_data_manifest(_config(), _policy(), data_root=tmp_path)

    assert before.files[0].sha256 != after.files[0].sha256
    assert before.scenario_coverage[0].segment_sha256 != after.scenario_coverage[0].segment_sha256
    assert before.manifest_hash != after.manifest_hash


def test_manifest_rejects_missing_candle(tmp_path: Path):
    _write(_candles().drop(index=12).reset_index(drop=True), tmp_path)

    with pytest.raises(DataManifestError, match="47/48 candles"):
        build_data_manifest(_config(), _policy(), data_root=tmp_path)


def test_manifest_rejects_invalid_ohlcv(tmp_path: Path):
    frame = _candles()
    frame.loc[12, "high"] = frame.loc[12, "low"] - 1.0
    _write(frame, tmp_path)

    with pytest.raises(DataManifestError, match="high-price invariants"):
        build_data_manifest(_config(), _policy(), data_root=tmp_path)
