"""Tests for transactional, one-time FINAL_TEST data-cell usage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from genetic_algorithm.orchestration.data_manifest_v2 import (
    DataManifestV2,
    OHLCVFileV2,
    ScenarioDataCoverageV2,
)
from genetic_algorithm.orchestration.final_test_ledger_v2 import (
    FinalTestLedgerError,
    FinalTestReuseError,
    FinalTestUsageLedgerV2,
)
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ScenarioRequirementV2,
    ShadowGatePolicyV2,
)
from genetic_algorithm.orchestration.result_contract import AttemptManifestV2


NOW = datetime(2026, 7, 21, 17, 0, tzinfo=UTC)


def _policy(
    *,
    scenario_ids: tuple[str, ...] = ("final-btc",),
    cost_multiplier: float = 1.0,
) -> ShadowGatePolicyV2:
    return ShadowGatePolicyV2(
        policy_version="final-ledger-test",
        required_scenarios=[
            ScenarioRequirementV2(
                scenario_id=scenario_id,
                pair="BTC/USDT",
                timeframe="1h",
                role="FINAL_TEST",
                period_start=date(2025, 1, 1),
                period_end=date(2025, 1, 2),
                cost_multiplier=cost_multiplier + index,
            )
            for index, scenario_id in enumerate(scenario_ids)
        ],
    )


def _data_manifest(
    scenario_ids: tuple[str, ...] = ("final-btc",),
    *,
    segment_hash: str = "2" * 64,
) -> DataManifestV2:
    return DataManifestV2(
        exchange="binance",
        files=[
            OHLCVFileV2(
                pair="BTC/USDT",
                timeframe="1h",
                relative_path="BTC_USDT-1h.feather",
                sha256="1" * 64,
                size_bytes=100,
                candle_count=48,
                first_candle=datetime(2025, 1, 1, tzinfo=UTC),
                last_candle=datetime(2025, 1, 2, 23, tzinfo=UTC),
            )
        ],
        scenario_coverage=[
            ScenarioDataCoverageV2(
                scenario_id=scenario_id,
                pair="BTC/USDT",
                timeframe="1h",
                period_start=date(2025, 1, 1),
                period_end=date(2025, 1, 2),
                expected_candles=48,
                actual_candles=48,
                first_candle=datetime(2025, 1, 1, tzinfo=UTC),
                last_candle=datetime(2025, 1, 2, 23, tzinfo=UTC),
                segment_sha256=segment_hash,
            )
            for scenario_id in scenario_ids
        ],
    )


def _manifest(attempt_id: str, data_manifest: DataManifestV2) -> AttemptManifestV2:
    root = Path("/tmp/final-ledger-tests") / attempt_id
    return AttemptManifestV2(
        attempt_id=attempt_id,
        wave_id=f"wave-{attempt_id}",
        experiment_id="experiment-final-ledger",
        created_at=NOW,
        config_hash="c" * 64,
        code_version="commit-final-ledger",
        data_manifest_hash=data_manifest.manifest_hash,
        fitness_policy_version="fitness-v2-test",
        seeds=[42],
        worker_count=1,
        resolved_config_path=str(root / "resolved_config.yaml"),
        artifact_root=str(root),
    )


def test_reservation_is_idempotent_for_same_attempt(tmp_path: Path):
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")
    policy = _policy(scenario_ids=("final-1x", "final-2x"))
    data = _data_manifest(("final-1x", "final-2x"))
    manifest = _manifest("attempt-001", data)

    first = ledger.reserve(
        manifest=manifest,
        policy=policy,
        data_manifest=data,
        reserved_at=NOW,
    )
    second = ledger.reserve(
        manifest=manifest,
        policy=policy,
        data_manifest=data,
        reserved_at=NOW + timedelta(minutes=1),
    )

    assert first == second
    assert first.status == "RESERVED"
    assert len(first.cells) == 1
    assert first.cells[0].scenario_ids == ["final-1x", "final-2x"]


def test_renaming_scenario_changing_cost_or_data_hash_cannot_bypass_claim(tmp_path: Path):
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")
    first_policy = _policy()
    first_data = _data_manifest()
    ledger.reserve(
        manifest=_manifest("attempt-first", first_data),
        policy=first_policy,
        data_manifest=first_data,
        reserved_at=NOW,
    )

    changed_policy = _policy(scenario_ids=("renamed-final",), cost_multiplier=4.0)
    changed_data = _data_manifest(("renamed-final",), segment_hash="9" * 64)
    with pytest.raises(FinalTestReuseError, match="attempt-first"):
        ledger.reserve(
            manifest=_manifest("attempt-second", changed_data),
            policy=changed_policy,
            data_manifest=changed_data,
            reserved_at=NOW,
        )


def test_unexposed_reservation_can_be_released(tmp_path: Path):
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")
    policy = _policy()
    data = _data_manifest()
    first = ledger.reserve(
        manifest=_manifest("attempt-first", data),
        policy=policy,
        data_manifest=data,
        reserved_at=NOW,
    )

    released = ledger.release_unexposed(
        first.reservation_id,
        released_at=NOW + timedelta(minutes=1),
    )
    reopened = ledger.reserve(
        manifest=_manifest("attempt-first", data),
        policy=policy,
        data_manifest=data,
        reserved_at=NOW + timedelta(minutes=2),
    )
    ledger.release_unexposed(
        reopened.reservation_id,
        released_at=NOW + timedelta(minutes=3),
    )
    second = ledger.reserve(
        manifest=_manifest("attempt-second", data),
        policy=policy,
        data_manifest=data,
        reserved_at=NOW + timedelta(minutes=4),
    )

    assert released.status == "RELEASED"
    assert reopened.status == "RESERVED"
    assert second.status == "RESERVED"
    with pytest.raises(FinalTestReuseError, match="attempt-second"):
        ledger.reserve(
            manifest=_manifest("attempt-first", data),
            policy=policy,
            data_manifest=data,
            reserved_at=NOW + timedelta(minutes=5),
        )


def test_exposure_permanently_burns_entire_panel(tmp_path: Path):
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")
    policy = _policy()
    data = _data_manifest()
    reservation = ledger.reserve(
        manifest=_manifest("attempt-first", data),
        policy=policy,
        data_manifest=data,
        reserved_at=NOW,
    )
    exposed = ledger.mark_exposed(
        reservation.reservation_id,
        exposed_at=NOW + timedelta(minutes=1),
    )

    assert exposed.status == "EXPOSED"
    assert (
        ledger.mark_exposed(
            reservation.reservation_id,
            exposed_at=NOW + timedelta(minutes=2),
        )
        == exposed
    )
    with pytest.raises(FinalTestLedgerError, match="can never be released"):
        ledger.release_unexposed(
            reservation.reservation_id,
            released_at=NOW + timedelta(minutes=2),
        )
    with pytest.raises(FinalTestReuseError):
        ledger.reserve(
            manifest=_manifest("attempt-second", data),
            policy=policy,
            data_manifest=data,
            reserved_at=NOW + timedelta(minutes=2),
        )


def test_parallel_reservations_have_exactly_one_winner(tmp_path: Path):
    ledger = FinalTestUsageLedgerV2(tmp_path / "final_test.sqlite3")
    policy = _policy()
    data = _data_manifest()

    def reserve(index: int) -> str:
        try:
            ledger.reserve(
                manifest=_manifest(f"attempt-{index:02d}", data),
                policy=policy,
                data_manifest=data,
                reserved_at=NOW,
            )
        except FinalTestReuseError:
            return "BLOCKED"
        return "RESERVED"

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(reserve, range(24)))

    assert outcomes.count("RESERVED") == 1
    assert outcomes.count("BLOCKED") == 23
