"""Transactional one-time-use ledger for blind FINAL_TEST market-data cells."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.data_manifest_v2 import DataManifestV2
from genetic_algorithm.orchestration.promotion_policy_v2 import ShadowGatePolicyV2
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    ScenarioRole,
    StrictV2Model,
)


class FinalTestLedgerError(ValueError):
    """Raised when FINAL_TEST usage cannot be recorded safely."""


class FinalTestReuseError(FinalTestLedgerError):
    """Raised when a blind data cell was already claimed by another attempt."""


class FinalTestCellV2(StrictV2Model):
    """One exchange x pair x timeframe x period blindness boundary."""

    exchange: str = Field(min_length=1)
    pair: str = Field(min_length=3)
    timeframe: str = Field(min_length=1)
    period_start: str = Field(min_length=10, max_length=10)
    period_end: str = Field(min_length=10, max_length=10)
    expected_candles: int = Field(ge=1)
    segment_sha256: str = Field(min_length=64, max_length=64)
    scenario_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _sorted_scenarios(self) -> FinalTestCellV2:
        if self.scenario_ids != sorted(set(self.scenario_ids)):
            raise ValueError("scenario_ids must be sorted and unique")
        return self

    @property
    def cell_hash(self) -> str:
        # Data hash, timeframe and cost/scenario IDs are deliberately excluded. A data
        # correction, resampling, renamed scenario, or different cost multiplier must
        # not make an already seen underlying market period blind again.
        return canonical_config_hash(
            {
                "exchange": self.exchange,
                "pair": self.pair,
                "period_start": self.period_start,
                "period_end": self.period_end,
            }
        )


class FinalTestReservationV2(StrictV2Model):
    schema_version: str = "2.0"
    reservation_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    wave_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    data_manifest_hash: str = Field(min_length=8)
    panel_hash: str = Field(min_length=64, max_length=64)
    status: Literal["RESERVED", "EXPOSED", "RELEASED"]
    reserved_at: datetime
    exposed_at: datetime | None = None
    released_at: datetime | None = None
    cells: list[FinalTestCellV2] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_status(self) -> FinalTestReservationV2:
        hashes = [cell.cell_hash for cell in self.cells]
        if hashes != sorted(set(hashes)):
            raise ValueError("final-test cells must be sorted and unique")
        if self.panel_hash != canonical_config_hash({"cell_hashes": hashes}):
            raise ValueError("panel_hash differs from final-test cells")
        if self.reserved_at.utcoffset() is None:
            raise ValueError("reserved_at must be timezone-aware")
        if self.status == "RESERVED" and (self.exposed_at or self.released_at):
            raise ValueError("RESERVED usage cannot have terminal timestamps")
        if self.status == "EXPOSED" and (self.exposed_at is None or self.released_at is not None):
            raise ValueError("EXPOSED usage requires only exposed_at")
        if self.status == "RELEASED" and (self.released_at is None or self.exposed_at is not None):
            raise ValueError("RELEASED usage requires only released_at")
        for value in (self.exposed_at, self.released_at):
            if value is not None:
                if value.utcoffset() is None:
                    raise ValueError("usage timestamps must be timezone-aware")
                if value < self.reserved_at:
                    raise ValueError("usage timestamp precedes reservation")
        return self


def _iso(value: datetime) -> str:
    if value.utcoffset() is None:
        raise FinalTestLedgerError("ledger timestamps must be timezone-aware")
    return value.isoformat()


def _final_test_cells(
    policy: ShadowGatePolicyV2,
    data_manifest: DataManifestV2,
) -> list[FinalTestCellV2]:
    coverage_by_id = {item.scenario_id: item for item in data_manifest.scenario_coverage}
    grouped: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    final_requirements = [
        item for item in policy.required_scenarios if item.role == ScenarioRole.FINAL_TEST
    ]
    if not final_requirements:
        raise FinalTestLedgerError("policy has no FINAL_TEST scenarios to reserve")

    for requirement in final_requirements:
        coverage = coverage_by_id.get(requirement.scenario_id)
        if coverage is None:
            raise FinalTestLedgerError(
                f"FINAL_TEST scenario lacks data coverage: {requirement.scenario_id}"
            )
        expected_identity = (
            requirement.pair,
            requirement.timeframe,
            requirement.period_start,
            requirement.period_end,
        )
        actual_identity = (
            coverage.pair,
            coverage.timeframe,
            coverage.period_start,
            coverage.period_end,
        )
        if actual_identity != expected_identity:
            raise FinalTestLedgerError(
                f"data coverage differs from FINAL_TEST scenario: {requirement.scenario_id}"
            )
        key = (
            data_manifest.exchange,
            requirement.pair,
            requirement.timeframe,
            str(requirement.period_start),
            str(requirement.period_end),
        )
        current = grouped.get(key)
        if current is None:
            grouped[key] = {
                "expected_candles": coverage.expected_candles,
                "segment_sha256": coverage.segment_sha256,
                "scenario_ids": [requirement.scenario_id],
            }
        else:
            if (
                current["expected_candles"] != coverage.expected_candles
                or current["segment_sha256"] != coverage.segment_sha256
            ):
                raise FinalTestLedgerError(
                    "duplicate FINAL_TEST cell has inconsistent data coverage"
                )
            scenario_ids = cast(list[str], current["scenario_ids"])
            scenario_ids.append(requirement.scenario_id)

    cells = []
    for key, evidence in grouped.items():
        exchange, pair, timeframe, period_start, period_end = key
        scenario_ids = cast(list[str], evidence["scenario_ids"])
        cells.append(
            FinalTestCellV2(
                exchange=exchange,
                pair=pair,
                timeframe=timeframe,
                period_start=period_start,
                period_end=period_end,
                expected_candles=int(evidence["expected_candles"]),
                segment_sha256=str(evidence["segment_sha256"]),
                scenario_ids=sorted(scenario_ids),
            )
        )
    return sorted(cells, key=lambda cell: cell.cell_hash)


class FinalTestUsageLedgerV2:
    """SQLite/WAL ledger whose cell claims prevent overlapping blind-test reuse."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 30_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = busy_timeout_ms
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, self.SCHEMA_VERSION}:
                raise FinalTestLedgerError(
                    f"unsupported final-test ledger schema version: {version}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id TEXT PRIMARY KEY,
                    attempt_id TEXT NOT NULL UNIQUE,
                    wave_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    data_manifest_hash TEXT NOT NULL,
                    panel_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('RESERVED', 'EXPOSED', 'RELEASED')),
                    reserved_at TEXT NOT NULL,
                    exposed_at TEXT,
                    released_at TEXT
                );
                CREATE TABLE IF NOT EXISTS reservation_cells (
                    reservation_id TEXT NOT NULL REFERENCES reservations(reservation_id),
                    cell_hash TEXT NOT NULL,
                    exchange_name TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    expected_candles INTEGER NOT NULL,
                    segment_sha256 TEXT NOT NULL,
                    scenario_ids_json TEXT NOT NULL,
                    PRIMARY KEY (reservation_id, cell_hash)
                );
                CREATE TABLE IF NOT EXISTS cell_claims (
                    cell_hash TEXT PRIMARY KEY,
                    reservation_id TEXT NOT NULL REFERENCES reservations(reservation_id)
                );
                CREATE INDEX IF NOT EXISTS idx_reservation_cells_hash
                    ON reservation_cells(cell_hash);
                """
            )
            connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        began = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            began = True
            yield connection
            connection.execute("COMMIT")
        except Exception:
            if began:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def reserve(
        self,
        *,
        manifest: AttemptManifestV2,
        policy: ShadowGatePolicyV2,
        data_manifest: DataManifestV2,
        reserved_at: datetime,
    ) -> FinalTestReservationV2:
        """Atomically claim every FINAL_TEST cell, or claim none."""

        if data_manifest.manifest_hash != manifest.data_manifest_hash:
            raise FinalTestLedgerError("data manifest differs from attempt manifest")
        cells = _final_test_cells(policy, data_manifest)
        panel_hash = canonical_config_hash({"cell_hashes": [cell.cell_hash for cell in cells]})
        reserved_at_iso = _iso(reserved_at)
        reservation_id = manifest.attempt_id

        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM reservations WHERE attempt_id = ?",
                (manifest.attempt_id,),
            ).fetchone()
            if existing is not None:
                receipt = self._read_receipt(connection, reservation_id)
                expected = (
                    manifest.wave_id,
                    policy.policy_version,
                    manifest.data_manifest_hash,
                    panel_hash,
                )
                actual = (
                    receipt.wave_id,
                    receipt.policy_version,
                    receipt.data_manifest_hash,
                    receipt.panel_hash,
                )
                if actual != expected or receipt.cells != cells:
                    raise FinalTestLedgerError(
                        "attempt_id already has a different final-test reservation"
                    )
                if receipt.status == "RELEASED":
                    placeholders = ",".join("?" for _ in cells)
                    conflicts = connection.execute(
                        f"""
                        SELECT r.attempt_id
                        FROM cell_claims c
                        JOIN reservations r ON r.reservation_id = c.reservation_id
                        WHERE c.cell_hash IN ({placeholders})
                        ORDER BY c.cell_hash
                        """,
                        tuple(cell.cell_hash for cell in cells),
                    ).fetchall()
                    if conflicts:
                        owners = sorted({row["attempt_id"] for row in conflicts})
                        raise FinalTestReuseError(
                            "FINAL_TEST market-data cell was claimed after release by "
                            + ", ".join(owners)
                        )
                    connection.execute(
                        """
                        UPDATE reservations
                        SET status = 'RESERVED', reserved_at = ?, released_at = NULL
                        WHERE reservation_id = ?
                        """,
                        (reserved_at_iso, reservation_id),
                    )
                    for cell in cells:
                        connection.execute(
                            "INSERT INTO cell_claims (cell_hash, reservation_id) VALUES (?, ?)",
                            (cell.cell_hash, reservation_id),
                        )
                    return self._read_receipt(connection, reservation_id)
                return receipt

            placeholders = ",".join("?" for _ in cells)
            conflicts = connection.execute(
                f"""
                SELECT c.cell_hash, c.reservation_id, r.attempt_id
                FROM cell_claims c
                JOIN reservations r ON r.reservation_id = c.reservation_id
                WHERE c.cell_hash IN ({placeholders})
                ORDER BY c.cell_hash
                """,
                tuple(cell.cell_hash for cell in cells),
            ).fetchall()
            if conflicts:
                owners = sorted({row["attempt_id"] for row in conflicts})
                raise FinalTestReuseError(
                    "FINAL_TEST market-data cell was already reserved/exposed by "
                    + ", ".join(owners)
                )

            connection.execute(
                """
                INSERT INTO reservations (
                    reservation_id, attempt_id, wave_id, policy_version,
                    data_manifest_hash, panel_hash, status, reserved_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'RESERVED', ?)
                """,
                (
                    reservation_id,
                    manifest.attempt_id,
                    manifest.wave_id,
                    policy.policy_version,
                    manifest.data_manifest_hash,
                    panel_hash,
                    reserved_at_iso,
                ),
            )
            for cell in cells:
                connection.execute(
                    """
                    INSERT INTO reservation_cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reservation_id,
                        cell.cell_hash,
                        cell.exchange,
                        cell.pair,
                        cell.timeframe,
                        cell.period_start,
                        cell.period_end,
                        cell.expected_candles,
                        cell.segment_sha256,
                        json.dumps(cell.scenario_ids, separators=(",", ":")),
                    ),
                )
                connection.execute(
                    "INSERT INTO cell_claims (cell_hash, reservation_id) VALUES (?, ?)",
                    (cell.cell_hash, reservation_id),
                )
            return self._read_receipt(connection, reservation_id)

    def mark_exposed(
        self,
        reservation_id: str,
        *,
        exposed_at: datetime,
    ) -> FinalTestReservationV2:
        """Permanently burn the whole panel before the first FINAL_TEST backtest."""

        exposed_at_iso = _iso(exposed_at)
        with self._transaction() as connection:
            receipt = self._read_receipt(connection, reservation_id)
            if receipt.status == "RELEASED":
                raise FinalTestLedgerError("released FINAL_TEST reservation cannot be exposed")
            if receipt.status == "RESERVED":
                if exposed_at < receipt.reserved_at:
                    raise FinalTestLedgerError("exposed_at precedes reservation")
                connection.execute(
                    """
                    UPDATE reservations
                    SET status = 'EXPOSED', exposed_at = ?
                    WHERE reservation_id = ?
                    """,
                    (exposed_at_iso, reservation_id),
                )
            return self._read_receipt(connection, reservation_id)

    def release_unexposed(
        self,
        reservation_id: str,
        *,
        released_at: datetime,
    ) -> FinalTestReservationV2:
        """Release only a panel that was never exposed to a backtest."""

        released_at_iso = _iso(released_at)
        with self._transaction() as connection:
            receipt = self._read_receipt(connection, reservation_id)
            if receipt.status == "EXPOSED":
                raise FinalTestLedgerError("exposed FINAL_TEST reservation can never be released")
            if receipt.status == "RESERVED":
                if released_at < receipt.reserved_at:
                    raise FinalTestLedgerError("released_at precedes reservation")
                connection.execute(
                    "DELETE FROM cell_claims WHERE reservation_id = ?",
                    (reservation_id,),
                )
                connection.execute(
                    """
                    UPDATE reservations
                    SET status = 'RELEASED', released_at = ?
                    WHERE reservation_id = ?
                    """,
                    (released_at_iso, reservation_id),
                )
            return self._read_receipt(connection, reservation_id)

    def get(self, reservation_id: str) -> FinalTestReservationV2:
        connection = self._connect()
        try:
            return self._read_receipt(connection, reservation_id)
        finally:
            connection.close()

    @staticmethod
    def _read_receipt(
        connection: sqlite3.Connection,
        reservation_id: str,
    ) -> FinalTestReservationV2:
        row = connection.execute(
            "SELECT * FROM reservations WHERE reservation_id = ?",
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise FinalTestLedgerError(f"unknown FINAL_TEST reservation: {reservation_id}")
        cell_rows = connection.execute(
            """
            SELECT * FROM reservation_cells
            WHERE reservation_id = ?
            ORDER BY cell_hash
            """,
            (reservation_id,),
        ).fetchall()
        cells = [
            FinalTestCellV2(
                exchange=cell["exchange_name"],
                pair=cell["pair"],
                timeframe=cell["timeframe"],
                period_start=cell["period_start"],
                period_end=cell["period_end"],
                expected_candles=cell["expected_candles"],
                segment_sha256=cell["segment_sha256"],
                scenario_ids=json.loads(cell["scenario_ids_json"]),
            )
            for cell in cell_rows
        ]
        return FinalTestReservationV2(
            reservation_id=row["reservation_id"],
            attempt_id=row["attempt_id"],
            wave_id=row["wave_id"],
            policy_version=row["policy_version"],
            data_manifest_hash=row["data_manifest_hash"],
            panel_hash=row["panel_hash"],
            status=row["status"],
            reserved_at=datetime.fromisoformat(row["reserved_at"]),
            exposed_at=(datetime.fromisoformat(row["exposed_at"]) if row["exposed_at"] else None),
            released_at=(
                datetime.fromisoformat(row["released_at"]) if row["released_at"] else None
            ),
            cells=cells,
        )
