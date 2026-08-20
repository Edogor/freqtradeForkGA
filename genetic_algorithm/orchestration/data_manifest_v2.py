"""Content-addressed OHLCV manifests for deterministic GA V2 replays."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from freqtrade.misc import pair_to_filename
from genetic_algorithm.core.strategy_gene import timeframe_to_minutes
from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.promotion_policy_v2 import ShadowGatePolicyV2
from genetic_algorithm.orchestration.result_contract import StrictV2Model


class DataManifestError(ValueError):
    """Raised when market data cannot prove exact replay coverage."""


class OHLCVFileV2(StrictV2Model):
    pair: str = Field(min_length=3)
    timeframe: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    format: str = "feather"
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=1)
    candle_count: int = Field(ge=1)
    first_candle: datetime
    last_candle: datetime


class ScenarioDataCoverageV2(StrictV2Model):
    scenario_id: str = Field(min_length=1)
    pair: str = Field(min_length=3)
    timeframe: str = Field(min_length=1)
    period_start: date
    period_end: date
    expected_candles: int = Field(ge=1)
    actual_candles: int = Field(ge=1)
    first_candle: datetime
    last_candle: datetime
    segment_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _complete(self) -> ScenarioDataCoverageV2:
        if self.period_end <= self.period_start:
            raise ValueError("period_end must be after period_start")
        if self.actual_candles != self.expected_candles:
            raise ValueError("scenario candle coverage is incomplete")
        return self


class DataManifestV2(StrictV2Model):
    schema_version: str = "2.0"
    exchange: str = Field(min_length=1)
    data_format: str = "feather"
    files: list[OHLCVFileV2] = Field(min_length=1)
    scenario_coverage: list[ScenarioDataCoverageV2] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_keys(self) -> DataManifestV2:
        file_keys = [(item.pair, item.timeframe) for item in self.files]
        scenario_ids = [item.scenario_id for item in self.scenario_coverage]
        if len(file_keys) != len(set(file_keys)):
            raise ValueError("OHLCV file keys must be unique")
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario coverage IDs must be unique")
        return self

    @property
    def manifest_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))


def resolve_spot_data_root(config: dict, pairs: list[str]) -> Path:
    """Mirror DirectBacktester's spot datadir selection without fallback."""

    unit_pairs = [pair for pair in pairs if "UNITTEST" in pair]
    if unit_pairs and len(unit_pairs) != len(pairs):
        raise DataManifestError("UNITTEST and exchange pairs cannot share one replay manifest")
    if unit_pairs:
        return Path("tests/testdata")
    exchange = str(config.get("backtesting", {}).get("exchange", "binance"))
    return Path("user_data/data") / exchange


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _segment_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    timestamps = frame["date"].astype("int64").to_numpy(dtype="<i8", copy=True)
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype="<f8", copy=True)
    digest.update(timestamps.tobytes(order="C"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _load_verified_feather(path: Path, pair: str, timeframe: str) -> pd.DataFrame:
    try:
        frame = pd.read_feather(path)
    except Exception as exc:
        raise DataManifestError(f"cannot read OHLCV file {path}") from exc
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataManifestError(f"{pair} {timeframe} missing columns: {', '.join(missing)}")
    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    if frame["date"].isna().any():
        raise DataManifestError(f"{pair} {timeframe} contains invalid timestamps")
    if frame["date"].duplicated().any():
        raise DataManifestError(f"{pair} {timeframe} contains duplicate timestamps")
    if not frame["date"].is_monotonic_increasing:
        raise DataManifestError(f"{pair} {timeframe} candles are not sorted")

    numeric = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise DataManifestError(f"{pair} {timeframe} contains non-finite OHLCV values")
    if (frame[["open", "high", "low", "close"]].to_numpy(dtype=float) <= 0).any():
        raise DataManifestError(f"{pair} {timeframe} contains non-positive prices")
    if (frame["volume"] < 0).any():
        raise DataManifestError(f"{pair} {timeframe} contains negative volume")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise DataManifestError(f"{pair} {timeframe} violates high-price invariants")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise DataManifestError(f"{pair} {timeframe} violates low-price invariants")
    return frame


def build_data_manifest(
    config: dict,
    policy: ShadowGatePolicyV2,
    *,
    data_root: str | Path | None = None,
) -> DataManifestV2:
    """Hash exact Feather files and prove every declared scenario is gap-free."""

    backtesting = config.get("backtesting", {})
    data_format = str(backtesting.get("dataformat_ohlcv", "feather"))
    if data_format != "feather":
        raise DataManifestError("V2 data manifests currently support feather OHLCV only")
    pairs = sorted({item.pair for item in policy.required_scenarios})
    root = Path(data_root) if data_root is not None else resolve_spot_data_root(config, pairs)
    if not root.is_dir():
        raise DataManifestError(f"OHLCV data root does not exist: {root}")

    exchange = str(backtesting.get("exchange", "binance"))
    files: list[OHLCVFileV2] = []
    coverage: list[ScenarioDataCoverageV2] = []
    loaded: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}

    for requirement in policy.required_scenarios:
        key = (requirement.pair, requirement.timeframe)
        if key not in loaded:
            filename = f"{pair_to_filename(requirement.pair)}-{requirement.timeframe}.feather"
            path = root / filename
            if not path.is_file():
                raise DataManifestError(f"required OHLCV file is missing: {path}")
            resolved = path.resolve()
            if root.resolve() not in resolved.parents:
                raise DataManifestError(f"OHLCV path escapes data root: {path}")
            frame = _load_verified_feather(path, *key)
            loaded[key] = (path, frame)
            files.append(
                OHLCVFileV2(
                    pair=requirement.pair,
                    timeframe=requirement.timeframe,
                    relative_path=path.relative_to(root).as_posix(),
                    sha256=_sha256_file(path),
                    size_bytes=path.stat().st_size,
                    candle_count=len(frame),
                    first_candle=frame["date"].iloc[0].to_pydatetime(),
                    last_candle=frame["date"].iloc[-1].to_pydatetime(),
                )
            )

        _, frame = loaded[key]
        minutes = timeframe_to_minutes(requirement.timeframe)
        if minutes <= 0:
            raise DataManifestError(f"unsupported timeframe: {requirement.timeframe}")
        interval = timedelta(minutes=minutes)
        start = datetime.combine(requirement.period_start, time.min, tzinfo=UTC)
        exclusive_end = datetime.combine(
            requirement.period_end + timedelta(days=1), time.min, tzinfo=UTC
        )
        period_seconds = (exclusive_end - start).total_seconds()
        interval_seconds = interval.total_seconds()
        if period_seconds % interval_seconds:
            raise DataManifestError("scenario period is not aligned to its timeframe")
        expected = int(period_seconds // interval_seconds)
        segment = frame[(frame["date"] >= start) & (frame["date"] < exclusive_end)]
        if len(segment) != expected:
            raise DataManifestError(
                f"{requirement.scenario_id} has {len(segment)}/{expected} candles"
            )
        expected_last = exclusive_end - interval
        actual_first = segment["date"].iloc[0].to_pydatetime()
        actual_last = segment["date"].iloc[-1].to_pydatetime()
        if actual_first != start or actual_last != expected_last:
            raise DataManifestError(
                f"{requirement.scenario_id} does not cover exact boundary candles"
            )
        deltas = segment["date"].diff().dropna()
        if not (deltas == interval).all():
            raise DataManifestError(f"{requirement.scenario_id} contains candle gaps")
        coverage.append(
            ScenarioDataCoverageV2(
                scenario_id=requirement.scenario_id,
                pair=requirement.pair,
                timeframe=requirement.timeframe,
                period_start=requirement.period_start,
                period_end=requirement.period_end,
                expected_candles=expected,
                actual_candles=len(segment),
                first_candle=actual_first,
                last_candle=actual_last,
                segment_sha256=_segment_hash(segment),
            )
        )

    return DataManifestV2(
        exchange=exchange,
        data_format=data_format,
        files=sorted(files, key=lambda item: (item.pair, item.timeframe)),
        scenario_coverage=sorted(coverage, key=lambda item: item.scenario_id),
    )
