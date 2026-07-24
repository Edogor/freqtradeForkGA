"""
Shared OHLCV Data Manager

Eliminates N× memory duplication across parallel worker processes by loading
OHLCV data once in the main process and sharing it via multiprocessing.shared_memory.

Architecture:
    Main process:
        1. Loads OHLCV DataFrames via FreqTrade data loading utilities
        2. Converts each pair's DataFrame to contiguous numpy arrays
        3. Creates SharedMemory blocks holding the raw array bytes
        4. Passes metadata (shm names, shapes, dtypes, columns) to workers via config

    Worker process:
        1. Attaches to existing SharedMemory blocks (zero-copy)
        2. Wraps numpy arrays pointing to shared buffer
        3. Pre-populates DirectBacktester._bt_data_cache with shared data
        4. .copy() is still called per-evaluation (indicators mutate DataFrames)
           but copies from shared memory, NOT from a per-worker duplicate

Memory savings:
    Before: 4 workers × ~1.5 GB OHLCV = ~6 GB
    After:  1 shared copy (~1.5 GB) + 4 × ~50 MB overhead = ~1.7 GB
"""

import atexit
import logging
import threading
import time
import uuid
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Module-level registry of SharedMemory objects for cleanup (main process: created blocks)
_shared_blocks: List[shared_memory.SharedMemory] = []
_shared_blocks_lock = threading.Lock()

# Worker-side: attached SharedMemory objects (kept alive for numpy view lifetime)
_worker_shm_blocks: List[shared_memory.SharedMemory] = []
_worker_shm_lock = threading.Lock()


def _cleanup_shared_memory():
    """Unlink all shared memory blocks created by this process (main only)."""
    with _shared_blocks_lock:
        for shm in _shared_blocks:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        _shared_blocks.clear()


def _cleanup_worker_shm():
    """Close (but don't unlink) shared memory attached by worker processes."""
    with _worker_shm_lock:
        for shm in _worker_shm_blocks:
            try:
                shm.close()
            except Exception:
                pass
        _worker_shm_blocks.clear()


atexit.register(_cleanup_shared_memory)
atexit.register(_cleanup_worker_shm)


def cleanup_stale_shared_memory():
    """Remove stale shared memory blocks from previous crashed GA runs.

    Scans /dev/shm for blocks matching the ``ga_ohlcv_`` prefix and unlinks
    any that don't belong to a currently running process.  Safe to call at
    startup before creating new blocks.
    """
    import pathlib
    shm_dir = pathlib.Path('/dev/shm')
    if not shm_dir.exists():
        return

    removed = 0
    for entry in shm_dir.iterdir():
        if entry.name.startswith('ga_ohlcv_'):
            try:
                old_shm = shared_memory.SharedMemory(name=entry.name, create=False)
                old_shm.close()
                old_shm.unlink()
                removed += 1
            except Exception:
                # Block may already be gone or in use
                pass
    if removed:
        logger.info(f"[SHARED] Cleaned up {removed} stale shared memory block(s) from /dev/shm")


# ─── Column layout for shared OHLCV arrays ───────────────────────────────────
# We store a fixed set of float64 columns plus date as int64 (epoch nanos).
# This is the raw OHLCV data that FreqTrade's data loading produces.
SHARED_FLOAT_COLS = ['open', 'high', 'low', 'close', 'volume']
DATE_COL = 'date'


def _resolve_data_dir(configured: str | Path) -> Path:
    """Resolve GA data paths independently from an attempt output cwd."""

    data_dir = Path(configured)
    if data_dir.is_absolute():
        return data_dir
    return Path(__file__).resolve().parents[2] / data_dir


class SharedDataManager:
    """
    Manages shared OHLCV data for parallel worker processes.

    Usage (main process):
        manager = SharedDataManager()
        manager.load_and_share(config)    # Loads data, creates shared memory
        metadata = manager.get_metadata()  # Pass this to workers via config
        ...
        manager.cleanup()                  # When done

    Usage (worker process):
        data = attach_shared_data(metadata)  # Returns Dict[str, DataFrame]
    """

    def __init__(self):
        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._metadata: Dict[str, Any] = {}
        self._loaded = False
        self._load_lock = threading.Lock()
        self._instance_id = uuid.uuid4().hex[:12]

    def load_and_share(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for all configured pairs and timeframes, then
        place it in shared memory.

        Args:
            config: GA config dict (must contain backtesting.pairs, backtesting.timerange, etc.)

        Returns:
            Dict[pair_name, DataFrame] — the loaded data (also now in shared memory)
        """
        with self._load_lock:
            if self._loaded:
                logger.warning("[SHARED] Data already loaded, skipping reload")
                return {}
            return self._load_and_share_locked(config)

    def _load_and_share_locked(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Internal load implementation called under lock.
        """

        bt_config = config.get('backtesting', {})
        pairs = bt_config.get('pairs', [])
        timerange_str = bt_config.get('timerange', '')
        configured_timeframes = list(
            config.get('strategy_constraints', {}).get('timeframes', ['5m'])
        )
        timeframes = list(dict.fromkeys(str(item) for item in configured_timeframes if item))

        if not pairs:
            logger.warning("[SHARED] No pairs configured, cannot load shared data")
            return {}
        if len(timeframes) != 1:
            logger.warning(
                "[SHARED] Shared memory requires exactly one strategy timeframe; "
                "configured=%s. Workers will load exact timeframe data from disk.",
                timeframes,
            )
            return {}
        timeframe = timeframes[0]

        logger.info(
            f"[SHARED] Loading {timeframe} OHLCV data for {len(pairs)} pairs, "
            f"timerange={timerange_str}"
        )
        start = time.time()

        # Use FreqTrade's data loading to get the same data workers would load
        data = self._load_ohlcv_data(config, pairs, timerange_str, timeframe)

        if not data:
            logger.warning("[SHARED] No data loaded — shared memory disabled")
            return {}

        # Convert each DataFrame to shared memory
        pair_metadata = {}
        for pair, df in data.items():
            shm_name, meta = self._dataframe_to_shared_memory(pair, df)
            pair_metadata[pair] = meta
            logger.debug(f"[SHARED] {pair}: {len(df)} rows, shm={shm_name}")

        self._metadata = {
            'pairs': pair_metadata,
            'timerange': timerange_str,
            'timeframe': timeframe,
            'loaded_at': time.time(),
        }
        self._loaded = True

        elapsed = time.time() - start
        total_rows = sum(len(df) for df in data.values())
        total_bytes = sum(m['nbytes'] for m in pair_metadata.values())
        logger.info(
            f"[SHARED] Loaded {len(data)} pairs ({total_rows:,} rows, "
            f"{total_bytes / 1024 / 1024:.1f} MB shared memory) in {elapsed:.1f}s"
        )
        return data

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata dict to pass to worker processes via config."""
        return dict(self._metadata)

    def cleanup(self):
        """Release all shared memory blocks."""
        for pair, shm in self._shm_blocks.items():
            try:
                shm.close()
                shm.unlink()
                logger.debug(f"[SHARED] Cleaned up shm for {pair}")
            except Exception as e:
                logger.debug(f"[SHARED] Cleanup error for {pair}: {e}")
            # Remove from module-level atexit list too
            with _shared_blocks_lock:
                if shm in _shared_blocks:
                    _shared_blocks.remove(shm)
        self._shm_blocks.clear()
        self._loaded = False

    def _load_ohlcv_data(
        self,
        config: Dict[str, Any],
        pairs: List[str],
        timerange_str: str,
        timeframe: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data using FreqTrade's data loading utilities.

        This mirrors what DirectBacktester does internally when it calls
        backtesting.load_bt_data(), but without creating a full Backtesting
        instance.
        """
        try:
            from freqtrade.configuration import TimeRange
            from freqtrade.data.history import load_pair_history

            bt_config = config.get('backtesting', {})
            data_dir = _resolve_data_dir(
                bt_config.get('datadir', 'user_data/data/binance')
            )
            dataformat = bt_config.get('dataformat_ohlcv', 'feather')

            # Parse timerange
            timerange = TimeRange.parse_timerange(timerange_str) if timerange_str else TimeRange()

            data = {}
            for pair in pairs:
                try:
                    df = load_pair_history(
                        pair=pair,
                        timeframe=timeframe,
                        datadir=data_dir,
                        timerange=timerange,
                        data_format=dataformat,
                    )
                    if df is not None and len(df) > 0:
                        data[pair] = df
                    else:
                        logger.warning(f"[SHARED] No data for {pair}")
                except Exception as e:
                    logger.error(f"[SHARED] Failed to load {pair}: {e}")

            return data

        except ImportError as e:
            logger.error(f"[SHARED] FreqTrade imports failed: {e}")
            return {}

    def _dataframe_to_shared_memory(
        self,
        pair: str,
        df: pd.DataFrame,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Convert a DataFrame to shared memory numpy arrays.

        Layout: one contiguous block containing:
            - date column as int64 (epoch nanoseconds)
            - OHLCV columns as float64

        Returns:
            (shm_name, metadata_dict)
        """
        n_rows = len(df)

        # Extract date as int64 nanoseconds
        date_values = df[DATE_COL].values.astype('datetime64[ns]').astype(np.int64)

        # Extract float columns — always include all SHARED_FLOAT_COLS for
        # deterministic layout; pad with zeros if a column is unexpectedly absent.
        float_arrays = []
        actual_float_cols = []
        for col in SHARED_FLOAT_COLS:
            if col in df.columns:
                float_arrays.append(df[col].values.astype(np.float64))
            else:
                logger.warning(f"[SHARED] Column '{col}' missing for {pair}, padding with zeros")
                float_arrays.append(np.zeros(n_rows, dtype=np.float64))
            actual_float_cols.append(col)

        # Calculate total size needed
        date_bytes = date_values.nbytes  # n_rows * 8
        float_bytes = sum(a.nbytes for a in float_arrays)  # n_rows * 8 * n_cols
        total_bytes = date_bytes + float_bytes

        # Create shared memory block with unique name (uuid avoids collisions)
        safe_pair = pair.replace('/', '_')
        shm_name = f"ga_ohlcv_{safe_pair}_{self._instance_id}"

        # Handle stale blocks from previous crashed runs
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=total_bytes)
        except FileExistsError:
            logger.warning(f"[SHARED] Stale block '{shm_name}' found, unlinking and recreating")
            try:
                old_shm = shared_memory.SharedMemory(name=shm_name, create=False)
                old_shm.close()
                old_shm.unlink()
            except Exception as e:
                logger.debug(f"[SHARED] Error unlinking stale block: {e}")
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=total_bytes)

        self._shm_blocks[pair] = shm
        with _shared_blocks_lock:
            _shared_blocks.append(shm)

        # Copy data into shared memory
        offset = 0
        buf = shm.buf

        # Date column
        np.copyto(
            np.ndarray(date_values.shape, dtype=np.int64, buffer=buf, offset=offset),
            date_values
        )
        offset += date_bytes

        # Float columns
        for arr in float_arrays:
            np.copyto(
                np.ndarray(arr.shape, dtype=np.float64, buffer=buf, offset=offset),
                arr
            )
            offset += arr.nbytes

        metadata = {
            'shm_name': shm_name,
            'n_rows': n_rows,
            'nbytes': total_bytes,
            'float_cols': actual_float_cols,
        }
        return shm_name, metadata


def attach_shared_data(metadata: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Attach to shared memory and reconstruct DataFrames (worker-side).

    The returned DataFrames' float columns are numpy views into shared memory —
    they are effectively READ-ONLY.  Callers MUST .copy() before modifying
    (e.g., adding indicator columns).  This is already done by the existing
    _bt_data_cache hit path in DirectBacktester.

    Args:
        metadata: Dict from SharedDataManager.get_metadata()

    Returns:
        Dict[pair, DataFrame] with OHLCV data backed by shared memory
    """
    if not metadata or 'pairs' not in metadata:
        return {}

    data = {}
    for pair, meta in metadata['pairs'].items():
        try:
            shm = shared_memory.SharedMemory(name=meta['shm_name'], create=False)
            with _worker_shm_lock:
                _worker_shm_blocks.append(shm)  # Keep alive for view lifetime

            n_rows = meta['n_rows']
            float_cols = meta['float_cols']

            offset = 0
            # Read date column — pd.to_datetime always creates a copy, fine
            date_arr = np.ndarray(
                (n_rows,), dtype=np.int64, buffer=shm.buf, offset=offset
            )
            offset += n_rows * 8

            # Read float columns as VIEWS into shared memory (NO copy)
            # These arrays share the underlying buffer with the main process.
            col_data = {'date': pd.to_datetime(date_arr, unit='ns', utc=True)}
            for col in float_cols:
                col_data[col] = np.ndarray(
                    (n_rows,), dtype=np.float64, buffer=shm.buf, offset=offset
                )
                offset += n_rows * 8

            df = pd.DataFrame(col_data)
            data[pair] = df

        except Exception as e:
            logger.error(f"[SHARED] Failed to attach {pair}: {e}")

    if data:
        logger.info(f"[SHARED-WORKER] Attached to {len(data)} pair(s) shared data")

    return data
