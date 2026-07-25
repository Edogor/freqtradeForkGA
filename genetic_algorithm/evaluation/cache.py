"""BacktestCache — disk-backed LRU cache for backtest results.

Extracted from :mod:`evaluation.direct_backtester` so it can be reused
independently (e.g. by parallel workers, CLI data commands).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from genetic_algorithm.evaluation.direct_backtester import BacktestResult

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 9
_CACHE_SCHEMA_KEY = "_cache_schema_version"


def _json_default(value: Any) -> Any:
    """Convert common scientific/Pandas scalar values to stable JSON values."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class BacktestCache:
    """Cache for backtest results with LRU disk eviction to prevent unbounded growth."""

    def __init__(self, cache_dir: Optional[Path] = None, max_disk_mb: int = 5000,
                 max_memory_entries: int = 300):
        """
        Args:
            cache_dir: Directory to store cache files.
            max_disk_mb: Maximum disk cache size in MB (default 5 GB).
                         When exceeded, oldest files are evicted.
            max_memory_entries: Maximum number of results kept in RAM.
                                When exceeded, least-recently-used entries are
                                dropped (they remain on disk). Default 300.
        """
        self.cache_dir = cache_dir or Path("genetic_algorithm/data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: OrderedDict[str, "BacktestResult"] = OrderedDict()
        self.max_memory_entries = max_memory_entries
        self.max_disk_bytes = max_disk_mb * 1024 * 1024

    # -- key generation ------------------------------------------------------

    def _get_cache_key(self, strategy_code: str, config: Dict[str, Any]) -> str:
        """Generate a deterministic cache key from strategy code and config."""
        config_json = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        cache_input = f"v{CACHE_SCHEMA_VERSION}_{strategy_code}_{config_json}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _delete_disk_entry(self, cache_key: str) -> None:
        """Remove a cache payload and its checksum together."""
        (self.cache_dir / f"{cache_key}.json").unlink(missing_ok=True)
        (self.cache_dir / f"{cache_key}.sha256").unlink(missing_ok=True)

    # -- get / put -----------------------------------------------------------

    def _memory_put(self, cache_key: str, result: "BacktestResult") -> None:
        """Insert into memory cache, evicting the LRU entry when over the limit."""
        self.cache[cache_key] = result
        self.cache.move_to_end(cache_key)
        while len(self.cache) > self.max_memory_entries:
            evicted_key, _ = self.cache.popitem(last=False)
            logger.debug(f"[CACHE] Evicted LRU memory entry: {evicted_key[:8]}...")

    def get(self, strategy_code: str, config: Dict[str, Any]) -> Optional["BacktestResult"]:
        """Return cached result or *None*."""
        cache_key = self._get_cache_key(strategy_code, config)

        # Memory cache — move to end to mark as recently used
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            logger.debug(f"Cache hit (memory): {cache_key[:8]}...")
            return self.cache[cache_key]

        # Disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                from genetic_algorithm.evaluation.direct_backtester import BacktestResult

                raw_bytes = cache_file.read_bytes()

                # Verify checksum if present
                checksum_file = self.cache_dir / f"{cache_key}.sha256"
                if checksum_file.exists():
                    expected = checksum_file.read_text().strip()
                    actual = hashlib.sha256(raw_bytes).hexdigest()
                    if expected != actual:
                        logger.warning(
                            f"Cache corruption detected for {cache_key[:8]}..., "
                            f"removing entry"
                        )
                        self._delete_disk_entry(cache_key)
                        return None

                data = json.loads(raw_bytes)
                if not isinstance(data, dict):
                    raise ValueError("cache payload must be a JSON object")
                schema_version = data.pop(_CACHE_SCHEMA_KEY, None)
                if schema_version != CACHE_SCHEMA_VERSION:
                    logger.info(
                        "Ignoring incompatible cache entry %s... (schema %r, expected %d)",
                        cache_key[:8], schema_version, CACHE_SCHEMA_VERSION,
                    )
                    self._delete_disk_entry(cache_key)
                    return None
                result = BacktestResult(**data)
                self._memory_put(cache_key, result)
                logger.debug(f"Cache hit (disk): {cache_key[:8]}...")
                return result
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_key[:8]}...: {e}, removing")
                self._delete_disk_entry(cache_key)

        return None

    def put(
        self,
        strategy_code: str,
        config: Dict[str, Any],
        result: "BacktestResult",
    ) -> None:
        """Store *result* in both memory and disk caches."""
        cache_key = self._get_cache_key(strategy_code, config)

        # LRU-bounded memory cache
        self._memory_put(cache_key, result)

        # Disk (atomic write)
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".json",
                dir=str(self.cache_dir),
                prefix=f".{cache_key[:16]}_",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    payload = result.to_dict()
                    payload[_CACHE_SCHEMA_KEY] = CACHE_SCHEMA_VERSION
                    json.dump(
                        payload,
                        f,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=_json_default,
                    )
                os.replace(tmp_path, str(cache_file))  # atomic on POSIX

                # Write checksum for corruption detection
                content_hash = hashlib.sha256(cache_file.read_bytes()).hexdigest()
                checksum_file = self.cache_dir / f"{cache_key}.sha256"
                checksum_file.write_text(content_hash)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            logger.debug(f"Cached result: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")

        self._maybe_evict()

    # -- eviction ------------------------------------------------------------

    def _maybe_evict(self) -> None:
        """Remove oldest cache files when total size exceeds *max_disk_bytes*."""
        try:
            cache_files = sorted(
                self.cache_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            total_size = sum(f.stat().st_size for f in cache_files)
            removed = 0
            while total_size > self.max_disk_bytes and cache_files:
                oldest = cache_files.pop(0)
                fsize = oldest.stat().st_size
                cache_key = oldest.stem
                self._delete_disk_entry(cache_key)
                total_size -= fsize
                removed += 1
            if removed:
                logger.info(
                    f"[CACHE] Evicted {removed} old cache files "
                    f"(limit: {self.max_disk_bytes / (1024**2):.0f} MB)"
                )
        except Exception as e:
            logger.debug(f"Cache eviction skipped: {e}")
