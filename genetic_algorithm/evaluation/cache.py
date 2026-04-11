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
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from genetic_algorithm.evaluation.direct_backtester import BacktestResult

logger = logging.getLogger(__name__)


class BacktestCache:
    """Cache for backtest results with LRU disk eviction to prevent unbounded growth."""

    def __init__(self, cache_dir: Optional[Path] = None, max_disk_mb: int = 5000):
        """
        Args:
            cache_dir: Directory to store cache files.
            max_disk_mb: Maximum disk cache size in MB (default 5 GB).
                         When exceeded, oldest files are evicted.
        """
        self.cache_dir = cache_dir or Path("genetic_algorithm/data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, "BacktestResult"] = {}
        self.max_disk_bytes = max_disk_mb * 1024 * 1024

    # -- key generation ------------------------------------------------------

    def _get_cache_key(self, strategy_code: str, config: Dict[str, Any]) -> str:
        """Generate a deterministic cache key from strategy code and config."""
        cache_input = f"{strategy_code}_{json.dumps(config, sort_keys=True)}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    # -- get / put -----------------------------------------------------------

    def get(self, strategy_code: str, config: Dict[str, Any]) -> Optional["BacktestResult"]:
        """Return cached result or *None*."""
        cache_key = self._get_cache_key(strategy_code, config)

        # Memory cache
        if cache_key in self.cache:
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
                        cache_file.unlink(missing_ok=True)
                        checksum_file.unlink(missing_ok=True)
                        return None

                data = json.loads(raw_bytes)
                result = BacktestResult(**data)
                self.cache[cache_key] = result
                logger.debug(f"Cache hit (disk): {cache_key[:8]}...")
                return result
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_key[:8]}...: {e}, removing")
                cache_file.unlink(missing_ok=True)
                checksum_path = self.cache_dir / f"{cache_key}.sha256"
                checksum_path.unlink(missing_ok=True)

        return None

    def put(
        self,
        strategy_code: str,
        config: Dict[str, Any],
        result: "BacktestResult",
    ) -> None:
        """Store *result* in both memory and disk caches."""
        cache_key = self._get_cache_key(strategy_code, config)

        # Memory
        self.cache[cache_key] = result

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
                    json.dump(result.to_dict(), f)
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
                oldest.unlink()
                total_size -= fsize
                removed += 1
            if removed:
                logger.info(
                    f"[CACHE] Evicted {removed} old cache files "
                    f"(limit: {self.max_disk_bytes / (1024**2):.0f} MB)"
                )
        except Exception as e:
            logger.debug(f"Cache eviction skipped: {e}")
