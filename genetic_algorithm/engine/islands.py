"""
IslandCoordinator: Unified entry point for island-model evolution.

Dispatches to the appropriate implementation based on configuration:

- **Regime-based** (``island_model.enabled: true``): Uses
  ``IslandModelEvolution`` — islands are specialized by market regime
  (bullish/bearish/sideways) with data segmentation.

- **Generic** (``generic_island_model.enabled: true``): Uses
  ``GenericIslandModelEvolution`` — islands are specialized by indicator
  pools, pair subsets, or seed diversity.

Usage::

    coordinator = IslandCoordinator.from_config("config.yaml")
    results = coordinator.evolve()

    # Or programmatically:
    coordinator = IslandCoordinator.from_config(
        "config.yaml", visualize=False, interactive=False,
    )
    results = coordinator.evolve()
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import yaml

from genetic_algorithm.engine.island_results import (
    IslandFinalistBatch,
    extract_island_finalists,
)

logger = logging.getLogger(__name__)


class IslandCoordinator:
    """Unified facade for island-model evolution.

    Wraps either ``IslandModelEvolution`` (regime-based) or
    ``GenericIslandModelEvolution`` (generic) based on which is enabled
    in the configuration file.

    The coordinator provides a single API regardless of which backend
    is active, and exposes the backend instance via :attr:`backend` for
    direct access when needed.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str,
        *,
        visualize: bool = False,
        interactive: bool = False,
    ) -> "IslandCoordinator":
        """Create an IslandCoordinator from a YAML config file.

        Examines the config to determine which island implementation to
        use, instantiates the appropriate backend, and wraps it.

        Raises:
            ValueError: If neither island model is enabled in config, or
                if both are enabled simultaneously.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        regime_enabled = config.get("island_model", {}).get("enabled", False)
        generic_enabled = config.get("generic_island_model", {}).get("enabled", False)

        if regime_enabled and generic_enabled:
            raise ValueError(
                "Both 'island_model' and 'generic_island_model' are enabled. "
                "Only one island model can be active at a time."
            )

        if regime_enabled:
            from genetic_algorithm.core.island_model import IslandModelEvolution

            backend = IslandModelEvolution(
                config_path, visualize=visualize, interactive=interactive,
            )
            logger.info("[ISLANDS] Using regime-based island model")
        elif generic_enabled:
            from genetic_algorithm.core.generic_island_model import (
                GenericIslandModelEvolution,
            )

            backend = GenericIslandModelEvolution(
                config_path, visualize=visualize, interactive=interactive,
            )
            logger.info("[ISLANDS] Using generic island model")
        else:
            raise ValueError(
                "No island model enabled in config. Set either "
                "'island_model.enabled: true' or "
                "'generic_island_model.enabled: true'."
            )

        return cls(backend)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend(self) -> Any:
        """Access the underlying island model implementation."""
        return self._backend

    @property
    def island_names(self) -> list:
        """List of island names."""
        if hasattr(self._backend, "island_configs"):
            return [ic.name for ic in self._backend.island_configs]
        return []

    @property
    def island_populations(self) -> dict:
        """Dict mapping island name → population."""
        return getattr(self._backend, "island_populations", {})

    def evolve(self) -> IslandFinalistBatch:
        """Run the full island-model evolution.

        Returns:
            A validated, flattened view whose finalists all come from one
            exact common replay panel.
        """
        raw = self._backend.evolve()
        return extract_island_finalists(
            raw,
            expected_islands=self.island_names,
            require_finalists=True,
        )

    def get_generation_stats(self) -> dict:
        """Return per-island generation stats if available."""
        return getattr(self._backend, "island_stats", {})

    def get_hall_of_fame(self):
        """Return Hall of Fame if available."""
        return getattr(self._backend, "hall_of_fame", None)
