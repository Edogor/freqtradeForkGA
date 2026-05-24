"""
Behavioral MAP-Elites Archive

Maintains a 2D grid of behavior cells, each holding the single best
strategy that exhibits that behavior. This explicitly preserves behavioral
diversity across the population.

Behavior dimensions:
  - Axis 0: Trade frequency (trades/month) — discretized into bins
  - Axis 1: Average hold duration (hours) — discretized into bins

Config:
    map_elites:
        enabled: true
        frequency_bins: 5           # Bins for trade frequency axis
        duration_bins: 5            # Bins for hold duration axis
        exploration_bonus: 0.05     # Fitness bonus for filling empty cells
        injection_count: 3          # Strategies to inject from archive per generation

Usage:
    archive = MAPElitesArchive(config)
    # After evaluation:
    archive.update(population)
    # For migration / diversity injection:
    diverse_individuals = archive.sample_diverse(n=3)
    # Exploration bonus:
    bonus = archive.get_exploration_bonus(individual)
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _compute_behavior(individual) -> Optional[Tuple[float, float]]:
    """Extract behavior descriptor from an evaluated individual.

    Returns:
        (trades_per_month, avg_hold_hours) or None if metrics unavailable
    """
    metrics = getattr(individual, 'metrics', {}) or {}

    # Trade frequency: trades per month
    total_trades = metrics.get('trade_count', metrics.get('total_trades', 0))
    # Estimate trading days from backtest duration
    trading_days = metrics.get('trading_days', metrics.get('backtest_days', 90))
    if trading_days and trading_days > 0:
        trades_per_month = (total_trades / trading_days) * 30.0
    else:
        trades_per_month = float(total_trades)

    # Average hold duration in hours
    avg_duration = metrics.get('avg_trade_duration', metrics.get('avg_duration_hours', None))
    if avg_duration is None:
        # Estimate from winning/losing durations
        win_dur = metrics.get('winning_avg_duration', 0)
        lose_dur = metrics.get('losing_avg_duration', 0)
        win_rate = metrics.get('win_rate', 0.5)
        if win_dur or lose_dur:
            avg_duration = win_dur * win_rate + lose_dur * (1 - win_rate)
        else:
            return None  # Can't compute behavior

    # Convert to hours if in minutes or seconds
    if isinstance(avg_duration, str):
        # Handle "HH:MM:SS" or "X days, HH:MM:SS" format
        try:
            parts = avg_duration.replace(' days, ', ':').replace(' day, ', ':').split(':')
            if len(parts) >= 3:
                avg_duration = float(parts[-3]) * 24 + float(parts[-2]) + float(parts[-1]) / 60
            else:
                avg_duration = float(parts[0])
        except (ValueError, IndexError):
            return None

    avg_duration = float(avg_duration)
    if avg_duration > 1000:  # Likely in minutes
        avg_duration /= 60.0

    return (trades_per_month, avg_duration)


class MAPElitesArchive:
    """2D behavior-fitness archive for quality-diversity preservation."""

    def __init__(self, config: Dict[str, Any]):
        me_config = config.get('map_elites', {})
        self.enabled = me_config.get('enabled', False)
        self.freq_bins = me_config.get('frequency_bins', 5)
        self.dur_bins = me_config.get('duration_bins', 5)
        self.exploration_bonus = me_config.get('exploration_bonus', 0.05)
        self.injection_count = me_config.get('injection_count', 3)

        # Bin edges
        # Trade frequency: [0, 2, 5, 10, 25, inf] -> 5 bins
        self._freq_edges = me_config.get('frequency_edges', [0, 2, 5, 10, 25])
        # Hold duration (hours): [0, 1, 4, 12, 48, inf] -> 5 bins
        self._dur_edges = me_config.get('duration_edges', [0, 1, 4, 12, 48])

        # Grid: (freq_bin, dur_bin) -> (individual, fitness)
        self._grid: Dict[Tuple[int, int], Tuple[Any, float]] = {}

        # Statistics
        self.total_placements = 0
        self.total_replacements = 0

    @property
    def occupancy(self) -> float:
        """Fraction of cells occupied."""
        total_cells = self.freq_bins * self.dur_bins
        return len(self._grid) / total_cells if total_cells > 0 else 0.0

    @property
    def filled_cells(self) -> int:
        return len(self._grid)

    @property
    def total_cells(self) -> int:
        return self.freq_bins * self.dur_bins

    def _discretize(self, trades_per_month: float, avg_hold_hours: float) -> Tuple[int, int]:
        """Map continuous behavior to grid cell indices."""
        freq_bin = 0
        for i, edge in enumerate(self._freq_edges):
            if trades_per_month >= edge:
                freq_bin = i
        freq_bin = min(freq_bin, self.freq_bins - 1)

        dur_bin = 0
        for i, edge in enumerate(self._dur_edges):
            if avg_hold_hours >= edge:
                dur_bin = i
        dur_bin = min(dur_bin, self.dur_bins - 1)

        return (freq_bin, dur_bin)

    def update(self, population) -> int:
        """Attempt to place each evaluated individual in its behavior cell.

        Args:
            population: Evaluated Population

        Returns:
            Number of new/improved placements
        """
        if not self.enabled:
            return 0

        placements = 0
        for ind in population:
            if not ind.evaluated or ind.fitness is None:
                continue

            behavior = _compute_behavior(ind)
            if behavior is None:
                continue

            cell = self._discretize(*behavior)
            fitness = float(ind.fitness)

            existing = self._grid.get(cell)
            if existing is None:
                # Empty cell — place unconditionally
                self._grid[cell] = (ind, fitness)
                self.total_placements += 1
                placements += 1
            elif fitness > existing[1]:
                # Better than current occupant — replace
                self._grid[cell] = (ind, fitness)
                self.total_replacements += 1
                placements += 1

        if placements > 0:
            logger.info(f"[MAP-ELITES] {placements} placements — "
                        f"occupancy {self.filled_cells}/{self.total_cells} "
                        f"({self.occupancy:.0%})")
        return placements

    def get_exploration_bonus(self, individual) -> float:
        """Return a fitness bonus if the individual would fill an empty cell.

        Designed to be added to fitness BEFORE selection.
        """
        if not self.enabled or self.exploration_bonus <= 0:
            return 0.0

        behavior = _compute_behavior(individual)
        if behavior is None:
            return 0.0

        cell = self._discretize(*behavior)
        if cell not in self._grid:
            return self.exploration_bonus
        return 0.0

    def sample_diverse(self, n: int = 3) -> list:
        """Sample individuals from diverse cells for migration/injection.

        Prefers cells that are far apart in the grid (maximize coverage).
        """
        if not self._grid:
            return []

        cells = list(self._grid.keys())
        if len(cells) <= n:
            return [self._grid[c][0] for c in cells]

        # Greedy farthest-point sampling for diversity
        import random
        selected = [random.choice(cells)]
        remaining = set(cells) - set(selected)

        while len(selected) < n and remaining:
            best_cell = None
            best_min_dist = -1
            for candidate in remaining:
                min_dist = min(
                    abs(candidate[0] - s[0]) + abs(candidate[1] - s[1])
                    for s in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_cell = candidate
            if best_cell is not None:
                selected.append(best_cell)
                remaining.discard(best_cell)

        return [self._grid[c][0] for c in selected]

    def to_hall_of_fame(
        self, n: Optional[int] = None, sort_by: str = "fitness"
    ) -> List[Any]:
        """T3.2 — Export the archive as a Hall-of-Fame list.

        The MAP-Elites grid is itself a *diversity-preserving* HoF:
        every cell holds the best strategy that exhibits a distinct
        behavior signature.  This method returns those individuals in
        a deterministic order.

        Args:
            n: Optional cap on the number of returned individuals.
               If ``None``, returns every occupant.
            sort_by: ``'fitness'`` (descending fitness, default) or
               ``'cell'`` (raster-scan over grid cells).

        Returns:
            List of evaluated Individual objects.  Empty list if the
            archive is empty or disabled.
        """
        if not self._grid:
            return []
        items = [(cell, ind, fit) for cell, (ind, fit) in self._grid.items()]
        if sort_by == "cell":
            items.sort(key=lambda x: (x[0][0], x[0][1]))
        else:
            items.sort(key=lambda x: x[2], reverse=True)
        out = [ind for _, ind, _ in items]
        if n is not None and n >= 0:
            out = out[:n]
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize archive state for checkpoint."""
        grid_data = {}
        for cell, (ind, fitness) in self._grid.items():
            key = f"{cell[0]}_{cell[1]}"
            grid_data[key] = {
                'fitness': fitness,
                'gene_dict': ind.strategy_gene.to_dict() if hasattr(ind, 'strategy_gene') else {},
            }
        return {
            'freq_bins': self.freq_bins,
            'dur_bins': self.dur_bins,
            'total_placements': self.total_placements,
            'total_replacements': self.total_replacements,
            'grid': grid_data,
        }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Restore archive from checkpoint data.

        Note: Only restores metadata and fitness; full Individual objects
        are not reconstructed (would require import of Individual class).
        The archive rebuilds organically from the next evaluated population.
        """
        self.total_placements = data.get('total_placements', 0)
        self.total_replacements = data.get('total_replacements', 0)
        # Grid is rebuilt from live population, not checkpoint

    def get_report(self) -> Dict[str, Any]:
        """Report for logging/analysis."""
        fitness_by_cell = {}
        for cell, (_, fitness) in self._grid.items():
            fitness_by_cell[f"{cell[0]}_{cell[1]}"] = round(fitness, 4)

        return {
            'enabled': self.enabled,
            'occupancy': round(self.occupancy, 3),
            'filled_cells': self.filled_cells,
            'total_cells': self.total_cells,
            'total_placements': self.total_placements,
            'total_replacements': self.total_replacements,
            'fitness_grid': fitness_by_cell,
        }
