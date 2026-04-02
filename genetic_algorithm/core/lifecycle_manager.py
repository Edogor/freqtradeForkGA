"""
Phase 4.2 — Strategy Lifecycle Manager

Track deployed strategies' performance over time.  Automatically retire
strategies whose drawdown exceeds a threshold, and promote strategies
that maintain their holdout performance for a sustained period.

The manager watches a *roster* of strategies (those promoted to
paper/live trading).  It is designed to run as a periodic background task
(e.g. cron or inside the queue daemon), reading the latest equity data
for each deployed strategy and writing a status file.

Config:
    lifecycle_manager:
        enabled: true
        roster_dir: "genetic_algorithm/data/lifecycle/roster"
        state_file: "genetic_algorithm/data/lifecycle/state.json"
        max_drawdown_retire: 0.30       # Retire when DD from peak > 30 %
        min_days_for_promotion: 30      # Days a strategy must be SAFE
        min_holdout_fitness: 0.10       # Minimum holdout fitness to promote
        check_interval_hours: 24        # How often to re-check

Usage:
    from genetic_algorithm.core.lifecycle_manager import LifecycleManager
    mgr = LifecycleManager(config)
    mgr.load_state()
    mgr.register_strategy(strategy_id, gene_dict, metrics)
    events = mgr.check_all()
    mgr.save_state()
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecord:
    """One tracked strategy in the lifecycle roster."""
    strategy_id: str
    status: str = 'active'          # active | retired | promoted
    registered_at: float = 0.0      # epoch
    last_checked: float = 0.0
    peak_equity: float = 1.0
    current_equity: float = 1.0
    max_drawdown_seen: float = 0.0
    days_active: int = 0
    holdout_fitness: Optional[float] = None
    retire_reason: Optional[str] = None
    gene_dict: Dict[str, Any] = field(default_factory=dict)
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)

    def drawdown_from_peak(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return 1.0 - (self.current_equity / self.peak_equity)


class LifecycleManager:
    """Manages the lifecycle of deployed trading strategies."""

    def __init__(self, config: Dict[str, Any]):
        lc_cfg = config.get('lifecycle_manager', {})
        self.enabled = lc_cfg.get('enabled', False)
        self.roster_dir = Path(lc_cfg.get('roster_dir',
                                          'genetic_algorithm/data/lifecycle/roster'))
        self.state_file = Path(lc_cfg.get('state_file',
                                           'genetic_algorithm/data/lifecycle/state.json'))
        self.max_dd_retire = lc_cfg.get('max_drawdown_retire', 0.30)
        self.min_days_promote = lc_cfg.get('min_days_for_promotion', 30)
        self.min_holdout_fitness = lc_cfg.get('min_holdout_fitness', 0.10)
        self.check_interval_hours = lc_cfg.get('check_interval_hours', 24)

        self._roster: Dict[str, StrategyRecord] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_strategy(self, strategy_id: str,
                          gene_dict: Dict[str, Any],
                          metrics: Optional[Dict[str, Any]] = None) -> StrategyRecord:
        """Add a strategy to the monitoring roster.

        Args:
            strategy_id: Unique id (e.g. "Gen11_Ind10").
            gene_dict:   Serialised StrategyGene (for reference / re-deployment).
            metrics:     Initial evaluation metrics.
        """
        if strategy_id in self._roster:
            logger.info(f"[LIFECYCLE] Strategy {strategy_id} already registered — updating")
            rec = self._roster[strategy_id]
            rec.gene_dict = gene_dict
            return rec

        rec = StrategyRecord(
            strategy_id=strategy_id,
            registered_at=time.time(),
            last_checked=time.time(),
            gene_dict=gene_dict,
            holdout_fitness=(metrics or {}).get('holdout_fitness'),
        )
        self._roster[strategy_id] = rec
        logger.info(f"[LIFECYCLE] Registered strategy {strategy_id}")
        return rec

    def unregister_strategy(self, strategy_id: str):
        """Remove a strategy entirely from the roster."""
        self._roster.pop(strategy_id, None)

    # ------------------------------------------------------------------
    # Equity update
    # ------------------------------------------------------------------

    def update_equity(self, strategy_id: str, equity: float,
                      snapshot: Optional[Dict[str, Any]] = None):
        """Push a new equity observation for a tracked strategy.

        Args:
            strategy_id: Which strategy.
            equity:      Current normalised equity (start = 1.0).
            snapshot:    Optional dict with extra metrics (trades today, etc.).
        """
        rec = self._roster.get(strategy_id)
        if rec is None:
            logger.warning(f"[LIFECYCLE] update_equity for unknown strategy {strategy_id}")
            return

        rec.current_equity = equity
        if equity > rec.peak_equity:
            rec.peak_equity = equity
        dd = rec.drawdown_from_peak()
        if dd > rec.max_drawdown_seen:
            rec.max_drawdown_seen = dd
        rec.last_checked = time.time()

        if snapshot:
            rec.metrics_history.append({
                'timestamp': time.time(),
                'equity': equity,
                'drawdown': dd,
                **snapshot,
            })

    # ------------------------------------------------------------------
    # Lifecycle checks
    # ------------------------------------------------------------------

    def check_all(self) -> List[Dict[str, Any]]:
        """Run lifecycle rules across every active strategy.

        Returns:
            List of event dicts, e.g. {'action': 'retire', 'id': '...', 'reason': '...'}
        """
        events: List[Dict[str, Any]] = []
        for sid, rec in list(self._roster.items()):
            if rec.status != 'active':
                continue

            # Update days active
            rec.days_active = int((time.time() - rec.registered_at) / 86400)

            # RETIRE check: drawdown exceeds threshold
            dd = rec.drawdown_from_peak()
            if dd >= self.max_dd_retire:
                rec.status = 'retired'
                rec.retire_reason = f"drawdown {dd:.1%} >= {self.max_dd_retire:.0%}"
                events.append({
                    'action': 'retire',
                    'strategy_id': sid,
                    'reason': rec.retire_reason,
                    'drawdown': dd,
                })
                logger.warning(f"[LIFECYCLE] RETIRED {sid}: {rec.retire_reason}")
                continue

            # PROMOTE check: sustained good holdout + enough days
            if (rec.days_active >= self.min_days_promote
                    and rec.holdout_fitness is not None
                    and rec.holdout_fitness >= self.min_holdout_fitness
                    and dd < self.max_dd_retire * 0.5):  # DD below half threshold
                rec.status = 'promoted'
                events.append({
                    'action': 'promote',
                    'strategy_id': sid,
                    'days_active': rec.days_active,
                    'holdout_fitness': rec.holdout_fitness,
                    'drawdown': dd,
                })
                logger.info(f"[LIFECYCLE] PROMOTED {sid}: {rec.days_active}d active, "
                            f"holdout={rec.holdout_fitness:.4f}, DD={dd:.1%}")

        return events

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def active_strategies(self) -> List[StrategyRecord]:
        return [r for r in self._roster.values() if r.status == 'active']

    @property
    def retired_strategies(self) -> List[StrategyRecord]:
        return [r for r in self._roster.values() if r.status == 'retired']

    @property
    def promoted_strategies(self) -> List[StrategyRecord]:
        return [r for r in self._roster.values() if r.status == 'promoted']

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total': len(self._roster),
            'active': len(self.active_strategies),
            'retired': len(self.retired_strategies),
            'promoted': len(self.promoted_strategies),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self):
        """Persist the full roster to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'version': 1,
            'saved_at': time.time(),
            'roster': {sid: asdict(rec) for sid, rec in self._roster.items()},
        }
        tmp = str(self.state_file) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        Path(tmp).replace(self.state_file)
        logger.debug(f"[LIFECYCLE] Saved state ({len(self._roster)} strategies)")

    def load_state(self):
        """Load roster from disk."""
        if not self.state_file.exists():
            logger.debug("[LIFECYCLE] No state file — starting fresh")
            return

        with open(self.state_file, 'r') as f:
            data = json.load(f)

        for sid, rec_data in data.get('roster', {}).items():
            # StrategyRecord from plain dict
            rec = StrategyRecord(
                strategy_id=rec_data.get('strategy_id', sid),
                status=rec_data.get('status', 'active'),
                registered_at=rec_data.get('registered_at', 0),
                last_checked=rec_data.get('last_checked', 0),
                peak_equity=rec_data.get('peak_equity', 1.0),
                current_equity=rec_data.get('current_equity', 1.0),
                max_drawdown_seen=rec_data.get('max_drawdown_seen', 0),
                days_active=rec_data.get('days_active', 0),
                holdout_fitness=rec_data.get('holdout_fitness'),
                retire_reason=rec_data.get('retire_reason'),
                gene_dict=rec_data.get('gene_dict', {}),
                metrics_history=rec_data.get('metrics_history', []),
            )
            self._roster[sid] = rec

        logger.info(f"[LIFECYCLE] Loaded {len(self._roster)} strategies from state file")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'summary': self.get_summary(),
        }
