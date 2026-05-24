"""T1.3 — Experiments database.

Persists every GA run, its resolved config, every HoF strategy and the
final metrics into a single SQLite database (``genetic_algorithm/data/experiments.db``
by default).  This is the canonical source for downstream meta-learning
(T4.4) and for the web dashboard.

Design goals
------------
* **Single write path** — the only producer is ``record_run_completion`` /
  ``record_strategy``; everything else is read-only.
* **Idempotent** — re-running an experiment with the same ``run_id`` updates
  rather than duplicates.
* **Resilient** — write failures are *logged*, never raised; the GA hot
  path must never crash because of telemetry.
* **Self-bootstrapping** — calling :meth:`ExperimentDB.connect()` creates
  the schema on demand.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("genetic_algorithm/data/experiments.db")

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    name TEXT,
    started_at REAL NOT NULL,
    completed_at REAL,
    status TEXT NOT NULL DEFAULT 'running',
    ga_type TEXT,
    generations_total INTEGER,
    best_fitness REAL,
    best_profit REAL,
    elapsed_seconds REAL,
    tags TEXT,
    notes TEXT,
    code_commit TEXT,
    config_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

CREATE TABLE IF NOT EXISTS configs_resolved (
    run_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_configs_hash ON configs_resolved(config_hash);

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    rank INTEGER,
    fitness REAL,
    profit REAL,
    sharpe REAL,
    drawdown REAL,
    trades INTEGER,
    winrate REAL,
    code_hash TEXT,
    code_pinned INTEGER NOT NULL DEFAULT 0,
    generation_found INTEGER,
    gene_json TEXT,
    metrics_json TEXT,
    strategy_code TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_strategies_run ON strategies(run_id);
CREATE INDEX IF NOT EXISTS idx_strategies_fitness ON strategies(fitness DESC);
CREATE INDEX IF NOT EXISTS idx_strategies_code_hash ON strategies(code_hash);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_json TEXT,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (run_id, metric_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
"""


class ExperimentDB:
    """Thin SQLite wrapper around the experiments schema."""

    def __init__(self, path: Path = DEFAULT_DB_PATH):
        self.path = Path(path)

    # ----- Connection / bootstrap ----------------------------------

    def connect(self) -> sqlite3.Connection:
        """Open a connection and ensure the schema exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA_SQL)
        # Stamp schema version on first init.
        cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
            conn.commit()
        return conn

    @contextmanager
    def session(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ----- Write API -----------------------------------------------

    def record_run_start(
        self,
        run_id: str,
        *,
        name: Optional[str] = None,
        ga_type: Optional[str] = None,
        generations_total: Optional[int] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        code_commit: Optional[str] = None,
    ) -> None:
        """Insert (or update) a row in ``runs`` and snapshot the resolved config.

        Idempotent: re-recording the same ``run_id`` keeps the original
        ``started_at`` and just refreshes the metadata."""
        try:
            with self.session() as conn:
                cur = conn.execute("SELECT started_at FROM runs WHERE run_id = ?", (run_id,))
                existing = cur.fetchone()
                if existing is None:
                    conn.execute(
                        """INSERT INTO runs(run_id, name, started_at, status, ga_type,
                                            generations_total, tags, code_commit, config_path)
                           VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
                        (
                            run_id, name or run_id, time.time(), ga_type,
                            generations_total,
                            json.dumps(tags) if tags else None,
                            code_commit, config_path,
                        ),
                    )
                else:
                    conn.execute(
                        """UPDATE runs SET name = COALESCE(?, name),
                                          ga_type = COALESCE(?, ga_type),
                                          generations_total = COALESCE(?, generations_total),
                                          tags = COALESCE(?, tags),
                                          code_commit = COALESCE(?, code_commit),
                                          config_path = COALESCE(?, config_path)
                           WHERE run_id = ?""",
                        (
                            name, ga_type, generations_total,
                            json.dumps(tags) if tags else None,
                            code_commit, config_path, run_id,
                        ),
                    )

                if config is not None:
                    cfg_str = json.dumps(config, default=str, sort_keys=True)
                    import hashlib
                    cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()
                    conn.execute(
                        """INSERT INTO configs_resolved(run_id, config_json, config_hash)
                           VALUES (?, ?, ?)
                           ON CONFLICT(run_id) DO UPDATE SET
                               config_json = excluded.config_json,
                               config_hash = excluded.config_hash""",
                        (run_id, cfg_str, cfg_hash),
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[ExperimentDB] record_run_start failed for {run_id}: {exc}")

    def record_run_completion(
        self,
        run_id: str,
        *,
        best_fitness: Optional[float] = None,
        best_profit: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        status: str = "completed",
        notes: Optional[str] = None,
    ) -> None:
        try:
            with self.session() as conn:
                conn.execute(
                    """UPDATE runs SET completed_at = ?, status = ?,
                                       best_fitness = COALESCE(?, best_fitness),
                                       best_profit = COALESCE(?, best_profit),
                                       elapsed_seconds = COALESCE(?, elapsed_seconds),
                                       notes = COALESCE(?, notes)
                       WHERE run_id = ?""",
                    (time.time(), status, best_fitness, best_profit,
                     elapsed_seconds, notes, run_id),
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[ExperimentDB] record_run_completion failed for {run_id}: {exc}")

    def record_strategy(
        self,
        run_id: str,
        *,
        strategy_id: str,
        rank: Optional[int] = None,
        fitness: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        gene_dict: Optional[Dict[str, Any]] = None,
        strategy_code: Optional[str] = None,
        code_hash: Optional[str] = None,
        generation_found: Optional[int] = None,
    ) -> None:
        """Upsert a strategy row (HoF member or final winner)."""
        try:
            m = metrics or {}
            with self.session() as conn:
                conn.execute(
                    """INSERT INTO strategies(strategy_id, run_id, rank, fitness, profit,
                                              sharpe, drawdown, trades, winrate, code_hash,
                                              code_pinned, generation_found, gene_json,
                                              metrics_json, strategy_code, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(strategy_id) DO UPDATE SET
                           run_id=excluded.run_id, rank=excluded.rank,
                           fitness=excluded.fitness, profit=excluded.profit,
                           sharpe=excluded.sharpe, drawdown=excluded.drawdown,
                           trades=excluded.trades, winrate=excluded.winrate,
                           code_hash=excluded.code_hash, code_pinned=excluded.code_pinned,
                           generation_found=excluded.generation_found,
                           gene_json=excluded.gene_json, metrics_json=excluded.metrics_json,
                           strategy_code=excluded.strategy_code""",
                    (
                        strategy_id, run_id, rank, fitness,
                        float(m.get("profit", 0) or 0),
                        float(m.get("sharpe_ratio", 0) or 0),
                        float(m.get("max_drawdown", 0) or 0),
                        int(m.get("total_trades", 0) or 0),
                        float(m.get("win_rate", 0) or 0),
                        code_hash,
                        1 if strategy_code else 0,
                        generation_found,
                        json.dumps(gene_dict, default=str) if gene_dict is not None else None,
                        json.dumps(m, default=str) if m else None,
                        strategy_code,
                        time.time(),
                    ),
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"[ExperimentDB] record_strategy failed (run={run_id} sid={strategy_id}): {exc}"
            )

    def record_metric(self, run_id: str, name: str,
                      value: Optional[float] = None,
                      payload: Optional[Any] = None) -> None:
        """Persist an aggregate run-level metric (or a JSON-serialisable payload)."""
        try:
            with self.session() as conn:
                conn.execute(
                    """INSERT INTO metrics(run_id, metric_name, metric_value, metric_json, recorded_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(run_id, metric_name) DO UPDATE SET
                           metric_value = excluded.metric_value,
                           metric_json = excluded.metric_json,
                           recorded_at = excluded.recorded_at""",
                    (run_id, name, value,
                     json.dumps(payload, default=str) if payload is not None else None,
                     time.time()),
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[ExperimentDB] record_metric failed ({run_id}/{name}): {exc}")

    # ----- Bulk helpers --------------------------------------------

    def record_hof_entries(self, run_id: str, hof_entries: Iterable[Any]) -> int:
        """Persist a HoF iterable.

        Each entry may be a :class:`HallOfFameEntry` (preferred) or a raw
        dict from the JSON file.  Returns the number of rows written.
        """
        n = 0
        for rank, e in enumerate(hof_entries, start=1):
            if hasattr(e, "to_dict"):
                d = e.to_dict()
            else:
                d = dict(e)
            sid = d.get("id") or d.get("entry_id") or f"{run_id}_rank{rank}"
            self.record_strategy(
                run_id,
                strategy_id=sid,
                rank=rank,
                fitness=d.get("fitness"),
                metrics=d.get("metrics") or {},
                gene_dict=d.get("strategy_gene"),
                strategy_code=d.get("strategy_code"),
                code_hash=d.get("code_hash"),
                generation_found=d.get("generation_found"),
            )
            n += 1
        return n

    # ----- Read API -------------------------------------------------

    def list_runs(self, *, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self.session() as conn:
            if status:
                cur = conn.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY started_at DESC LIMIT ?",
                    (status, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
                )
            return [dict(row) for row in cur.fetchall()]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.session() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def get_strategies_for_run(self, run_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self.session() as conn:
            cur = conn.execute(
                """SELECT strategy_id, rank, fitness, profit, sharpe, drawdown,
                          trades, winrate, code_hash, code_pinned, generation_found
                   FROM strategies WHERE run_id = ?
                   ORDER BY fitness DESC LIMIT ?""",
                (run_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def top_strategies_overall(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self.session() as conn:
            cur = conn.execute(
                """SELECT s.strategy_id, s.run_id, s.fitness, s.profit, s.sharpe,
                          s.drawdown, s.trades, s.winrate, r.started_at, r.name
                   FROM strategies s JOIN runs r ON r.run_id = s.run_id
                   ORDER BY s.fitness DESC LIMIT ?""",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]
