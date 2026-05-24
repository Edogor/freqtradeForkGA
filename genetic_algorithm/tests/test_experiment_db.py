"""Tests for T1.3 ExperimentDB."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from genetic_algorithm.data.experiment_db import ExperimentDB


@pytest.fixture()
def db(tmp_path: Path) -> ExperimentDB:
    return ExperimentDB(path=tmp_path / "experiments.db")


def test_schema_bootstrap(db: ExperimentDB):
    with db.session() as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "configs_resolved", "strategies", "metrics", "schema_version"} <= names


def test_record_run_start_creates_row_with_config(db: ExperimentDB):
    db.record_run_start("run1", name="Run 1", ga_type="island",
                        generations_total=10, config={"a": 1})
    run = db.get_run("run1")
    assert run is not None
    assert run["status"] == "running"
    assert run["ga_type"] == "island"
    assert run["generations_total"] == 10
    # Config snapshot exists
    with db.session() as conn:
        row = conn.execute("SELECT config_json FROM configs_resolved WHERE run_id=?",
                           ("run1",)).fetchone()
    assert row is not None
    assert '"a": 1' in row["config_json"]


def test_record_run_start_is_idempotent(db: ExperimentDB):
    db.record_run_start("run1", name="orig", ga_type="standard", config={"x": 1})
    first = db.get_run("run1")
    db.record_run_start("run1", name="updated", ga_type="island", config={"x": 2})
    second = db.get_run("run1")
    assert second["started_at"] == first["started_at"]  # preserved
    assert second["name"] == "updated"
    assert second["ga_type"] == "island"


def test_record_run_completion(db: ExperimentDB):
    db.record_run_start("run1", config={})
    db.record_run_completion("run1", best_fitness=1.23, best_profit=45.6,
                             elapsed_seconds=99.0)
    run = db.get_run("run1")
    assert run["status"] == "completed"
    assert run["best_fitness"] == pytest.approx(1.23)
    assert run["best_profit"] == pytest.approx(45.6)
    assert run["elapsed_seconds"] == pytest.approx(99.0)


def test_record_strategy_and_upsert(db: ExperimentDB):
    db.record_run_start("run1", config={})
    db.record_strategy("run1", strategy_id="s1", rank=1, fitness=0.5,
                       metrics={"profit": 10, "sharpe_ratio": 1.5,
                                "max_drawdown": 5, "total_trades": 30,
                                "win_rate": 0.6})
    rows = db.get_strategies_for_run("run1")
    assert len(rows) == 1
    assert rows[0]["profit"] == pytest.approx(10)
    assert rows[0]["sharpe"] == pytest.approx(1.5)

    # Upsert
    db.record_strategy("run1", strategy_id="s1", rank=1, fitness=0.9,
                       metrics={"profit": 20})
    rows = db.get_strategies_for_run("run1")
    assert len(rows) == 1
    assert rows[0]["fitness"] == pytest.approx(0.9)
    assert rows[0]["profit"] == pytest.approx(20)


def test_record_strategy_with_code_sets_pinned_flag(db: ExperimentDB):
    db.record_run_start("run1", config={})
    db.record_strategy("run1", strategy_id="s1", strategy_code="class S: pass",
                       code_hash="abc", metrics={})
    rows = db.get_strategies_for_run("run1")
    assert rows[0]["code_pinned"] == 1
    assert rows[0]["code_hash"] == "abc"


def test_record_hof_entries_from_dicts(db: ExperimentDB):
    db.record_run_start("run1", config={})
    entries = [
        {"id": "e1", "fitness": 0.8,
         "metrics": {"profit": 12, "sharpe_ratio": 1.1, "max_drawdown": 4,
                     "total_trades": 25, "win_rate": 0.55},
         "strategy_gene": {"foo": "bar"}},
        {"id": "e2", "fitness": 0.6, "metrics": {"profit": 8}},
    ]
    n = db.record_hof_entries("run1", entries)
    assert n == 2
    rows = db.get_strategies_for_run("run1")
    assert {r["strategy_id"] for r in rows} == {"e1", "e2"}
    assert rows[0]["rank"] in (1, 2)


def test_record_metric(db: ExperimentDB):
    db.record_run_start("run1", config={})
    db.record_metric("run1", "convergence_gen", 42.0)
    db.record_metric("run1", "convergence_gen", 50.0)  # upsert
    with db.session() as conn:
        rows = conn.execute("SELECT metric_value FROM metrics WHERE run_id='run1'").fetchall()
    assert len(rows) == 1
    assert rows[0]["metric_value"] == pytest.approx(50.0)


def test_list_runs_orders_by_started_at_desc_and_filters(db: ExperimentDB):
    db.record_run_start("a", config={})
    db.record_run_start("b", config={})
    db.record_run_completion("a")  # completed
    runs = db.list_runs()
    ids = [r["run_id"] for r in runs]
    assert set(ids) == {"a", "b"}
    # Filter by status
    only_completed = db.list_runs(status="completed")
    assert [r["run_id"] for r in only_completed] == ["a"]


def test_top_strategies_overall_orders_by_fitness(db: ExperimentDB):
    db.record_run_start("run1", config={})
    db.record_run_start("run2", config={})
    db.record_strategy("run1", strategy_id="s1", fitness=0.5, metrics={})
    db.record_strategy("run2", strategy_id="s2", fitness=0.9, metrics={})
    db.record_strategy("run1", strategy_id="s3", fitness=0.7, metrics={})
    top = db.top_strategies_overall(limit=10)
    assert [r["strategy_id"] for r in top] == ["s2", "s3", "s1"]


def test_get_run_missing_returns_none(db: ExperimentDB):
    assert db.get_run("nope") is None
