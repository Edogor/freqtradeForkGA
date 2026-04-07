"""Tests for ExperimentRegistry (orchestration/registry.py)."""

import json
import tempfile
from pathlib import Path

import pytest

from genetic_algorithm.orchestration.registry import ExperimentRegistry


@pytest.fixture
def registry(tmp_path):
    """Create a registry backed by a temp file."""
    return ExperimentRegistry(path=tmp_path / "registry.json")


class TestRegister:
    def test_register_new(self, registry):
        entry = registry.register("exp1", config_path="c.yaml", tags=["a", "b"])
        assert entry["experiment_id"] == "exp1"
        assert entry["status"] == "queued"
        assert entry["config_path"] == "c.yaml"
        assert entry["tags"] == ["a", "b"]
        assert entry["created_at"] is not None

    def test_register_duplicate_raises(self, registry):
        registry.register("exp1")
        with pytest.raises(ValueError, match="already registered"):
            registry.register("exp1")

    def test_register_ga_type(self, registry):
        entry = registry.register("exp1", ga_type="island")
        assert entry["ga_type"] == "island"

    def test_register_extra_fields(self, registry):
        entry = registry.register("exp1", extra={"custom_key": 42})
        assert entry["custom_key"] == 42


class TestLifecycle:
    def test_full_lifecycle(self, registry):
        registry.register("exp1")
        registry.start("exp1", pid=999, log_path="/tmp/exp1.log")
        exp = registry.get("exp1")
        assert exp["status"] == "running"
        assert exp["pid"] == 999

        registry.update("exp1", generation=3, best_fitness=0.65)
        exp = registry.get("exp1")
        assert exp["generation"] == 3
        assert exp["best_fitness"] == 0.65

        registry.complete("exp1", best_fitness=0.80, best_profit=12.5)
        exp = registry.get("exp1")
        assert exp["status"] == "completed"
        assert exp["best_fitness"] == 0.80
        assert exp["best_profit"] == 12.5
        assert exp["pid"] is None
        assert exp["finished_at"] is not None

    def test_fail(self, registry):
        registry.register("exp1")
        registry.start("exp1")
        registry.fail("exp1", error="OOM")
        exp = registry.get("exp1")
        assert exp["status"] == "failed"
        assert exp["error"] == "OOM"
        assert exp["pid"] is None

    def test_cancel(self, registry):
        registry.register("exp1")
        registry.cancel("exp1")
        assert registry.get("exp1")["status"] == "cancelled"

    def test_update_nonexistent_raises(self, registry):
        with pytest.raises(KeyError, match="not found"):
            registry.update("nope", generation=1)

    def test_invalid_status_raises(self, registry):
        registry.register("exp1")
        with pytest.raises(ValueError, match="Invalid status"):
            registry.update("exp1", status="exploded")


class TestQuery:
    @pytest.fixture(autouse=True)
    def _populate(self, registry):
        registry.register("a", tags=["batch1"])
        registry.start("a")
        registry.register("b", tags=["batch1", "scalp"])
        registry.register("c", tags=["batch2"])
        registry.start("c")
        registry.complete("c")

    def test_list_all(self, registry):
        assert len(registry.list()) == 3

    def test_list_by_status(self, registry):
        running = registry.list(status="running")
        assert len(running) == 1
        assert running[0]["experiment_id"] == "a"

    def test_list_by_multiple_statuses(self, registry):
        results = registry.list(status=["running", "queued"])
        ids = {e["experiment_id"] for e in results}
        assert ids == {"a", "b"}

    def test_list_by_tag(self, registry):
        results = registry.list(tags=["batch1"])
        ids = {e["experiment_id"] for e in results}
        assert ids == {"a", "b"}

    def test_list_by_tag_intersection(self, registry):
        results = registry.list(tags=["batch1", "scalp"])
        assert len(results) == 1
        assert results[0]["experiment_id"] == "b"

    def test_list_limit(self, registry):
        results = registry.list(limit=2)
        assert len(results) == 2

    def test_get_nonexistent(self, registry):
        assert registry.get("missing") is None

    def test_count(self, registry):
        assert registry.count() == 3
        assert registry.count(status="running") == 1
        assert registry.count(status="queued") == 1
        assert registry.count(status="completed") == 1

    def test_summary(self, registry):
        s = registry.summary()
        assert s["running"] == 1
        assert s["queued"] == 1
        assert s["completed"] == 1


class TestRemove:
    def test_remove_existing(self, registry):
        registry.register("exp1")
        assert registry.remove("exp1") is True
        assert registry.get("exp1") is None

    def test_remove_nonexistent(self, registry):
        assert registry.remove("nope") is False


class TestPersistence:
    def test_reopen_preserves_data(self, tmp_path):
        path = tmp_path / "registry.json"
        r1 = ExperimentRegistry(path=path)
        r1.register("exp1", tags=["t1"])
        r1.start("exp1", pid=42)

        r2 = ExperimentRegistry(path=path)
        exp = r2.get("exp1")
        assert exp["status"] == "running"
        assert exp["pid"] == 42
        assert exp["tags"] == ["t1"]

    def test_file_is_valid_json(self, tmp_path):
        path = tmp_path / "registry.json"
        r = ExperimentRegistry(path=path)
        r.register("exp1")
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert "exp1" in data["experiments"]
