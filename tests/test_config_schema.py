"""Tests for config/schema.py — defaults, deep merge, presets, validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from genetic_algorithm.config.schema import (
    DEFAULTS,
    deep_merge,
    load_config,
    resolve_preset,
    validate_config,
)


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        result = deep_merge(base, {"b": 99})
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        result = deep_merge(base, {"x": {"b": 99}})
        assert result == {"x": {"a": 1, "b": 99}}

    def test_new_keys_kept(self):
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        base = {"x": {"a": 1}}
        deep_merge(base, {"x": {"a": 999}})
        assert base["x"]["a"] == 1

    def test_nested_new_key(self):
        base = {"x": {"a": 1}}
        result = deep_merge(base, {"x": {"z": 42}})
        assert result["x"]["z"] == 42
        assert result["x"]["a"] == 1


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_valid_defaults(self):
        errors, warnings = validate_config(DEFAULTS)
        assert errors == []

    def test_empty_pairs_is_error(self):
        cfg = deep_merge(DEFAULTS, {"backtesting": {"pairs": []}})
        errors, _ = validate_config(cfg)
        assert any("pairs" in e for e in errors)

    def test_elite_ge_pop_is_error(self):
        cfg = deep_merge(DEFAULTS, {"genetic_algorithm": {
            "population_size": 10, "elite_size": 10
        }})
        errors, _ = validate_config(cfg)
        assert any("elite_size" in e for e in errors)

    def test_bad_weights_warns(self):
        cfg = deep_merge(DEFAULTS, {"fitness_weights": {
            "profit": 0.9, "sharpe_ratio": 0.9,
            "sortino_ratio": 0, "profit_factor": 0,
            "drawdown": 0, "win_rate": 0,
            "trade_frequency": 0, "monthly_stability": 0, "cross_pair": 0,
        }})
        _, warnings = validate_config(cfg)
        assert any("fitness_weights" in w for w in warnings)

    def test_both_islands_error(self):
        cfg = deep_merge(DEFAULTS, {
            "generic_island_model": {"enabled": True},
            "island_model": {"enabled": True},
        })
        errors, _ = validate_config(cfg)
        assert any("island" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_simple_yaml(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({
            "backtesting": {"pairs": ["ETH/USDT"]},
            "genetic_algorithm": {"generations": 5},
        }))
        config = load_config(cfg_file)
        assert config["backtesting"]["pairs"] == ["ETH/USDT"]
        assert config["genetic_algorithm"]["generations"] == 5
        # Defaults filled in
        assert config["genetic_algorithm"]["population_size"] == 30
        assert config["parallel_evaluation"]["enabled"] is True

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_load_overrides(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"backtesting": {"pairs": ["BTC/USDT"]}}))
        config = load_config(cfg_file, overrides={
            "genetic_algorithm": {"generations": 99}
        })
        assert config["genetic_algorithm"]["generations"] == 99

    def test_load_validation_error(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(yaml.dump({
            "backtesting": {"pairs": []},
        }))
        with pytest.raises(ValueError, match="validation failed"):
            load_config(cfg_file)

    def test_load_empty_yaml(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        config = load_config(cfg_file)
        # Should get all defaults
        assert config["genetic_algorithm"]["population_size"] == 30


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    def test_quick_test_preset_loads(self):
        preset_path = Path("genetic_algorithm/config/presets/quick_test.yaml")
        if not preset_path.exists():
            pytest.skip("Preset not found")
        config = load_config(preset_path)
        assert config["genetic_algorithm"]["population_size"] == 5
        assert config["genetic_algorithm"]["generations"] == 2

    def test_standard_preset_loads(self):
        preset_path = Path("genetic_algorithm/config/presets/standard.yaml")
        if not preset_path.exists():
            pytest.skip("Preset not found")
        config = load_config(preset_path)
        assert config["genetic_algorithm"]["population_size"] == 30
        assert config["holdout_validation"]["enabled"] is True

    def test_island_preset_loads(self):
        preset_path = Path("genetic_algorithm/config/presets/island.yaml")
        if not preset_path.exists():
            pytest.skip("Preset not found")
        config = load_config(preset_path)
        assert config["generic_island_model"]["enabled"] is True

    def test_production_preset_loads(self):
        preset_path = Path("genetic_algorithm/config/presets/production.yaml")
        if not preset_path.exists():
            pytest.skip("Preset not found")
        config = load_config(preset_path)
        assert config["walk_forward"]["enabled"] is True
        assert config["monte_carlo"]["enabled"] is True

    def test_nsga2_preset_loads(self):
        preset_path = Path("genetic_algorithm/config/presets/nsga2.yaml")
        if not preset_path.exists():
            pytest.skip("Preset not found")
        config = load_config(preset_path)
        assert config["genetic_algorithm"]["mode"] == "nsga2"

    def test_preset_key_resolution(self, tmp_path):
        """A config with 'preset: quick_test' loads the preset + overrides."""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({
            "preset": "quick_test",
            "genetic_algorithm": {"generations": 99},
        }))
        config = load_config(cfg_file)
        # generations overridden
        assert config["genetic_algorithm"]["generations"] == 99
        # rest from quick_test preset
        assert config["genetic_algorithm"]["population_size"] == 5

    def test_bad_preset_raises(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"preset": "nonexistent_preset_xyz"}))
        with pytest.raises(FileNotFoundError, match="nonexistent_preset_xyz"):
            load_config(cfg_file)


# ---------------------------------------------------------------------------
# DEFAULTS integrity
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_all_top_level_keys_are_dicts(self):
        for key, val in DEFAULTS.items():
            assert isinstance(val, dict), f"DEFAULTS['{key}'] should be a dict"

    def test_fitness_weights_sum_to_one(self):
        total = sum(DEFAULTS["fitness_weights"].values())
        assert abs(total - 1.0) < 0.05, f"fitness_weights sum={total}"
