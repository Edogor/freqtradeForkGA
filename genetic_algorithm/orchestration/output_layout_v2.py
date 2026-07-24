"""Canonical, immutable filesystem layout for one V2 attempt.

The worker spec persists this model, so every operational output path is part
of the command-bound input contract.  Components may use different
subdirectories, but none may choose a path outside the attempt artifact root.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from genetic_algorithm.orchestration.artifact_store_v2 import canonical_config_hash
from genetic_algorithm.orchestration.result_contract import StrictV2Model


class AttemptOutputLayoutV2(StrictV2Model):
    """All writable locations available to canonical evolution/replay workers."""

    schema_version: Literal["attempt-output-layout-v2"] = "attempt-output-layout-v2"
    artifact_root: str = Field(min_length=1)
    evolution_root: str = Field(min_length=1)
    evolution_work_dir: str = Field(min_length=1)
    engine_config_path: str = Field(min_length=1)
    diagnostics_dir: str = Field(min_length=1)
    checkpoint_dir: str = Field(min_length=1)
    runs_dir: str = Field(min_length=1)
    evolution_cache_dir: str = Field(min_length=1)
    evolution_strategy_dir: str = Field(min_length=1)
    evolution_backtest_export_dir: str = Field(min_length=1)
    evolution_user_data_dir: str = Field(min_length=1)
    hall_of_fame_dir: str = Field(min_length=1)
    engine_log_path: str = Field(min_length=1)
    plot_dir: str = Field(min_length=1)
    trade_plot_dir: str = Field(min_length=1)
    replay_root: str = Field(min_length=1)
    replay_work_dir: str = Field(min_length=1)
    replay_cache_dir: str = Field(min_length=1)
    replay_strategy_dir: str = Field(min_length=1)
    replay_backtest_export_dir: str = Field(min_length=1)
    replay_user_data_dir: str = Field(min_length=1)

    @staticmethod
    def _canonical_values(artifact_root: str | Path) -> dict[str, str]:
        root = Path(artifact_root).resolve()
        evolution = root / "evolution"
        replay = root / "replay"
        return {
            "artifact_root": str(root),
            "evolution_root": str(evolution),
            "evolution_work_dir": str(evolution / "work"),
            "engine_config_path": str(evolution / "engine_config.yaml"),
            "diagnostics_dir": str(evolution / "diagnostics"),
            "checkpoint_dir": str(evolution / "checkpoints"),
            "runs_dir": str(evolution / "runs"),
            "evolution_cache_dir": str(evolution / "cache"),
            "evolution_strategy_dir": str(evolution / "generated_strategies"),
            "evolution_backtest_export_dir": str(evolution / "backtest_results"),
            "evolution_user_data_dir": str(evolution / "freqtrade_user_data"),
            "hall_of_fame_dir": str(evolution / "hall_of_fame"),
            "engine_log_path": str(evolution / "engine.log"),
            "plot_dir": str(evolution / "plots"),
            "trade_plot_dir": str(evolution / "trade_plots"),
            "replay_root": str(replay),
            "replay_work_dir": str(replay / "work"),
            "replay_cache_dir": str(replay / "cache"),
            "replay_strategy_dir": str(replay / "generated_strategies"),
            "replay_backtest_export_dir": str(replay / "backtest_results"),
            "replay_user_data_dir": str(replay / "freqtrade_user_data"),
        }

    @classmethod
    def for_artifact_root(cls, artifact_root: str | Path) -> AttemptOutputLayoutV2:
        return cls(**cls._canonical_values(artifact_root))

    @model_validator(mode="after")
    def _canonical_attempt_paths(self) -> AttemptOutputLayoutV2:
        root = Path(self.artifact_root)
        if not root.is_absolute():
            raise ValueError("artifact_root must be absolute")
        canonical = type(self)._canonical_values(root)
        for field_name, expected in canonical.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} is not the canonical attempt-local path")
        return self

    @property
    def layout_hash(self) -> str:
        return canonical_config_hash(self.model_dump(mode="json"))

    def apply_to_engine_config(self, config: dict[str, Any]) -> None:
        """Set every writable path used by the supported standard engine."""

        config.setdefault("output", {}).update(
            {"dir": self.diagnostics_dir, "plots_dir": self.plot_dir}
        )
        config.setdefault("storage", {}).update(
            {
                "checkpoint_dir": self.checkpoint_dir,
                "runs_dir": self.runs_dir,
                "cache_dir": self.evolution_cache_dir,
                "generated_strategy_dir": self.evolution_strategy_dir,
                "backtest_export_dir": self.evolution_backtest_export_dir,
                "backtest_user_data_dir": self.evolution_user_data_dir,
            }
        )
        config.setdefault("hall_of_fame", {})["directory"] = self.hall_of_fame_dir
        config.setdefault("logging", {}).update({"file": self.engine_log_path, "console": False})
        # These sections are optional in source configs.  Only update them when
        # present so the strict config schema does not gain operational aliases.
        if "trade_visualization" in config:
            config["trade_visualization"]["output_dir"] = self.trade_plot_dir

    def assert_engine_config(self, config: Mapping[str, Any]) -> None:
        expected = {
            ("output", "dir"): self.diagnostics_dir,
            ("output", "plots_dir"): self.plot_dir,
            ("storage", "checkpoint_dir"): self.checkpoint_dir,
            ("storage", "runs_dir"): self.runs_dir,
            ("storage", "cache_dir"): self.evolution_cache_dir,
            ("storage", "generated_strategy_dir"): self.evolution_strategy_dir,
            ("storage", "backtest_export_dir"): self.evolution_backtest_export_dir,
            ("storage", "backtest_user_data_dir"): self.evolution_user_data_dir,
            ("hall_of_fame", "directory"): self.hall_of_fame_dir,
            ("logging", "file"): self.engine_log_path,
        }
        mismatches: dict[str, tuple[str, Any]] = {}
        for (section, key), required in expected.items():
            actual = config.get(section, {}).get(key)
            if actual != required:
                mismatches[f"{section}.{key}"] = (required, actual)
        if mismatches:
            raise ValueError(f"engine config violates attempt output layout: {mismatches}")

    def isolated_backtester_config(
        self,
        config: Mapping[str, Any],
        *,
        phase: Literal["evolution", "replay"],
    ) -> dict[str, Any]:
        import copy

        isolated = copy.deepcopy(dict(config))
        storage = isolated.setdefault("storage", {})
        if phase == "evolution":
            storage["cache_dir"] = self.evolution_cache_dir
            storage["generated_strategy_dir"] = self.evolution_strategy_dir
            storage["backtest_export_dir"] = self.evolution_backtest_export_dir
            storage["backtest_user_data_dir"] = self.evolution_user_data_dir
        else:
            storage["cache_dir"] = self.replay_cache_dir
            storage["generated_strategy_dir"] = self.replay_strategy_dir
            storage["backtest_export_dir"] = self.replay_backtest_export_dir
            storage["backtest_user_data_dir"] = self.replay_user_data_dir
        return isolated


@contextmanager
def attempt_output_scope(
    layout: AttemptOutputLayoutV2,
    *,
    phase: Literal["evolution", "replay"],
) -> Iterator[Path]:
    """Contain relative writes and ignore the removed ``GA_OUTPUT_DIR`` alias."""

    work_dir = Path(layout.evolution_work_dir if phase == "evolution" else layout.replay_work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    inherited_output = os.environ.pop("GA_OUTPUT_DIR", None)
    try:
        os.chdir(work_dir)
        yield work_dir
    finally:
        os.chdir(previous_cwd)
        if inherited_output is not None:
            os.environ["GA_OUTPUT_DIR"] = inherited_output
