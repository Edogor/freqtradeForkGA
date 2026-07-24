"""Execute frozen candidates over a predeclared V2 shadow scenario panel."""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import Field, model_validator

from genetic_algorithm.evaluation.direct_backtester import BacktestResult, DirectBacktester
from genetic_algorithm.orchestration.artifact_store_v2 import (
    ArtifactIntegrityError,
    canonical_config_hash,
)
from genetic_algorithm.orchestration.code_manifest_v2 import (
    CodeManifestV2,
    capture_code_manifest,
)
from genetic_algorithm.orchestration.data_manifest_v2 import (
    DataManifestV2,
    build_data_manifest,
)
from genetic_algorithm.orchestration.final_test_ledger_v2 import FinalTestUsageLedgerV2
from genetic_algorithm.orchestration.promotion_policy_v2 import (
    ShadowGatePolicyV2,
    shadow_gate_policy_from_config,
)
from genetic_algorithm.orchestration.result_adapter import (
    BacktestContextV2,
    adapt_shadow_backtest_result,
)
from genetic_algorithm.orchestration.result_contract import (
    AttemptManifestV2,
    AttemptResultV2,
    ScenarioRole,
    StrictV2Model,
)
from genetic_algorithm.orchestration.shadow_attempt_v2 import ShadowAttemptRecorderV2
from genetic_algorithm.orchestration.split_contract_v2 import (
    EvaluationSplitPlanV2,
    build_evaluation_split_plan,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class FrozenCandidateV2(StrictV2Model):
    """One immutable executable strategy submitted to the replay panel."""

    candidate_id: str = Field(min_length=1)
    strategy_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    strategy_code: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    max_open_trades: int = Field(ge=1)
    execution_parameters: dict[str, Any] = Field(default_factory=dict)
    phenotype_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash_matches_executable(self) -> FrozenCandidateV2:
        expected = executable_phenotype_hash(
            strategy_name=self.strategy_name,
            strategy_code=self.strategy_code,
            timeframe=self.timeframe,
            max_open_trades=self.max_open_trades,
            execution_parameters=self.execution_parameters,
        )
        if self.phenotype_hash != expected:
            raise ValueError("phenotype_hash differs from executable candidate content")
        return self


def executable_phenotype_hash(
    *,
    strategy_name: str,
    strategy_code: str,
    timeframe: str,
    max_open_trades: int,
    execution_parameters: Mapping[str, Any] | None = None,
) -> str:
    """Hash code plus every execution parameter outside generated code."""

    normalized_code = strategy_code.replace("\r\n", "\n").replace("\r", "\n")
    try:
        tree = ast.parse(normalized_code)
    except SyntaxError as exc:
        raise ValueError("strategy_code is not valid Python") from exc
    strategy_classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == strategy_name
    ]
    if not strategy_classes:
        raise ValueError(f"strategy_code does not define class {strategy_name}")
    declared_timeframes: list[str] = []
    for node in strategy_classes[0].body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "timeframe" for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "timeframe"
        ):
            value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            declared_timeframes.append(value.value)
    if declared_timeframes != [timeframe]:
        raise ValueError(
            "strategy_code must declare exactly one literal timeframe matching the candidate"
        )
    if max_open_trades < 1:
        raise ValueError("max_open_trades must be positive")
    return canonical_config_hash(
        {
            "strategy_name": strategy_name,
            "strategy_code": normalized_code,
            "timeframe": timeframe,
            "max_open_trades": max_open_trades,
            "execution_parameters": dict(execution_parameters or {}),
        }
    )


def freeze_candidate(
    *,
    candidate_id: str,
    strategy_name: str,
    strategy_code: str,
    timeframe: str,
    max_open_trades: int,
    execution_parameters: Mapping[str, Any] | None = None,
) -> FrozenCandidateV2:
    parameters = dict(execution_parameters or {})
    return FrozenCandidateV2(
        candidate_id=candidate_id,
        strategy_name=strategy_name,
        strategy_code=strategy_code,
        timeframe=timeframe,
        max_open_trades=max_open_trades,
        execution_parameters=parameters,
        phenotype_hash=executable_phenotype_hash(
            strategy_name=strategy_name,
            strategy_code=strategy_code,
            timeframe=timeframe,
            max_open_trades=max_open_trades,
            execution_parameters=parameters,
        ),
    )


class ShadowReplayRunnerV2:
    """Run one-seed shadow replays and commit records through the V2 recorder."""

    def __init__(
        self,
        *,
        manifest: AttemptManifestV2,
        resolved_config: Mapping[str, Any],
        policy: ShadowGatePolicyV2,
        data_manifest: DataManifestV2,
        code_manifest: CodeManifestV2,
        final_test_ledger: FinalTestUsageLedgerV2,
        repo_root: str,
        backtester_factory: Callable[[dict[str, Any]], Any] = DirectBacktester,
        data_manifest_builder: Callable[
            [dict, ShadowGatePolicyV2], DataManifestV2
        ] = build_data_manifest,
        code_manifest_builder: Callable[[str], CodeManifestV2] = capture_code_manifest,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.manifest = manifest
        self.config = copy.deepcopy(dict(resolved_config))
        self.policy = policy
        self.data_manifest = data_manifest
        self.code_manifest = code_manifest
        self.final_test_ledger = final_test_ledger
        self.repo_root = repo_root
        self.backtester_factory = backtester_factory
        self.data_manifest_builder = data_manifest_builder
        self.code_manifest_builder = code_manifest_builder
        self.clock = clock
        self.split_manifest: EvaluationSplitPlanV2 | None = None
        self._validate_preflight()

    def _validate_preflight(self) -> None:
        if canonical_config_hash(self.config) != self.manifest.config_hash:
            raise ArtifactIntegrityError("resolved config differs from manifest config_hash")
        if len(self.manifest.seeds) != 1:
            raise ArtifactIntegrityError("replay runner currently requires exactly one seed")
        configured_policy = shadow_gate_policy_from_config(self.config)
        if configured_policy != self.policy:
            raise ArtifactIntegrityError("runtime policy differs from resolved promotion_v2 config")
        self._validate_code_preflight()
        self._validate_data_preflight()
        self._validate_split_preflight()
        self._validate_runtime_profile()

    def _validate_code_preflight(self) -> None:
        if self.code_manifest.code_version != self.manifest.code_version:
            raise ArtifactIntegrityError("code manifest differs from manifest code_version")
        if self.code_manifest.dirty_patch_hash != self.manifest.dirty_patch_hash:
            raise ArtifactIntegrityError("code manifest differs from manifest dirty_patch_hash")
        if self.code_manifest_builder(self.repo_root) != self.code_manifest:
            raise ArtifactIntegrityError("repository files differ from the frozen code manifest")

    def _validate_data_preflight(self) -> None:
        if self.data_manifest.manifest_hash != self.manifest.data_manifest_hash:
            raise ArtifactIntegrityError("data manifest differs from manifest data_manifest_hash")
        current_data_manifest = self.data_manifest_builder(self.config, self.policy)
        if current_data_manifest != self.data_manifest:
            raise ArtifactIntegrityError("OHLCV files differ from the frozen data manifest")

    def _validate_split_preflight(self) -> None:
        split_manifest = build_evaluation_split_plan(self.config, self.policy)
        if self.manifest.split_manifest_hash != split_manifest.split_hash:
            raise ArtifactIntegrityError("split plan differs from manifest split_manifest_hash")
        self.split_manifest = split_manifest

    def _validate_runtime_profile(self) -> None:
        safety = self.config.get("safety_profile", {})
        evaluation = self.config.get("evaluation_v2", {})
        backtesting = self.config.get("backtesting", {})
        if not safety.get("shadow_mode", False):
            raise ArtifactIntegrityError("V2 replay runner currently requires shadow_mode")
        if safety.get("automation_eligible", False):
            raise ArtifactIntegrityError("shadow replay cannot be automation_eligible")
        if not evaluation.get("enabled", False) or not evaluation.get("mark_to_market", False):
            raise ArtifactIntegrityError("V2 replay requires mark-to-market evaluation")
        if backtesting.get("fee_noise_std", 0.0) != 0.0:
            raise ArtifactIntegrityError("V2 replay forbids fee noise")
        if backtesting.get("dynamic_slippage", False):
            raise ArtifactIntegrityError("dynamic slippage lacks exact scenario provenance")
        if float(backtesting.get("spread_pct", 0.0) or 0.0) != 0.0:
            raise ArtifactIntegrityError("separate spread modeling is not implemented")
        if float(backtesting.get("funding_rate", 0.0) or 0.0) != 0.0:
            raise ArtifactIntegrityError("funding cost replay is not implemented")
        if self.config.get("short_selling", {}).get("enabled", False):
            raise ArtifactIntegrityError("current mark-to-market replay supports spot-long only")

    def _backtester_for_multiplier(self, multiplier: float) -> Any:
        scenario_config = copy.deepcopy(self.config)
        backtesting = scenario_config.setdefault("backtesting", {})
        base_fee = float(self.config.get("backtesting", {}).get("fee", 0.0))
        base_slippage = float(self.config.get("backtesting", {}).get("slippage_pct", 0.0))
        backtesting["fee"] = base_fee * multiplier
        backtesting["slippage_pct"] = base_slippage * multiplier
        backtesting["fee_noise_std"] = 0.0
        backtesting["pairs"] = sorted({item.pair for item in self.policy.required_scenarios})
        return self.backtester_factory(scenario_config)

    def _reserve_final_test_if_declared(self):
        if not any(
            item.role == ScenarioRole.FINAL_TEST
            for item in self.policy.required_scenarios
        ):
            return None
        return self.final_test_ledger.reserve(
            manifest=self.manifest,
            policy=self.policy,
            data_manifest=self.data_manifest,
            reserved_at=self.clock(),
        )

    def _expose_final_test_if_needed(self, requirement, reservation, exposed):
        if requirement.role != ScenarioRole.FINAL_TEST or exposed:
            return reservation, exposed
        if reservation is None:
            raise ArtifactIntegrityError("FINAL_TEST scenario lacks a ledger reservation")
        return (
            self.final_test_ledger.mark_exposed(
                reservation.reservation_id,
                exposed_at=self.clock(),
            ),
            True,
        )

    def execute(
        self,
        candidates: Sequence[FrozenCandidateV2],
        *,
        finished_at: datetime,
    ) -> AttemptResultV2:
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ArtifactIntegrityError("replay candidate_id values must be unique")
        reservation = self._reserve_final_test_if_declared()
        exposed = reservation is not None and reservation.status == "EXPOSED"
        try:
            recorder = ShadowAttemptRecorderV2(
                self.manifest,
                self.config,
                self.policy,
                self.data_manifest,
                self.code_manifest,
                self.split_manifest,
            )
            evaluation = self.config["evaluation_v2"]
            backtesting = self.config["backtesting"]
            seed = self.manifest.seeds[0]
            backtesters: dict[float, Any] = {}

            for candidate in candidates:
                recorder.add_frozen_candidate(candidate)
                for requirement in self.policy.required_scenarios:
                    if candidate.timeframe != requirement.timeframe:
                        result = BacktestResult(
                            success=False,
                            strategy_name=candidate.strategy_name,
                            error_message=(
                                f"candidate timeframe {candidate.timeframe} differs from "
                                f"scenario timeframe {requirement.timeframe}"
                            ),
                        )
                    else:
                        reservation, exposed = self._expose_final_test_if_needed(
                            requirement,
                            reservation,
                            exposed,
                        )
                        backtester = backtesters.get(requirement.cost_multiplier)
                        if backtester is None:
                            backtester = self._backtester_for_multiplier(
                                requirement.cost_multiplier
                            )
                            backtesters[requirement.cost_multiplier] = backtester
                        # Freqtrade timerange ends are exclusive; add one day so
                        # the declared inclusive period_end is fully evaluated.
                        exclusive_end = requirement.period_end + timedelta(days=1)
                        timerange = f"{requirement.period_start:%Y%m%d}-{exclusive_end:%Y%m%d}"
                        try:
                            result = backtester.backtest_strategy(
                                candidate.strategy_code,
                                candidate.strategy_name,
                                strategy_max_open_trades=candidate.max_open_trades,
                                timerange_override=timerange,
                                pairs_override=[requirement.pair],
                            )
                        except Exception as exc:
                            result = BacktestResult(
                                success=False,
                                strategy_name=candidate.strategy_name,
                                error_message=f"replay raised {type(exc).__name__}: {exc}",
                            )

                    multiplier = requirement.cost_multiplier
                    context = BacktestContextV2(
                        attempt_id=self.manifest.attempt_id,
                        wave_id=self.manifest.wave_id,
                        experiment_id=self.manifest.experiment_id,
                        candidate_id=candidate.candidate_id,
                        config_hash=self.manifest.config_hash,
                        phenotype_hash=candidate.phenotype_hash,
                        code_version=self.manifest.code_version,
                        data_manifest_hash=self.manifest.data_manifest_hash,
                        fitness_policy_version=self.manifest.fitness_policy_version,
                        seed=seed,
                        worker_count=self.manifest.worker_count,
                        scenario_id=requirement.scenario_id,
                        pair=requirement.pair,
                        timeframe=requirement.timeframe,
                        role=requirement.role,
                        period_start=requirement.period_start,
                        period_end=requirement.period_end,
                        cost_multiplier=multiplier,
                        fee_rate=float(backtesting.get("fee", 0.0)) * multiplier,
                        slippage_rate=float(backtesting.get("slippage_pct", 0.0)) * multiplier,
                        spread_rate=0.0,
                        funding_rate=0.0,
                    )
                    record = adapt_shadow_backtest_result(
                        result,
                        context,
                        bootstrap_samples=int(evaluation["bootstrap_samples"]),
                        bootstrap_block_days=int(evaluation["bootstrap_block_days"]),
                        trade_bootstrap_block=int(evaluation["trade_bootstrap_block"]),
                        expectancy_cluster_days=int(
                            evaluation.get("expectancy_cluster_days", 1)
                        ),
                        bootstrap_confidence=float(evaluation["bootstrap_confidence"]),
                    )
                    recorder.add_backtest(record)

            if reservation is not None and not exposed:
                reservation = self.final_test_ledger.release_unexposed(
                    reservation.reservation_id,
                    released_at=self.clock(),
                )
            if reservation is not None:
                recorder.add_final_test_usage(reservation)
            return recorder.finalize(finished_at=finished_at)
        except Exception:
            if reservation is not None and not exposed:
                self.final_test_ledger.release_unexposed(
                    reservation.reservation_id,
                    released_at=self.clock(),
                )
            raise
