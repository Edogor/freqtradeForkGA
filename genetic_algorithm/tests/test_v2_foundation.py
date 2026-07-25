"""Regression tests for the first GA V2 foundation slice."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from genetic_algorithm.tests.v2_metric_fixtures import trade_evidence


def _complete_backtest_result():
    from genetic_algorithm.evaluation.direct_backtester import BacktestResult

    return BacktestResult(
        success=True,
        strategy_name="StableStrategy",
        total_profit=123.45,
        profit_percent=12.345,
        total_trades=3,
        wins=2,
        losses=1,
        win_rate=2 / 3,
        max_drawdown=0.08,
        max_drawdown_abs=80.0,
        sharpe_ratio=1.2,
        sortino_ratio=1.7,
        profit_factor=1.9,
        avg_profit=0.004,
        median_profit=0.003,
        avg_duration="0 days 04:00:00",
        error_message=None,
        execution_time=1.25,
        no_trades=False,
        trades=[
            {"pair": "BTC/USDT", "profit_ratio": 0.01},
            {"pair": "ETH/USDT", "profit_ratio": -0.005},
            {"pair": "BTC/USDT", "profit_ratio": 0.02},
        ],
        ohlcv_data={"must": "not be cached"},
        per_pair_profit={"BTC/USDT": 3.0, "ETH/USDT": -0.5},
        per_pair_trades={"BTC/USDT": 2, "ETH/USDT": 1},
        monthly_profits=[1.0, -0.2, 2.4],
        monthly_periods=["2025-01", "2025-02", "2025-03"],
        starting_balance=100.0,
        final_balance=112.345,
        backtest_start="2025-01-01 00:00:00",
        backtest_end="2025-01-03 00:00:00",
        daily_profit_abs=[["2025-01-01", 5.0], ["2025-01-02", -1.0], ["2025-01-03", 8.345]],
        daily_net_returns=[0.05, -1.0 / 105.0, 8.345 / 104.0],
        equity_curve=[100.0, 105.0, 104.0, 112.345],
        equity_method="REALIZED_CLOSE",
        equity_error_message=None,
        annualized_net_return=2.5,
        daily_sharpe_ratio=1.1,
        daily_sortino_ratio=1.5,
        daily_expected_shortfall_5=1.0 / 105.0,
        calmar_ratio=20.0,
        ulcer_index=0.006,
        time_under_water_ratio=1.0 / 3.0,
        daily_max_drawdown=1.0 / 105.0,
        max_consecutive_losses=1,
        max_drawdown_duration_days=4.5,
        trade_profit_ratios=[0.01, -0.005, 0.02],
    )


class TestBacktestCacheV2:
    def test_result_dict_preserves_every_evaluation_field(self):
        result = _complete_backtest_result()
        payload = result.to_dict()

        assert "ohlcv_data" not in payload
        assert payload["trades"] == result.trades
        assert payload["per_pair_profit"] == result.per_pair_profit
        assert payload["per_pair_trades"] == result.per_pair_trades
        assert payload["monthly_profits"] == result.monthly_profits
        assert payload["monthly_periods"] == result.monthly_periods
        assert payload["daily_net_returns"] == result.daily_net_returns
        assert payload["equity_curve"] == result.equity_curve
        assert payload["equity_method"] == "REALIZED_CLOSE"
        assert payload["mark_to_market_error_message"] is None
        assert payload["daily_expected_shortfall_5"] == result.daily_expected_shortfall_5
        assert payload["risk_metric_contract_version"] == "calendar-effective-v1"
        assert payload["risk_periods_per_year"] == 365
        assert payload["annual_risk_free_rate"] == 0.0
        assert payload["periodic_risk_free_rate"] == 0.0
        assert payload["max_consecutive_losses"] == 1
        assert payload["max_drawdown_duration_days"] == 4.5
        assert payload["trade_profit_ratios"] == result.trade_profit_ratios
        assert payload["no_trades"] is False
        assert payload["profit_factor_censored"] is False
        assert (
            payload["profit_factor_contract_version"]
            == "right-censored-profit-factor-v1"
        )

    def test_ram_and_disk_hits_have_identical_metric_payloads(self, tmp_path: Path):
        from genetic_algorithm.evaluation.cache import BacktestCache, CACHE_SCHEMA_VERSION

        result = _complete_backtest_result()
        strategy_code = "class StableStrategy: pass"
        config = {"pairs": ["BTC/USDT", "ETH/USDT"], "seed": 42}

        cache = BacktestCache(cache_dir=tmp_path)
        cache.put(strategy_code, config, result)
        ram_hit = cache.get(strategy_code, config)

        restarted_cache = BacktestCache(cache_dir=tmp_path)
        disk_hit = restarted_cache.get(strategy_code, config)

        assert ram_hit is result
        assert disk_hit is not result
        assert disk_hit.to_dict() == result.to_dict()
        assert disk_hit.ohlcv_data is None

        payload = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert payload["_cache_schema_version"] == CACHE_SCHEMA_VERSION

    def test_unversioned_payload_is_rejected_fail_closed(self, tmp_path: Path):
        from genetic_algorithm.evaluation.cache import BacktestCache

        cache = BacktestCache(cache_dir=tmp_path)
        code = "class Legacy: pass"
        config = {"legacy": True}
        key = cache._get_cache_key(code, config)
        (tmp_path / f"{key}.json").write_text(json.dumps({
            "success": True,
            "strategy_name": "Legacy",
            "profit_percent": 99.0,
        }))

        assert cache.get(code, config) is None
        assert not (tmp_path / f"{key}.json").exists()


class TestDeterminismV2:
    def test_fee_noise_seed_has_fixed_process_independent_value(self):
        from genetic_algorithm.evaluation.direct_backtester import _stable_fee_noise_seed

        assert _stable_fee_noise_seed(42, "Strategy_A") == 8724337150007245567
        assert _stable_fee_noise_seed(42, "Strategy_A") != _stable_fee_noise_seed(43, "Strategy_A")


class TestFitnessMetricContractV2:
    @staticmethod
    def _penalty_evaluator(**penalty_overrides):
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        evaluator = object.__new__(FitnessEvaluator)
        evaluator.fitness_penalties = {
            "min_trades": 0,
            "max_drawdown": 10.0,
            "min_win_rate": 0.0,
            "complexity_weight": 0.0,
            "unused_indicator_weight": 0.0,
            "min_penalty_floor": 0.0,
            **penalty_overrides,
        }
        evaluator.backtest_config = {"fee": 0.001, "slippage_pct": 0.0, "pairs": ["BTC/USDT"]}
        evaluator.min_trades_per_month = 0
        evaluator.timerange_months = 12
        return evaluator

    @staticmethod
    def _strategy_stub():
        return SimpleNamespace(
            timeframe="1h",
            calculate_complexity=lambda: 0,
            indicators=[],
            entry_conditions=[],
            exit_conditions=[],
        )

    def test_mapper_keeps_penalty_inputs(self):
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        evaluator = object.__new__(FitnessEvaluator)
        metrics = evaluator._backtest_result_to_metrics(_complete_backtest_result())

        assert metrics["avg_profit"] == 0.004
        assert metrics["avg_duration"] == "0 days 04:00:00"
        assert metrics["trades"] == _complete_backtest_result().trades
        assert metrics["per_pair_trades"] == {"BTC/USDT": 2, "ETH/USDT": 1}
        assert metrics["worst_pair_trades"] == 1
        assert metrics["active_pair_ratio"] == 1.0

    def test_pair_trade_coverage_penalizes_the_worst_pair_smoothly(self):
        evaluator = self._penalty_evaluator(
            target_trades_per_pair=60,
            pair_trade_penalty_floor=0.01,
        )
        base = {
            "num_trades": 120,
            "max_drawdown": 0.0,
            "win_rate": 0.5,
        }

        baseline = evaluator._apply_penalties(1.0, base)
        balanced = evaluator._apply_penalties(
            1.0,
            {**base, "per_pair_trades": {"BTC/USDT": 60, "SOL/USDT": 60}},
        )
        sparse = evaluator._apply_penalties(
            1.0,
            {**base, "per_pair_trades": {"BTC/USDT": 119, "SOL/USDT": 1}},
        )
        absent = evaluator._apply_penalties(
            1.0,
            {**base, "per_pair_trades": {"BTC/USDT": 120, "SOL/USDT": 0}},
        )

        assert balanced == pytest.approx(baseline)
        assert sparse == pytest.approx(baseline * 0.0265)
        assert absent == pytest.approx(baseline * 0.01)

    def test_pair_trade_coverage_is_not_weakened_by_general_penalty_floor(self):
        evaluator = self._penalty_evaluator(
            target_trades_per_pair=60,
            pair_trade_penalty_floor=0.01,
            min_penalty_floor=0.10,
        )
        metrics = {
            "num_trades": 120,
            "max_drawdown": 0.0,
            "win_rate": 0.5,
            "per_pair_trades": {"BTC/USDT": 0, "SOL/USDT": 120},
        }

        baseline = evaluator._apply_penalties(
            1.0,
            {key: value for key, value in metrics.items() if key != "per_pair_trades"},
        )
        fitness = evaluator._apply_penalties(1.0, metrics)

        assert fitness == pytest.approx(baseline * 0.01)
        assert metrics["pair_trade_coverage_multiplier"] == pytest.approx(0.01)

    def test_pair_summary_accepts_freqtrade_json_trade_records(self):
        from genetic_algorithm.evaluation.direct_backtester import (
            _summarize_trades_by_pair,
        )

        profit, counts = _summarize_trades_by_pair(
            [
                {"pair": "SOL/USDT", "profit_ratio": 0.02},
                {"pair": "SOL/USDT", "profit_ratio": -0.005},
                {"pair": "ETH/USDT", "profit_ratio": 0.01},
            ],
            ["BTC/USDT", "SOL/USDT"],
        )

        assert counts == {"BTC/USDT": 0, "SOL/USDT": 2, "ETH/USDT": 1}
        assert profit == pytest.approx(
            {"BTC/USDT": 0.0, "SOL/USDT": 1.5, "ETH/USDT": 1.0}
        )

    def test_mapper_does_not_invent_missing_tail_risk(self):
        from genetic_algorithm.evaluation.direct_backtester import BacktestResult
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        evaluator = object.__new__(FitnessEvaluator)
        metrics = evaluator._backtest_result_to_metrics(
            BacktestResult(success=True, strategy_name="MissingTail")
        )

        assert "max_consecutive_losses" not in metrics
        assert "max_drawdown_duration_days" not in metrics

    def test_spread_penalty_uses_decimal_avg_profit_without_double_conversion(self):
        evaluator = self._penalty_evaluator(
            spread_aware_enabled=True,
            spread_aware_min_edge_multiplier=2.0,
        )

        metrics = {
            "num_trades": 100,
            "max_drawdown": 0.0,
            "win_rate": 0.5,
            # 0.20% average edge versus a required 0.40% (2x round-trip cost).
            "avg_profit": 0.002,
        }
        evaluator.fitness_penalties["spread_aware_enabled"] = False
        baseline = evaluator._apply_penalties(1.0, metrics)
        evaluator.fitness_penalties["spread_aware_enabled"] = True
        penalized = evaluator._apply_penalties(1.0, metrics)

        assert penalized == pytest.approx(baseline * 0.5, abs=1e-12)

    def test_duration_penalty_is_reachable_from_mapped_duration(self):
        evaluator = self._penalty_evaluator(
            max_avg_trade_candles=2,
            trade_duration_penalty_weight=0.3,
        )
        strategy = self._strategy_stub()
        metrics = {
            "num_trades": 20,
            "max_drawdown": 0.0,
            "win_rate": 0.5,
            "avg_duration": "0 days 04:00:00",
        }

        evaluator.fitness_penalties["max_avg_trade_candles"] = 0
        baseline = evaluator._apply_penalties(1.0, metrics, strategy)
        evaluator.fitness_penalties["max_avg_trade_candles"] = 2
        penalized = evaluator._apply_penalties(1.0, metrics, strategy)

        assert penalized == pytest.approx(baseline * 0.7, abs=1e-12)

    def test_clustering_penalty_is_reachable_from_trade_records(self):
        evaluator = self._penalty_evaluator(
            trade_clustering_enabled=True,
            trade_clustering_max_per_window=3,
            trade_clustering_window_candles=5,
        )
        strategy = self._strategy_stub()
        start = datetime(2025, 1, 1)
        trades = [{"open_date": start + timedelta(minutes=i)} for i in range(12)]
        metrics = {
            "num_trades": len(trades),
            "max_drawdown": 0.0,
            "win_rate": 0.5,
            "trades": trades,
        }

        evaluator.fitness_penalties["trade_clustering_enabled"] = False
        baseline = evaluator._apply_penalties(1.0, metrics, strategy)
        evaluator.fitness_penalties["trade_clustering_enabled"] = True
        penalized = evaluator._apply_penalties(1.0, metrics, strategy)

        assert evaluator._compute_cluster_ratio(trades, 3, 5, "1h") > 0.3
        assert penalized < baseline


class TestSafeV2Preset:
    def test_safe_v2_resolves_to_enforced_shadow_baseline(self):
        from genetic_algorithm.config.schema import load_config

        config = load_config("genetic_algorithm/config/presets/safe_v2.yaml")

        assert config["config_schema_version"] == 2
        assert config["safety_profile"] == {
            "name": "safe_v2",
            "enforce": True,
            "shadow_mode": True,
            "automation_eligible": False,
        }
        assert config["strategy_constraints"]["timeframes"] == ["1h"]
        assert config["genetic_algorithm"]["mode"] == "single_objective"
        assert config["backtesting"]["fee_noise_std"] == 0.0
        assert config["evaluation_v2"]["enabled"] is True
        assert config["evaluation_v2"]["mark_to_market"] is True
        assert config["promotion_v2"]["enabled"] is False
        for section in (
            "walk_forward", "holdout_validation", "holdout_monitoring",
            "monte_carlo", "deflated_sharpe", "surrogate", "regime_aware",
            "pair_validation", "cpcv", "generic_island_model", "island_model",
            "sis", "llm", "short_selling",
        ):
            assert config[section]["enabled"] is False
        assert config["advanced"]["llm"]["enabled"] is False

    def test_safe_v2_rejects_reenabling_untrusted_feature(self, tmp_path: Path):
        from genetic_algorithm.config.schema import load_config

        config_path = tmp_path / "unsafe_override.yaml"
        config_path.write_text(yaml.safe_dump({
            "preset": "safe_v2",
            "walk_forward": {"enabled": True},
        }))

        with pytest.raises(ValueError, match="safe_v2 forbids walk_forward"):
            load_config(config_path)

    def test_enabled_promotion_v2_requires_predeclared_scenario_matrix(self, tmp_path: Path):
        from genetic_algorithm.config.schema import load_config

        config_path = tmp_path / "missing_panel.yaml"
        config_path.write_text(yaml.safe_dump({
            "preset": "safe_v2",
            "promotion_v2": {"enabled": True, "required_scenarios": []},
        }))

        with pytest.raises(ValueError, match="invalid promotion_v2 policy"):
            load_config(config_path)


class TestStrictV2Contracts:
    def test_valid_scenario_cannot_hide_missing_risk_metrics(self):
        from pydantic import ValidationError
        from genetic_algorithm.orchestration.result_contract import ScenarioMetricsV2

        with pytest.raises(ValidationError, match="missing metrics"):
            ScenarioMetricsV2(
                scenario_id="btc-inner-baseline",
                pair="BTC/USDT",
                timeframe="1h",
                role="INNER_VALIDATION",
                period_start=date(2025, 1, 1),
                period_end=date(2025, 6, 1),
                cost_multiplier=1.0,
                status="VALID",
                success=True,
                trade_count=20,
                active_months=5,
                net_return=0.04,
            )

    def test_invalid_scenario_requires_machine_readable_reason(self):
        from pydantic import ValidationError
        from genetic_algorithm.orchestration.result_contract import ScenarioMetricsV2

        with pytest.raises(ValidationError, match="error_code"):
            ScenarioMetricsV2(
                scenario_id="btc-final",
                pair="BTC/USDT",
                timeframe="1h",
                role="FINAL_TEST",
                period_start=date(2025, 1, 1),
                period_end=date(2025, 6, 1),
                cost_multiplier=2.0,
                status="INVALID",
                success=False,
                trade_count=0,
                active_months=0,
            )

    def test_shadow_decision_cannot_authorize_promotion(self):
        from datetime import datetime, timezone
        from pydantic import ValidationError
        from genetic_algorithm.orchestration.result_contract import PromotionDecisionV2

        with pytest.raises(ValidationError, match="shadow decisions cannot authorize"):
            PromotionDecisionV2(
                decision_id="decision-1",
                wave_id="wave-1",
                candidate_id="candidate-1",
                policy_version="shadow-1",
                created_at=datetime.now(timezone.utc),
                shadow_mode=True,
                outcome="PASS",
                promotion_authorized=True,
                gate_results=[{
                    "gate_id": "return-lcb",
                    "passed": True,
                    "status": "VALID",
                    "reason_code": "RETURN_LCB_POSITIVE",
                }],
                reason_codes=["ALL_GATES_PASS"],
            )


class TestLegacyV2Adapter:
    @staticmethod
    def _context():
        from genetic_algorithm.orchestration.legacy_adapter import LegacyBacktestContextV2

        return LegacyBacktestContextV2(
            attempt_id="legacy-attempt-1",
            wave_id="wave-44",
            experiment_id="wave44-A",
            candidate_id="candidate-A",
            config_hash="config-hash-123",
            phenotype_hash="phenotype-hash-123",
            code_version="legacy-dirty-tree",
            data_manifest_hash="data-hash-123",
            fitness_policy_version="legacy-import-1",
            seed=42,
            worker_count=4,
            scenario_id="btc-legacy-train",
            pair="BTC/USDT",
            timeframe="1h",
            role="TRAIN",
            period_start=date(2021, 1, 1),
            period_end=date(2024, 12, 31),
            cost_multiplier=1.0,
            fee_rate=0.001,
            slippage_rate=0.0005,
            spread_rate=0.0,
            funding_rate=0.0,
        )

    def test_successful_legacy_result_still_requires_v2_replay(self):
        from genetic_algorithm.orchestration.legacy_adapter import adapt_legacy_backtest_result

        record = adapt_legacy_backtest_result(_complete_backtest_result(), self._context())

        assert record.metrics.status.value == "INCONCLUSIVE"
        assert record.metrics.error_code == "LEGACY_REPLAY_REQUIRED"
        assert record.metrics.net_return == pytest.approx(0.12345)
        assert record.daily_net_returns == []
        assert record.equity_curve == []

    def test_failed_legacy_result_stays_failed(self):
        from genetic_algorithm.evaluation.direct_backtester import BacktestResult
        from genetic_algorithm.orchestration.legacy_adapter import adapt_legacy_backtest_result

        result = BacktestResult(
            success=False,
            strategy_name="BrokenLegacy",
            error_message="timeout",
        )
        record = adapt_legacy_backtest_result(result, self._context())

        assert record.metrics.status.value == "FAIL"
        assert record.metrics.error_code == "LEGACY_BACKTEST_FAILED"


class TestShadowResultAdapter:
    @staticmethod
    def _context():
        from genetic_algorithm.orchestration.result_adapter import BacktestContextV2

        return BacktestContextV2(
            attempt_id="attempt-shadow-1",
            wave_id="wave-shadow-1",
            experiment_id="experiment-shadow-1",
            candidate_id="candidate-shadow-1",
            config_hash="config-shadow-hash",
            phenotype_hash="phenotype-shadow-hash",
            code_version="commit-shadow",
            data_manifest_hash="manifest-shadow-hash",
            fitness_policy_version="shadow-policy-1",
            seed=42,
            worker_count=4,
            scenario_id="btc-inner-baseline",
            pair="BTC/USDT",
            timeframe="1h",
            role="INNER_VALIDATION",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 4, 30),
            cost_multiplier=1.0,
            fee_rate=0.001,
            slippage_rate=0.0005,
            spread_rate=0.0,
            funding_rate=0.0,
        )

    @staticmethod
    def _long_realized_result():
        from genetic_algorithm.evaluation.direct_backtester import BacktestResult
        from genetic_algorithm.evaluation.equity_metrics_v2 import (
            build_realized_close_equity,
            calculate_equity_risk_metrics,
        )

        daily_pnl = []
        for offset in range(120):
            pnl = 0.2 if offset % 10 else -1.0
            daily_pnl.append((date(2025, 1, 1) + timedelta(days=offset), pnl))
        series = build_realized_close_equity(
            period_start=date(2025, 1, 1),
            period_end=date(2025, 4, 30),
            starting_balance=100.0,
            daily_profit_abs=daily_pnl,
        )
        risk = calculate_equity_risk_metrics(series)
        return BacktestResult(
            success=True,
            strategy_name="ShadowStrategy",
            total_profit=series.final_balance - series.starting_balance,
            profit_percent=(series.final_balance / series.starting_balance - 1.0) * 100.0,
            total_trades=30,
            wins=20,
            losses=10,
            win_rate=2 / 3,
            profit_factor=1.5,
            avg_profit=0.001,
            trades=[{"profit_ratio": 0.001}] * 30,
            starting_balance=series.starting_balance,
            final_balance=series.final_balance,
            backtest_start="2025-01-01",
            backtest_end="2025-04-30",
            daily_profit_abs=[
                [day.isoformat(), pnl]
                for day, pnl in zip(series.dates, series.daily_profit_abs)
            ],
            daily_net_returns=list(series.daily_net_returns),
            equity_curve=list(series.equity_curve),
            equity_method="REALIZED_CLOSE",
            annualized_net_return=risk.annualized_net_return,
            daily_sharpe_ratio=risk.sharpe_ratio,
            daily_sortino_ratio=risk.sortino_ratio,
            daily_expected_shortfall_5=risk.daily_expected_shortfall_5,
            calmar_ratio=risk.calmar_ratio,
            ulcer_index=risk.ulcer_index,
            time_under_water_ratio=risk.time_under_water_ratio,
            daily_max_drawdown=risk.max_drawdown,
            max_consecutive_losses=1,
            max_drawdown_duration_days=risk.max_drawdown_duration_days,
        )

    @classmethod
    def _complete_mark_to_market_result(cls):
        result = cls._long_realized_result()
        trade_returns = [0.012, 0.012, 0.012, -0.02, 0.003, 0.01, -0.015] * 5
        result.equity_method = "MARK_TO_MARKET"
        result.total_trades = len(trade_returns)
        result.avg_profit = sum(trade_returns) / len(trade_returns)
        result.trade_profit_ratios = trade_returns
        committed_capital = result.total_profit / (
            sum(trade_returns) * 1.001
        )
        result.trades = trade_evidence(
            trade_returns,
            committed_capital=committed_capital,
        )
        return result

    def test_realized_equity_gets_bounds_but_cannot_be_valid(self):
        from genetic_algorithm.orchestration.result_adapter import adapt_shadow_backtest_result

        first = adapt_shadow_backtest_result(
            self._long_realized_result(),
            self._context(),
            bootstrap_samples=200,
        )
        second = adapt_shadow_backtest_result(
            self._long_realized_result(),
            self._context(),
            bootstrap_samples=200,
        )

        assert first == second
        assert first.metrics.status.value == "INCONCLUSIVE"
        assert first.metrics.error_code == "MARK_TO_MARKET_REPLAY_REQUIRED"
        assert first.metrics.annualized_net_return_lcb is not None
        assert first.metrics.max_drawdown_ucb is not None
        assert first.metrics.daily_expected_shortfall_5_ucb is not None
        assert first.equity_method == "REALIZED_CLOSE"

    def test_complete_mark_to_market_scenario_is_valid_with_trade_confidence(self):
        from genetic_algorithm.orchestration.result_adapter import adapt_shadow_backtest_result

        result = self._complete_mark_to_market_result()

        record = adapt_shadow_backtest_result(
            result,
            self._context(),
            bootstrap_samples=200,
            trade_bootstrap_block=5,
        )

        assert record.metrics.status.value == "VALID"
        assert record.metrics.error_code is None
        assert record.metrics.net_expectancy == pytest.approx(result.avg_profit)
        assert record.metrics.net_expectancy_lcb < record.metrics.net_expectancy
        assert record.metrics.net_expectancy_on_committed_capital == pytest.approx(
            result.avg_profit
        )
        assert (
            record.metrics.net_expectancy_on_committed_capital_lcb
            < record.metrics.net_expectancy_on_committed_capital
        )
        assert (
            record.metrics.expectancy_contract_version
            == "net-expectancy-clustered-v1"
        )
        assert 1 <= record.metrics.effective_sample_size <= result.total_trades
        assert record.risk_metric_contract_version == "calendar-effective-v1"
        assert record.risk_periods_per_year == 365
        assert record.annual_risk_free_rate == 0.0
        assert record.periodic_risk_free_rate == 0.0
        assert record.metrics.daily_sharpe_ratio == result.daily_sharpe_ratio
        assert record.metrics.daily_sortino_ratio == result.daily_sortino_ratio
        assert record.metrics.calmar_ratio == result.calmar_ratio

    def test_trade_ledger_profit_mismatch_is_invalid(self):
        from genetic_algorithm.orchestration.result_adapter import adapt_shadow_backtest_result

        result = self._complete_mark_to_market_result()
        result.trades[0]["profit_abs"] += 1.0
        result.trades[0]["profit_ratio"] = result.trades[0]["profit_abs"] / (
            result.trades[0]["max_stake_amount"]
            * (1.0 + result.trades[0]["fee_open"])
        )

        record = adapt_shadow_backtest_result(
            result,
            self._context(),
            bootstrap_samples=200,
            trade_bootstrap_block=5,
        )

        assert record.metrics.status.value == "INVALID"
        assert record.metrics.error_code == "TRADE_PROFIT_MISMATCH"

    def test_trade_ledger_wallet_mismatch_is_invalid(self):
        from genetic_algorithm.orchestration.result_adapter import adapt_shadow_backtest_result

        result = self._complete_mark_to_market_result()
        result.final_balance += 1.0

        record = adapt_shadow_backtest_result(
            result,
            self._context(),
            bootstrap_samples=200,
            trade_bootstrap_block=5,
        )

        assert record.metrics.status.value == "INVALID"
        assert record.metrics.error_code == "TRADE_WALLET_PROFIT_MISMATCH"

    def test_missing_equity_is_invalid_not_zero_risk(self):
        from genetic_algorithm.evaluation.direct_backtester import BacktestResult
        from genetic_algorithm.orchestration.result_adapter import adapt_shadow_backtest_result

        result = BacktestResult(
            success=True,
            strategy_name="MissingEquity",
            profit_percent=10.0,
            total_trades=20,
            wins=15,
            losses=5,
            win_rate=0.75,
        )
        record = adapt_shadow_backtest_result(result, self._context(), bootstrap_samples=200)

        assert record.metrics.status.value == "INVALID"
        assert record.metrics.error_code == "MISSING_EQUITY_EVIDENCE"
        assert record.metrics.max_drawdown is None
        assert record.metrics.daily_expected_shortfall_5 is None
