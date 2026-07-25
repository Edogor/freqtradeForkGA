"""
Direct Backtesting Integration Module

Uses FreqTrade Python API directly with mocked exchange to avoid network calls.
"""

import gc
import json
import logging
import math
import os
import signal
import sys
import tempfile
import time
import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch
from dataclasses import dataclass

from genetic_algorithm.core.strategy_gene import timeframe_to_minutes as _timeframe_to_minutes_util
from genetic_algorithm.evaluation.cache import BacktestCache
from genetic_algorithm.evaluation.equity_metrics_v2 import (
    CALENDAR_DAYS_PER_YEAR,
    DEFAULT_ANNUAL_RISK_FREE_RATE,
    EquityDataError,
    RISK_METRIC_CONTRACT_VERSION,
    build_equity_from_freqtrade_stats,
    build_mark_to_market_equity,
    calculate_equity_risk_metrics,
)
from genetic_algorithm.evaluation.profit_factor_v2 import (
    PROFIT_FACTOR_CONTRACT_VERSION,
    normalize_profit_factor,
)

logger = logging.getLogger(__name__)


def _summarize_trades_by_pair(
    trades: Any,
    expected_pairs: List[str],
) -> tuple[Dict[str, float], Dict[str, int]]:
    """Return pair profit percentages and counts from dataframe or JSON records.

    ``generate_backtest_stats`` exposes serialized trades as a list on the
    active Freqtrade version.  The old extractor only handled a pandas
    dataframe, silently leaving every declared pair at zero even when the
    aggregate result contained hundreds of trades.
    """

    per_pair_profit = {pair: 0.0 for pair in expected_pairs}
    per_pair_trades = {pair: 0 for pair in expected_pairs}
    if trades is None:
        return per_pair_profit, per_pair_trades

    if hasattr(trades, "groupby") and hasattr(trades, "__len__"):
        if len(trades) == 0:
            return per_pair_profit, per_pair_trades
        for pair, group in trades.groupby("pair"):
            pair_name = str(pair)
            per_pair_profit.setdefault(pair_name, 0.0)
            per_pair_trades.setdefault(pair_name, 0)
            if "profit_ratio" in group.columns:
                per_pair_profit[pair_name] = float(group["profit_ratio"].sum()) * 100
            per_pair_trades[pair_name] = len(group)
        return per_pair_profit, per_pair_trades

    if isinstance(trades, list):
        for trade in trades:
            if not isinstance(trade, Mapping):
                continue
            pair = trade.get("pair")
            if not isinstance(pair, str) or not pair:
                continue
            per_pair_profit.setdefault(pair, 0.0)
            per_pair_trades.setdefault(pair, 0)
            per_pair_trades[pair] += 1
            try:
                profit_ratio = float(trade.get("profit_ratio", 0.0))
            except (TypeError, ValueError):
                profit_ratio = 0.0
            if math.isfinite(profit_ratio):
                per_pair_profit[pair] += profit_ratio * 100
        return per_pair_profit, per_pair_trades

    logger.warning(
        "Unsupported trade payload type for pair metrics: %s",
        type(trades).__name__,
    )
    return per_pair_profit, per_pair_trades


def _monthly_returns_from_daily_equity(
    daily_profit_abs: Optional[list],
    daily_net_returns: Optional[list],
) -> tuple[Optional[list], Optional[list]]:
    """Compound complete daily wallet returns into dated monthly percentages."""

    if not daily_profit_abs or not daily_net_returns:
        return None, None
    if len(daily_profit_abs) != len(daily_net_returns):
        raise EquityDataError("daily PnL dates and returns have different lengths")

    monthly_growth: OrderedDict[str, float] = OrderedDict()
    previous_day: Optional[date] = None
    for index, (dated_pnl, raw_return) in enumerate(
        zip(daily_profit_abs, daily_net_returns, strict=True)
    ):
        if not isinstance(dated_pnl, (list, tuple)) or len(dated_pnl) != 2:
            raise EquityDataError(f"daily_profit_abs[{index}] must be a [date, pnl] pair")
        try:
            day = date.fromisoformat(str(dated_pnl[0])[:10])
        except ValueError as exc:
            raise EquityDataError(f"daily_profit_abs[{index}] has no ISO date") from exc
        if previous_day is not None and day != previous_day + timedelta(days=1):
            raise EquityDataError("daily PnL dates must be consecutive")
        previous_day = day
        month = day.isoformat()[:7]
        try:
            daily_return = float(raw_return)
        except (TypeError, ValueError) as exc:
            raise EquityDataError(f"daily_net_returns[{index}] must be numeric") from exc
        if not math.isfinite(daily_return) or daily_return < -1.0:
            raise EquityDataError(f"daily_net_returns[{index}] is invalid")
        if daily_return == -1.0 and index != len(daily_net_returns) - 1:
            raise EquityDataError("a total wallet loss must be the final daily return")
        monthly_growth[month] = monthly_growth.get(month, 1.0) * (1.0 + daily_return)

    periods = list(monthly_growth)
    profits_pct = [(growth - 1.0) * 100.0 for growth in monthly_growth.values()]
    return periods, profits_pct


def _stable_fee_noise_seed(random_seed: Any, strategy_name: str) -> int:
    """Derive a process-independent RNG seed for fee-noise injection.

    Python's built-in :func:`hash` is intentionally randomized per process.
    Using it here made the same evaluation produce different fees in sequential
    and worker processes.  SHA-256 keeps the mapping stable across processes and
    Python invocations.
    """
    payload = json.dumps(
        {"random_seed": random_seed, "strategy_name": strategy_name[:64]},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


@dataclass
class BacktestResult:
    """
    Container for backtest results.
    """

    success: bool
    strategy_name: str

    # Performance metrics
    total_profit: float = 0.0
    profit_percent: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_abs: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    profit_factor_censored: bool = False
    profit_factor_contract_version: str = PROFIT_FACTOR_CONTRACT_VERSION

    # Trade metrics
    avg_profit: float = 0.0
    median_profit: float = 0.0
    avg_duration: str = ""

    # Additional info
    error_message: Optional[str] = None
    execution_time: float = 0.0
    no_trades: bool = False  # True when backtest ran successfully but produced zero trades

    # Trade visualization data (optional, only populated when requested)
    trades: Optional[list] = None  # List of trade dicts for visualization
    ohlcv_data: Optional[Dict[str, Any]] = None  # Dict of pair_timeframe -> DataFrame

    # Per-pair performance breakdown
    per_pair_profit: Optional[Dict[str, float]] = None  # pair -> profit percentage
    per_pair_trades: Optional[Dict[str, int]] = None  # pair -> completed trade count

    # Monthly return breakdown for stability analysis
    monthly_profits: Optional[list] = None  # List of monthly profit percentages
    monthly_periods: Optional[list] = None  # Matching ISO YYYY-MM labels

    # Calendar-aligned V2 realised-close equity evidence.  Missing is None,
    # never a favourable zero.  This is not mark-to-market equity.
    starting_balance: Optional[float] = None
    final_balance: Optional[float] = None
    backtest_start: Optional[str] = None
    backtest_end: Optional[str] = None
    daily_profit_abs: Optional[list] = None  # [[ISO date, absolute wallet PnL], ...]
    daily_net_returns: Optional[list] = None  # Decimal, capital-weighted by prior equity
    equity_curve: Optional[list] = None  # Initial balance plus one close-equity value per day
    equity_method: Optional[str] = None
    equity_error_message: Optional[str] = None
    mark_to_market_error_message: Optional[str] = None
    annualized_net_return: Optional[float] = None
    daily_sharpe_ratio: Optional[float] = None
    daily_sortino_ratio: Optional[float] = None
    daily_expected_shortfall_5: Optional[float] = None
    calmar_ratio: Optional[float] = None
    ulcer_index: Optional[float] = None
    time_under_water_ratio: Optional[float] = None
    daily_max_drawdown: Optional[float] = None
    risk_metric_contract_version: str = RISK_METRIC_CONTRACT_VERSION
    risk_periods_per_year: int = CALENDAR_DAYS_PER_YEAR
    annual_risk_free_rate: float = DEFAULT_ANNUAL_RISK_FREE_RATE
    periodic_risk_free_rate: Optional[float] = 0.0

    # Tail-risk metrics
    max_consecutive_losses: Optional[int] = None  # Missing is distinct from measured zero
    max_drawdown_duration_days: Optional[float] = None  # Missing is distinct from measured zero

    # Lightweight trade profit list for Monte Carlo analysis
    trade_profit_ratios: Optional[list] = None  # List of per-trade profit ratios

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete cache- and evaluation-relevant result payload.

        ``ohlcv_data`` is deliberately excluded: it can contain large pandas
        DataFrames and is visualization input, not part of the evaluation
        result.  Every other dataclass field is preserved so a disk-cache hit
        has the same metric semantics as a fresh backtest.
        """
        return {
            "success": self.success,
            "strategy_name": self.strategy_name,
            "total_profit": self.total_profit,
            "profit_percent": self.profit_percent,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_abs": self.max_drawdown_abs,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "profit_factor": self.profit_factor,
            "profit_factor_censored": self.profit_factor_censored,
            "profit_factor_contract_version": self.profit_factor_contract_version,
            "avg_profit": self.avg_profit,
            "median_profit": self.median_profit,
            "avg_duration": self.avg_duration,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "no_trades": self.no_trades,
            "trades": self.trades,
            "per_pair_profit": self.per_pair_profit,
            "per_pair_trades": self.per_pair_trades,
            "monthly_profits": self.monthly_profits,
            "monthly_periods": self.monthly_periods,
            "starting_balance": self.starting_balance,
            "final_balance": self.final_balance,
            "backtest_start": self.backtest_start,
            "backtest_end": self.backtest_end,
            "daily_profit_abs": self.daily_profit_abs,
            "daily_net_returns": self.daily_net_returns,
            "equity_curve": self.equity_curve,
            "equity_method": self.equity_method,
            "equity_error_message": self.equity_error_message,
            "mark_to_market_error_message": self.mark_to_market_error_message,
            "annualized_net_return": self.annualized_net_return,
            "daily_sharpe_ratio": self.daily_sharpe_ratio,
            "daily_sortino_ratio": self.daily_sortino_ratio,
            "daily_expected_shortfall_5": self.daily_expected_shortfall_5,
            "calmar_ratio": self.calmar_ratio,
            "ulcer_index": self.ulcer_index,
            "time_under_water_ratio": self.time_under_water_ratio,
            "daily_max_drawdown": self.daily_max_drawdown,
            "risk_metric_contract_version": self.risk_metric_contract_version,
            "risk_periods_per_year": self.risk_periods_per_year,
            "annual_risk_free_rate": self.annual_risk_free_rate,
            "periodic_risk_free_rate": self.periodic_risk_free_rate,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "trade_profit_ratios": self.trade_profit_ratios,
        }


class DirectBacktester:
    """
    Direct backtester that uses FreqTrade's Python API with mocked exchange.

    This avoids network calls and allows offline backtesting.
    """

    # Default starting balances by stake currency type
    DEFAULT_BTC_BALANCE = 10  # 10 BTC (reasonable starting balance for BTC-denominated strategies)
    DEFAULT_STABLECOIN_BALANCE = 10000  # $10k for stablecoin-denominated strategies
    DEFAULT_EXCHANGE = "binance"  # Default exchange for real pairs

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize direct backtester.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_config = config.get("backtesting", {})
        self.freqtrade_root = Path(__file__).parent.parent.parent
        storage_config = config.get("storage", {})
        configured_strategy_dir = storage_config.get("generated_strategy_dir")
        self.strategy_dir = (
            Path(configured_strategy_dir)
            if configured_strategy_dir
            else self.freqtrade_root / "user_data" / "strategies" / "ga_generated"
        )
        self.strategy_dir.mkdir(parents=True, exist_ok=True)

        # Per-backtest timeout (seconds). Defence-in-depth for non-parallel paths.
        # The parallel evaluator also wraps each call with its own future timeout,
        # so this is a safety net that fires only if the parallel timeout misses.
        self.backtest_timeout = self.backtest_config.get("backtest_timeout", 300)

        # Initialize cache if enabled
        self.cache = None
        if self.backtest_config.get("enable_cache", True):
            max_cache_mb = storage_config.get("max_cache_disk_mb", 5000)
            configured_cache_dir = storage_config.get("cache_dir")
            cache_dir = Path(configured_cache_dir) if configured_cache_dir else None
            self.cache = BacktestCache(cache_dir=cache_dir, max_disk_mb=max_cache_mb)

        # Persistent Backtesting instance cache: reuse the heavy Backtesting
        # object (exchange mock, OHLCV data, pairlists) across evaluations
        # when config hasn't changed.  Key = (pairs_tuple, timerange, timeframe).
        self._bt_instance_cache: Dict[tuple, Any] = {}
        # LRU-capped OHLCV data cache: max 3 entries to bound memory.
        # Shared-memory-backed entries (from SharedDataManager) count as ~0 MB;
        # disk-loaded entries are ~300-500 MB each.
        self._bt_data_cache: OrderedDict[tuple, tuple] = (
            OrderedDict()
        )  # key → (data_dict, timerange_obj)
        self._bt_data_cache_max = 3

        # Log backtester initialization summary
        logger.info("[INIT] DirectBacktester initialized")
        logger.info(f"  Pairs: {self.backtest_config.get('pairs')}")
        logger.info(f"  Timerange: {self.backtest_config.get('timerange')}")
        logger.info(f"  Stake amount: {self.backtest_config.get('stake_amount')}")
        logger.info(f"  Max open trades: {self.backtest_config.get('max_open_trades')}")
        logger.info(f"  Fee: {self.backtest_config.get('fee')}")

        # Validate and auto-download data if enabled
        self._validate_and_download_data()

    def get_available_data_range(self) -> Optional[str]:
        """
        Detect the actual available data range from disk for the configured pairs.

        Checks all configured timeframes and uses the widest (union) range across
        them, then intersects with the configured timerange to produce an effective
        timerange that can be used for walk-forward window creation.

        Returns:
            Effective timerange string (YYYYMMDD-YYYYMMDD), or None if no data found.
        """
        pairs = self.backtest_config.get("pairs", [])
        timeframes = list(self.config.get("strategy_constraints", {}).get("timeframes", ["5m"]))
        if not timeframes:
            timeframes = ["5m"]

        # Include multi-timeframe timeframes when enabled (consistent with
        # _validate_data_exists) so that higher-TF data with wider history
        # contributes to the effective range instead of being ignored.
        multi_tf_config = self.config.get("multi_timeframe", {})
        if multi_tf_config.get("enabled", False):
            multi_tf_available = multi_tf_config.get("available", [])
            for tf in multi_tf_available:
                if tf not in timeframes:
                    timeframes.append(tf)

        # Skip for test pairs
        if any("UNITTEST" in p for p in pairs):
            return self.backtest_config.get("timerange", "")

        exchange_name = self.backtest_config.get("exchange", self.DEFAULT_EXCHANGE)
        datadir = self.freqtrade_root / "user_data" / "data" / exchange_name

        try:
            from freqtrade.data.history.datahandlers import get_datahandler
            from freqtrade.enums import CandleType
            from genetic_algorithm.utils.timerange import parse_timerange, format_date

            data_handler = get_datahandler(datadir)

            # Find the common data range across ALL configured pairs and timeframes.
            # For each timeframe, compute the intersection across pairs (common range
            # that all pairs share), then intersect across timeframes to guarantee
            # every pair has data for every timeframe in the final range.
            overall_min = None
            overall_max = None

            for timeframe in timeframes:
                tf_min = None
                tf_max = None

                for pair in pairs:
                    min_date, max_date, length = data_handler.ohlcv_data_min_max(
                        pair, timeframe, CandleType.SPOT
                    )

                    if length == 0:
                        logger.debug(f"No data on disk for {pair} {timeframe}")
                        continue

                    # Make datetimes naive for comparison (ohlcv_data_min_max may return tz-aware)
                    min_dt = min_date.replace(tzinfo=None) if min_date.tzinfo else min_date
                    max_dt = max_date.replace(tzinfo=None) if max_date.tzinfo else max_date

                    # Intersection across pairs for this timeframe
                    if tf_min is None or min_dt > tf_min:
                        tf_min = min_dt
                    if tf_max is None or max_dt < tf_max:
                        tf_max = max_dt

                if tf_min is not None and tf_max is not None and tf_min < tf_max:
                    logger.debug(
                        f"Data range for timeframe {timeframe}: "
                        f"{format_date(tf_min)}-{format_date(tf_max)}"
                    )
                    # Intersection across timeframes: keep the latest start
                    # and earliest end so ALL timeframes have data in range
                    if overall_min is None or tf_min > overall_min:
                        overall_min = tf_min
                    if overall_max is None or tf_max < overall_max:
                        overall_max = tf_max
                else:
                    logger.warning(f"No data on disk for timeframe {timeframe} across all pairs")

            if overall_min is None or overall_max is None:
                logger.warning("Could not determine data range - no data files found")
                return None

            logger.info(
                f"Actual data range on disk: {format_date(overall_min)}-{format_date(overall_max)}"
            )

            # Intersect with configured timerange
            config_timerange = self.backtest_config.get("timerange", "")
            if config_timerange:
                config_start, config_end = parse_timerange(config_timerange)

                effective_start = max(overall_min, config_start)
                effective_end = min(overall_max, config_end)

                if effective_start >= effective_end:
                    logger.error(
                        f"No overlap between config timerange ({config_timerange}) and "
                        f"available data ({format_date(overall_min)}-{format_date(overall_max)})"
                    )
                    return None

                effective_timerange = f"{format_date(effective_start)}-{format_date(effective_end)}"
            else:
                effective_timerange = f"{format_date(overall_min)}-{format_date(overall_max)}"

            logger.info(f"Effective data range: {effective_timerange}")
            return effective_timerange

        except Exception as e:
            logger.warning(f"Failed to detect data range: {e}. Using config timerange as fallback.")
            return self.backtest_config.get("timerange", "")

    def _validate_data_exists(self) -> Dict[str, list]:
        """
        Check if required data files exist for backtesting.

        Returns:
            Dictionary with 'missing' list of (pair, timeframe) tuples that are missing
        """
        pairs = self.backtest_config.get("pairs", [])
        # Use same config path as StrategyGenerator to ensure we check all timeframes
        # that might be used by generated strategies
        timeframes = list(self.config.get("strategy", {}).get("timeframes", ["5m", "15m", "1h"]))

        # Include multi-timeframe timeframes when enabled
        multi_tf_config = self.config.get("multi_timeframe", {})
        if multi_tf_config.get("enabled", False):
            multi_tf_available = multi_tf_config.get("available", [])
            for tf in multi_tf_available:
                if tf not in timeframes:
                    timeframes.append(tf)
            logger.info(f"Multi-timeframe enabled, validating timeframes: {timeframes}")

        # Skip validation for test pairs
        if any("UNITTEST" in p for p in pairs):
            logger.debug("Using test pairs (UNITTEST), skipping data validation")
            return {"missing": []}

        # Determine exchange and data directory
        exchange = self.backtest_config.get("exchange", self.DEFAULT_EXCHANGE)
        datadir = self.freqtrade_root / "user_data" / "data" / exchange

        missing = []

        for pair in pairs:
            for timeframe in timeframes:
                # Convert pair format for filename (BTC/USDT -> BTC_USDT)
                pair_filename = pair.replace("/", "_")

                # Check for data file in various formats FreqTrade uses
                data_file_json = datadir / f"{pair_filename}-{timeframe}.json"
                data_file_feather = datadir / f"{pair_filename}-{timeframe}.feather"
                data_file_parquet = datadir / f"{pair_filename}-{timeframe}.parquet"

                if not (
                    data_file_json.exists()
                    or data_file_feather.exists()
                    or data_file_parquet.exists()
                ):
                    missing.append((pair, timeframe))
                    logger.debug(f"Missing data: {pair} {timeframe}")

        if missing:
            logger.info(f"Found {len(missing)} missing data file(s)")
        else:
            logger.info("All required data files exist")

        return {"missing": missing}

    def _auto_download_data(self, missing_data: list) -> bool:
        """
        Automatically download missing data files.

        Args:
            missing_data: List of (pair, timeframe) tuples to download

        Returns:
            True if download successful, False otherwise
        """
        if not missing_data:
            return True

        try:
            from freqtrade.resolvers import ExchangeResolver
            from freqtrade.data.history import refresh_backtest_ohlcv_data
            from freqtrade.enums import CandleType, TradingMode

            # Get unique pairs and timeframes
            pairs = list(set(p for p, _ in missing_data))
            timeframes = list(set(tf for _, tf in missing_data))

            exchange_name = self.backtest_config.get("exchange", self.DEFAULT_EXCHANGE)
            datadir = self.freqtrade_root / "user_data" / "data" / exchange_name
            datadir.mkdir(parents=True, exist_ok=True)

            # Use the configured timerange so the download covers the full requested
            # history instead of only what the exchange provides by default.
            configured_timerange = self.backtest_config.get("timerange", None)

            logger.info(
                f"Auto-downloading missing data for {len(pairs)} pair(s) and {len(timeframes)} timeframe(s)..."
            )
            logger.info(f"  Pairs: {pairs}")
            logger.info(f"  Timeframes: {timeframes}")
            logger.info(f"  Exchange: {exchange_name}")
            if configured_timerange:
                logger.info(f"  Timerange: {configured_timerange}")

            # Determine stake currency from configured pairs
            stake_currency = "USDT"
            config_pairs = self.backtest_config.get("pairs", [])
            if config_pairs and "/" in config_pairs[0]:
                stake_currency = config_pairs[0].split("/")[1]
            else:
                logger.warning(
                    f"Could not derive stake currency from pairs {config_pairs}, "
                    f"defaulting to '{stake_currency}'"
                )

            # Determine trading mode from short_selling config
            short_selling_cfg = self.config.get("short_selling", {})
            if short_selling_cfg.get("enabled", False):
                _trading_mode_str = "futures"
                _margin_mode_str = "isolated"
                _trading_mode_enum = TradingMode.FUTURES
                _candle_type = CandleType.FUTURES
            else:
                _trading_mode_str = "spot"
                _margin_mode_str = ""
                _trading_mode_enum = TradingMode.SPOT
                _candle_type = CandleType.SPOT

            # Create exchange configuration
            exchange_config = {
                "exchange": {
                    "name": exchange_name,
                    "key": "",
                    "secret": "",
                    "ccxt_config": {},
                    "ccxt_async_config": {},
                },
                "datadir": datadir,
                "user_data_dir": self.freqtrade_root / "user_data",
                "trading_mode": _trading_mode_str,
                "margin_mode": _margin_mode_str,
                "stake_currency": stake_currency,
                "dry_run": True,
                "runmode": "other",
                "entry_pricing": {
                    "price_side": "same",
                    "use_order_book": False,
                    "order_book_top": 1,
                },
                "exit_pricing": {
                    "price_side": "same",
                    "use_order_book": False,
                    "order_book_top": 1,
                },
            }

            # Initialize exchange
            exchange = ExchangeResolver.load_exchange(exchange_config)

            # Convert the configured timerange string to a TimeRange object so the
            # download covers the full requested history rather than just the default
            # number of candles the exchange returns without an explicit range.
            from freqtrade.configuration import TimeRange as FTTimeRange

            ft_timerange = (
                FTTimeRange.parse_timerange(configured_timerange) if configured_timerange else None
            )

            # Download data
            refresh_backtest_ohlcv_data(
                exchange=exchange,
                pairs=pairs,
                timeframes=timeframes,
                datadir=datadir,
                timerange=ft_timerange,
                erase=False,
                trading_mode=_trading_mode_enum,
                candle_types=[_candle_type],
            )

            logger.info("✓ Data download completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to auto-download data: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _validate_and_download_data(self):
        """
        Validate data exists and auto-download if enabled and missing.
        """
        # Check if auto-download is enabled
        auto_download = self.backtest_config.get("auto_download_data", True)

        # Validate data exists
        validation_result = self._validate_data_exists()
        missing = validation_result["missing"]

        if not missing:
            logger.debug("Data validation passed - all required files exist")
            return

        # If data is missing
        if auto_download:
            logger.info(f"Missing {len(missing)} data file(s), attempting auto-download...")
            success = self._auto_download_data(missing)

            if not success:
                logger.warning("Auto-download failed, but continuing with existing data")
        else:
            # Auto-download disabled, show helpful error
            pairs = list(set(p for p, _ in missing))
            timeframes = list(set(tf for _, tf in missing))

            logger.warning("=" * 80)
            logger.warning("❌ Missing data files detected:")
            for pair, timeframe in missing:
                logger.warning(f"   • {pair} {timeframe}")
            logger.warning("")
            logger.warning("To fix this:")
            logger.warning(
                "1. Enable auto-download in config: set 'backtesting.auto_download_data: true'"
            )
            logger.warning("2. Or manually download:")
            logger.warning(
                f"   freqtrade download-data --pairs {' '.join(pairs)} "
                f"--timeframes {' '.join(timeframes)} --days 90"
            )
            logger.warning("=" * 80)

    def backtest_strategy(
        self,
        strategy_code: str,
        strategy_name: str,
        max_retries: int = 2,
        strategy_max_open_trades: Optional[int] = None,
        timerange_override: Optional[str] = None,
        pairs_override: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Run backtest for a strategy using direct Python API.

        Args:
            strategy_code: Python code for strategy
            strategy_name: Name of the strategy
            max_retries: Maximum number of retries on failure
            strategy_max_open_trades: Optional per-strategy max open trades
            timerange_override: Optional timerange override
            pairs_override: Optional list of pairs to use instead of config pairs

        Returns:
            BacktestResult object
        """
        start_time = time.time()

        # Check cache first (skip cache when timerange or pairs are overridden —
        # the cache key uses self.backtest_config which doesn't reflect the override)
        if self.cache and not timerange_override and not pairs_override:
            cached_result = self.cache.get(strategy_code, self.backtest_config)
            if cached_result:
                logger.debug(f"Using cached result for {strategy_name}")
                return cached_result

        # Try multiple times in case of transient errors
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{max_retries} for {strategy_name}")
                    time.sleep(1)

                result = self._run_backtest_direct(
                    strategy_code,
                    strategy_name,
                    strategy_max_open_trades,
                    timerange_override=timerange_override,
                    pairs_override=pairs_override,
                )

                result.execution_time = time.time() - start_time

                # Cache successful result (skip when timerange or pairs overridden)
                if result.success and self.cache and not timerange_override and not pairs_override:
                    self.cache.put(strategy_code, self.backtest_config, result)

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Backtest attempt {attempt + 1} failed: {e}")
                import traceback

                traceback.print_exc()

        # All retries failed
        execution_time = time.time() - start_time
        return BacktestResult(
            success=False,
            strategy_name=strategy_name,
            error_message=f"Failed after {max_retries + 1} attempts: {str(last_error)}",
            execution_time=execution_time,
        )

    def _run_backtest_direct(
        self,
        strategy_code: str,
        strategy_name: str,
        strategy_max_open_trades: Optional[int] = None,
        collect_trades: bool = False,
        timerange_override: Optional[str] = None,
        pairs_override: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Run backtest using FreqTrade Python API with mocked exchange.

        Args:
            strategy_code: Python code for strategy
            strategy_name: Name of the strategy
            collect_trades: Whether to collect detailed trade data for visualization

        Returns:
            BacktestResult object
        """
        # Write strategy to file atomically to prevent race conditions
        # when multiple workers write to the same strategy file
        strategy_file = self.strategy_dir / f"{strategy_name}.py"
        try:
            # Write to a temp file first, then atomically rename
            fd, tmp_path = tempfile.mkstemp(
                suffix=".py", dir=str(self.strategy_dir), prefix=f".{strategy_name}_"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(strategy_code)
                os.replace(tmp_path, str(strategy_file))  # atomic on POSIX
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            # Invalidate any cached .pyc for this module
            cache_dir = self.strategy_dir / "__pycache__"
            if cache_dir.exists():
                for pyc in cache_dir.glob(f"{strategy_name}.*.pyc"):
                    try:
                        pyc.unlink()
                    except OSError:
                        pass
            logger.debug(f"Wrote strategy file (atomic): {strategy_file}")
        except Exception as e:
            logger.error(f"Failed to write strategy file: {e}")
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Failed to write strategy file: {e}",
            )

        # Validate generated Python syntax before backtesting
        try:
            compile(strategy_code, str(strategy_file), "exec")
        except SyntaxError as e:
            logger.error(f"Generated strategy has syntax error at line {e.lineno}: {e.msg}")
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Generated strategy syntax error at line {e.lineno}: {e.msg}",
            )

        # Deep validation: actually import the module to catch runtime errors
        # (e.g., missing talib functions, NameError, ImportError)
        try:
            import importlib
            import importlib.util

            importlib.invalidate_caches()  # ensure fresh file is picked up
            spec = importlib.util.spec_from_file_location(strategy_name, str(strategy_file))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Verify the expected class exists in the module
                if not hasattr(module, strategy_name):
                    logger.error(f"Strategy module loaded but class '{strategy_name}' not found")
                    return BacktestResult(
                        success=False,
                        strategy_name=strategy_name,
                        error_message=f"Strategy class '{strategy_name}' not found in module",
                    )
        except (ImportError, ModuleNotFoundError) as e:
            # Missing optional dependencies (psutil, rapidjson, etc.) — non-fatal.
            # The actual backtesting call below will succeed if freqtrade itself can load.
            logger.debug(f"Strategy '{strategy_name}' pre-import skipped (missing dep): {e}")
        except (NameError, AttributeError) as e:
            logger.error(f"Strategy '{strategy_name}' has runtime import error: {e}")
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Strategy runtime import error: {e}",
            )
        except Exception as e:
            # Don't block on unexpected errors during validation — let FreqTrade try
            logger.warning(f"Strategy pre-import validation warning for '{strategy_name}': {e}")

        try:
            # Import FreqTrade modules
            from freqtrade.optimize.backtesting import Backtesting
            from freqtrade.exchange.exchange import Exchange
            import io

            # Extract timeframe from strategy code for dynamic slippage
            import re as _re_module

            _tf_match = _re_module.search(r"timeframe\s*=\s*['\"](\w+)['\"]", strategy_code)
            _strategy_tf = _tf_match.group(1) if _tf_match else None

            # Create configuration
            config_dict = self._create_backtest_config(
                strategy_name,
                strategy_max_open_trades,
                timerange_override=timerange_override,
                pairs_override=pairs_override,
                strategy_timeframe=_strategy_tf,
            )

            # Suppress FreqTrade's verbose output by redirecting stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            # Also suppress FreqTrade's verbose logging during backtesting
            freqtrade_loggers = [
                "freqtrade.exchange.exchange",
                "freqtrade.resolvers",
                "freqtrade.resolvers.strategy_resolver",
                "freqtrade.resolvers.exchange_resolver",
                "freqtrade.resolvers.iresolver",
                "freqtrade.configuration",
                "freqtrade.configuration.config_validation",
                "freqtrade.optimize.backtesting",
                "freqtrade.data.dataprovider",
                "freqtrade.data.history",
                "freqtrade.strategy",
                "freqtrade.strategy.hyper",
                "freqtrade.misc",
            ]
            old_log_levels = {}
            for logger_name in freqtrade_loggers:
                ft_logger = logging.getLogger(logger_name)
                old_log_levels[logger_name] = ft_logger.level
                ft_logger.setLevel(logging.WARNING)

            try:
                # Mock the exchange to avoid network calls
                with (
                    patch.object(Exchange, "_load_async_markets", return_value={}),
                    patch.object(
                        Exchange, "markets", PropertyMock(return_value=self._get_mock_markets())
                    ),
                    patch.object(Exchange, "validate_config", MagicMock()),
                    patch.object(Exchange, "validate_timeframes", MagicMock()),
                    patch.object(Exchange, "_init_ccxt", MagicMock()),
                    patch.object(
                        Exchange,
                        "get_fee",
                        return_value=config_dict.get("exchange", {}).get("fee", 0.001),
                    ),
                    patch.object(Exchange, "precisionMode", PropertyMock(return_value=2)),
                    patch.object(Exchange, "precision_mode_price", PropertyMock(return_value=2)),
                    patch.object(
                        Exchange,
                        "timeframes",
                        PropertyMock(return_value=["1m", "5m", "15m", "1h", "1d"]),
                    ),
                    patch.object(Exchange, "get_min_pair_stake_amount", return_value=0.0),
                    patch.object(Exchange, "get_max_pair_stake_amount", return_value=float("inf")),
                ):
                    # Initialize backtesting
                    backtesting = Backtesting(config_dict)

                    # Skip prior backtest loading - GA generates unique strategies
                    # and the parallel workers corrupt each other's .meta files
                    backtesting.load_prior_backtest = lambda: None

                    # --- OHLCV data caching: load once, reuse across evaluations ---
                    _cache_key = (
                        tuple(sorted(config_dict.get("exchange", {}).get("pair_whitelist", []))),
                        config_dict.get("timerange", ""),
                        config_dict.get("timeframe", "5m"),
                        timerange_override or "",
                        tuple(sorted(pairs_override)) if pairs_override else (),
                    )
                    cached = self._bt_data_cache.get(_cache_key)
                    if cached is not None:
                        # Move to end (most-recently-used) for LRU ordering
                        self._bt_data_cache.move_to_end(_cache_key)
                        raw_data, timerange_obj = cached
                        # Shallow-copy each DataFrame so indicator columns
                        # from previous strategies don't leak across runs
                        data = {pair: df.copy() for pair, df in raw_data.items()}
                        # Restore timerange and required fields on the instance
                        backtesting.timerange = timerange_obj
                    else:
                        data, timerange_obj = backtesting.load_bt_data()
                        # Cache the raw data (before any indicator additions)
                        self._bt_data_cache[_cache_key] = (
                            {pair: df.copy() for pair, df in data.items()},
                            timerange_obj,
                        )
                        # Evict oldest entry if cache exceeds max size
                        while len(self._bt_data_cache) > self._bt_data_cache_max:
                            evicted_key, _ = self._bt_data_cache.popitem(last=False)
                            logger.debug(
                                f"[CACHE] Evicted LRU data cache entry: {evicted_key[:2]}…"
                            )

                    # Run backtest with timeout safety net.
                    # signal.alarm / SIGALRM only work on the main thread;
                    # skip them when called from a worker thread (e.g. web dashboard).
                    import threading as _threading

                    _is_main_thread = _threading.current_thread() is _threading.main_thread()

                    def _timeout_handler(signum, frame):
                        raise TimeoutError(f"Backtest exceeded {self.backtest_timeout}s timeout")

                    old_handler = None
                    _alarm_set = False
                    if self.backtest_timeout > 0 and hasattr(signal, "SIGALRM") and _is_main_thread:
                        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                        signal.alarm(self.backtest_timeout)
                        _alarm_set = True
                    try:
                        strat = backtesting.strategylist[0]
                        min_date, max_date = backtesting.backtest_one_strategy(
                            strat,
                            data,
                            timerange_obj,
                        )
                        # Generate results
                        from freqtrade.optimize.optimize_reports import generate_backtest_stats

                        if backtesting.all_bt_content:
                            backtesting.results = generate_backtest_stats(
                                data,
                                backtesting.all_bt_content,
                                min_date=min_date,
                                max_date=max_date,
                            )
                    except TimeoutError:
                        logger.warning(
                            f"[TIMEOUT] Backtest for {strategy_name} timed out after {self.backtest_timeout}s"
                        )
                        return BacktestResult(
                            success=False,
                            strategy_name=strategy_name,
                            error_message=f"Backtest timed out after {self.backtest_timeout}s",
                        )
                    finally:
                        if _alarm_set and old_handler is not None:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)

                    # Get results from the backtest results
                    logger.debug(
                        f"Backtest results structure: {backtesting.results.keys() if backtesting.results else 'None'}"
                    )

                    if not backtesting.results or "strategy" not in backtesting.results:
                        logger.warning(f"No results available from backtest for {strategy_name}")
                        return BacktestResult(
                            success=True,
                            strategy_name=strategy_name,
                            total_trades=0,
                            no_trades=True,
                            error_message="No trades generated - strategy may be too restrictive",
                        )

                    # Parse results from the strategy results
                    strategy_results = backtesting.results["strategy"].get(strategy_name, {})

                    # Fallback: if the expected key is missing, use the first available strategy key
                    if not strategy_results and backtesting.results["strategy"]:
                        available_keys = list(backtesting.results["strategy"].keys())
                        logger.warning(
                            f"Strategy key '{strategy_name}' not found in results. "
                            f"Available keys: {available_keys}. Using first key."
                        )
                        strategy_results = backtesting.results["strategy"][available_keys[0]]

                    logger.debug(
                        f"Strategy results keys: {strategy_results.keys() if strategy_results else 'None'}"
                    )

                    if not strategy_results:
                        logger.warning(f"Empty strategy results for {strategy_name}")
                        expected_pairs = config_dict.get("exchange", {}).get(
                            "pair_whitelist", []
                        )
                        return BacktestResult(
                            success=True,
                            strategy_name=strategy_name,
                            total_trades=0,
                            no_trades=True,
                            error_message="No trades generated - check strategy conditions",
                            per_pair_profit={pair: 0.0 for pair in expected_pairs},
                            per_pair_trades={pair: 0 for pair in expected_pairs},
                        )

                    result = self._parse_stats(strategy_results, strategy_name)
                    trades_df = strategy_results.get("trades", None)
                    self._attach_mark_to_market_equity(result, trades_df, data)
                    try:
                        (
                            result.monthly_periods,
                            result.monthly_profits,
                        ) = _monthly_returns_from_daily_equity(
                            result.daily_profit_abs,
                            result.daily_net_returns,
                        )
                    except EquityDataError as e:
                        logger.debug(f"Could not derive monthly wallet returns: {e}")

                    # Preserve a lightweight trade record for evaluation logic
                    # (clustering, confidence, loss tails) on every backtest.
                    # Visualization requests replace this with the full records
                    # below.  Keeping only stable, relevant columns bounds cache
                    # size while avoiding a metrics contract that changes by path.
                    try:
                        if trades_df is not None and hasattr(trades_df, "to_dict"):
                            trade_columns = [
                                col
                                for col in (
                                    "trade_id",
                                    "pair",
                                    "stake_amount",
                                    "max_stake_amount",
                                    "amount",
                                    "open_date",
                                    "close_date",
                                    "open_rate",
                                    "close_rate",
                                    "open_timestamp",
                                    "close_timestamp",
                                    "fee_open",
                                    "fee_close",
                                    "profit_ratio",
                                    "profit_abs",
                                    "leverage",
                                    "is_short",
                                    "is_open",
                                    "funding_fees",
                                    "funding_fee_events",
                                    "orders",
                                    "min_rate",
                                    "max_rate",
                                    "exit_reason",
                                )
                                if col in trades_df.columns
                            ]
                            result.trades = trades_df[trade_columns].to_dict("records")
                        elif isinstance(trades_df, list):
                            result.trades = trades_df
                    except Exception as e:
                        logger.debug(f"Could not extract lightweight trade records: {e}")

                    # Extract per-pair performance breakdown
                    try:
                        trades_df = strategy_results.get("trades", None)
                        expected_pairs = list(
                            strategy_results.get("pairlist")
                            or config_dict.get("exchange", {}).get("pair_whitelist", [])
                        )
                        per_pair, per_pair_trades = _summarize_trades_by_pair(
                            trades_df,
                            expected_pairs,
                        )
                        result.per_pair_profit = per_pair
                        result.per_pair_trades = per_pair_trades
                        logger.debug(
                            "Per-pair profits/trades: %s / %s",
                            per_pair,
                            per_pair_trades,
                        )
                    except Exception as e:
                        logger.debug(f"Could not extract per-pair metrics: {e}")

                    # Extract tail-risk metrics from trade data
                    try:
                        trades_df = strategy_results.get("trades", None)
                        if (
                            trades_df is not None
                            and hasattr(trades_df, "__len__")
                            and len(trades_df) > 0
                        ):
                            # --- Max consecutive losses ---
                            if "profit_ratio" in trades_df.columns:
                                is_loss = (trades_df["profit_ratio"] < 0).values
                                max_streak = 0
                                current_streak = 0
                                for loss in is_loss:
                                    if loss:
                                        current_streak += 1
                                        max_streak = max(max_streak, current_streak)
                                    else:
                                        current_streak = 0
                                result.max_consecutive_losses = max_streak

                            # --- Max drawdown duration (days) ---
                            if "profit_ratio" in trades_df.columns:
                                import pandas as pd

                                date_col = None
                                for col in ["close_date", "sell_date", "exit_date"]:
                                    if col in trades_df.columns:
                                        date_col = col
                                        break
                                if date_col:
                                    dates = pd.to_datetime(trades_df[date_col])
                                    equity = (1 + trades_df["profit_ratio"]).cumprod()
                                    running_max = equity.cummax()
                                    in_drawdown = equity < running_max
                                    max_dd_dur = 0.0
                                    dd_start = None
                                    for i in range(len(in_drawdown)):
                                        if in_drawdown.iloc[i]:
                                            if dd_start is None:
                                                dd_start = dates.iloc[i]
                                        else:
                                            if dd_start is not None:
                                                dur = (
                                                    dates.iloc[i] - dd_start
                                                ).total_seconds() / 86400.0
                                                max_dd_dur = max(max_dd_dur, dur)
                                                dd_start = None
                                    # Handle ongoing drawdown at end of data
                                    if dd_start is not None:
                                        dur = (dates.iloc[-1] - dd_start).total_seconds() / 86400.0
                                        max_dd_dur = max(max_dd_dur, dur)
                                    # Prefer the calendar-aligned V2 duration
                                    # when available.  This trade-close fallback
                                    # is retained for legacy/incomplete stats.
                                    if result.max_drawdown_duration_days is None:
                                        result.max_drawdown_duration_days = max_dd_dur
                    except Exception as e:
                        logger.debug(f"Could not extract tail-risk metrics: {e}")

                    # Extract lightweight trade profit ratios for Monte Carlo analysis
                    try:
                        trades_df = strategy_results.get("trades", None)
                        if (
                            trades_df is not None
                            and hasattr(trades_df, "__len__")
                            and len(trades_df) > 0
                        ):
                            if "profit_ratio" in trades_df.columns:
                                result.trade_profit_ratios = trades_df["profit_ratio"].tolist()
                    except Exception as e:
                        logger.debug(f"Could not extract trade profit ratios: {e}")

                    # Collect detailed trade data for visualization if requested
                    if collect_trades:
                        trades_list = []
                        ohlcv_dict = {}

                        try:
                            # Extract trades from backtest results
                            # FreqTrade stores trades as a DataFrame in strategy_results
                            trades_df = strategy_results.get("trades", None)
                            if trades_df is not None and hasattr(trades_df, "to_dict"):
                                # Convert DataFrame to list of dicts
                                trades_list = trades_df.to_dict("records")
                                logger.debug(
                                    f"Collected {len(trades_list)} trades for visualization"
                                )
                            elif isinstance(strategy_results.get("trades"), list):
                                trades_list = strategy_results["trades"]

                            # Try to get OHLCV data from backtesting object
                            # FreqTrade stores data in different attributes depending on version
                            ohlcv_data_source = None
                            if hasattr(backtesting, "processed") and backtesting.processed:
                                ohlcv_data_source = backtesting.processed
                            elif hasattr(backtesting, "data") and backtesting.data:
                                ohlcv_data_source = backtesting.data
                            elif hasattr(backtesting, "_data") and backtesting._data:
                                ohlcv_data_source = backtesting._data

                            if ohlcv_data_source:
                                for key, df in ohlcv_data_source.items():
                                    if hasattr(df, "copy"):
                                        ohlcv_dict[key] = df.copy()
                                logger.debug(f"Collected OHLCV data for {len(ohlcv_dict)} pair(s)")
                            else:
                                # Fallback: load OHLCV data directly from disk
                                logger.debug(
                                    "No OHLCV data in backtesting object, loading from disk..."
                                )
                                ohlcv_dict = self._load_ohlcv_for_pairs(config_dict)

                        except Exception as e:
                            logger.warning(f"Could not extract trade details: {e}")

                        result.trades = trades_list
                        result.ohlcv_data = ohlcv_dict

                    return result
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                # Restore logging levels
                for logger_name, level in old_log_levels.items():
                    logging.getLogger(logger_name).setLevel(level)
                # Cleanup backtesting engine to free memory
                if "backtesting" in locals():
                    try:
                        del backtesting
                    except Exception:
                        pass
                gc.collect()

        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            import traceback

            traceback.print_exc()
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Execution error: {str(e)}",
            )

    def _create_backtest_config(
        self,
        strategy_name: str,
        strategy_max_open_trades: Optional[int] = None,
        timerange_override: Optional[str] = None,
        pairs_override: Optional[List[str]] = None,
        strategy_timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create FreqTrade config for backtesting from GA config.

        This is called automatically for each strategy backtest.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Configuration dictionary
        """
        # Read from GA config (stored in self.backtest_config)
        ga_cfg = self.backtest_config

        # Extract values from GA config
        pairs = pairs_override if pairs_override else ga_cfg.get("pairs", ["UNITTEST/BTC"])
        timerange = ga_cfg.get("timerange", "")
        stake_amount = ga_cfg.get("stake_amount", 0.05)
        # Use strategy-specific max_open_trades if provided, otherwise use global config
        if strategy_max_open_trades is not None:
            max_open_trades = strategy_max_open_trades
        else:
            max_open_trades = ga_cfg.get("max_open_trades", 3)
        fee = ga_cfg.get("fee", 0.001)

        # Add slippage on top of exchange fee for realistic cost modeling
        # Slippage accounts for spread, market impact, and execution delays
        slippage_pct = ga_cfg.get("slippage_pct", 0.0)
        if slippage_pct > 0:
            # Dynamic slippage: scale based on timeframe (shorter TF = higher slippage)
            if ga_cfg.get("dynamic_slippage", False) and strategy_timeframe:
                tf_minutes = _timeframe_to_minutes_util(strategy_timeframe)
                # Base slippage is calibrated for 1h; scale inversely with sqrt of TF
                # 5m → 3.46x, 15m → 2x, 30m → 1.41x, 1h → 1x, 4h → 0.5x
                slippage_scale = (60.0 / max(tf_minutes, 1)) ** 0.5
                slippage_pct = slippage_pct * slippage_scale
                logger.debug(
                    f"Dynamic slippage: TF={strategy_timeframe}, scale={slippage_scale:.2f}, "
                    f"slippage={slippage_pct:.6f}"
                )
            fee = fee + slippage_pct
            logger.debug(
                f"Fee adjusted with slippage: base={ga_cfg.get('fee', 0.001)}, "
                f"slippage={slippage_pct}, total={fee}"
            )

        # Fee noise injection: jitter fee per evaluation to prevent overfitting
        # to the exact fee level. Configured via 'fee_noise_std' (default 0 = off).
        import random as _random_module

        # Fee noise is opt-in.  A hidden non-zero default makes nominally equal
        # evaluations use different cost models and undermines reproducibility.
        fee_noise_std = ga_cfg.get("fee_noise_std", 0.0)
        if fee_noise_std > 0:
            # Prefer the GA seed.  Legacy configs sometimes stored it under
            # backtesting, so keep that location as a compatibility fallback.
            global_seed = self.config.get("genetic_algorithm", {}).get(
                "random_seed", ga_cfg.get("random_seed", 42)
            )
            if global_seed is None:
                global_seed = 42
            _fee_rng_seed = _stable_fee_noise_seed(global_seed, strategy_name)
            _fee_rng = _random_module.Random(_fee_rng_seed)
            noise = _fee_rng.gauss(0, fee_noise_std)
            fee = max(0.0, fee + noise)
            logger.debug(f"Fee after noise injection: {fee:.6f} (noise={noise:+.6f})")

        # Determine stake currency from pairs
        # Extract quote currency from pairs (format: BASE/QUOTE)
        stake_currency = "BTC"  # Default
        if pairs:
            # Use the quote currency from the first pair
            # All pairs should use the same quote currency for consistent backtesting
            first_pair = pairs[0]
            if "/" in first_pair:
                stake_currency = first_pair.split("/")[1]

        # Set reasonable starting balance based on stake currency
        if stake_currency == "BTC":
            starting_balance = self.DEFAULT_BTC_BALANCE
        elif stake_currency in ("USDT", "USD", "USDC", "BUSD"):
            starting_balance = self.DEFAULT_STABLECOIN_BALANCE
        else:
            starting_balance = (
                self.DEFAULT_STABLECOIN_BALANCE
            )  # Default to stablecoin balance for other currencies
            logger.warning(
                f"Unknown stake currency '{stake_currency}', using default balance of {self.DEFAULT_STABLECOIN_BALANCE}"
            )

        # Convert fractional stake_amount to actual currency amount
        # If stake_amount is between 0 and 1, treat it as a fraction of starting balance
        # e.g., 0.15 means 15% of wallet = $1500 for a $10,000 wallet
        # Values >= 1 are treated as literal amounts (e.g., 100 = $100 per trade)
        if isinstance(stake_amount, (int, float)) and 0 < stake_amount < 1:
            actual_stake = stake_amount * starting_balance
            logger.info(
                f"Stake amount {stake_amount} interpreted as {stake_amount:.0%} of "
                f"{starting_balance} {stake_currency} = {actual_stake:.2f} {stake_currency} per trade"
            )
            stake_amount = actual_stake

        # Determine exchange and data directory from pairs
        # Check if exchange is specified in GA config, otherwise use default
        exchange_name = self.backtest_config.get("exchange", self.DEFAULT_EXCHANGE)

        # For test pairs (UNITTEST/BTC), use test data directory
        if any("UNITTEST" in p for p in pairs):
            datadir = self.freqtrade_root / "tests" / "testdata"
        else:
            # Use user_data directory for real pairs
            datadir = self.freqtrade_root / "user_data" / "data" / exchange_name

        storage_config = self.config.get("storage", {})
        configured_export_dir = storage_config.get("backtest_export_dir")
        exportdir = (
            Path(configured_export_dir)
            if configured_export_dir
            else self.freqtrade_root / "user_data" / "backtest_results"
        )
        exportdir.mkdir(parents=True, exist_ok=True)
        configured_user_data_dir = storage_config.get("backtest_user_data_dir")
        backtest_user_data_dir = (
            Path(configured_user_data_dir)
            if configured_user_data_dir
            else self.freqtrade_root / "user_data"
        )
        backtest_user_data_dir.mkdir(parents=True, exist_ok=True)

        # Determine trading mode from short_selling config
        short_selling_cfg = self.config.get("short_selling", {})
        if short_selling_cfg.get("enabled", False):
            _bt_trading_mode = "futures"
            _bt_margin_mode = "isolated"
        else:
            _bt_trading_mode = "spot"
            _bt_margin_mode = ""

        # Build FreqTrade config
        config = {
            "strategy": strategy_name,
            "strategy_path": str(self.strategy_dir),
            "user_data_dir": backtest_user_data_dir,
            "datadir": datadir,  # Path object - uses calculated directory based on pairs/exchange
            "exportdirectory": exportdir,  # Path object for storing results
            "runmode": "backtest",  # Required for FreqTrade
            # Data format - CRITICAL: must match the format of data files on disk
            # Read from GA config's backtesting section, default to feather
            "dataformat_ohlcv": self.backtest_config.get("dataformat_ohlcv", "feather"),
            "dataformat_trades": self.backtest_config.get("dataformat_trades", "feather"),
            # Critical config values from GA config
            "stake_currency": stake_currency,  # Calculated from pairs
            "stake_amount": stake_amount,  # From GA config
            "dry_run_wallet": starting_balance,  # Calculated based on stake currency
            "max_open_trades": max_open_trades,  # From GA config
            "fee": fee,  # From GA config
            # Position stacking: configurable from GA config (default False to match live trading)
            "position_stacking": self.backtest_config.get("position_stacking", False),
            # Don't set timeframe here - let the strategy define it
            # "timeframe": "5m",  # Removed - strategy's timeframe will be used
            "timerange": timerange_override
            if timerange_override
            else (timerange if timerange else None),
            # Exchange configuration
            "exchange": {
                "name": exchange_name,
                "pair_whitelist": pairs,  # From GA config
                "ccxt_config": {},
                "ccxt_async_config": {},
            },
            "pairlists": [{"method": "StaticPairList"}],
            "trading_mode": _bt_trading_mode,
            "margin_mode": _bt_margin_mode,
            "dry_run": True,
        }

        # Store original config reference (required for backtest storage)
        config["original_config"] = config.copy()

        # Log at debug level to avoid spam - full config is shown at initialization
        logger.debug(
            f"Backtest config for {strategy_name}: pairs={pairs}, timerange={timerange}, max_open_trades={max_open_trades}"
        )

        return config

    def _get_mock_markets(self) -> Dict[str, Any]:
        """
        Get mock markets data for offline backtesting.

        Dynamically builds market definitions from configured pairs to support
        both test pairs (UNITTEST/BTC) and real pairs (BTC/USDT, ETH/USDT, etc.).

        Returns:
            Mock markets dictionary
        """
        mock_markets = {}

        # Get pairs from config
        config_pairs = self.backtest_config.get("pairs", [])

        # Include common test pairs for backward compatibility
        test_pairs = [
            "UNITTEST/BTC",
            "ETH/BTC",
            "LTC/BTC",
            "XRP/BTC",
            "ADA/BTC",
            "DASH/BTC",
            "ETC/BTC",
            "XLM/BTC",
            "XMR/BTC",
            "NXT/BTC",
            "ZEC/BTC",
            "TRX/BTC",
        ]

        # Combine config pairs with test pairs (config pairs take precedence)
        all_pairs = list(set(config_pairs + test_pairs))

        for pair in all_pairs:
            if "/" not in pair:
                logger.warning(f"Invalid pair format: {pair} (expected BASE/QUOTE)")
                continue

            base, quote = pair.split("/", 1)  # maxsplit=1 to handle pairs like 'BTC/USDT'
            mock_markets[pair] = {
                "id": pair.replace("/", "_"),
                "symbol": pair,
                "base": base,
                "quote": quote,
                "active": True,
                "spot": True,
                "precision": {"amount": 8, "price": 8},
                "limits": {
                    "amount": {"min": 0.001, "max": 10000},
                    "price": {"min": 0.00000001, "max": 100000},
                    "cost": {"min": 0.001, "max": None},
                },
                "info": {},
            }

        logger.debug(
            f"Created mock markets for {len(mock_markets)} pairs: {list(mock_markets.keys())}"
        )

        return mock_markets

    def _parse_stats(self, stats: Dict[str, Any], strategy_name: str) -> BacktestResult:
        """
        Parse backtest statistics into BacktestResult.

        Args:
            stats: Statistics dictionary from backtesting
            strategy_name: Name of strategy

        Returns:
            BacktestResult object
        """
        # Log the raw stats for debugging profit issues
        logger.debug(f"Raw backtest stats for {strategy_name}: {stats}")

        # Extract metrics from stats - handle both percentage and absolute values
        profit_total = stats.get("profit_total", 0.0)
        profit_total_abs = stats.get("profit_total_abs", 0.0)

        # Log raw profit values for debugging
        logger.debug(
            f"Raw profit for {strategy_name}: profit_total={profit_total}, "
            f"profit_total_abs={profit_total_abs}, "
            f"max_drawdown_account={stats.get('max_drawdown_account', 'N/A')}, "
            f"max_drawdown_abs={stats.get('max_drawdown_abs', 'N/A')}"
        )

        # Convert profit_total to percentage
        # FreqTrade always returns profit_total as a ratio (e.g., 0.05 = 5%)
        # and profit_total_pct as percentage. We use profit_total * 100 for consistency.
        # See freqtrade/optimize/optimize_reports/optimize_reports.py:
        #   profit_total = result["profit_abs"].sum() / starting_balance  (ratio)
        #   "profit_total_pct": round(profit_total * 100.0, 2)  (percentage)
        profit_percent = stats.get("profit_total_pct", profit_total * 100)

        total_trades = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)

        # Calculate win rate if not provided
        win_rate = stats.get("winrate", 0.0)
        if win_rate == 0.0 and total_trades > 0:
            win_rate = wins / total_trades

        # Log key results at debug level (summary logged elsewhere)
        logger.debug(
            f"Parsed {strategy_name}: profit={profit_percent:.4f}%, trades={total_trades}, win_rate={win_rate:.2%}"
        )

        # Extract metrics from stats
        # Note: FreqTrade uses 'max_drawdown_account' for percentage drawdown (as ratio, e.g., 0.15 = 15%)
        max_drawdown = stats.get("max_drawdown_account", stats.get("max_drawdown", 0.0))
        profit_factor, profit_factor_censored = normalize_profit_factor(
            stats.get("profit_factor", 0.0),
            total_trades=total_trades,
            losses=losses,
            net_profit=max(float(profit_total_abs), float(profit_total)),
        )

        config = getattr(self, "config", {})
        evaluation_v2 = config.get("evaluation_v2", {}) if isinstance(config, dict) else {}
        annual_risk_free_rate = evaluation_v2.get(
            "annual_risk_free_rate",
            DEFAULT_ANNUAL_RISK_FREE_RATE,
        )
        equity_payload: Dict[str, Any] = {
            "starting_balance": None,
            "final_balance": None,
            "backtest_start": stats.get("backtest_start"),
            "backtest_end": stats.get("backtest_end"),
            "daily_profit_abs": None,
            "daily_net_returns": None,
            "equity_curve": None,
            "equity_method": None,
            "equity_error_message": None,
            "annualized_net_return": None,
            "daily_sharpe_ratio": None,
            "daily_sortino_ratio": None,
            "daily_expected_shortfall_5": None,
            "calmar_ratio": None,
            "ulcer_index": None,
            "time_under_water_ratio": None,
            "daily_max_drawdown": None,
            "risk_metric_contract_version": RISK_METRIC_CONTRACT_VERSION,
            "risk_periods_per_year": CALENDAR_DAYS_PER_YEAR,
            "annual_risk_free_rate": annual_risk_free_rate,
            "periodic_risk_free_rate": None,
            "max_consecutive_losses": stats.get("max_consecutive_losses"),
            "max_drawdown_duration_days": None,
        }
        try:
            daily_equity = build_equity_from_freqtrade_stats(stats)
            equity_metrics = calculate_equity_risk_metrics(
                daily_equity,
                annual_risk_free_rate=annual_risk_free_rate,
            )
            equity_payload.update(
                {
                    "starting_balance": daily_equity.starting_balance,
                    "final_balance": daily_equity.final_balance,
                    "daily_profit_abs": [
                        [day.isoformat(), pnl]
                        for day, pnl in zip(daily_equity.dates, daily_equity.daily_profit_abs)
                    ],
                    "daily_net_returns": list(daily_equity.daily_net_returns),
                    "equity_curve": list(daily_equity.equity_curve),
                    "equity_method": daily_equity.equity_method,
                    "annualized_net_return": equity_metrics.annualized_net_return,
                    "daily_sharpe_ratio": equity_metrics.sharpe_ratio,
                    "daily_sortino_ratio": equity_metrics.sortino_ratio,
                    "daily_expected_shortfall_5": equity_metrics.daily_expected_shortfall_5,
                    "calmar_ratio": equity_metrics.calmar_ratio,
                    "ulcer_index": equity_metrics.ulcer_index,
                    "time_under_water_ratio": equity_metrics.time_under_water_ratio,
                    "daily_max_drawdown": equity_metrics.max_drawdown,
                    "risk_metric_contract_version": (
                        equity_metrics.risk_metric_contract_version
                    ),
                    "risk_periods_per_year": equity_metrics.periods_per_year,
                    "annual_risk_free_rate": equity_metrics.annual_risk_free_rate,
                    "periodic_risk_free_rate": equity_metrics.periodic_risk_free_rate,
                    "max_drawdown_duration_days": equity_metrics.max_drawdown_duration_days,
                }
            )
        except EquityDataError as exc:
            # Legacy/mocked stats often lack daily wallet evidence.  Preserve
            # that absence explicitly; never replace it with good-looking zeroes.
            equity_payload["equity_error_message"] = str(exc)

        return BacktestResult(
            success=True,
            strategy_name=strategy_name,
            total_profit=profit_total_abs,
            profit_percent=profit_percent,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            max_drawdown_abs=stats.get("max_drawdown_abs", 0.0),
            sharpe_ratio=stats.get("sharpe", 0.0),
            sortino_ratio=stats.get("sortino", 0.0),
            profit_factor=profit_factor,
            profit_factor_censored=profit_factor_censored,
            avg_profit=stats.get("profit_mean", 0.0),
            median_profit=stats.get("profit_median", 0.0),
            # FreqTrade uses 'holding_avg' (timedelta) at strategy level, not 'duration_avg'
            avg_duration=str(stats.get("holding_avg", ""))
            if stats.get("holding_avg")
            else stats.get("duration_avg", ""),
            **equity_payload,
        )

    def _attach_mark_to_market_equity(
        self,
        result: BacktestResult,
        trades: object,
        ohlcv_by_pair: Dict[str, Any],
    ) -> None:
        """Replace realised-close evidence with MTM evidence when V2 requests it.

        A replay failure is evidence, not a backtest failure: the realised-close
        series remains available for diagnostics while the V2 adapter refuses
        to mark the scenario valid.
        """

        evaluation = self.config.get("evaluation_v2", {})
        if not evaluation.get("enabled", False) or not evaluation.get("mark_to_market", False):
            return
        if (
            result.starting_balance is None
            or result.final_balance is None
            or result.backtest_start is None
            or result.backtest_end is None
        ):
            result.mark_to_market_error_message = "realised wallet evidence is incomplete"
            return

        try:
            series = build_mark_to_market_equity(
                period_start=result.backtest_start,
                period_end=result.backtest_end,
                starting_balance=result.starting_balance,
                trades=trades if trades is not None else [],
                ohlcv_by_pair=ohlcv_by_pair,
                expected_final_balance=result.final_balance,
            )
            metrics = calculate_equity_risk_metrics(
                series,
                annual_risk_free_rate=evaluation.get(
                    "annual_risk_free_rate",
                    DEFAULT_ANNUAL_RISK_FREE_RATE,
                ),
            )
        except EquityDataError as exc:
            result.mark_to_market_error_message = str(exc)
            return

        result.daily_profit_abs = [
            [day.isoformat(), pnl] for day, pnl in zip(series.dates, series.daily_profit_abs)
        ]
        result.daily_net_returns = list(series.daily_net_returns)
        result.equity_curve = list(series.equity_curve)
        result.equity_method = series.equity_method
        result.mark_to_market_error_message = None
        result.annualized_net_return = metrics.annualized_net_return
        result.daily_sharpe_ratio = metrics.sharpe_ratio
        result.daily_sortino_ratio = metrics.sortino_ratio
        result.daily_expected_shortfall_5 = metrics.daily_expected_shortfall_5
        result.calmar_ratio = metrics.calmar_ratio
        result.ulcer_index = metrics.ulcer_index
        result.time_under_water_ratio = metrics.time_under_water_ratio
        result.daily_max_drawdown = metrics.max_drawdown
        result.max_drawdown_duration_days = metrics.max_drawdown_duration_days
        result.risk_metric_contract_version = metrics.risk_metric_contract_version
        result.risk_periods_per_year = metrics.periods_per_year
        result.annual_risk_free_rate = metrics.annual_risk_free_rate
        result.periodic_risk_free_rate = metrics.periodic_risk_free_rate

    def backtest_strategy_with_trades(
        self,
        strategy_code: str,
        strategy_name: str,
        max_retries: int = 2,
        strategy_max_open_trades: Optional[int] = None,
    ) -> BacktestResult:
        """
        Run backtest and collect detailed trade data for visualization.

        This method is similar to backtest_strategy() but also collects
        the individual trades and OHLCV data needed for trade visualization.

        Args:
            strategy_code: Python code for strategy
            strategy_name: Name of the strategy
            max_retries: Maximum number of retries on failure
            strategy_max_open_trades: Optional max open trades override

        Returns:
            BacktestResult object with trades and ohlcv_data populated
        """
        start_time = time.time()

        # Note: We don't use cache for trade visualization as we need fresh data

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.debug(f"Retry {attempt}/{max_retries} for {strategy_name}")
                    time.sleep(1)

                # Run backtest with trade collection enabled
                result = self._run_backtest_direct(
                    strategy_code, strategy_name, strategy_max_open_trades, collect_trades=True
                )
                result.execution_time = time.time() - start_time

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Backtest with trades attempt {attempt + 1} failed: {e}")
                import traceback

                traceback.print_exc()

        # All retries failed
        execution_time = time.time() - start_time
        return BacktestResult(
            success=False,
            strategy_name=strategy_name,
            error_message=f"Failed after {max_retries + 1} attempts: {str(last_error)}",
            execution_time=execution_time,
        )

    def _load_ohlcv_for_pairs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load OHLCV data from disk for trade visualization.

        Args:
            config: Backtest configuration with pairs, datadir, etc.

        Returns:
            Dictionary mapping "pair_timeframe" to DataFrame
        """
        ohlcv_dict = {}

        try:
            from freqtrade.data.history import load_pair_history
            from freqtrade.enums import CandleType

            pairs = config.get("exchange", {}).get("pair_whitelist", [])
            datadir = config.get("datadir")

            # Get the base timeframe from strategy config
            # Use 5m as default since it's the most common
            timeframe = "5m"

            for pair in pairs:
                try:
                    df = load_pair_history(
                        pair=pair,
                        timeframe=timeframe,
                        datadir=datadir,
                        candle_type=CandleType.SPOT,
                        timerange=None,  # Load all data
                    )

                    if df is not None and len(df) > 0:
                        key = f"{pair.replace('/', '_')}_{timeframe}"
                        ohlcv_dict[key] = df
                        logger.debug(f"Loaded {len(df)} candles for {pair} {timeframe}")

                except Exception as e:
                    logger.warning(f"Could not load OHLCV data for {pair}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load OHLCV data from disk: {e}")

        return ohlcv_dict
