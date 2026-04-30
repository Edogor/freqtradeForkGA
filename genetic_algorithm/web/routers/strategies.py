"""
Strategies API — inspect individual strategies, view code, compare.
"""

from __future__ import annotations

import copy
import threading
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from genetic_algorithm.web.models.strategy import StrategyDetail, LineageResponse
from genetic_algorithm.web.routers.backtest import _backtests, _backtest_lock, _active_count

logger = __import__("logging").getLogger(__name__)
router = APIRouter(prefix="/api", tags=["strategies"])


def _data(request: Request):
    return request.app.state.data_service


# ── Read endpoints ──────────────────────────────────────────────

@router.get("/runs/{run_id}/strategies/{strategy_id}", response_model=StrategyDetail)
async def get_strategy(run_id: str, strategy_id: str, request: Request):
    """Get detailed info about a specific strategy."""
    detail = _data(request).get_strategy(run_id, strategy_id)
    if not detail:
        raise HTTPException(404, f"Strategy {strategy_id} not found in run {run_id}")
    return detail


@router.get("/runs/{run_id}/strategies/{strategy_id}/code")
async def get_strategy_code(run_id: str, strategy_id: str, request: Request):
    """Get generated Python code for a strategy."""
    code = _data(request).get_strategy_code(run_id, strategy_id)
    if code is None:
        raise HTTPException(404, f"Cannot generate code for {strategy_id}")
    return {"strategy_id": strategy_id, "code": code}


@router.get("/hall-of-fame")
async def get_hall_of_fame(request: Request):
    """Get all Hall of Fame entries."""
    return _data(request).get_hall_of_fame()


@router.get("/runs/{run_id}/lineage/{strategy_id}", response_model=LineageResponse)
async def get_lineage(run_id: str, strategy_id: str, request: Request):
    """Trace a strategy's ancestral lineage through parent chain."""
    chain = _data(request).get_lineage(run_id, strategy_id)
    return LineageResponse(strategy_id=strategy_id, run_id=run_id, chain=chain)


# ── Interactive test endpoint ───────────────────────────────────

class StrategyTestRequest(BaseModel):
    """Request body for interactive strategy parameter testing."""

    # Partial gene overrides — only include fields you want to change.
    # Merged on top of the original strategy gene.
    gene_overrides: Dict[str, Any] = Field(default_factory=dict)

    # Optional backtest window; defaults to the run's configured timerange
    timerange: str = ""
    pairs: List[str] = Field(default_factory=list)
    exchange: str = ""


@router.post("/runs/{run_id}/strategies/{strategy_id}/test")
async def test_strategy(
    run_id: str,
    strategy_id: str,
    body: StrategyTestRequest,
    request: Request,
):
    """
    Re-backtest a strategy with optional gene parameter overrides.

    Returns a backtest_id that can be polled via GET /api/backtest/{id}.
    Trade markers will use a different colour on the chart to distinguish
    the test run from the original backtest.
    """
    global _active_count
    max_concurrent = 2
    try:
        max_concurrent = request.app.state.web_config.max_concurrent_backtests
    except Exception:
        pass

    with _backtest_lock:
        if _active_count >= max_concurrent:
            raise HTTPException(429, "Too many concurrent backtests")
        _active_count += 1

    # Load original strategy gene
    detail = _data(request).get_strategy(run_id, strategy_id)
    if not detail or not detail.gene:
        with _backtest_lock:
            _active_count -= 1
        raise HTTPException(404, f"Strategy {strategy_id} not found or has no gene in run {run_id}")

    # Load run config for timerange + pairs
    run_detail = _data(request).get_run_detail(run_id)
    run_config: dict = {}
    if run_detail and hasattr(run_detail, "config"):
        run_config = run_detail.config or {}

    # Resolve backtest params
    bt_cfg = run_config.get("backtesting", {}) if isinstance(run_config, dict) else {}
    timerange = body.timerange or bt_cfg.get("timerange", "20250101-20260101")
    pairs = body.pairs or bt_cfg.get("pairs", [])
    exchange = body.exchange or bt_cfg.get("exchange", "binance")

    # Merge gene overrides on top of the original gene dict
    original_gene = detail.gene.model_dump() if hasattr(detail.gene, "model_dump") else dict(detail.gene)
    merged_gene = copy.deepcopy(original_gene)
    if body.gene_overrides:
        _deep_merge(merged_gene, body.gene_overrides)

    # Create backtest entry
    backtest_id = f"test_{uuid.uuid4().hex[:8]}"
    from genetic_algorithm.web.models.strategy import BacktestResultModel
    result = BacktestResultModel(backtest_id=backtest_id, status="running")
    with _backtest_lock:
        _backtests[backtest_id] = result

    # Build a BacktestRequest-compatible object and reuse _run_backtest
    from genetic_algorithm.web.routers.backtest import _run_backtest
    from genetic_algorithm.web.models.strategy import BacktestRequest

    bt_body = BacktestRequest(
        strategy_gene=merged_gene,
        timerange=timerange,
        pairs=pairs if isinstance(pairs, list) else [],
        exchange=exchange,
    )

    thread = threading.Thread(
        target=_run_backtest,
        args=(backtest_id, bt_body),
        daemon=True,
        name=f"strategy-test-{backtest_id}",
    )
    thread.start()

    return {"backtest_id": backtest_id, "status": "running"}


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into base dict in-place."""
    for key, val in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
