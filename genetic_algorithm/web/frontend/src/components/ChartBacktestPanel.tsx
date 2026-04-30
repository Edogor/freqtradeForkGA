/**
 * ChartBacktestPanel
 *
 * Self-contained panel that replaces the "Price Data" card on StrategyPage.
 *
 * Features:
 * - Pair / Timeframe / Exchange selectors
 * - Candlestick chart with indicator overlays
 * - Click-to-select range mode: click start candle → click end candle
 * - Quick config: Stake ($), Stoploss (%)
 * - Advanced collapsible: Max Open Trades, Trailing Stop
 * - Run Backtest → polls until done → inline results + trade markers on chart
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play, Loader2, Check, ChevronDown, ChevronUp,
  Crosshair, RotateCcw, TrendingUp, TrendingDown,
  BarChart3, Calendar, DollarSign, ShieldAlert, Eye,
} from 'lucide-react';
import { api } from '../api/client';
import { CandlestickChart, parseOHLCVCandles } from './CandlestickChart';
import type { SelectionRange } from './CandlestickChart';
import type { Candle, IndicatorLine } from './CandlestickChart';
import type {
  StrategyGene,
  PairInfo,
  BacktestTrade,
  BacktestTradesResponse,
} from '../types';

// ── Helpers ───────────────────────────────────────────────────

function fmtUnix(unix: number): string {
  const d = new Date(unix * 1000);
  return d.toISOString().slice(0, 10);
}

function unixToYYYYMMDD(unix: number): string {
  const d = new Date(unix * 1000);
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, '0');
  const day = String(d.getUTCDate()).padStart(2, '0');
  return `${y}${m}${day}`;
}

function yyyymmddToUnix(s: string): number | null {
  if (s.length < 8) return null;
  const year = parseInt(s.slice(0, 4), 10);
  const month = parseInt(s.slice(4, 6), 10) - 1;
  const day = parseInt(s.slice(6, 8), 10);
  const d = new Date(Date.UTC(year, month, day));
  return isNaN(d.getTime()) ? null : Math.floor(d.getTime() / 1000);
}

function StatChip({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-surface-2 rounded-lg px-3 py-2 min-w-0">
      <div className="text-[10px] text-gray-500 uppercase truncate">{label}</div>
      <div className={`text-sm font-mono truncate ${color ?? 'text-gray-200'}`}>{value}</div>
    </div>
  );
}

import type { IndicatorPreview } from './StrategyGeneTree';

// ── Props ─────────────────────────────────────────────────────

interface ChartBacktestPanelProps {
  gene: StrategyGene;
  availablePairs: PairInfo[];
  defaultPair?: string;
  defaultTimeframe?: string;
  defaultExchange?: string;
  /** Called when a backtest completes — passes back trades for cross-card use */
  onTradesLoaded?: (trades: BacktestTrade[], backtestId: string) => void;
  /** If set, a specific indicator is highlighted/focused on the chart */
  highlightIndicator?: IndicatorPreview | null;
}

// ── Component ─────────────────────────────────────────────────

export function ChartBacktestPanel({
  gene,
  availablePairs,
  defaultPair = '',
  defaultTimeframe = '',
  defaultExchange = 'binance',
  onTradesLoaded,
  highlightIndicator,
}: ChartBacktestPanelProps) {

  // ── Chart data ──────────────────────────────────────────────
  const [selectedPair, setSelectedPair] = useState(defaultPair);
  const [selectedTimeframe, setSelectedTimeframe] = useState(defaultTimeframe || gene.timeframe || '5m');
  const [selectedExchange, setSelectedExchange] = useState(defaultExchange);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [indicatorLines, setIndicatorLines] = useState<IndicatorLine[]>([]);
  const [chartLoading, setChartLoading] = useState(false);

  // ── Range selection ─────────────────────────────────────────
  type RangeMode = 'off' | 'picking_start' | 'picking_end';
  const [rangeMode, setRangeMode] = useState<RangeMode>('off');
  const [rangeStart, setRangeStart] = useState<number | null>(null);
  const [rangeEnd, setRangeEnd] = useState<number | null>(null);
  // Manual text inputs (YYYYMMDD)
  const [manualStart, setManualStart] = useState('');
  const [manualEnd, setManualEnd] = useState('');

  // ── Backtest config ─────────────────────────────────────────
  const [stakeAmount, setStakeAmount] = useState('100');
  const [stoplossInput, setStoplossInput] = useState(
    gene.stoploss != null ? String(Math.abs(gene.stoploss * 100).toFixed(1)) : '5',
  );
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxOpenTrades, setMaxOpenTrades] = useState(
    gene.max_open_trades != null ? String(gene.max_open_trades) : '3',
  );
  const [trailingStop, setTrailingStop] = useState(gene.trailing_stop ?? false);

  // ── Backtest execution ──────────────────────────────────────
  const [btId, setBtId] = useState<string | null>(null);
  const [btStatus, setBtStatus] = useState<string | null>(null);
  const [btResult, setBtResult] = useState<Record<string, any> | null>(null);
  const [trades, setTrades] = useState<BacktestTrade[]>([]);
  const [btRunning, setBtRunning] = useState(false);
  const [btError, setBtError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Sync defaults when parent props change ──────────────────
  useEffect(() => { if (defaultPair && !selectedPair) setSelectedPair(defaultPair); }, [defaultPair]);
  useEffect(() => { if (defaultExchange) setSelectedExchange(defaultExchange); }, [defaultExchange]);

  // ── Load candles ────────────────────────────────────────────
  useEffect(() => {
    if (!selectedPair || !selectedTimeframe) return;
    setChartLoading(true);
    setCandles([]);
    api.getOHLCV({
      pair: selectedPair,
      timeframe: selectedTimeframe,
      exchange: selectedExchange,
      limit: 1000,
    })
      .then((r) => setCandles(parseOHLCVCandles(r.candles)))
      .catch(() => {})
      .finally(() => setChartLoading(false));
  }, [selectedPair, selectedTimeframe, selectedExchange]);

  // ── Load indicator overlays ─────────────────────────────────
  useEffect(() => {
    if (!selectedPair || !selectedTimeframe || !gene.indicators?.length) {
      setIndicatorLines([]);
      return;
    }
    const indNames = gene.indicators
      .map((i) => i.type)
      .filter(Boolean)
      .join(',');
    api.getIndicators({
      pair: selectedPair,
      timeframe: selectedTimeframe,
      exchange: selectedExchange,
      indicators: indNames,
      limit: 1000,
    })
      .then((r) => {
        const lines: IndicatorLine[] = Object.entries(r.indicators || {}).map(([name, data]) => ({
          name,
          data: (data as any[]).map(([ts, v]: [number, number]) => ({
            time: Math.floor(ts / 1000),
            value: v,
          })),
          pane: 'price' as const,
        }));
        setIndicatorLines(lines);
      })
      .catch(() => setIndicatorLines([]));
  }, [selectedPair, selectedTimeframe, selectedExchange, gene.indicators]);

  // ── Highlighted indicator lines (T3-6) ──────────────────────
  // When a specific indicator is highlighted, give it a thicker line
  const visibleIndicatorLines: IndicatorLine[] = highlightIndicator
    ? indicatorLines.map((l) => {
        const nameMatch = l.name.toLowerCase().startsWith(highlightIndicator.type.toLowerCase());
        return nameMatch ? { ...l, lineWidth: 3 } : { ...l, lineWidth: 1 };
      })
    : indicatorLines;

  // ── Candle click handler ────────────────────────────────────
  const handleCandleClick = useCallback((unixSec: number) => {
    if (rangeMode === 'picking_start') {
      setRangeStart(unixSec);
      setManualStart(unixToYYYYMMDD(unixSec));
      setRangeMode('picking_end');
    } else if (rangeMode === 'picking_end') {
      const start = rangeStart ?? unixSec;
      const end = unixSec;
      const [s, e] = start <= end ? [start, end] : [end, start];
      setRangeStart(s);
      setRangeEnd(e);
      setManualStart(unixToYYYYMMDD(s));
      setManualEnd(unixToYYYYMMDD(e));
      setRangeMode('off');
    }
  }, [rangeMode, rangeStart]);

  // Selection range for the chart highlight
  const selectionRange: SelectionRange | null =
    rangeStart != null
      ? { start: rangeStart, end: rangeEnd ?? rangeStart }
      : null;

  // ── Backtest polling ────────────────────────────────────────
  useEffect(() => {
    if (!btId) return;
    const poll = async () => {
      try {
        const r = await api.getBacktestResult(btId);
        setBtStatus(r.status);
        if (r.status === 'completed' || r.status === 'failed') {
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
          if (r.status === 'completed') {
            setBtResult(r.result ?? null);
            // Load all trades
            const allT: BacktestTrade[] = [];
            let offset = 0;
            let hasMore = true;
            while (hasMore) {
              const resp: BacktestTradesResponse = await api.getBacktestTrades(btId, { offset, limit: 500 });
              allT.push(...resp.trades);
              offset += 500;
              hasMore = offset < resp.total;
            }
            setTrades(allT);
            onTradesLoaded?.(allT, btId);
          } else if (r.error) {
            setBtError(r.error);
          }
        }
      } catch { /* ignore */ }
    };
    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [btId]);

  // ── Run backtest ────────────────────────────────────────────
  const runBacktest = async () => {
    if (!selectedPair || !selectedTimeframe) {
      setBtError('Select a pair and timeframe first.');
      return;
    }

    // Resolve timerange
    let timerange = '';
    const s = rangeStart ?? (manualStart ? yyyymmddToUnix(manualStart) : null);
    const e = rangeEnd ?? (manualEnd ? yyyymmddToUnix(manualEnd) : null);
    if (s && e) {
      timerange = `${unixToYYYYMMDD(s)}-${unixToYYYYMMDD(e)}`;
    } else if (manualStart && manualEnd) {
      timerange = `${manualStart}-${manualEnd}`;
    } else {
      // Default: last 3 months of visible candles
      if (candles.length >= 2) {
        const last = candles[candles.length - 1].time;
        const approxStart = last - 90 * 24 * 3600;
        timerange = `${unixToYYYYMMDD(approxStart)}-${unixToYYYYMMDD(last)}`;
      } else {
        setBtError('No time range selected. Click "Select Range" or enter dates manually.');
        return;
      }
    }

    const stoplossVal = -(Math.abs(parseFloat(stoplossInput) || 5) / 100);
    const stakeVal = parseFloat(stakeAmount) || 100;

    setBtRunning(true);
    setBtError(null);
    setBtStatus(null);
    setBtResult(null);
    setTrades([]);

    try {
      const res = await api.startBacktest({
        strategy_gene: gene as any,
        timerange,
        pairs: [selectedPair],
        timeframe: selectedTimeframe,
        exchange: selectedExchange,
        stake_amount: stakeVal,
        stoploss: stoplossVal,
        max_open_trades: parseInt(maxOpenTrades, 10) || 3,
        trailing_stop: trailingStop,
      } as any);
      setBtId((res as any).backtest_id);
      setBtStatus('running');
    } catch (err) {
      setBtError(err instanceof Error ? err.message : String(err));
    } finally {
      setBtRunning(false);
    }
  };

  // ── Derived state ───────────────────────────────────────────
  const uniquePairs = [...new Set(availablePairs.map((p) => p.pair))].sort();
  const uniqueTimeframes = [
    ...new Set(
      availablePairs
        .filter((p) => !selectedPair || p.pair === selectedPair)
        .map((p) => p.timeframe),
    ),
  ].sort();
  const uniqueExchanges = [...new Set(availablePairs.map((p) => p.exchange))].sort();

  const currentPairTrades = trades.filter((t) => t.pair === selectedPair);
  const profit = Number(btResult?.profit_percent ?? 0);
  const winRate = btResult?.total_trades
    ? (Number(btResult.wins ?? 0) / Number(btResult.total_trades)) * 100
    : 0;

  // ── Render ──────────────────────────────────────────────────
  return (
    <div className="space-y-4">
      {/* ── Pair / Timeframe / Exchange row ── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <div>
          <label className="text-[10px] text-gray-500 uppercase block mb-1">Pair</label>
          <select
            value={selectedPair}
            onChange={(e) => setSelectedPair(e.target.value)}
            className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
          >
            <option value="">Select pair…</option>
            {uniquePairs.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div>
          <label className="text-[10px] text-gray-500 uppercase block mb-1">Timeframe</label>
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
          >
            <option value="">Select…</option>
            {uniqueTimeframes.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
          </select>
        </div>
        <div>
          <label className="text-[10px] text-gray-500 uppercase block mb-1">Exchange</label>
          <select
            value={selectedExchange}
            onChange={(e) => setSelectedExchange(e.target.value)}
            className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
          >
            {uniqueExchanges.length > 0
              ? uniqueExchanges.map((ex) => <option key={ex} value={ex}>{ex}</option>)
              : <option value="binance">binance</option>}
          </select>
        </div>
      </div>

      {/* ── Chart ── */}
      {chartLoading && (
        <div className="flex items-center justify-center py-12 text-gray-500 gap-2">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading chart…
        </div>
      )}

      {!chartLoading && candles.length > 0 && (
        <>
          {highlightIndicator && (
            <div className="flex items-center gap-2 px-2 py-1 mb-1 rounded-lg bg-accent/10 border border-accent/20 text-xs text-accent">
              <Eye className="w-3 h-3 flex-shrink-0" />
              Previewing <span className="font-mono font-semibold">{highlightIndicator.type}</span>
              <span className="text-gray-500">
                ({Object.entries(highlightIndicator.parameters).map(([k,v]) => `${k}=${v}`).join(', ')})
              </span>
            </div>
          )}
          <CandlestickChart
            candles={candles}
            trades={currentPairTrades}
            indicators={visibleIndicatorLines}
            height={400}
            onCandleClick={rangeMode !== 'off' ? handleCandleClick : undefined}
            selectionRange={selectionRange}
          />
        </>
      )}

      {!chartLoading && candles.length === 0 && selectedPair && (
        <div className="flex items-center justify-center py-12 text-gray-500 text-sm">
          No candle data available for {selectedPair} / {selectedTimeframe}
        </div>
      )}

      {/* ── Range selection controls ── */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          {rangeMode === 'off' ? (
            <button
              onClick={() => { setRangeMode('picking_start'); setRangeStart(null); setRangeEnd(null); }}
              className="flex items-center gap-1.5 text-xs bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 px-3 py-1.5 rounded-lg hover:bg-indigo-500/25 transition-colors"
            >
              <Crosshair className="w-3 h-3" /> Select Range
            </button>
          ) : (
            <button
              onClick={() => setRangeMode('off')}
              className="flex items-center gap-1.5 text-xs bg-yellow-500/15 text-yellow-300 border border-yellow-500/30 px-3 py-1.5 rounded-lg hover:bg-yellow-500/25 transition-colors animate-pulse"
            >
              <Crosshair className="w-3 h-3" />
              {rangeMode === 'picking_start' ? 'Click start candle…' : 'Click end candle…'}
            </button>
          )}

          {(rangeStart || rangeEnd) && (
            <button
              onClick={() => { setRangeStart(null); setRangeEnd(null); setManualStart(''); setManualEnd(''); }}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
              title="Clear range"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          )}
        </div>

        {/* Date chip / manual inputs */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <Calendar className="w-3 h-3 text-gray-500 shrink-0" />
          <input
            type="text"
            placeholder="YYYYMMDD start"
            value={manualStart}
            onChange={(e) => {
              setManualStart(e.target.value);
              const unix = yyyymmddToUnix(e.target.value);
              if (unix) setRangeStart(unix);
            }}
            className="w-28 bg-surface-2 border border-white/10 rounded px-2 py-1 text-xs font-mono text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
          />
          <span className="text-gray-600 text-xs">→</span>
          <input
            type="text"
            placeholder="YYYYMMDD end"
            value={manualEnd}
            onChange={(e) => {
              setManualEnd(e.target.value);
              const unix = yyyymmddToUnix(e.target.value);
              if (unix) setRangeEnd(unix);
            }}
            className="w-28 bg-surface-2 border border-white/10 rounded px-2 py-1 text-xs font-mono text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
          />
          {rangeStart && rangeEnd && (
            <span className="text-[10px] text-indigo-400 shrink-0">
              {fmtUnix(rangeStart)} → {fmtUnix(rangeEnd)}
            </span>
          )}
        </div>
      </div>

      {/* ── Quick config row ── */}
      <div className="flex flex-wrap items-end gap-3 pt-1">
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-500 uppercase flex items-center gap-1">
            <DollarSign className="w-3 h-3" /> Stake Amount
          </label>
          <input
            type="number"
            min="1"
            step="10"
            value={stakeAmount}
            onChange={(e) => setStakeAmount(e.target.value)}
            className="w-28 bg-surface-2 border border-white/10 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-500 uppercase flex items-center gap-1">
            <ShieldAlert className="w-3 h-3" /> Stoploss (%)
          </label>
          <input
            type="number"
            min="0.1"
            max="50"
            step="0.1"
            value={stoplossInput}
            onChange={(e) => setStoplossInput(e.target.value)}
            className="w-24 bg-surface-2 border border-white/10 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
          />
        </div>

        {/* Advanced toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 border border-white/10 rounded px-3 py-1.5 bg-surface-2 transition-colors self-end"
        >
          Advanced {showAdvanced ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>

        {/* Run button */}
        <button
          onClick={runBacktest}
          disabled={btRunning || btStatus === 'running' || !selectedPair}
          className="flex items-center gap-1.5 text-xs bg-accent/20 text-accent border border-accent/30 px-4 py-2 rounded-lg hover:bg-accent/30 transition-colors disabled:opacity-50 self-end ml-auto"
        >
          {btRunning || btStatus === 'running' ? (
            <><Loader2 className="w-3 h-3 animate-spin" /> Running…</>
          ) : (
            <><Play className="w-3 h-3" /> Run Backtest</>
          )}
        </button>
      </div>

      {/* ── Advanced section ── */}
      {showAdvanced && (
        <div className="bg-surface-2/40 border border-white/5 rounded-xl p-4 space-y-3">
          <p className="text-[10px] text-gray-500 uppercase tracking-wider">Advanced Configuration</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-gray-500 uppercase">Max Open Trades</label>
              <input
                type="number"
                min="1"
                max="20"
                step="1"
                value={maxOpenTrades}
                onChange={(e) => setMaxOpenTrades(e.target.value)}
                className="bg-surface-2 border border-white/10 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-gray-500 uppercase">Trailing Stop</label>
              <button
                type="button"
                onClick={() => setTrailingStop(!trailingStop)}
                className={[
                  'flex items-center gap-2 px-3 py-1.5 rounded border text-xs transition-colors',
                  trailingStop
                    ? 'bg-accent/20 border-accent/30 text-accent'
                    : 'bg-surface-2 border-white/10 text-gray-400',
                ].join(' ')}
              >
                <span className={[
                  'w-3.5 h-3.5 rounded-full border-2 transition-colors',
                  trailingStop ? 'bg-accent border-accent' : 'bg-transparent border-gray-500',
                ].join(' ')} />
                {trailingStop ? 'Enabled' : 'Disabled'}
              </button>
            </div>
            <div className="col-span-2 text-[10px] text-gray-600 self-end pb-1">
              All other parameters (ROI table, indicators, conditions) are inherited from the strategy gene.
            </div>
          </div>
        </div>
      )}

      {/* ── Error ── */}
      {btError && (
        <div className="text-xs text-loss bg-loss/10 border border-loss/20 rounded-lg px-3 py-2">
          {btError}
        </div>
      )}

      {/* ── Backtest results ── */}
      {btStatus === 'completed' && btResult && (
        <div className="border border-accent/20 rounded-xl bg-accent/5 p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-accent" /> Backtest Results
            </h4>
            {btId && (
              <a
                href={`/backtest/${btId}`}
                className="text-xs text-accent hover:underline"
              >
                Full detail →
              </a>
            )}
          </div>

          {/* Summary chips */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
            <StatChip
              label="Total Profit"
              value={`${profit >= 0 ? '+' : ''}${profit.toFixed(2)}%`}
              color={profit >= 0 ? 'text-profit' : 'text-loss'}
            />
            <StatChip label="Total Trades" value={String(btResult.total_trades ?? 0)} />
            <StatChip
              label="Win Rate"
              value={btResult.total_trades ? `${winRate.toFixed(1)}%` : '—'}
              color={winRate >= 50 ? 'text-profit' : 'text-loss'}
            />
            <StatChip
              label="Max Drawdown"
              value={`${(Number(btResult.max_drawdown ?? 0) * 100).toFixed(1)}%`}
              color="text-loss"
            />
            <StatChip
              label="Sharpe"
              value={btResult.sharpe_ratio != null ? Number(btResult.sharpe_ratio).toFixed(2) : '—'}
            />
            <StatChip
              label="Profit Factor"
              value={btResult.profit_factor != null ? Number(btResult.profit_factor).toFixed(2) : '—'}
              color={Number(btResult.profit_factor ?? 0) >= 1 ? 'text-profit' : 'text-loss'}
            />
          </div>

          {/* Trade count note */}
          {currentPairTrades.length > 0 && (
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <Check className="w-3 h-3 text-profit" />
              {currentPairTrades.length} trade markers shown on chart for {selectedPair}
              <span className="text-gray-600">·</span>
              <span className="text-profit flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                {currentPairTrades.filter((t) => t.profit_ratio > 0).length} wins
              </span>
              <span className="text-loss flex items-center gap-1">
                <TrendingDown className="w-3 h-3" />
                {currentPairTrades.filter((t) => t.profit_ratio <= 0).length} losses
              </span>
            </div>
          )}

          {/* No trades warning */}
          {btResult.total_trades === 0 && (
            <div className="text-xs text-yellow-400 bg-yellow-500/10 border border-yellow-500/20 rounded-lg px-3 py-2">
              No trades generated — try widening the date range or reducing the stoploss.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
