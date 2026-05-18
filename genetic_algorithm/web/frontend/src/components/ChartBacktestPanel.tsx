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
  RotateCcw, TrendingUp, TrendingDown,
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
  const [strategyTimeframe, setStrategyTimeframe] = useState(gene.timeframe || '5m');
  const [showROITable, setShowROITable] = useState(false);
  // Track overrides: path → value (e.g., "indicators.0.parameters.period" → 20)
  const [parameterOverrides, setParameterOverrides] = useState<Record<string, unknown>>({});

  // ── Backtest execution ──────────────────────────────────────
  const [btId, setBtId] = useState<string | null>(null);
  const [btStatus, setBtStatus] = useState<string | null>(null);
  const [btResult, setBtResult] = useState<Record<string, any> | null>(null);
  const [trades, setTrades] = useState<BacktestTrade[]>([]);
  const [btRunning, setBtRunning] = useState(false);
  const [btError, setBtError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Infinite scroll state ───────────────────────────────────
  const [moreDataLoading, setMoreDataLoading] = useState(false);
  const [noMoreData, setNoMoreData] = useState(false);
  const oldestCandleTimeRef = useRef<number>(0);
  const newestCandleTimeRef = useRef<number>(0);

  // ── Sync defaults when parent props change ──────────────────
  useEffect(() => { if (defaultPair && !selectedPair) setSelectedPair(defaultPair); }, [defaultPair]);
  useEffect(() => { if (defaultExchange) setSelectedExchange(defaultExchange); }, [defaultExchange]);

  // ── Load candles ────────────────────────────────────────────
  useEffect(() => {
    if (!selectedPair || !selectedTimeframe) return;
    setChartLoading(true);
    setCandles([]);
    setNoMoreData(false);
    oldestCandleTimeRef.current = 0;
    api.getOHLCV({
      pair: selectedPair,
      timeframe: selectedTimeframe,
      exchange: selectedExchange,
      limit: 5000,
    })
      .then((r) => {
        const parsed = parseOHLCVCandles(r.candles);
        setCandles(parsed);
        if (parsed.length > 0) {
          oldestCandleTimeRef.current = parsed[0].time;
          newestCandleTimeRef.current = parsed[parsed.length - 1].time;
        }
      })
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
          data: (data as unknown as [number, number][]).map(([ts, v]: [number, number]) => ({
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
    // First click sets start (or resets an existing range)
    if (rangeStart == null || rangeEnd != null) {
      setRangeStart(unixSec);
      setRangeEnd(null);
      setManualStart(unixToYYYYMMDD(unixSec));
      setManualEnd('');
      return;
    }

    // Second click sets end
    if (rangeStart != null && rangeEnd == null) {
      const start = rangeStart ?? unixSec;
      const end = unixSec;
      const [s, e] = start <= end ? [start, end] : [end, start];
      setRangeStart(s);
      setRangeEnd(e);
      setManualStart(unixToYYYYMMDD(s));
      setManualEnd(unixToYYYYMMDD(e));
    }
  }, [rangeStart, rangeEnd]);

  // Selection range for the chart highlight
  const selectionRange: SelectionRange | null =
    rangeStart != null
      ? { start: rangeStart, end: rangeEnd ?? rangeStart }
      : null;

  // ── Infinite scroll: load older candles ────────────────────
  const handleNeedMoreData = useCallback(async () => {
    if (moreDataLoading || noMoreData || !oldestCandleTimeRef.current || !selectedPair || !selectedTimeframe) return;
    setMoreDataLoading(true);
    try {
      // Fetch candles ending one day before our oldest candle
      const endStr = unixToYYYYMMDD(oldestCandleTimeRef.current - 86400);
      const [ohlcvResp, indResp] = await Promise.all([
        api.getOHLCV({ pair: selectedPair, timeframe: selectedTimeframe, exchange: selectedExchange, end: endStr, limit: 3000 }),
        gene.indicators?.length
          ? api.getIndicators({
              pair: selectedPair,
              timeframe: selectedTimeframe,
              exchange: selectedExchange,
              indicators: gene.indicators.map((i) => i.type).join(','),
              end: endStr,
              limit: 3000,
            })
          : Promise.resolve(null),
      ]);

      const newCandles = parseOHLCVCandles(ohlcvResp.candles);
      if (newCandles.length === 0) {
        setNoMoreData(true);
        return;
      }

      // Prepend (deduplication by time)
      setCandles((prev) => {
        const existingTimes = new Set(prev.map((c) => c.time));
        const fresh = newCandles.filter((c) => !existingTimes.has(c.time));
        if (fresh.length === 0) {
          setNoMoreData(true);
          return prev;
        }
        const merged = [...fresh, ...prev];
        oldestCandleTimeRef.current = merged[0].time;
        newestCandleTimeRef.current = merged[merged.length - 1].time;
        return merged;
      });

      // Merge indicator data for older range
      if (indResp) {
        const newLines: IndicatorLine[] = Object.entries(indResp.indicators || {}).map(([name, data]) => ({
          name,
          data: (data as unknown as [number, number][]).map(([ts, v]) => ({ time: Math.floor(ts / 1000), value: v })),
          pane: 'price' as const,
        }));
        setIndicatorLines((prev) =>
          prev.map((existing) => {
            const fresh = newLines.find((l) => l.name === existing.name);
            if (!fresh) return existing;
            const existingTimes = new Set(existing.data.map((d) => d.time));
            return {
              ...existing,
              data: [...fresh.data.filter((d) => !existingTimes.has(d.time)), ...existing.data],
            };
          }),
        );
      }

      if (newCandles.length < 50) setNoMoreData(true);
    } catch {
      // Ignore transient network errors while panning
    } finally {
      setMoreDataLoading(false);
    }
  }, [moreDataLoading, noMoreData, selectedPair, selectedTimeframe, selectedExchange, gene.indicators]);

  // ── Infinite scroll: load newer candles ────────────────────
  const handleNeedNewerData = useCallback(async () => {
    if (moreDataLoading || !newestCandleTimeRef.current || !selectedPair || !selectedTimeframe) return;
    setMoreDataLoading(true);
    try {
      // Fetch candles starting one day after our newest candle
      const startStr = unixToYYYYMMDD(newestCandleTimeRef.current + 86400);
      const [ohlcvResp, indResp] = await Promise.all([
        api.getOHLCV({ pair: selectedPair, timeframe: selectedTimeframe, exchange: selectedExchange, start: startStr, limit: 3000 }),
        gene.indicators?.length
          ? api.getIndicators({
              pair: selectedPair,
              timeframe: selectedTimeframe,
              exchange: selectedExchange,
              indicators: gene.indicators.map((i) => i.type).join(','),
              start: startStr,
              limit: 3000,
            })
          : Promise.resolve(null),
      ]);

      const newCandles = parseOHLCVCandles(ohlcvResp.candles);
      if (newCandles.length === 0) return;

      // Append (deduplication by time)
      setCandles((prev) => {
        const existingTimes = new Set(prev.map((c) => c.time));
        const fresh = newCandles.filter((c) => !existingTimes.has(c.time));
        if (fresh.length === 0) return prev;
        const merged = [...prev, ...fresh];
        oldestCandleTimeRef.current = merged[0].time;
        newestCandleTimeRef.current = merged[merged.length - 1].time;
        return merged;
      });

      // Merge indicator data for newer range
      if (indResp) {
        const newLines: IndicatorLine[] = Object.entries(indResp.indicators || {}).map(([name, data]) => ({
          name,
          data: (data as unknown as [number, number][]).map(([ts, v]) => ({ time: Math.floor(ts / 1000), value: v })),
          pane: 'price' as const,
        }));
        setIndicatorLines((prev) =>
          prev.map((existing) => {
            const fresh = newLines.find((l) => l.name === existing.name);
            if (!fresh) return existing;
            const existingTimes = new Set(existing.data.map((d) => d.time));
            return {
              ...existing,
              data: [...existing.data, ...fresh.data.filter((d) => !existingTimes.has(d.time))],
            };
          }),
        );
      }
    } catch {
      // Ignore transient network errors while panning
    } finally {
      setMoreDataLoading(false);
    }
  }, [moreDataLoading, selectedPair, selectedTimeframe, selectedExchange, gene.indicators]);

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

  // ── Merge parameter overrides into gene ────────────────────
  const buildGeneWithOverrides = (): StrategyGene => {
    const overridden = JSON.parse(JSON.stringify(gene)) as StrategyGene;
    
    // Direct overrides
    if (strategyTimeframe) overridden.timeframe = strategyTimeframe;
    
    // Apply parameter overrides (e.g., indicator period changes, ROI changes)
    Object.entries(parameterOverrides).forEach(([path, value]) => {
      const parts = path.split('.');
      let obj: any = overridden;
      for (let i = 0; i < parts.length - 1; i++) {
        const part = parts[i];
        if (part.match(/^\d+$/)) {
          obj = obj[parseInt(part, 10)];
        } else {
          obj = obj[part] = obj[part] || {};
        }
      }
      const lastKey = parts[parts.length - 1];
      if (lastKey.match(/^\d+$/)) {
        obj[parseInt(lastKey, 10)] = value;
      } else {
        obj[lastKey] = value;
      }
    });
    
    return overridden;
  };

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
        setBtError('No time range selected. Click the chart to set start/end or enter dates manually.');
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
      // Build gene with parameter overrides applied
      const geneWithOverrides = buildGeneWithOverrides();
      
      const res = await api.startBacktest({
        strategy_gene: geneWithOverrides as any,
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
            onCandleClick={handleCandleClick}
            selectionRange={selectionRange}
            onNeedMoreData={handleNeedMoreData}
            onNeedNewerData={handleNeedNewerData}
            moreDataLoading={moreDataLoading}
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
          <span className="text-xs text-indigo-300 bg-indigo-500/10 border border-indigo-500/30 px-3 py-1.5 rounded-lg">
            {rangeStart == null || rangeEnd != null
              ? 'Click chart to set range start'
              : 'Click chart again to set range end'}
          </span>

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
        <div className="bg-surface-2/40 border border-white/5 rounded-xl p-4 space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-[10px] text-gray-500 uppercase tracking-wider">Advanced Configuration</p>
            {Object.keys(parameterOverrides).length > 0 && (
              <button
                onClick={() => setParameterOverrides({})}
                className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300 transition-colors"
                title="Reset all overrides to defaults"
              >
                <RotateCcw className="w-3 h-3" /> Reset
              </button>
            )}
          </div>

          {/* ── Core Strategy Parameters ── */}
          <div className="bg-surface-3/50 border border-white/5 rounded-lg p-3 space-y-3">
            <p className="text-[9px] text-gray-600 uppercase tracking-wider">Core Strategy</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {/* Strategy Timeframe (independent from chart) */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-gray-500 uppercase">Strategy Timeframe</label>
                <select
                  value={strategyTimeframe}
                  onChange={(e) => setStrategyTimeframe(e.target.value)}
                  className="bg-surface-2 border border-white/10 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
                >
                  {['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d'].map((tf) => (
                    <option key={tf} value={tf}>{tf}</option>
                  ))}
                </select>
                <p className="text-[8px] text-gray-600 mt-0.5">Chart timeframe is independent</p>
              </div>

              {/* Stoploss (from quick config, shown again here for reference) */}
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-gray-500 uppercase">Stoploss (%)</label>
                <input
                  type="number"
                  min="0.1"
                  max="50"
                  step="0.1"
                  value={stoplossInput}
                  onChange={(e) => setStoplossInput(e.target.value)}
                  className="bg-surface-2 border border-white/10 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
                />
              </div>

              {/* Max Open Trades */}
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

              {/* Trailing Stop */}
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
            </div>
          </div>

          {/* ── Indicators ── */}
          {gene.indicators && gene.indicators.length > 0 && (
            <div className="bg-surface-3/50 border border-white/5 rounded-lg p-3 space-y-2">
              <p className="text-[9px] text-gray-600 uppercase tracking-wider">Indicators</p>
              <div className="space-y-2">
                {gene.indicators.map((ind, idx) => {
                  const hasParams = ind.parameters && Object.keys(ind.parameters).length > 0;
                  return (
                    <div key={`ind-${idx}`} className="bg-surface-2/50 border border-white/5 rounded p-2">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <div>
                          <p className="text-xs font-mono text-gray-300">{ind.type}</p>
                          <p className="text-[8px] text-gray-600">ID: {ind.instance_id ?? '—'}</p>
                        </div>
                        <span className="text-[9px] text-gray-600">w: {(ind.weight ?? 1).toFixed(2)}</span>
                      </div>
                      {hasParams && (
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-2">
                          {Object.entries(ind.parameters).map(([key, val]) => {
                            const overrideKey = `indicators.${idx}.parameters.${key}`;
                            const currentVal = (parameterOverrides[overrideKey] ?? val) as any;
                            return (
                              <div key={`${idx}-${key}`} className="flex flex-col gap-0.5">
                                <label className="text-[8px] text-gray-600 uppercase">{key}</label>
                                <input
                                  type={typeof val === 'number' ? 'number' : 'text'}
                                  value={String(currentVal)}
                                  onChange={(e) => {
                                    const parsed = typeof val === 'number'
                                      ? parseFloat(e.target.value) || 0
                                      : e.target.value;
                                    setParameterOverrides((prev) => ({
                                      ...prev,
                                      [overrideKey]: parsed,
                                    }));
                                  }}
                                  className="bg-surface-1 border border-white/10 rounded px-1.5 py-0.5 text-[10px] font-mono text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
                                />
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── ROI Table ── */}
          {gene.minimal_roi && Object.keys(gene.minimal_roi).length > 0 && (
            <div className="bg-surface-3/50 border border-white/5 rounded-lg p-3 space-y-2">
              <button
                type="button"
                onClick={() => setShowROITable(!showROITable)}
                className="flex items-center gap-2 text-[9px] text-gray-600 uppercase tracking-wider hover:text-gray-400 transition-colors"
              >
                {showROITable ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                ROI Table
              </button>
              {showROITable && (
                <div className="space-y-2">
                  {Object.entries(gene.minimal_roi).map(([timeStr, roi]) => {
                    const overrideKey = `minimal_roi.${timeStr}`;
                    const currentVal = parameterOverrides[overrideKey] ?? roi;
                    return (
                      <div key={`roi-${timeStr}`} className="flex items-center gap-2">
                        <span className="text-[9px] text-gray-600 w-12">{timeStr}m</span>
                        <input
                          type="number"
                          min="0"
                          step="0.01"
                          value={String(currentVal)}
                          onChange={(e) => {
                            setParameterOverrides((prev) => ({
                              ...prev,
                              [overrideKey]: parseFloat(e.target.value) || 0,
                            }));
                          }}
                          className="flex-1 bg-surface-2 border border-white/10 rounded px-2 py-1 text-[10px] font-mono text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
                        />
                        <span className="text-[9px] text-gray-600">
                          {((currentVal as number) * 100).toFixed(2)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* ── Info ── */}
          <div className="text-[9px] text-gray-600 bg-surface-1/50 border border-white/5 rounded p-2">
            💡 Override indicator parameters above and they'll be applied to the backtest. Leave unchanged to use defaults from the strategy gene.
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
