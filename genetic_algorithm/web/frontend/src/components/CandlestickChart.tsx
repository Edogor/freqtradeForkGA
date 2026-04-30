/**
 * Interactive candlestick chart using lightweight-charts (v4).
 *
 * Features:
 * - OHLCV candlestick rendering with volume histogram
 * - Trade markers overlay (buy/sell arrows, green/red by P/L)
 * - Indicator line overlays on price pane (EMA, SMA, Bollinger Bands)
 * - Responsive resize + dark theme
 */

import { useEffect, useRef, useCallback, memo } from 'react';
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type HistogramData,
  type LineData,
  type SeriesMarker,
  type Time,
  ColorType,
  CrosshairMode,
  LogicalRange,
} from 'lightweight-charts';
import type { BacktestTrade } from '../types';

// ── Types ─────────────────────────────────────────────────────

export interface Candle {
  time: number;  // unix timestamp in seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndicatorLine {
  name: string;
  data: { time: number; value: number }[];
  color?: string;
  lineWidth?: number;
  pane?: 'price' | 'separate';  // 'price' overlays on candlestick pane
}

interface SelectionRange {
  start: number;  // unix seconds
  end: number;    // unix seconds
}

interface CandlestickChartProps {
  candles: Candle[];
  trades?: BacktestTrade[];
  /** Test trades (from interactive panel) — rendered in purple */
  testTrades?: BacktestTrade[];
  indicators?: IndicatorLine[];
  height?: number;
  /** Show volume histogram below candles */
  showVolume?: boolean;
  /** Called when user clicks "Set as Backtest Range" with the visible time range (YYYYMMDD strings) */
  onTimeRangeSelect?: (start: string, end: string) => void;
  /** Called when user clicks a candle — receives unix seconds */
  onCandleClick?: (unixSeconds: number) => void;
  /** Highlight a selected time range on the chart (unix seconds) */
  selectionRange?: SelectionRange | null;
  /** Called when user scrolls near the left edge — parent should fetch older candles */
  onNeedMoreData?: () => void;
  /** Show a "Loading older data…" spinner overlay in top-left */
  moreDataLoading?: boolean;
}

export type { SelectionRange };

// ── Helpers ───────────────────────────────────────────────────

/** Convert unix seconds → lightweight-charts Time (UTC seconds) */
function toTime(ts: number): Time {
  return ts as Time;
}

/** Parse date string "YYYY-MM-DD HH:MM:SS" or ISO to unix seconds */
function parseDateToUnix(dateStr: string): number {
  const d = new Date(dateStr);
  return Math.floor(d.getTime() / 1000);
}

const INDICATOR_COLORS = [
  '#f59e0b', // amber
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#10b981', // emerald
  '#f97316', // orange
];

// ── Component ─────────────────────────────────────────────────

export const CandlestickChart = memo(function CandlestickChart({
  candles,
  trades,
  testTrades,
  indicators,
  height = 500,
  showVolume = true,
  onTimeRangeSelect,
  onCandleClick,
  selectionRange,
  onNeedMoreData,
  moreDataLoading = false,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const indicatorSeriesRefs = useRef<ISeriesApi<'Line'>[]>([]);
  const selectionOverlayRef = useRef<HTMLDivElement>(null);
  // Keep latest callback in a ref to avoid re-subscribing on every render
  const onCandleClickRef = useRef(onCandleClick);
  useEffect(() => { onCandleClickRef.current = onCandleClick; }, [onCandleClick]);

  // Infinite scroll — left-edge detection refs
  const onNeedMoreDataRef = useRef(onNeedMoreData);
  useEffect(() => { onNeedMoreDataRef.current = onNeedMoreData; }, [onNeedMoreData]);
  const needMoreDataCooldownRef = useRef(false);
  const logicalRangeHandlerRef = useRef<((r: LogicalRange | null) => void) | null>(null);
  // Track oldest candle time to detect prepend vs full refresh
  const prevOldestTimeRef = useRef<number>(0);

  // Create chart on mount
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: '#0f1117' },
        textColor: '#9ca3af',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.04)' },
        horzLines: { color: 'rgba(255,255,255,0.04)' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: 'rgba(255,255,255,0.15)', width: 1, style: 2, labelBackgroundColor: '#1e2030' },
        horzLine: { color: 'rgba(255,255,255,0.15)', width: 1, style: 2, labelBackgroundColor: '#1e2030' },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.08)',
        scaleMargins: { top: 0.05, bottom: showVolume ? 0.25 : 0.05 },
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.08)',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      borderVisible: false,
    });
    candleSeriesRef.current = candleSeries;

    // Volume histogram
    if (showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
      volumeSeriesRef.current = volumeSeries;
    }

    chartRef.current = chart;

    // Click handler — passes clicked candle time to parent
    chart.subscribeClick((param) => {
      if (param.time != null) {
        onCandleClickRef.current?.(param.time as number);
      }
    });

    // Responsive resize
    const resizeHandler = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    const ro = new ResizeObserver(resizeHandler);
    ro.observe(containerRef.current);

    // Left-edge scroll detection — triggers onNeedMoreData when user is near bar 50
    const logicalRangeHandler = (range: LogicalRange | null) => {
      if (!range || needMoreDataCooldownRef.current) return;
      if (range.from <= 50) {
        needMoreDataCooldownRef.current = true;
        onNeedMoreDataRef.current?.();
        setTimeout(() => { needMoreDataCooldownRef.current = false; }, 3000);
      }
    };
    logicalRangeHandlerRef.current = logicalRangeHandler;
    chart.timeScale().subscribeVisibleLogicalRangeChange(logicalRangeHandler);

    return () => {
      if (logicalRangeHandlerRef.current) {
        chart.timeScale().unsubscribeVisibleLogicalRangeChange(logicalRangeHandlerRef.current);
      }
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      indicatorSeriesRefs.current = [];
    };
  }, [height, showVolume]);

  // Update candle data
  useEffect(() => {
    if (!candleSeriesRef.current || candles.length === 0) return;

    const candleData: CandlestickData[] = candles.map((c) => ({
      time: toTime(c.time),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleSeriesRef.current.setData(candleData);

  // Detect prepend: if the new oldest candle is older than previous oldest,
  // this is a backwards data load — preserve viewport instead of fitContent
  const newOldest = candles[0].time;
  const isPrepend = prevOldestTimeRef.current > 0 && newOldest < prevOldestTimeRef.current;
  prevOldestTimeRef.current = newOldest;

    // Volume
    if (volumeSeriesRef.current) {
      const volumeData: HistogramData[] = candles.map((c) => ({
        time: toTime(c.time),
        value: c.volume,
        color: c.close >= c.open
          ? 'rgba(34,197,94,0.25)'
          : 'rgba(239,68,68,0.25)',
      }));
      volumeSeriesRef.current.setData(volumeData);
    }

    // Only fit content on initial/full load, not when prepending older candles
    if (!isPrepend) {
      chartRef.current?.timeScale().fitContent();
    }
  }, [candles]);

  // Update trade markers (original + test overlay)
  useEffect(() => {
    if (!candleSeriesRef.current) return;

    const markers: SeriesMarker<Time>[] = [];

    const addMarkers = (tradeList: BacktestTrade[], isTest: boolean) => {
      for (const t of tradeList) {
        const openTime = parseDateToUnix(t.open_date);
        const closeTime = parseDateToUnix(t.close_date);
        const profitable = t.profit_ratio > 0;

        // Test trades use purple; regular use blue/green/red
        const entryColor = isTest ? '#a855f7' : (t.is_short ? '#8b5cf6' : '#3b82f6');
        const exitColor = isTest ? (profitable ? '#c084fc' : '#f472b6') : (profitable ? '#22c55e' : '#ef4444');

        markers.push({
          time: toTime(openTime),
          position: t.is_short ? 'aboveBar' : 'belowBar',
          color: entryColor,
          shape: t.is_short ? 'arrowDown' : 'arrowUp',
          text: isTest ? '◆' : (t.is_short ? 'S' : 'B'),
        });

        markers.push({
          time: toTime(closeTime),
          position: t.is_short ? 'belowBar' : 'aboveBar',
          color: exitColor,
          shape: t.is_short ? 'arrowUp' : 'arrowDown',
          text: `${profitable ? '+' : ''}${(t.profit_ratio * 100).toFixed(1)}%`,
        });
      }
    };

    if (trades && trades.length > 0) addMarkers(trades, false);
    if (testTrades && testTrades.length > 0) addMarkers(testTrades, true);

    if (markers.length === 0) {
      candleSeriesRef.current.setMarkers([]);
      return;
    }

    // Sort by time (required by lightweight-charts)
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    candleSeriesRef.current.setMarkers(markers);
  }, [trades, testTrades]);

  // Update indicator overlays
  useEffect(() => {
    if (!chartRef.current) return;

    // Remove old indicator series
    for (const series of indicatorSeriesRefs.current) {
      try {
        chartRef.current.removeSeries(series);
      } catch { /* series may already be removed */ }
    }
    indicatorSeriesRefs.current = [];

    if (!indicators || indicators.length === 0) return;

    indicators.forEach((ind, idx) => {
      if (!chartRef.current || ind.data.length === 0) return;

      const color = ind.color || INDICATOR_COLORS[idx % INDICATOR_COLORS.length];

      // For separate pane indicators, we'd need a different approach
      // For now, all overlays go on the price pane
      const series = chartRef.current.addLineSeries({
        color,
        lineWidth: (ind.lineWidth ?? 1) as 1 | 2 | 3 | 4,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: true,
        title: ind.name,
      });

      const lineData: LineData[] = ind.data.map((d) => ({
        time: toTime(d.time),
        value: d.value,
      }));
      series.setData(lineData);
      indicatorSeriesRefs.current.push(series);
    });
  }, [indicators]);

  // Selection range overlay — position an absolutely-placed highlight div
  useEffect(() => {
    const overlay = selectionOverlayRef.current;
    const chart = chartRef.current;
    if (!overlay || !chart) return;

    if (!selectionRange) {
      overlay.style.display = 'none';
      return;
    }

    const positionOverlay = () => {
      const ts = chart.timeScale();
      const xStart = ts.timeToCoordinate(toTime(selectionRange.start));
      const xEnd   = ts.timeToCoordinate(toTime(selectionRange.end));
      if (xStart == null || xEnd == null) {
        overlay.style.display = 'none';
        return;
      }
      const left  = Math.min(xStart, xEnd);
      const width = Math.abs(xEnd - xStart);
      overlay.style.display = 'block';
      overlay.style.left   = `${left}px`;
      overlay.style.width  = `${width}px`;
    };

    positionOverlay();
    // Re-position when the user scrolls/zooms
    chart.timeScale().subscribeVisibleTimeRangeChange(positionOverlay);
    return () => { chart.timeScale().unsubscribeVisibleTimeRangeChange(positionOverlay); };
  }, [selectionRange]);

  return (
    <div className="relative">
      {onTimeRangeSelect && candles.length > 0 && (
        <button
          onClick={() => {
                  {/* Loading older data indicator */}
                  {moreDataLoading && (
                    <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5 px-2 py-1 rounded bg-black/60 text-gray-400 text-[10px] pointer-events-none">
                      <span className="w-2.5 h-2.5 border-2 border-gray-500/60 border-t-gray-300 rounded-full animate-spin flex-shrink-0" />
                      Loading older data…
                    </div>
                  )}
                  {onTimeRangeSelect && candles.length > 0 && (
                    <button
                      onClick={() => {
            const range = chartRef.current?.timeScale().getVisibleRange();
            if (range) {
              const fmt = (t: Time) => {
                const d = new Date((t as number) * 1000);
                const y = d.getUTCFullYear();
                const m = String(d.getUTCMonth() + 1).padStart(2, '0');
                const day = String(d.getUTCDate()).padStart(2, '0');
                return `${y}${m}${day}`;
              };
              onTimeRangeSelect(fmt(range.from), fmt(range.to));
            }
          }}
          className="absolute top-2 right-2 z-10 text-[10px] bg-accent/20 text-accent border border-accent/30 px-2 py-1 rounded hover:bg-accent/30 transition-colors"
        >
          Set as Backtest Range
        </button>
      )}
      {/* Selection range highlight overlay */}
      <div
        ref={selectionOverlayRef}
        className="absolute top-0 bottom-0 pointer-events-none z-10"
        style={{
          display: 'none',
          background: 'rgba(99,102,241,0.15)',
          borderLeft: '2px solid rgba(99,102,241,0.6)',
          borderRight: '2px solid rgba(99,102,241,0.6)',
        }}
      />
      <div
        ref={containerRef}
        className="w-full rounded-lg overflow-hidden border border-white/5"
        style={{ minHeight: height, cursor: onCandleClick ? 'crosshair' : 'default' }}
      />
    </div>
  );
  return (
    <div className="relative">
      {/* Loading older data indicator (top-left) */}
      {moreDataLoading && (
        <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5 px-2 py-1 rounded bg-black/60 text-gray-400 text-[10px] pointer-events-none">
          <span className="w-2.5 h-2.5 border-2 border-gray-500/60 border-t-gray-300 rounded-full animate-spin flex-shrink-0" />
          Loading older data…
        </div>
      )}
      {/* "Set as Backtest Range" button (top-right) */}
      {onTimeRangeSelect && candles.length > 0 && (
        <button
          onClick={() => {
            const range = chartRef.current?.timeScale().getVisibleRange();
            if (range) {
              const fmt = (t: Time) => {
                const d = new Date((t as number) * 1000);
                const y = d.getUTCFullYear();
                const m = String(d.getUTCMonth() + 1).padStart(2, '0');
                const day = String(d.getUTCDate()).padStart(2, '0');
                return `${y}${m}${day}`;
              };
              onTimeRangeSelect(fmt(range.from), fmt(range.to));
            }
          }}
          className="absolute top-2 right-2 z-10 text-[10px] bg-accent/20 text-accent border border-accent/30 px-2 py-1 rounded hover:bg-accent/30 transition-colors"
        >
          Set as Backtest Range
        </button>
      )}
      {/* Selection range highlight overlay */}
      <div
        ref={selectionOverlayRef}
        className="absolute top-0 bottom-0 pointer-events-none z-10"
        style={{
          display: 'none',
          background: 'rgba(99,102,241,0.15)',
          borderLeft: '2px solid rgba(99,102,241,0.6)',
          borderRight: '2px solid rgba(99,102,241,0.6)',
        }}
      />
      <div
        ref={containerRef}
        className="w-full rounded-lg overflow-hidden border border-white/5"
        style={{ minHeight: height, cursor: onCandleClick ? 'crosshair' : 'default' }}
      />
    </div>
  );
});

// ── Utility: convert raw OHLCV arrays to Candle objects ───────

/**
 * Convert OHLCV candle arrays from the API response to Candle objects.
 * API returns: [timestamp_ms, open, high, low, close, volume]
 */
export function parseOHLCVCandles(raw: number[][]): Candle[] {
  return raw.map(([ts, o, h, l, c, v]) => ({
    time: Math.floor(ts / 1000),  // ms → seconds
    open: o,
    high: h,
    low: l,
    close: c,
    volume: v,
  }));
}
