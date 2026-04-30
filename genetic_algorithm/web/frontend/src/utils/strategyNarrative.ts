/**
 * Strategy Narrative Utility
 *
 * Generates a human-readable performance summary from strategy metrics
 * and gene structure. Entirely rule-based — no LLM required.
 */

import type { StrategyMetrics, StrategyGene } from '../types';

export type VerdictLevel = 'excellent' | 'good' | 'average' | 'weak' | 'risky';

export interface StrategyNarrative {
  verdict: VerdictLevel;
  verdictLabel: string;
  summary: string;
  strengths: string[];
  warnings: string[];
  details: string[];
}

// ── Helpers ────────────────────────────────────────────────────

function pct(v: number): string {
  return `${v > 0 ? '+' : ''}${v.toFixed(1)}%`;
}

function getNum(metrics: StrategyMetrics, ...keys: string[]): number | null {
  for (const k of keys) {
    const v = metrics[k];
    if (typeof v === 'number' && isFinite(v)) return v;
  }
  return null;
}

// ── Main narrative generator ───────────────────────────────────

export function generateNarrative(
  metrics: StrategyMetrics,
  gene: StrategyGene | null | undefined,
): StrategyNarrative {
  const profit = getNum(metrics, 'profit', 'total_profit', 'profit_total');
  const sharpe = getNum(metrics, 'sharpe_ratio');
  const winRate = getNum(metrics, 'win_rate');
  const numTrades = getNum(metrics, 'num_trades');
  const maxDD = getNum(metrics, 'max_drawdown');
  const profitFactor = getNum(metrics, 'profit_factor');
  const sortino = getNum(metrics, 'sortino_ratio');
  const holdoutDeg = getNum(metrics, 'holdout_degradation', 'train_val_gap');
  const mcRobustness = getNum(metrics, 'mc_robustness');

  const strengths: string[] = [];
  const warnings: string[] = [];
  const details: string[] = [];

  // ── Profit ──
  let profitScore = 0;
  if (profit !== null) {
    if (profit > 30) { strengths.push(`Strong profit: ${pct(profit)}`); profitScore = 3; }
    else if (profit > 10) { strengths.push(`Solid profit: ${pct(profit)}`); profitScore = 2; }
    else if (profit > 0) { details.push(`Modest profit: ${pct(profit)}`); profitScore = 1; }
    else { warnings.push(`Negative profit: ${pct(profit)}`); profitScore = -1; }
  }

  // ── Sharpe ──
  let sharpeScore = 0;
  if (sharpe !== null) {
    if (sharpe > 2) { strengths.push(`Excellent Sharpe ratio: ${sharpe.toFixed(2)}`); sharpeScore = 3; }
    else if (sharpe > 1) { strengths.push(`Good Sharpe ratio: ${sharpe.toFixed(2)}`); sharpeScore = 2; }
    else if (sharpe > 0.5) { details.push(`Average Sharpe ratio: ${sharpe.toFixed(2)}`); sharpeScore = 1; }
    else { warnings.push(`Low Sharpe ratio: ${sharpe.toFixed(2)} — poor risk-adjusted returns`); sharpeScore = -1; }
  }

  // ── Win rate ──
  let winScore = 0;
  if (winRate !== null) {
    const wp = winRate * 100;
    if (wp > 65) { strengths.push(`High win rate: ${wp.toFixed(1)}%`); winScore = 3; }
    else if (wp > 55) { details.push(`Good win rate: ${wp.toFixed(1)}%`); winScore = 2; }
    else if (wp > 45) { details.push(`Average win rate: ${wp.toFixed(1)}%`); winScore = 1; }
    else if (wp > 35) { warnings.push(`Below-average win rate: ${wp.toFixed(1)}%`); winScore = -1; }
    else { warnings.push(`Low win rate: ${wp.toFixed(1)}% — needs strong reward:risk to compensate`); winScore = -2; }
  }

  // ── Trade count ──
  let tradeScore = 0;
  if (numTrades !== null) {
    if (numTrades < 10) { warnings.push(`Very few trades: ${numTrades} — results not statistically significant`); tradeScore = -2; }
    else if (numTrades < 30) { warnings.push(`Low trade count: ${numTrades} — treat with caution`); tradeScore = -1; }
    else if (numTrades < 100) { details.push(`Moderate trade frequency: ${numTrades} trades`); tradeScore = 1; }
    else { details.push(`Active strategy: ${numTrades} trades`); tradeScore = 2; }
  }

  // ── Drawdown ──
  let ddScore = 0;
  if (maxDD !== null) {
    const ddPct = maxDD * 100;
    if (ddPct < 5) { strengths.push(`Very low drawdown: ${pct(-ddPct)}`); ddScore = 3; }
    else if (ddPct < 10) { details.push(`Low drawdown: ${pct(-ddPct)}`); ddScore = 2; }
    else if (ddPct < 20) { details.push(`Moderate drawdown: ${pct(-ddPct)}`); ddScore = 1; }
    else if (ddPct < 30) { warnings.push(`High drawdown: ${pct(-ddPct)}`); ddScore = -1; }
    else { warnings.push(`Very high drawdown: ${pct(-ddPct)} — significant capital risk`); ddScore = -2; }
  }

  // ── Profit factor ──
  if (profitFactor !== null) {
    if (profitFactor > 2) strengths.push(`Strong profit factor: ${profitFactor.toFixed(2)}`);
    else if (profitFactor > 1.3) details.push(`Decent profit factor: ${profitFactor.toFixed(2)}`);
    else if (profitFactor < 1) warnings.push(`Profit factor below 1.0: ${profitFactor.toFixed(2)} — losing trades outweigh winners`);
  }

  // ── Sortino ──
  if (sortino !== null && sortino > 2) {
    strengths.push(`Strong Sortino ratio: ${sortino.toFixed(2)} — low downside volatility`);
  }

  // ── Robustness ──
  if (holdoutDeg !== null) {
    const degradation = Math.abs(holdoutDeg) * 100;
    if (degradation > 40) warnings.push(`Overfitting risk: holdout degradation ${degradation.toFixed(0)}%`);
    else if (degradation > 20) warnings.push(`Mild overfitting: holdout degradation ${degradation.toFixed(0)}%`);
    else details.push(`Holds up on unseen data: degradation ${degradation.toFixed(0)}%`);
  }

  if (mcRobustness !== null) {
    const robPct = mcRobustness * 100;
    if (robPct > 75) strengths.push(`Monte Carlo robust: ${robPct.toFixed(0)}% of simulations profitable`);
    else if (robPct < 40) warnings.push(`Monte Carlo fragile: only ${robPct.toFixed(0)}% of simulations profitable`);
  }

  // ── Gene-based description ──
  if (gene) {
    const indicatorTypes = gene.indicators?.map((i) => i.type) ?? [];
    const uniqueInds = [...new Set(indicatorTypes)];
    if (uniqueInds.length > 0) {
      details.push(`Uses ${uniqueInds.length} indicator${uniqueInds.length > 1 ? 's' : ''}: ${uniqueInds.slice(0, 4).join(', ')}${uniqueInds.length > 4 ? '…' : ''}`);
    }
    if (gene.timeframe) details.push(`Timeframe: ${gene.timeframe}`);
    if (gene.stoploss) details.push(`Stoploss: ${(gene.stoploss * 100).toFixed(1)}%`);
    if (gene.trailing_stop) details.push('Uses trailing stop');
    if (gene.can_short) details.push('Can enter short positions');
    const entryCount = gene.entry_conditions?.length ?? 0;
    const exitCount = gene.exit_conditions?.length ?? 0;
    if (entryCount > 0 || exitCount > 0) {
      details.push(`${entryCount} entry condition${entryCount !== 1 ? 's' : ''}, ${exitCount} exit condition${exitCount !== 1 ? 's' : ''}`);
    }
  }

  // ── Overall verdict ──
  const totalScore = profitScore + sharpeScore * 1.5 + winScore * 0.5 + tradeScore * 0.5 + ddScore;
  const criticalWarnings = warnings.filter(
    (w) => w.includes('Overfitting') || w.includes('Very few') || w.includes('Negative'),
  ).length;

  let verdict: VerdictLevel;
  let verdictLabel: string;

  if (criticalWarnings >= 2 || (criticalWarnings >= 1 && totalScore < 2)) {
    verdict = 'risky';
    verdictLabel = 'Risky';
  } else if (totalScore >= 9) {
    verdict = 'excellent';
    verdictLabel = 'Excellent';
  } else if (totalScore >= 5) {
    verdict = 'good';
    verdictLabel = 'Good';
  } else if (totalScore >= 1) {
    verdict = 'average';
    verdictLabel = 'Average';
  } else {
    verdict = 'weak';
    verdictLabel = 'Weak';
  }

  // ── Summary sentence ──
  const tradeSummary = numTrades !== null ? `${numTrades} trades` : 'unknown trade count';
  const profitSummary =
    profit !== null ? `${pct(profit)} total profit` : 'unknown profit';
  const sharpeSummary = sharpe !== null ? `, Sharpe ${sharpe.toFixed(2)}` : '';
  const timeframeSummary = gene?.timeframe ? ` on ${gene.timeframe} candles` : '';

  const summary = `${verdictLabel} strategy${timeframeSummary}. ${profitSummary} across ${tradeSummary}${sharpeSummary}.`;

  return { verdict, verdictLabel, summary, strengths, warnings, details };
}
