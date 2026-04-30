/**
 * StrategyNarrativeCard — human-readable performance summary.
 *
 * Generates a narrative from strategy metrics and gene structure
 * without requiring an LLM — purely rule-based.
 */

import { CheckCircle, AlertTriangle, TrendingUp, Info, Star } from 'lucide-react';
import { generateNarrative, type VerdictLevel } from '../utils/strategyNarrative';
import type { StrategyMetrics, StrategyGene } from '../types';

interface Props {
  metrics: StrategyMetrics;
  gene?: StrategyGene | null;
  collapsed?: boolean;
}

const VERDICT_STYLES: Record<VerdictLevel, { bg: string; border: string; badge: string; icon: React.ElementType }> = {
  excellent: {
    bg: 'bg-profit/5',
    border: 'border-profit/20',
    badge: 'bg-profit/20 text-profit',
    icon: Star,
  },
  good: {
    bg: 'bg-accent/5',
    border: 'border-accent/20',
    badge: 'bg-accent/20 text-accent',
    icon: TrendingUp,
  },
  average: {
    bg: 'bg-white/[0.02]',
    border: 'border-white/10',
    badge: 'bg-gray-500/20 text-gray-300',
    icon: Info,
  },
  weak: {
    bg: 'bg-yellow-500/5',
    border: 'border-yellow-500/20',
    badge: 'bg-yellow-500/20 text-yellow-400',
    icon: AlertTriangle,
  },
  risky: {
    bg: 'bg-loss/5',
    border: 'border-loss/20',
    badge: 'bg-loss/20 text-loss',
    icon: AlertTriangle,
  },
};

export function StrategyNarrativeCard({ metrics, gene, collapsed = false }: Props) {
  const narrative = generateNarrative(metrics, gene);
  const style = VERDICT_STYLES[narrative.verdict];
  const VerdictIcon = style.icon;

  return (
    <div className={`rounded-xl border p-4 space-y-3 ${style.bg} ${style.border}`}>
      {/* Header */}
      <div className="flex items-center gap-3">
        <VerdictIcon className="w-4 h-4 flex-shrink-0 opacity-80" />
        <span className="text-sm font-medium text-gray-200 flex-1">{narrative.summary}</span>
        <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${style.badge}`}>
          {narrative.verdictLabel}
        </span>
      </div>

      {!collapsed && (
        <>
          {/* Strengths */}
          {narrative.strengths.length > 0 && (
            <div className="space-y-1">
              {narrative.strengths.map((s, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-profit">
                  <CheckCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <span>{s}</span>
                </div>
              ))}
            </div>
          )}

          {/* Warnings */}
          {narrative.warnings.length > 0 && (
            <div className="space-y-1">
              {narrative.warnings.map((w, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-yellow-400">
                  <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <span>{w}</span>
                </div>
              ))}
            </div>
          )}

          {/* Details */}
          {narrative.details.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {narrative.details.map((d, i) => (
                <span
                  key={i}
                  className="text-[11px] px-2 py-0.5 rounded bg-white/[0.04] text-gray-400"
                >
                  {d}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
