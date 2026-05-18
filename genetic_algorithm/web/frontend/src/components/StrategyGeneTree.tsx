import type { StrategyGene } from '../types';
import { clsx } from 'clsx';
import { Eye, EyeOff } from 'lucide-react';
import { useState } from 'react';

export interface IndicatorPreview {
  type: string;
  parameters: Record<string, unknown>;
}

interface StrategyGeneTreeProps {
  gene: StrategyGene;
  /** Called when user clicks an indicator node to preview it on the chart */
  onIndicatorClick?: (ind: IndicatorPreview) => void;
  /** Currently highlighted indicator type (shown with active state) */
  activeIndicator?: string | null;
}

export function StrategyGeneTree({ gene, onIndicatorClick, activeIndicator }: StrategyGeneTreeProps) {
  return (
    <div className="card space-y-4">
      <h3 className="text-sm font-medium text-gray-300">Strategy Gene</h3>

      {/* Parameters */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <ParamPill label="Timeframe" value={gene.timeframe} />
        <ParamPill label="Stoploss" value={`${(gene.stoploss * 100).toFixed(1)}%`} negative />
        <ParamPill label="Max Trades" value={gene.max_open_trades.toString()} />
        <ParamPill label="Trailing" value={gene.trailing_stop ? 'Yes' : 'No'} />
      </div>

      {/* Indicators */}
      <Section title={`Indicators (${gene.indicators.length})`}>
        {onIndicatorClick && (
          <p className="text-[10px] text-gray-500 mb-1.5">
            Click an indicator to preview it on the chart ↓
          </p>
        )}
        <div className="space-y-1.5">
          {gene.indicators.map((ind, i) => {
            const isActive = activeIndicator === ind.type;
            return (
              <div
                key={i}
                onClick={() => onIndicatorClick?.({ type: ind.type, parameters: ind.parameters })}
                className={clsx(
                  'flex items-center gap-2 text-xs px-2 py-1.5 rounded-lg transition-colors',
                  onIndicatorClick
                    ? 'cursor-pointer select-none'
                    : '',
                  isActive
                    ? 'bg-accent/20 border border-accent/40'
                    : 'bg-surface-2 hover:bg-surface-2/80',
                )}
              >
                <span className={clsx('font-mono font-medium', isActive ? 'text-accent' : 'text-accent')}>
                  {ind.type}
                </span>
                <span className="text-gray-500">
                  {Object.entries(ind.parameters)
                    .map(([k, v]) => `${k}=${v}`)
                    .join(', ')}
                </span>
                {ind.timeframe && (
                  <span className="text-gray-600 text-[10px]">{ind.timeframe}</span>
                )}
                {onIndicatorClick && (
                  <span className="ml-auto flex-shrink-0">
                    {isActive
                      ? <EyeOff className="w-3 h-3 text-accent" />
                      : <Eye className="w-3 h-3 text-gray-600" />}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </Section>

      {/* Entry Conditions */}
      <Section title={`Entry Conditions (${gene.entry_conditions.length})`}>
        <ConditionList conditions={gene.entry_conditions} color="text-profit" />
      </Section>

      {/* Exit Conditions */}
      <Section title={`Exit Conditions (${gene.exit_conditions.length})`}>
        <ConditionList conditions={gene.exit_conditions} color="text-loss" />
      </Section>

      {/* ROI Table */}
      {Object.keys(gene.minimal_roi).length > 0 && (
        <Section title="Minimal ROI">
          <div className="flex flex-wrap gap-2">
            {Object.entries(gene.minimal_roi)
              .sort(([a], [b]) => parseInt(a) - parseInt(b))
              .map(([mins, pct]) => (
                <span key={mins} className="text-xs bg-surface-2 px-2 py-1 rounded font-mono">
                  {mins}m → {(pct * 100).toFixed(1)}%
                </span>
              ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1.5">{title}</h4>
      {children}
    </div>
  );
}

function ParamPill({
  label,
  value,
  negative,
}: {
  label: string;
  value: string;
  negative?: boolean;
}) {
  return (
    <div className="bg-surface-2 rounded-lg px-3 py-2">
      <div className="text-[10px] text-gray-500 uppercase">{label}</div>
      <div className={clsx('text-sm font-mono font-medium', negative ? 'text-loss' : 'text-gray-200')}>
        {value}
      </div>
    </div>
  );
}

function ConditionList({
  conditions,
  color,
}: {
  conditions: { indicator: string; operator: string; threshold: unknown; logic: string }[];
  color: string;
}) {
  return (
    <div className="space-y-1">
      {conditions.map((c, i) => (
        <div key={i} className="flex items-center gap-1.5 text-xs">
          {i > 0 && (
            <span className="text-gray-600 font-mono text-[10px] w-6">{c.logic}</span>
          )}
          <span className={clsx('font-mono font-medium', color)}>{c.indicator}</span>
          <span className="text-gray-400">{c.operator}</span>
          <span className="text-gray-300 font-mono">{String(c.threshold ?? '')}</span>
        </div>
      ))}
    </div>
  );
}
