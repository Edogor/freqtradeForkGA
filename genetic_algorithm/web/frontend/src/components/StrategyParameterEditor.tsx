/**
 * StrategyParameterEditor — inline gene parameter editor.
 *
 * Renders each gene field as an appropriate input control.
 * Emits an `overrides` object containing only the changed values
 * (suitable for the POST /api/strategies/{id}/test endpoint).
 */

import { useState, useCallback } from 'react';
import { RotateCcw } from 'lucide-react';
import type { StrategyGene, IndicatorModel, ConditionModel } from '../types';

interface Props {
  gene: StrategyGene;
  onChange: (overrides: Record<string, unknown>) => void;
}

const OPERATORS = ['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'];
const TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d'];

// ── Utilities ──────────────────────────────────────────────────

function NumInput({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[10px] text-gray-500 uppercase">{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step ?? 1}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-full bg-surface-2 border border-white/10 rounded px-2 py-1 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
      />
    </div>
  );
}

function SelectInput({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[10px] text-gray-500 uppercase">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-surface-2 border border-white/10 rounded px-2 py-1 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

function Toggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => onChange(!value)}
        className={[
          'w-8 h-4 rounded-full transition-colors relative',
          value ? 'bg-accent' : 'bg-surface-2 border border-white/10',
        ].join(' ')}
      >
        <span
          className={[
            'absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform',
            value ? 'translate-x-4' : 'translate-x-0.5',
          ].join(' ')}
        />
      </button>
      <span className="text-xs text-gray-400">{label}</span>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────

export function StrategyParameterEditor({ gene, onChange }: Props) {
  const [overrides, setOverrides] = useState<Record<string, unknown>>({});

  const update = useCallback(
    (path: string, value: unknown) => {
      setOverrides((prev) => {
        const next = { ...prev, [path]: value };
        onChange(buildOverridesObject(next, gene));
        return next;
      });
    },
    [onChange, gene],
  );

  const reset = () => {
    setOverrides({});
    onChange({});
  };

  // Get current effective value (override takes priority)
  const v = <T,>(key: string, fallback: T): T =>
    (overrides[key] as T) ?? fallback;

  return (
    <div className="space-y-4">
      {/* Reset button */}
      {Object.keys(overrides).length > 0 && (
        <div className="flex justify-end">
          <button
            onClick={reset}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
          >
            <RotateCcw className="w-3 h-3" /> Reset all
          </button>
        </div>
      )}

      {/* ── Core params ── */}
      <section>
        <h4 className="text-[11px] text-gray-500 uppercase tracking-wider mb-2">Core</h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <SelectInput
            label="Timeframe"
            value={v('timeframe', gene.timeframe || '5m')}
            options={TIMEFRAMES}
            onChange={(val) => update('timeframe', val)}
          />
          <NumInput
            label="Stoploss (%)"
            value={parseFloat(((v('stoploss', gene.stoploss || -0.1)) * 100).toFixed(2))}
            min={-50}
            max={-0.1}
            step={0.1}
            onChange={(val) => update('stoploss', val / 100)}
          />
          <NumInput
            label="Max Open Trades"
            value={v('max_open_trades', gene.max_open_trades || 3)}
            min={1}
            max={10}
            onChange={(val) => update('max_open_trades', Math.round(val))}
          />
          <Toggle
            label="Trailing Stop"
            value={v('trailing_stop', gene.trailing_stop || false)}
            onChange={(val) => update('trailing_stop', val)}
          />
        </div>
      </section>

      {/* ── Indicator parameters ── */}
      {gene.indicators && gene.indicators.length > 0 && (
        <section>
          <h4 className="text-[11px] text-gray-500 uppercase tracking-wider mb-2">
            Indicator Parameters
          </h4>
          <div className="space-y-3">
            {gene.indicators.map((ind, idx) => (
              <IndicatorEditor
                key={`${ind.type}-${idx}`}
                indicator={ind}
                index={idx}
                overrides={overrides}
                onUpdate={update}
              />
            ))}
          </div>
        </section>
      )}

      {/* ── Entry conditions ── */}
      {gene.entry_conditions && gene.entry_conditions.length > 0 && (
        <section>
          <h4 className="text-[11px] text-gray-500 uppercase tracking-wider mb-2">
            Entry Conditions
          </h4>
          <div className="space-y-2">
            {gene.entry_conditions.map((cond, idx) => (
              <ConditionEditor
                key={`entry-${idx}`}
                condition={cond}
                index={idx}
                prefix="entry_conditions"
                overrides={overrides}
                onUpdate={update}
              />
            ))}
          </div>
        </section>
      )}

      {/* ── Exit conditions ── */}
      {gene.exit_conditions && gene.exit_conditions.length > 0 && (
        <section>
          <h4 className="text-[11px] text-gray-500 uppercase tracking-wider mb-2">
            Exit Conditions
          </h4>
          <div className="space-y-2">
            {gene.exit_conditions.map((cond, idx) => (
              <ConditionEditor
                key={`exit-${idx}`}
                condition={cond}
                index={idx}
                prefix="exit_conditions"
                overrides={overrides}
                onUpdate={update}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

// ── Sub-editors ────────────────────────────────────────────────

function IndicatorEditor({
  indicator,
  index,
  overrides,
  onUpdate,
}: {
  indicator: IndicatorModel;
  index: number;
  overrides: Record<string, unknown>;
  onUpdate: (path: string, val: unknown) => void;
}) {
  const paramKeys = Object.keys(indicator.parameters || {});

  if (paramKeys.length === 0) return null;

  return (
    <div className="bg-surface-2/50 rounded-lg px-3 py-2 space-y-2">
      <span className="text-xs font-medium text-gray-300 font-mono">
        {indicator.type}
        {indicator.timeframe ? <span className="text-gray-500 ml-1 text-[10px]">({indicator.timeframe})</span> : null}
      </span>
      <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
        {paramKeys.map((param) => {
          const path = `__ind_${index}_${param}`;
          const origVal = indicator.parameters[param];
          const curVal = (overrides[path] as number) ?? (typeof origVal === 'number' ? origVal : 0);
          return (
            <NumInput
              key={param}
              label={param}
              value={curVal}
              min={1}
              step={1}
              onChange={(val) => onUpdate(path, val)}
            />
          );
        })}
      </div>
    </div>
  );
}

function ConditionEditor({
  condition,
  index,
  prefix,
  overrides,
  onUpdate,
}: {
  condition: ConditionModel;
  index: number;
  prefix: string;
  overrides: Record<string, unknown>;
  onUpdate: (path: string, val: unknown) => void;
}) {
  const opPath = `__${prefix}_${index}_operator`;
  const thPath = `__${prefix}_${index}_threshold`;

  const curOp = (overrides[opPath] as string) ?? condition.operator;
  const curTh = (overrides[thPath] as number) ?? (typeof condition.threshold === 'number' ? condition.threshold : 0);

  return (
    <div className="flex flex-wrap items-end gap-2 bg-surface-2/50 rounded-lg px-3 py-2">
      <span className="text-[10px] text-gray-500 w-full">{condition.indicator}</span>
      <SelectInput
        label="Operator"
        value={curOp}
        options={OPERATORS}
        onChange={(val) => onUpdate(opPath, val)}
      />
      <NumInput
        label="Threshold"
        value={curTh}
        step={0.1}
        onChange={(val) => onUpdate(thPath, val)}
      />
    </div>
  );
}

// ── Build overrides object ─────────────────────────────────────
// Converts flat "__ind_0_period" keys back to nested gene structure

function buildOverridesObject(
  flatOverrides: Record<string, unknown>,
  gene: StrategyGene,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  for (const [key, val] of Object.entries(flatOverrides)) {
    if (key.startsWith('__ind_')) {
      // Indicator param: __ind_{index}_{paramKey}
      const parts = key.slice(6).split('_');
      const idx = parseInt(parts[0], 10);
      const paramKey = parts.slice(1).join('_');

      if (!result.indicators) {
        // Deep-clone the original indicators array
        result.indicators = gene.indicators?.map((ind) => ({
          ...ind,
          parameters: { ...ind.parameters },
        })) ?? [];
      }
      const inds = result.indicators as IndicatorModel[];
      if (inds[idx]) {
        (inds[idx].parameters as Record<string, unknown>)[paramKey] = val;
      }
    } else if (key.startsWith('__entry_conditions_')) {
      const parts = key.replace('__entry_conditions_', '').split('_');
      const idx = parseInt(parts[0], 10);
      const field = parts.slice(1).join('_');
      if (!result.entry_conditions) {
        result.entry_conditions = gene.entry_conditions?.map((c) => ({ ...c })) ?? [];
      }
      const conds = result.entry_conditions as ConditionModel[];
      if (conds[idx]) (conds[idx] as Record<string, unknown>)[field] = val;
    } else if (key.startsWith('__exit_conditions_')) {
      const parts = key.replace('__exit_conditions_', '').split('_');
      const idx = parseInt(parts[0], 10);
      const field = parts.slice(1).join('_');
      if (!result.exit_conditions) {
        result.exit_conditions = gene.exit_conditions?.map((c) => ({ ...c })) ?? [];
      }
      const conds = result.exit_conditions as ConditionModel[];
      if (conds[idx]) (conds[idx] as Record<string, unknown>)[field] = val;
    } else {
      // Top-level gene param
      result[key] = val;
    }
  }

  return result;
}
