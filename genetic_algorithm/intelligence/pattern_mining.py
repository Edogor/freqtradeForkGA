"""
Strategy DNA Pattern Mining

Discovers which gene patterns (indicator combos, condition parameters, risk settings)
are associated with high-fitness strategies. Produces a ranked "recipe book" of
building blocks for strategy construction.

Key analyses:
  - Indicator co-occurrence in top vs bottom strategies
  - Condition operator/threshold distributions by indicator
  - ROI/stoploss parameter heatmaps
  - Cross-indicator synergy scores (lift)

Usage:
    from genetic_algorithm.intelligence.pattern_mining import PatternMiner
    miner = PatternMiner(corpus_df)
    miner.mine()
    print(miner.report())
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PatternMiner:
    """Discovers gene patterns that predict strategy success."""

    def __init__(self, df: pd.DataFrame, top_pct: float = 0.2, bottom_pct: float = 0.2):
        """
        Args:
            df: Strategy corpus DataFrame.
            top_pct: Fraction of strategies considered "top" by fitness.
            bottom_pct: Fraction considered "bottom".
        """
        self.df = df.dropna(subset=["fitness"]).copy()
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct

        # Results
        self.indicator_enrichment: Optional[pd.DataFrame] = None
        self.indicator_synergies: Optional[pd.DataFrame] = None
        self.operator_patterns: Optional[Dict[str, Any]] = None
        self.risk_param_analysis: Optional[Dict[str, Any]] = None

    def mine(self) -> Dict[str, Any]:
        """Run all pattern mining analyses.

        Returns:
            Summary dict with key findings.
        """
        self._mine_indicator_enrichment()
        self._mine_indicator_synergies()
        self._mine_operator_patterns()
        self._mine_risk_parameters()

        summary = {
            "n_strategies": len(self.df),
            "n_top": int(len(self.df) * self.top_pct),
            "n_bottom": int(len(self.df) * self.bottom_pct),
            "top_indicators": (
                self.indicator_enrichment.head(5)["indicator"].tolist()
                if self.indicator_enrichment is not None else []
            ),
            "top_synergies": (
                self.indicator_synergies.head(3)[["ind_a", "ind_b", "lift"]].to_dict("records")
                if self.indicator_synergies is not None else []
            ),
        }

        logger.info(f"[PATTERN] Mined patterns from {len(self.df)} strategies")
        return summary

    def _mine_indicator_enrichment(self):
        """Find indicators enriched in top strategies vs bottom."""
        from genetic_algorithm.evaluation.surrogate import _INDICATOR_TYPES

        n = len(self.df)
        top_n = max(1, int(n * self.top_pct))
        bottom_n = max(1, int(n * self.bottom_pct))

        top = self.df.nlargest(top_n, "fitness")
        bottom = self.df.nsmallest(bottom_n, "fitness")

        records = []
        for ind_type in _INDICATOR_TYPES:
            col = f"ind_{ind_type}"
            if col not in self.df.columns:
                continue

            top_rate = top[col].mean()
            bottom_rate = bottom[col].mean()
            global_rate = self.df[col].mean()

            # Enrichment ratio: how much more common is this indicator in top vs bottom?
            enrichment = (top_rate + 0.01) / (bottom_rate + 0.01)
            # Lift: how much more common in top vs global?
            lift = (top_rate + 0.01) / (global_rate + 0.01)

            # Mean fitness when indicator is present vs absent
            present = self.df[self.df[col] > 0.5]
            absent = self.df[self.df[col] < 0.5]
            fitness_present = present["fitness"].mean() if len(present) > 5 else None
            fitness_absent = absent["fitness"].mean() if len(absent) > 5 else None

            records.append({
                "indicator": ind_type,
                "top_rate": top_rate,
                "bottom_rate": bottom_rate,
                "global_rate": global_rate,
                "enrichment": enrichment,
                "lift": lift,
                "fitness_when_present": fitness_present,
                "fitness_when_absent": fitness_absent,
                "n_present": len(present),
            })

        self.indicator_enrichment = pd.DataFrame(records).sort_values(
            "enrichment", ascending=False
        )

    def _mine_indicator_synergies(self):
        """Find indicator pairs that co-occur in top strategies more than expected."""
        from genetic_algorithm.evaluation.surrogate import _INDICATOR_TYPES

        n = len(self.df)
        top_n = max(1, int(n * self.top_pct))
        top = self.df.nlargest(top_n, "fitness")

        ind_cols = [f"ind_{t}" for t in _INDICATOR_TYPES if f"ind_{t}" in self.df.columns]

        synergies = []
        for i, col_a in enumerate(ind_cols):
            for col_b in ind_cols[i + 1:]:
                # Co-occurrence rate in top vs global
                top_cooccur = ((top[col_a] > 0.5) & (top[col_b] > 0.5)).mean()
                global_cooccur = ((self.df[col_a] > 0.5) & (self.df[col_b] > 0.5)).mean()
                global_a = (self.df[col_a] > 0.5).mean()
                global_b = (self.df[col_b] > 0.5).mean()

                # Expected co-occurrence under independence
                expected = global_a * global_b
                lift = (top_cooccur + 0.001) / (expected + 0.001) if expected > 0 else 0

                # Only report meaningful synergies
                if top_cooccur > 0.05 and lift > 1.2:
                    # Mean fitness when both present
                    both_present = self.df[
                        (self.df[col_a] > 0.5) & (self.df[col_b] > 0.5)
                    ]
                    fitness_both = both_present["fitness"].mean() if len(both_present) > 3 else None

                    synergies.append({
                        "ind_a": col_a.replace("ind_", ""),
                        "ind_b": col_b.replace("ind_", ""),
                        "top_cooccurrence": top_cooccur,
                        "global_cooccurrence": global_cooccur,
                        "lift": lift,
                        "fitness_when_both": fitness_both,
                        "n_both": len(both_present),
                    })

        self.indicator_synergies = pd.DataFrame(synergies).sort_values(
            "lift", ascending=False
        ) if synergies else pd.DataFrame()

    def _mine_operator_patterns(self):
        """Analyze operator usage patterns in top vs bottom strategies."""
        from genetic_algorithm.evaluation.surrogate import _OPERATOR_TYPES

        n = len(self.df)
        top_n = max(1, int(n * self.top_pct))
        bottom_n = max(1, int(n * self.bottom_pct))

        top = self.df.nlargest(top_n, "fitness")
        bottom = self.df.nsmallest(bottom_n, "fitness")

        entry_ops = {}
        exit_ops = {}

        for op in _OPERATOR_TYPES:
            entry_col = f"entry_op_{op}"
            exit_col = f"exit_op_{op}"

            if entry_col in self.df.columns:
                entry_ops[op] = {
                    "top_mean": float(top[entry_col].mean()),
                    "bottom_mean": float(bottom[entry_col].mean()),
                    "global_mean": float(self.df[entry_col].mean()),
                }

            if exit_col in self.df.columns:
                exit_ops[op] = {
                    "top_mean": float(top[exit_col].mean()),
                    "bottom_mean": float(bottom[exit_col].mean()),
                    "global_mean": float(self.df[exit_col].mean()),
                }

        self.operator_patterns = {
            "entry_operators": entry_ops,
            "exit_operators": exit_ops,
        }

    def _mine_risk_parameters(self):
        """Analyze risk parameter distributions in top strategies."""
        n = len(self.df)
        top_n = max(1, int(n * self.top_pct))
        bottom_n = max(1, int(n * self.bottom_pct))

        top = self.df.nlargest(top_n, "fitness")
        bottom = self.df.nsmallest(bottom_n, "fitness")

        params = {}
        for col in ["stoploss", "roi_max", "roi_min", "max_open_trades",
                     "trailing_stop", "n_indicators", "n_entry_conds", "n_exit_conds"]:
            if col not in self.df.columns:
                continue
            vals_top = top[col].dropna()
            vals_bottom = bottom[col].dropna()
            vals_all = self.df[col].dropna()

            params[col] = {
                "top_mean": float(vals_top.mean()) if len(vals_top) > 0 else None,
                "top_median": float(vals_top.median()) if len(vals_top) > 0 else None,
                "bottom_mean": float(vals_bottom.mean()) if len(vals_bottom) > 0 else None,
                "bottom_median": float(vals_bottom.median()) if len(vals_bottom) > 0 else None,
                "global_mean": float(vals_all.mean()) if len(vals_all) > 0 else None,
            }

        self.risk_param_analysis = params

    def report(self) -> str:
        """Generate a human-readable pattern mining report."""
        lines = ["=" * 70, "  STRATEGY DNA PATTERN MINING", "=" * 70, ""]

        # Indicator Enrichment
        if self.indicator_enrichment is not None and len(self.indicator_enrichment) > 0:
            lines.append("  INDICATOR ENRICHMENT (top vs bottom strategies)")
            lines.append(f"  {'Indicator':<20s} {'Top%':>6s} {'Bot%':>6s} {'Enrich':>8s} {'Fitness+':>9s} {'Fitness-':>9s}")
            lines.append("  " + "-" * 60)
            for _, row in self.indicator_enrichment.head(15).iterrows():
                fp = row.get("fitness_when_present")
                fa = row.get("fitness_when_absent")
                fp_s = f"{fp:.3f}" if fp is not None else "N/A"
                fa_s = f"{fa:.3f}" if fa is not None else "N/A"
                lines.append(
                    f"  {row['indicator']:<20s} {row['top_rate']:>5.1%} {row['bottom_rate']:>5.1%} "
                    f"{row['enrichment']:>7.2f}x {fp_s:>9s} {fa_s:>9s}"
                )
            lines.append("")

        # Indicator Synergies
        if self.indicator_synergies is not None and len(self.indicator_synergies) > 0:
            lines.append("  INDICATOR SYNERGIES (co-occurrence lift in top strategies)")
            lines.append(f"  {'Pair':<30s} {'Lift':>7s} {'Top Co%':>8s} {'Fitness':>9s} {'N':>5s}")
            lines.append("  " + "-" * 60)
            for _, row in self.indicator_synergies.head(10).iterrows():
                pair = f"{row['ind_a']} + {row['ind_b']}"
                fb = row.get("fitness_when_both")
                fb_s = f"{fb:.3f}" if fb is not None else "N/A"
                lines.append(
                    f"  {pair:<30s} {row['lift']:>6.2f}x {row['top_cooccurrence']:>7.1%} "
                    f"{fb_s:>9s} {row['n_both']:>5d}"
                )
            lines.append("")

        # Risk Parameters
        if self.risk_param_analysis:
            lines.append("  RISK PARAMETER PATTERNS")
            lines.append(f"  {'Parameter':<20s} {'Top Mean':>10s} {'Bot Mean':>10s} {'Global':>10s}")
            lines.append("  " + "-" * 50)
            for param, vals in self.risk_param_analysis.items():
                tm = vals.get("top_mean")
                bm = vals.get("bottom_mean")
                gm = vals.get("global_mean")
                tm_s = f"{tm:>10.4f}" if tm is not None else f"{'N/A':>10s}"
                bm_s = f"{bm:>10.4f}" if bm is not None else f"{'N/A':>10s}"
                gm_s = f"{gm:>10.4f}" if gm is not None else f"{'N/A':>10s}"
                lines.append(f"  {param:<20s} {tm_s} {bm_s} {gm_s}")
            lines.append("")

        # Operator Patterns
        if self.operator_patterns:
            lines.append("  ENTRY OPERATOR USAGE")
            entry_ops = self.operator_patterns.get("entry_operators", {})
            for op, vals in sorted(entry_ops.items(), key=lambda x: -(x[1].get("top_mean", 0))):
                tm = vals.get("top_mean", 0)
                bm = vals.get("bottom_mean", 0)
                if tm > 0.1 or bm > 0.1:
                    trend = "↑" if tm > bm * 1.2 else "↓" if tm < bm * 0.8 else "="
                    lines.append(f"    {op:<20s} top={tm:.2f}  bot={bm:.2f}  {trend}")
            lines.append("")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, models_dir=None) -> None:
        """Persist mining results so SISIntegrator can auto-load live weights."""
        import pickle
        from pathlib import Path
        out_dir = Path(models_dir) if models_dir else Path("genetic_algorithm/ml/models")
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "indicator_enrichment": self.indicator_enrichment,
            "indicator_synergies": self.indicator_synergies,
            "operator_patterns": self.operator_patterns,
            "risk_param_analysis": self.risk_param_analysis,
        }
        with open(out_dir / "pattern_miner.pkl", "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"[PATTERNS] Saved mining results to {out_dir}/pattern_miner.pkl")

    @classmethod
    def load(cls, models_dir=None):
        """Load a previously saved PatternMiner. Returns None if not found."""
        import pickle
        from pathlib import Path
        pkl_path = Path(models_dir) / "pattern_miner.pkl" if models_dir else \
            Path("genetic_algorithm/ml/models/pattern_miner.pkl")
        if not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.df = None
        obj.top_pct = 0.2
        obj.bottom_pct = 0.2
        obj.indicator_enrichment = data.get("indicator_enrichment")
        obj.indicator_synergies = data.get("indicator_synergies")
        obj.operator_patterns = data.get("operator_patterns")
        obj.risk_param_analysis = data.get("risk_param_analysis")
        return obj
