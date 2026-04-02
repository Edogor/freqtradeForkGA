"""
Temporal Performance Analysis & Complementary Strategy Discovery

Analyzes when and where strategies fail by examining per-month and per-pair
performance breakdowns. Finds complementary strategies whose strengths
cover other strategies' weaknesses.

Key outputs:
  - Per-strategy monthly profit heatmaps
  - Weak period identification + regime correlation
  - Complementary strategy pairs/portfolios
  - Ensemble Sharpe improvement estimates

Usage:
    from genetic_algorithm.intelligence.temporal_analysis import TemporalAnalyzer
    analyzer = TemporalAnalyzer(corpus_df)
    report = analyzer.analyze_weaknesses(top_n=20)
    combos = analyzer.find_complementary_pairs(top_n=10)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analyzes temporal performance patterns and finds complementary strategies."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a corpus DataFrame that has monthly_profit columns."""
        self.df = df
        self._monthly_matrix: Optional[pd.DataFrame] = None
        self._weakness_report: Optional[Dict[str, Any]] = None

    def build_monthly_matrix(self) -> pd.DataFrame:
        """Build a strategy × month profit matrix from available monthly data.

        Uses monthly_profit_* summary stats from the corpus. For detailed analysis,
        the full monthly_profits vectors would need to be stored in the corpus.

        Returns:
            DataFrame with strategies as rows and summary stats as columns.
        """
        monthly_cols = [c for c in self.df.columns if c.startswith("monthly_profit_")]
        if not monthly_cols:
            logger.warning("[TEMPORAL] No monthly profit data available in corpus")
            return pd.DataFrame()

        matrix = self.df[["fingerprint", "fitness"] + monthly_cols].copy()
        matrix = matrix.dropna(subset=["monthly_profit_mean"])
        self._monthly_matrix = matrix
        logger.info(f"[TEMPORAL] Monthly matrix: {len(matrix)} strategies with monthly data")
        return matrix

    def analyze_weaknesses(self, top_n: int = 20) -> Dict[str, Any]:
        """Identify performance weaknesses in top strategies.

        Analyzes:
          - Strategies with high variation (monthly_profit_std)
          - Strategies with negative minimum months
          - Pair-level weaknesses (per_pair_profit analysis)

        Args:
            top_n: Number of top strategies to analyze.

        Returns:
            Dict with weakness analysis per strategy.
        """
        # Get top strategies by fitness
        top = self.df.nlargest(top_n, "fitness").copy()

        report: Dict[str, Any] = {
            "n_analyzed": len(top),
            "strategies": [],
        }

        for _, row in top.iterrows():
            entry: Dict[str, Any] = {
                "fingerprint": row.get("fingerprint", ""),
                "fitness": row.get("fitness"),
                "profit": row.get("profit"),
                "run_id": row.get("run_id", ""),
            }

            # Monthly stability analysis
            monthly_std = row.get("monthly_profit_std")
            monthly_min = row.get("monthly_profit_min")
            monthly_mean = row.get("monthly_profit_mean")
            n_neg_months = row.get("n_negative_months")
            n_months = row.get("n_months_total")

            if monthly_std is not None and monthly_mean is not None:
                # Coefficient of variation — higher = more unstable
                cv = abs(monthly_std / monthly_mean) if monthly_mean != 0 else float("inf")
                entry["monthly_cv"] = cv
                entry["monthly_std"] = monthly_std
                entry["monthly_min"] = monthly_min
                entry["monthly_mean"] = monthly_mean

                weaknesses = []
                if cv > 2.0:
                    weaknesses.append("HIGH_VARIANCE")
                if monthly_min is not None and monthly_min < -5:
                    weaknesses.append("DEEP_LOSS_MONTH")
                if n_neg_months and n_months and n_neg_months / n_months > 0.4:
                    weaknesses.append("MANY_LOSING_MONTHS")
                entry["weaknesses"] = weaknesses

            # Pair-level analysis
            pair_std = row.get("per_pair_profit_std")
            pair_min = row.get("per_pair_profit_min")
            n_profitable = row.get("n_profitable_pairs")
            n_pairs = row.get("n_pairs")

            if pair_std is not None:
                entry["pair_profit_std"] = pair_std
                entry["pair_profit_min"] = pair_min
                entry["n_profitable_pairs"] = n_profitable
                entry["n_pairs"] = n_pairs

                if pair_min is not None and pair_min < -2:
                    entry.setdefault("weaknesses", []).append("WEAK_PAIR")
                if n_profitable and n_pairs and n_profitable < n_pairs * 0.5:
                    entry.setdefault("weaknesses", []).append("LOW_PAIR_COVERAGE")

            # Drawdown analysis
            max_dd = row.get("max_drawdown")
            dd_days = row.get("max_drawdown_duration_days")
            consec_losses = row.get("max_consecutive_losses")
            if max_dd and max_dd > 0.2:
                entry.setdefault("weaknesses", []).append("HIGH_DRAWDOWN")
            if dd_days and dd_days > 30:
                entry.setdefault("weaknesses", []).append("LONG_RECOVERY")
            if consec_losses and consec_losses > 8:
                entry.setdefault("weaknesses", []).append("LONG_LOSING_STREAK")

            report["strategies"].append(entry)

        # Aggregate weakness frequency
        all_weaknesses = []
        for s in report["strategies"]:
            all_weaknesses.extend(s.get("weaknesses", []))
        report["weakness_frequency"] = dict(
            sorted(
                ((w, all_weaknesses.count(w)) for w in set(all_weaknesses)),
                key=lambda x: -x[1],
            )
        )

        self._weakness_report = report
        return report

    def find_complementary_pairs(
        self, top_n: int = 10, method: str = "correlation"
    ) -> List[Dict[str, Any]]:
        """Find strategy pairs that complement each other.

        Strategies are complementary when one performs well during the other's
        weak periods. Uses monthly profit statistics and per-pair breakdowns.

        Args:
            top_n: Analyze top N strategies.
            method: 'correlation' (monthly profit pattern) or 'pair_coverage' (per-pair).

        Returns:
            List of {strategy_a, strategy_b, score, description} dicts.
        """
        top = self.df.nlargest(top_n * 3, "fitness").copy()  # Broader pool

        if method == "pair_coverage":
            return self._find_pair_complementary(top, top_n)
        return self._find_monthly_complementary(top, top_n)

    def _find_monthly_complementary(
        self, pool: pd.DataFrame, top_n: int
    ) -> List[Dict[str, Any]]:
        """Find complementary pairs based on monthly profit pattern anti-correlation."""
        # Use available monthly stats as proxy features
        features = ["monthly_profit_mean", "monthly_profit_std",
                     "monthly_profit_min", "monthly_profit_max",
                     "n_positive_months", "n_negative_months"]
        available = [f for f in features if f in pool.columns]
        if len(available) < 3:
            logger.warning("[TEMPORAL] Insufficient monthly data for complementary analysis")
            return []

        pool_clean = pool.dropna(subset=available).head(top_n * 2)
        if len(pool_clean) < 4:
            return []

        # Build feature matrix and compute pairwise correlation
        X = pool_clean[available].values
        # Normalize
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        combos = []
        indices = pool_clean.index.tolist()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                # Anti-correlation score: strategies whose "bad" features anti-correlate
                # High std in A but low in B → complementary
                a = X_norm[i]
                b = X_norm[j]
                # Negative dot product = anti-correlated profiles
                score = -np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

                # Also consider: combined fitness should be good
                fa = pool_clean.iloc[i].get("fitness", 0) or 0
                fb = pool_clean.iloc[j].get("fitness", 0) or 0
                combined_quality = (fa + fb) / 2

                combos.append({
                    "strategy_a": pool_clean.iloc[i].get("fingerprint", ""),
                    "strategy_b": pool_clean.iloc[j].get("fingerprint", ""),
                    "fitness_a": fa,
                    "fitness_b": fb,
                    "complementary_score": float(score),
                    "combined_quality": float(combined_quality),
                    # Composite: balance complementarity and quality
                    "composite_score": float(0.6 * score + 0.4 * combined_quality),
                })

        combos.sort(key=lambda x: -x["composite_score"])
        return combos[:top_n]

    def _find_pair_complementary(
        self, pool: pd.DataFrame, top_n: int
    ) -> List[Dict[str, Any]]:
        """Find strategies that cover each other's weak pairs."""
        pair_cols = ["per_pair_profit_mean", "per_pair_profit_std",
                     "per_pair_profit_min", "n_profitable_pairs", "n_pairs"]
        available = [c for c in pair_cols if c in pool.columns]
        if not available:
            logger.warning("[TEMPORAL] No per-pair data available")
            return []

        pool_clean = pool.dropna(subset=available).head(top_n * 2)
        if len(pool_clean) < 4:
            return []

        combos = []
        for i in range(len(pool_clean)):
            for j in range(i + 1, len(pool_clean)):
                a = pool_clean.iloc[i]
                b = pool_clean.iloc[j]

                # Complementary if one has weak pairs where other is strong
                a_min = a.get("per_pair_profit_min", 0) or 0
                b_min = b.get("per_pair_profit_min", 0) or 0
                a_mean = a.get("per_pair_profit_mean", 0) or 0
                b_mean = b.get("per_pair_profit_mean", 0) or 0

                # Score: if A's worst pair is offset by B's average (and vice versa)
                coverage_score = min(b_mean - a_min, a_mean - b_min) if a_mean and b_mean else 0

                combos.append({
                    "strategy_a": a.get("fingerprint", ""),
                    "strategy_b": b.get("fingerprint", ""),
                    "fitness_a": a.get("fitness", 0),
                    "fitness_b": b.get("fitness", 0),
                    "coverage_score": float(coverage_score),
                })

        combos.sort(key=lambda x: -x["coverage_score"])
        return combos[:top_n]

    def report(self) -> str:
        """Generate human-readable weakness + complementary analysis report."""
        lines = ["=" * 70, "  TEMPORAL PERFORMANCE ANALYSIS", "=" * 70, ""]

        if self._weakness_report:
            wr = self._weakness_report
            lines.append(f"  Analyzed: {wr['n_analyzed']} top strategies")
            lines.append("")

            # Weakness frequency
            lines.append("  Weakness Frequency:")
            for weakness, count in wr.get("weakness_frequency", {}).items():
                pct = count / wr["n_analyzed"] * 100
                bar = "█" * int(pct / 5)
                lines.append(f"    {weakness:<25s} {count:>3d} ({pct:>5.1f}%) {bar}")
            lines.append("")

            # Per-strategy details
            lines.append("  Strategy Details:")
            for s in wr["strategies"][:10]:  # Top 10
                fp = s["fingerprint"][:12]
                fitness = s.get("fitness", 0)
                ws = s.get("weaknesses", [])
                ws_str = ", ".join(ws) if ws else "none"
                lines.append(f"    {fp}  fitness={fitness:.3f}  weaknesses: {ws_str}")
        else:
            lines.append("  No weakness analysis yet. Call analyze_weaknesses() first.")

        return "\n".join(lines)
