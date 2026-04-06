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

import json
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
        """Find complementary pairs based on monthly profit anti-correlation.

        Uses raw monthly profit vectors (``monthly_profits_raw``) when available
        for true month-by-month Pearson correlation.  Falls back to summary-stat
        proxy when raw data is missing.
        """
        # Try raw-vector approach first
        if 'monthly_profits_raw' in pool.columns:
            raw_result = self._find_monthly_complementary_raw(pool, top_n)
            if raw_result:
                return raw_result

        # Fallback: summary-stat proxy
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
                a = X_norm[i]
                b = X_norm[j]
                score = -np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

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
                    "composite_score": float(0.6 * score + 0.4 * combined_quality),
                })

        combos.sort(key=lambda x: -x["composite_score"])
        return combos[:top_n]

    def _find_monthly_complementary_raw(
        self, pool: pd.DataFrame, top_n: int
    ) -> List[Dict[str, Any]]:
        """True month-by-month anti-correlation using raw profit vectors."""
        # Parse raw JSON vectors
        parsed: List[tuple] = []  # (idx, fingerprint, fitness, profit_vec)
        for idx, row in pool.iterrows():
            raw = row.get('monthly_profits_raw')
            if not raw or pd.isna(raw):
                continue
            try:
                vec = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(vec, list) and len(vec) >= 3:
                    parsed.append((
                        idx,
                        row.get('fingerprint', ''),
                        row.get('fitness', 0) or 0,
                        np.array(vec, dtype=float),
                    ))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        if len(parsed) < 4:
            return []

        # Trim to top_n*2 by fitness
        parsed.sort(key=lambda x: -x[2])
        parsed = parsed[:top_n * 2]

        combos = []
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                _, fp_a, fa, va = parsed[i]
                _, fp_b, fb, vb = parsed[j]

                # Align vectors to same length (truncate longer)
                min_len = min(len(va), len(vb))
                if min_len < 3:
                    continue
                va_t, vb_t = va[:min_len], vb[:min_len]

                # Pearson correlation
                corr = np.corrcoef(va_t, vb_t)[0, 1]
                if np.isnan(corr):
                    corr = 0.0

                # Ensemble Sharpe estimate (equal-weight portfolio)
                ensemble = (va_t + vb_t) / 2
                ens_mean = float(np.mean(ensemble))
                ens_std = float(np.std(ensemble))
                ensemble_sharpe = ens_mean / (ens_std + 1e-8)

                # Individual Sharpes for comparison
                sharpe_a = float(np.mean(va_t)) / (float(np.std(va_t)) + 1e-8)
                sharpe_b = float(np.mean(vb_t)) / (float(np.std(vb_t)) + 1e-8)
                sharpe_improvement = ensemble_sharpe - max(sharpe_a, sharpe_b)

                combined_quality = (fa + fb) / 2
                # Score: anti-correlation + quality + Sharpe improvement
                anti_corr = -corr  # higher is more anti-correlated
                composite = (0.35 * anti_corr
                             + 0.30 * combined_quality
                             + 0.35 * max(0, sharpe_improvement))

                combos.append({
                    "strategy_a": fp_a,
                    "strategy_b": fp_b,
                    "fitness_a": fa,
                    "fitness_b": fb,
                    "monthly_correlation": round(float(corr), 3),
                    "ensemble_sharpe": round(ensemble_sharpe, 3),
                    "sharpe_improvement": round(sharpe_improvement, 3),
                    "complementary_score": round(float(anti_corr), 3),
                    "combined_quality": round(float(combined_quality), 3),
                    "composite_score": round(float(composite), 3),
                })

        combos.sort(key=lambda x: -x["composite_score"])
        return combos[:top_n]

    def find_portfolio(
        self, max_size: int = 5, top_n: int = 30
    ) -> Dict[str, Any]:
        """Greedy portfolio construction from top strategies.

        Selects strategies one-by-one that maximally improve portfolio Sharpe
        using their raw monthly profit vectors.

        Args:
            max_size: Maximum number of strategies in the portfolio.
            top_n: Candidate pool size (top N by fitness).

        Returns:
            Dict with 'strategies', 'ensemble_sharpe', 'individual_sharpes',
            'monthly_correlation_matrix'.
        """
        if 'monthly_profits_raw' not in self.df.columns:
            return {"error": "No raw monthly profit data in corpus"}

        # Parse profit vectors for top strategies
        pool = self.df.nlargest(top_n, "fitness")
        candidates: List[Dict[str, Any]] = []
        for _, row in pool.iterrows():
            raw = row.get('monthly_profits_raw')
            if not raw or pd.isna(raw):
                continue
            try:
                vec = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(vec, list) and len(vec) >= 3:
                    candidates.append({
                        'fingerprint': row.get('fingerprint', ''),
                        'fitness': row.get('fitness', 0) or 0,
                        'vec': np.array(vec, dtype=float),
                    })
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        if len(candidates) < 2:
            return {"error": f"Only {len(candidates)} strategies with raw monthly data"}

        # Find min common length
        min_len = min(len(c['vec']) for c in candidates)
        for c in candidates:
            c['vec'] = c['vec'][:min_len]

        # Greedy selection: pick strategy that maximizes ensemble Sharpe
        selected: List[Dict[str, Any]] = []

        # Start with best individual Sharpe
        best_idx = max(
            range(len(candidates)),
            key=lambda i: np.mean(candidates[i]['vec']) / (np.std(candidates[i]['vec']) + 1e-8)
        )
        selected.append(candidates.pop(best_idx))

        while len(selected) < max_size and candidates:
            best_improvement = -float('inf')
            best_j = -1

            # Current portfolio returns
            port_vec = np.mean([s['vec'] for s in selected], axis=0)
            current_sharpe = float(np.mean(port_vec)) / (float(np.std(port_vec)) + 1e-8)

            for j, cand in enumerate(candidates):
                trial_vecs = [s['vec'] for s in selected] + [cand['vec']]
                trial_port = np.mean(trial_vecs, axis=0)
                trial_sharpe = float(np.mean(trial_port)) / (float(np.std(trial_port)) + 1e-8)
                improvement = trial_sharpe - current_sharpe
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_j = j

            if best_j >= 0 and best_improvement > -0.01:
                selected.append(candidates.pop(best_j))
            else:
                break

        # Build result
        port_vec = np.mean([s['vec'] for s in selected], axis=0)
        ensemble_sharpe = float(np.mean(port_vec)) / (float(np.std(port_vec)) + 1e-8)

        # Correlation matrix
        vecs = np.array([s['vec'] for s in selected])
        corr_matrix = np.corrcoef(vecs) if len(vecs) > 1 else np.array([[1.0]])

        return {
            'strategies': [
                {'fingerprint': s['fingerprint'], 'fitness': s['fitness']}
                for s in selected
            ],
            'ensemble_sharpe': round(ensemble_sharpe, 3),
            'individual_sharpes': [
                round(float(np.mean(s['vec'])) / (float(np.std(s['vec'])) + 1e-8), 3)
                for s in selected
            ],
            'monthly_correlation_matrix': np.round(corr_matrix, 3).tolist(),
            'n_months': min_len,
        }

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

        # Complementary pairs
        lines.append("")
        lines.append("-" * 70)
        lines.append("  COMPLEMENTARY PAIRS")
        lines.append("-" * 70)
        try:
            pairs = self.find_complementary_pairs(top_n=5)
            if pairs:
                for i, p in enumerate(pairs, 1):
                    fp_a = p['strategy_a'][:10]
                    fp_b = p['strategy_b'][:10]
                    score = p.get('composite_score', 0)
                    corr = p.get('monthly_correlation', None)
                    sharpe_imp = p.get('sharpe_improvement', None)
                    detail = f"  {i}. {fp_a} + {fp_b}  composite={score:.3f}"
                    if corr is not None:
                        detail += f"  corr={corr:.2f}"
                    if sharpe_imp is not None:
                        detail += f"  sharpe_Δ={sharpe_imp:+.3f}"
                    lines.append(detail)
            else:
                lines.append("  No complementary pairs found.")
        except Exception as e:
            lines.append(f"  Complementary analysis failed: {e}")

        # Portfolio
        lines.append("")
        lines.append("-" * 70)
        lines.append("  PORTFOLIO CONSTRUCTION")
        lines.append("-" * 70)
        try:
            portfolio = self.find_portfolio(max_size=5)
            if 'error' in portfolio:
                lines.append(f"  {portfolio['error']}")
            else:
                lines.append(
                    f"  Ensemble Sharpe: {portfolio['ensemble_sharpe']:.3f} "
                    f"({portfolio['n_months']} months)"
                )
                for i, (strat, sharpe) in enumerate(
                    zip(portfolio['strategies'], portfolio['individual_sharpes']), 1
                ):
                    fp = strat['fingerprint'][:12]
                    lines.append(f"    {i}. {fp}  fitness={strat['fitness']:.3f}  sharpe={sharpe:.3f}")
        except Exception as e:
            lines.append(f"  Portfolio construction failed: {e}")

        return "\n".join(lines)
