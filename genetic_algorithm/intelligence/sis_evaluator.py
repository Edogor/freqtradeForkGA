"""
SIS Evaluation Framework

Measures and reports on the health and accuracy of all SIS components:
- Predictor accuracy (per-target R², AUC on held-out data)
- Prediction drift (does model accuracy degrade on newer waves?)
- Archetype stability (do clusters remain consistent across runs?)
- Overall SIS Health Report for pre-run and post-run analysis

Usage:
    from genetic_algorithm.intelligence.sis_evaluator import SISEvaluator
    evaluator = SISEvaluator(corpus_df, predictor, classifier)
    report = evaluator.health_report()
    evaluator.post_run_report(new_strategies_df)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SISEvaluator:
    """Evaluates and monitors SIS component health."""

    def __init__(
        self,
        corpus_df: Optional[pd.DataFrame] = None,
        predictor: Optional[Any] = None,
        classifier: Optional[Any] = None,
    ):
        self._corpus_df = corpus_df
        self._predictor = predictor
        self._classifier = classifier

    # ── Predictor evaluation ────────────────────────────────────────

    def evaluate_predictor_health(self) -> Dict[str, Any]:
        """Assess predictor model quality and identify unreliable models.

        Returns a dict with:
        - reliable_models: models with R²>0.1 or AUC>0.52
        - unreliable_models: models that should not be used for decisions
        - total_models: count of all models
        - health_score: 0-1 score summarizing overall predictor quality
        """
        if self._predictor is None:
            return {"error": "No predictor loaded", "health_score": 0.0}

        reliable = self._predictor.get_reliable_models()
        metrics = self._predictor.metrics

        all_model_names = set(metrics.keys())
        reliable_names = set(reliable.keys())
        unreliable_names = all_model_names - reliable_names

        # Health score = fraction of models that are reliable,
        # weighted by their quality
        if not all_model_names:
            health_score = 0.0
        else:
            quality_sum = sum(reliable.values())
            # Normalize: avg R²/AUC of reliable models * fraction reliable
            frac_reliable = len(reliable_names) / len(all_model_names)
            avg_quality = quality_sum / len(reliable_names) if reliable_names else 0
            health_score = round(frac_reliable * avg_quality, 3)

        result = {
            "reliable_models": {
                name: {"score": score, "type": metrics[name].get("type", "unknown")}
                for name, score in reliable.items()
            },
            "unreliable_models": {
                name: {
                    "score": metrics[name].get("r2", metrics[name].get("auc", 0)),
                    "type": metrics[name].get("type", "unknown"),
                    "reason": self._unreliable_reason(name, metrics[name]),
                }
                for name in unreliable_names
            },
            "total_models": len(all_model_names),
            "n_reliable": len(reliable_names),
            "n_unreliable": len(unreliable_names),
            "health_score": health_score,
        }
        return result

    @staticmethod
    def _unreliable_reason(name: str, met: Dict) -> str:
        model_type = met.get("type", "")
        if model_type == "regression":
            r2 = met.get("r2", -1)
            if r2 < 0:
                return f"Negative R²={r2:.3f} (anti-predictive, worse than mean)"
            return f"Low R²={r2:.3f} (below 0.1 threshold)"
        elif model_type == "classification":
            auc = met.get("auc", 0.5)
            return f"Low AUC={auc:.3f} (near random, below 0.52 threshold)"
        return "Unknown model type"

    # ── Prediction drift detection ──────────────────────────────────

    def evaluate_prediction_drift(
        self,
        new_strategies_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compare predictions vs actual outcomes on new strategies.

        Args:
            new_strategies_df: DataFrame with both features AND actual outcomes
                (fitness, profit, win_rate etc.) from a completed evolution run.

        Returns:
            Dict with per-target prediction accuracy on new data vs training data.
        """
        if self._predictor is None:
            return {"error": "No predictor loaded"}

        from genetic_algorithm.intelligence.predictors import _add_interaction_features

        df = _add_interaction_features(new_strategies_df)
        results = {}

        # Check regression models
        for target, model in self._predictor.models.items():
            if target not in df.columns:
                continue

            mask = df[target].notna()
            if mask.sum() < 10:
                continue

            try:
                from sklearn.metrics import r2_score, mean_absolute_error

                X = self._predictor._get_feature_matrix(df[mask])
                y_true = df.loc[mask, target].values
                y_pred = model.predict(X)

                new_r2 = r2_score(y_true, y_pred)
                new_mae = mean_absolute_error(y_true, y_pred)
                train_r2 = self._predictor.metrics.get(target, {}).get("r2", None)

                drift = (train_r2 - new_r2) if train_r2 is not None else None

                results[target] = {
                    "type": "regression",
                    "new_r2": round(new_r2, 4),
                    "train_r2": round(train_r2, 4) if train_r2 is not None else None,
                    "drift": round(drift, 4) if drift is not None else None,
                    "new_mae": round(new_mae, 4),
                    "n_samples": int(mask.sum()),
                    "degraded": drift is not None and drift > 0.1,
                }
            except Exception as e:
                results[target] = {"error": str(e)}

        # Check classification models
        for clf_name, clf_model in self._predictor.classifiers.items():
            source_col_map = {
                "clf_fitness": "fitness",
                "clf_win_rate": "win_rate",
                "clf_profit": "profit",
                "clf_sharpe_ratio": "sharpe_ratio",
                "clf_overfit_risk": "overfit_risk",
            }
            source_col = source_col_map.get(clf_name)
            if source_col is None or source_col not in df.columns:
                continue

            threshold = self._predictor._quartile_thresholds.get(clf_name)
            if threshold is None:
                continue

            mask = df[source_col].notna()
            if mask.sum() < 10:
                continue

            try:
                from sklearn.metrics import roc_auc_score

                X = self._predictor._get_feature_matrix(df[mask])
                if source_col == "overfit_risk":
                    y_true = (df.loc[mask, source_col] <= threshold).astype(int).values
                else:
                    y_true = (df.loc[mask, source_col] >= threshold).astype(int).values

                if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
                    continue

                y_prob = clf_model.predict_proba(X)[:, 1]
                new_auc = roc_auc_score(y_true, y_prob)
                train_auc = self._predictor.metrics.get(clf_name, {}).get("auc", None)
                drift = (train_auc - new_auc) if train_auc is not None else None

                results[clf_name] = {
                    "type": "classification",
                    "new_auc": round(new_auc, 4),
                    "train_auc": round(train_auc, 4) if train_auc is not None else None,
                    "drift": round(drift, 4) if drift is not None else None,
                    "n_samples": int(mask.sum()),
                    "degraded": drift is not None and drift > 0.05,
                }
            except Exception as e:
                results[clf_name] = {"error": str(e)}

        return results

    # ── Archetype evaluation ────────────────────────────────────────

    def evaluate_archetype_health(self) -> Dict[str, Any]:
        """Assess archetype (clustering) quality.

        Returns:
            Dict with cluster count, noise ratio, size distribution.
        """
        if self._classifier is None:
            return {"error": "No classifier loaded"}

        labels = self._classifier.archetype_labels
        stats = self._classifier.archetype_stats

        if not labels:
            return {
                "error": "No archetypes fitted",
                "health_score": 0.0,
            }

        n_archetypes = len([k for k in labels if k != -1])
        arch_sizes = {
            k: stats.get(k, {}).get("count", 0)
            for k in labels if k != -1
        }
        total_assigned = sum(arch_sizes.values())
        noise_count = stats.get(-1, {}).get("count", 0)
        total = total_assigned + noise_count

        noise_ratio = noise_count / total if total > 0 else 1.0
        # Good clustering has 3-15 archetypes and <40% noise
        cluster_score = 1.0 if 3 <= n_archetypes <= 15 else 0.5
        noise_score = max(0, 1.0 - noise_ratio / 0.4)  # 0% noise=1.0, 40%+=0.0
        health_score = round((cluster_score + noise_score) / 2, 3)

        return {
            "n_archetypes": n_archetypes,
            "archetype_sizes": arch_sizes,
            "noise_count": noise_count,
            "noise_ratio": round(noise_ratio, 3),
            "labels": labels,
            "health_score": health_score,
        }

    # ── Corpus evaluation ───────────────────────────────────────────

    def evaluate_corpus_health(self) -> Dict[str, Any]:
        """Assess corpus quality and staleness."""
        if self._corpus_df is None or len(self._corpus_df) == 0:
            return {"error": "No corpus loaded", "health_score": 0.0}

        df = self._corpus_df
        n = len(df)

        waves = []
        if "wave" in df.columns:
            waves = sorted(df["wave"].dropna().unique().tolist())

        # Feature completeness
        from genetic_algorithm.intelligence.corpus import SURROGATE_FEATURE_NAMES
        present_features = [f for f in SURROGATE_FEATURE_NAMES if f in df.columns]
        feature_completeness = len(present_features) / len(SURROGATE_FEATURE_NAMES)

        # Target availability
        target_cols = ["fitness", "profit", "win_rate", "max_drawdown", "sharpe_ratio"]
        target_availability = {
            col: int(df[col].notna().sum()) if col in df.columns else 0
            for col in target_cols
        }

        # Size score: 200+ strategies is decent, 1000+ is good, 3000+ is excellent
        if n >= 3000:
            size_score = 1.0
        elif n >= 1000:
            size_score = 0.8
        elif n >= 200:
            size_score = 0.5
        else:
            size_score = 0.2

        health_score = round((size_score + feature_completeness) / 2, 3)

        return {
            "n_strategies": n,
            "n_waves": len(waves),
            "waves": waves,
            "feature_completeness": round(feature_completeness, 3),
            "target_availability": target_availability,
            "size_score": size_score,
            "health_score": health_score,
        }

    # ── Overall health report ───────────────────────────────────────

    def health_report(self) -> Dict[str, Any]:
        """Generate comprehensive SIS health report.

        Combines corpus, predictor, and archetype health into a single report.
        """
        corpus_health = self.evaluate_corpus_health()
        predictor_health = self.evaluate_predictor_health()
        archetype_health = self.evaluate_archetype_health()
        temporal_coverage = self.evaluate_temporal_coverage()
        complementary = self.evaluate_complementary_potential()

        # Overall health = weighted average of component scores
        scores = [
            corpus_health.get("health_score", 0),
            predictor_health.get("health_score", 0),
            archetype_health.get("health_score", 0),
        ]
        overall = round(sum(scores) / len(scores), 3)

        report = {
            "overall_health_score": overall,
            "corpus": corpus_health,
            "predictor": predictor_health,
            "archetypes": archetype_health,
            "temporal": temporal_coverage,
            "complementary": complementary,
            "recommendations": self._generate_recommendations(
                corpus_health, predictor_health, archetype_health
            ),
        }
        return report

    def _generate_recommendations(
        self,
        corpus_health: Dict,
        predictor_health: Dict,
        archetype_health: Dict,
    ) -> List[str]:
        """Generate actionable recommendations based on health scores."""
        recs = []

        # Corpus recommendations
        corpus_score = corpus_health.get("health_score", 0)
        if corpus_score < 0.5:
            recs.append(
                "Corpus is small or incomplete. Run more evolution waves and "
                "rebuild the corpus to improve SIS accuracy."
            )

        # Predictor recommendations
        n_unreliable = predictor_health.get("n_unreliable", 0)
        n_reliable = predictor_health.get("n_reliable", 0)
        if n_unreliable > n_reliable and n_reliable > 0:
            recs.append(
                f"{n_unreliable} of {n_unreliable + n_reliable} models are unreliable. "
                f"SIS decisions are based on only {n_reliable} trustworthy model(s). "
                f"Consider retraining with more data or different features."
            )
        if n_reliable == 0:
            recs.append(
                "No reliable predictor models available. SIS seed filtering and "
                "quality gating are effectively disabled. Retrain predictors or "
                "gather more strategy data."
            )

        # Archetype recommendations
        n_arch = archetype_health.get("n_archetypes", 0)
        noise_ratio = archetype_health.get("noise_ratio", 1.0)
        if n_arch < 3:
            recs.append(
                f"Only {n_arch} archetypes discovered. The corpus may be too small "
                f"or homogeneous for meaningful clustering."
            )
        if noise_ratio > 0.4:
            recs.append(
                f"High noise ratio ({noise_ratio:.1%}). Many strategies don't fit "
                f"any archetype. Consider adjusting min_cluster_size or gathering "
                f"more diverse strategies."
            )

        if not recs:
            recs.append("SIS health is good. All components are operational.")

        return recs

    # ── Post-run analysis ───────────────────────────────────────────

    def post_run_report(
        self,
        new_strategies_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Generate post-evolution-run analysis.

        Compares SIS predictions against actual outcomes from the run.

        Args:
            new_strategies_df: DataFrame with features AND actual outcomes
                from a completed evolution run.

        Returns:
            Dict with prediction drift, archetype discovery, and recommendations.
        """
        drift = self.evaluate_prediction_drift(new_strategies_df)

        # Check which archetypes appeared in the new run
        archetype_discovery = {}
        if self._classifier is not None:
            try:
                from genetic_algorithm.intelligence.predictors import (
                    _add_interaction_features,
                )
                from genetic_algorithm.intelligence.corpus import SURROGATE_FEATURE_NAMES

                feat_cols = list(SURROGATE_FEATURE_NAMES)
                available_cols = [c for c in feat_cols if c in new_strategies_df.columns]
                if len(available_cols) == len(feat_cols):
                    pred_df = self._classifier.predict(
                        new_strategies_df[feat_cols].fillna(0)
                    )
                    if "archetype" in pred_df.columns:
                        counts = pred_df["archetype"].value_counts().to_dict()
                        known = set(self._classifier.archetype_labels.keys()) - {-1}
                        found = set(counts.keys()) - {-1}
                        archetype_discovery = {
                            "known_archetypes": list(known),
                            "found_in_run": list(found),
                            "missing_from_run": list(known - found),
                            "distribution": {str(k): v for k, v in counts.items()},
                        }
            except Exception as e:
                archetype_discovery = {"error": str(e)}

        # Identify degraded models
        degraded = [
            name for name, info in drift.items()
            if isinstance(info, dict) and info.get("degraded", False)
        ]

        report = {
            "prediction_drift": drift,
            "archetype_discovery": archetype_discovery,
            "degraded_models": degraded,
            "recommendations": [],
        }

        if degraded:
            report["recommendations"].append(
                f"Models {degraded} show significant drift. Consider retraining "
                f"the SIS predictors with data from this run included."
            )
        if archetype_discovery.get("missing_from_run"):
            missing = archetype_discovery["missing_from_run"]
            report["recommendations"].append(
                f"Archetypes {missing} were not discovered in this run. "
                f"Future runs could target these gaps for portfolio diversity."
            )

        return report

    # ── Formatted report output ─────────────────────────────────────

    @staticmethod
    def format_health_report(report: Dict[str, Any]) -> str:
        """Format a health report dict into a human-readable string."""
        lines = [
            "=" * 70,
            "  SIS HEALTH REPORT",
            "=" * 70,
            "",
            f"  Overall Health Score: {report.get('overall_health_score', 0):.1%}",
            "",
        ]

        # Corpus section
        corpus = report.get("corpus", {})
        lines.append("  CORPUS")
        lines.append(f"    Strategies: {corpus.get('n_strategies', 0)}")
        lines.append(f"    Waves: {corpus.get('n_waves', 0)}")
        lines.append(f"    Feature completeness: {corpus.get('feature_completeness', 0):.1%}")
        lines.append(f"    Health: {corpus.get('health_score', 0):.1%}")
        lines.append("")

        # Predictor section
        pred = report.get("predictor", {})
        lines.append("  PREDICTORS")
        lines.append(f"    Reliable: {pred.get('n_reliable', 0)} / {pred.get('total_models', 0)}")
        for name, info in pred.get("reliable_models", {}).items():
            score = info.get("score", 0)
            lines.append(f"      [OK] {name}: {score:.3f}")
        for name, info in pred.get("unreliable_models", {}).items():
            reason = info.get("reason", "unknown")
            lines.append(f"      [!!] {name}: {reason}")
        lines.append(f"    Health: {pred.get('health_score', 0):.1%}")
        lines.append("")

        # Archetype section
        arch = report.get("archetypes", {})
        lines.append("  ARCHETYPES")
        lines.append(f"    Clusters: {arch.get('n_archetypes', 0)}")
        lines.append(f"    Noise ratio: {arch.get('noise_ratio', 0):.1%}")
        lines.append(f"    Health: {arch.get('health_score', 0):.1%}")
        lines.append("")

        # Recommendations
        recs = report.get("recommendations", [])
        if recs:
            lines.append("  RECOMMENDATIONS")
            for i, rec in enumerate(recs, 1):
                lines.append(f"    {i}. {rec}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ── Temporal correlation analysis ───────────────────────────────

    def evaluate_temporal_coverage(self) -> Dict[str, Any]:
        """Assess how much raw temporal data is available in the corpus.

        Returns stats on monthly_profits_raw and per_pair_profit_raw coverage,
        which are needed for complementary strategy analysis.
        """
        if self._corpus_df is None or len(self._corpus_df) == 0:
            return {"error": "No corpus loaded", "coverage": 0.0}

        import json as _json

        df = self._corpus_df
        n = len(df)

        monthly_available = 0
        pair_available = 0
        monthly_lengths = []

        if "monthly_profits_raw" in df.columns:
            for val in df["monthly_profits_raw"].dropna():
                try:
                    vec = _json.loads(val)
                    if vec:
                        monthly_available += 1
                        monthly_lengths.append(len(vec))
                except (ValueError, TypeError):
                    pass

        if "per_pair_profit_raw" in df.columns:
            pair_available = int(df["per_pair_profit_raw"].notna().sum())

        coverage = monthly_available / n if n > 0 else 0.0

        return {
            "n_with_monthly_vectors": monthly_available,
            "n_with_pair_maps": pair_available,
            "total_strategies": n,
            "monthly_coverage": round(coverage, 3),
            "avg_monthly_length": (
                round(np.mean(monthly_lengths), 1) if monthly_lengths else 0
            ),
            "ready_for_correlation": monthly_available >= 50,
        }

    def evaluate_complementary_potential(self) -> Dict[str, Any]:
        """Run complementary strategy analysis and portfolio construction.

        Uses TemporalAnalyzer to find anti-correlated pairs and build
        a greedy Sharpe-maximizing portfolio.
        """
        if self._corpus_df is None or len(self._corpus_df) < 10:
            return {"error": "Insufficient corpus data"}

        try:
            from genetic_algorithm.intelligence.temporal_analysis import TemporalAnalyzer
            analyzer = TemporalAnalyzer(self._corpus_df)

            result: Dict[str, Any] = {}

            # Top complementary pairs
            pairs = analyzer.find_complementary_pairs(top_n=5)
            result['top_pairs'] = pairs
            result['n_pairs_found'] = len(pairs)

            # Portfolio construction
            portfolio = analyzer.find_portfolio(max_size=5)
            result['portfolio'] = portfolio

            # Summary
            if pairs:
                best = pairs[0]
                result['best_pair_score'] = best.get('composite_score', 0)
                result['best_pair_correlation'] = best.get('monthly_correlation', None)
            if 'ensemble_sharpe' in portfolio:
                result['portfolio_sharpe'] = portfolio['ensemble_sharpe']

            return result
        except Exception as e:
            return {"error": str(e)}
