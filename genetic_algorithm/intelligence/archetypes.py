"""
Strategy Archetype Clustering

Automatically discovers strategy families (archetypes) by clustering
vectorized strategy representations. Uses HDBSCAN for density-based
clustering and optional dimensionality reduction for visualization.

Each archetype gets:
  - Auto-generated label based on dominant indicator/condition patterns
  - Performance statistics (avg fitness, drawdown, per-pair breakdown)
  - Representative strategies (closest to cluster centroid)

Usage:
    from genetic_algorithm.intelligence.archetypes import ArchetypeClassifier
    ac = ArchetypeClassifier()
    df = ac.fit(corpus_df)             # adds 'archetype' column
    ac.report()                        # print summary
    ac.save()                          # persist to disk
"""

import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("genetic_algorithm/ml/models")


def _get_feature_columns() -> List[str]:
    from genetic_algorithm.intelligence.corpus import SURROGATE_FEATURE_NAMES
    return list(SURROGATE_FEATURE_NAMES)


class ArchetypeClassifier:
    """Discovers and labels strategy archetypes via clustering."""

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        models_dir: Optional[Path] = None,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self._feature_columns = _get_feature_columns()
        self._scaler = None
        self._clusterer = None
        self._reducer = None  # For 2D visualization
        self._centroids: Dict[int, np.ndarray] = {}  # cluster_id -> centroid vector
        self._centroid_thresholds: Dict[int, float] = {}  # cluster_id -> max assign distance
        self.labels_: Optional[np.ndarray] = None
        self.archetype_stats: Dict[int, Dict[str, Any]] = {}
        self.archetype_labels: Dict[int, str] = {}

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cluster strategies and assign archetype labels.

        Args:
            df: Strategy corpus with surrogate feature columns and metrics.

        Returns:
            Input df with added columns: 'archetype' (int), 'archetype_label' (str),
            'cluster_x', 'cluster_y' (2D embedding for visualization).
        """
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import StandardScaler

        X = df[self._feature_columns].fillna(0).values

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Cluster
        self._clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
        )
        self.labels_ = self._clusterer.fit_predict(X_scaled)

        df = df.copy()
        df["archetype"] = self.labels_

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = (self.labels_ == -1).sum()
        logger.info(
            f"[ARCHETYPES] Found {n_clusters} archetypes, {n_noise} noise points "
            f"({n_noise / len(df) * 100:.1f}%)"
        )

        # Auto-label clusters
        self.archetype_labels = self._generate_labels(df)
        df["archetype_label"] = df["archetype"].map(
            lambda c: self.archetype_labels.get(c, "noise")
        )

        # Compute per-archetype statistics
        self.archetype_stats = self._compute_stats(df)

        # Compute cluster centroids and distance thresholds for predict()
        self._centroids = {}
        self._centroid_thresholds = {}
        for cid in set(self.labels_):
            if cid == -1:
                continue
            mask = self.labels_ == cid
            members = X_scaled[mask]
            centroid = members.mean(axis=0)
            self._centroids[cid] = centroid
            # Threshold: 2x mean intra-cluster distance
            dists = np.linalg.norm(members - centroid, axis=1)
            self._centroid_thresholds[cid] = float(dists.mean() * 2.0)

        # 2D embedding for visualization (PCA — always available, no extra deps)
        from sklearn.decomposition import PCA
        self._reducer = PCA(n_components=2, random_state=42)
        coords = self._reducer.fit_transform(X_scaled)
        df["cluster_x"] = coords[:, 0]
        df["cluster_y"] = coords[:, 1]

        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign archetype labels to new strategies using the fitted model.

        Uses approximate prediction via nearest cluster centroid.
        """
        if self._scaler is None or self._clusterer is None:
            raise RuntimeError("Must call fit() before predict()")

        X = df[self._feature_columns].fillna(0).values
        X_scaled = self._scaler.transform(X)

        # HDBSCAN approximate_predict if available, else nearest centroid
        try:
            from sklearn.cluster import HDBSCAN
            labels, strengths = HDBSCAN.approximate_predict(self._clusterer, X_scaled)
        except (AttributeError, TypeError):
            # Fallback: assign to nearest centroid
            labels = self._nearest_centroid_predict(X_scaled)

        df = df.copy()
        df["archetype"] = labels
        df["archetype_label"] = df["archetype"].map(
            lambda c: self.archetype_labels.get(c, "noise")
        )

        if self._reducer is not None:
            coords = self._reducer.transform(X_scaled)
            df["cluster_x"] = coords[:, 0]
            df["cluster_y"] = coords[:, 1]

        return df

    def report(self) -> str:
        """Generate a human-readable report of discovered archetypes."""
        lines = ["=" * 70, "  STRATEGY ARCHETYPE REPORT", "=" * 70, ""]

        if not self.archetype_stats:
            return "\n".join(lines + ["  No archetypes discovered yet."])

        # Overview
        n_clusters = sum(1 for k in self.archetype_stats if k != -1)
        n_noise = self.archetype_stats.get(-1, {}).get("count", 0)
        total = sum(s["count"] for s in self.archetype_stats.values())
        lines.append(f"  Archetypes: {n_clusters} | Noise: {n_noise} | Total: {total}")
        lines.append("")

        # Per-archetype details
        for cluster_id in sorted(self.archetype_stats.keys()):
            if cluster_id == -1:
                continue
            stats = self.archetype_stats[cluster_id]
            label = self.archetype_labels.get(cluster_id, f"Cluster {cluster_id}")
            lines.append(f"  ┌─ Archetype {cluster_id}: {label}")
            lines.append(f"  │  Count: {stats['count']} strategies")
            lines.append(
                f"  │  Fitness: {stats['fitness_mean']:.3f} ± {stats['fitness_std']:.3f} "
                f"(best: {stats['fitness_max']:.3f})"
            )
            if stats.get("profit_mean") is not None:
                lines.append(
                    f"  │  Profit:  {stats['profit_mean']:.1f}% ± {stats['profit_std']:.1f}%"
                )
            if stats.get("max_drawdown_mean") is not None:
                lines.append(f"  │  Drawdown: {stats['max_drawdown_mean']:.3f} avg")
            if stats.get("win_rate_mean") is not None:
                lines.append(f"  │  Win Rate: {stats['win_rate_mean']:.1%}")
            if stats.get("num_trades_mean") is not None:
                lines.append(f"  │  Trades:  {stats['num_trades_mean']:.0f} avg")

            top_inds = stats.get("top_indicators", [])
            if top_inds:
                lines.append(f"  │  Top indicators: {', '.join(top_inds[:5])}")

            top_tf = stats.get("dominant_timeframe", "")
            if top_tf:
                lines.append(f"  │  Timeframe: {top_tf}")

            lines.append(f"  └─")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: Optional[Path] = None):
        """Persist the fitted classifier to disk."""
        out_dir = Path(path) if path else self.models_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "archetype_classifier.pkl", "wb") as f:
            pickle.dump({
                "scaler": self._scaler,
                "clusterer": self._clusterer,
                "reducer": self._reducer,
                "labels": self.labels_,
                "centroids": self._centroids,
                "centroid_thresholds": self._centroid_thresholds,
                "archetype_labels": self.archetype_labels,
                "archetype_stats": self.archetype_stats,
                "feature_columns": self._feature_columns,
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
            }, f)

        logger.info(f"[ARCHETYPES] Saved classifier to {out_dir}")

    def load(self, path: Optional[Path] = None):
        """Load a saved classifier."""
        load_dir = Path(path) if path else self.models_dir
        pkl_path = load_dir / "archetype_classifier.pkl"
        if not pkl_path.exists():
            logger.warning(f"[ARCHETYPES] No classifier found at {pkl_path}")
            return

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        self._scaler = data["scaler"]
        self._clusterer = data["clusterer"]
        self._reducer = data.get("reducer")
        self.labels_ = data.get("labels")
        self._centroids = data.get("centroids", {})
        self._centroid_thresholds = data.get("centroid_thresholds", {})
        self.archetype_labels = data.get("archetype_labels", {})
        self.archetype_stats = data.get("archetype_stats", {})
        self._feature_columns = data.get("feature_columns", _get_feature_columns())
        logger.info(f"[ARCHETYPES] Loaded classifier with {len(self.archetype_labels)} archetypes")

    # ── Internal helpers ──────────────────────────────────────────

    def _generate_labels(self, df: pd.DataFrame) -> Dict[int, str]:
        """Auto-generate descriptive labels for each cluster."""
        from genetic_algorithm.evaluation.surrogate import _INDICATOR_TYPES

        labels = {}
        ind_cols = [f"ind_{t}" for t in _INDICATOR_TYPES]

        for cluster_id in sorted(set(self.labels_)):
            if cluster_id == -1:
                labels[-1] = "noise"
                continue

            cluster_df = df[df["archetype"] == cluster_id]

            # Find dominant indicators (highest mean presence in cluster vs global)
            global_means = df[ind_cols].mean()
            cluster_means = cluster_df[ind_cols].mean()
            # Indicators more prevalent in this cluster than globally
            enrichment = (cluster_means - global_means).sort_values(ascending=False)
            top_indicators = [
                col.replace("ind_", "") for col in enrichment.head(3).index
                if enrichment[col] > 0.1  # at least 10% more prevalent
            ]

            # Dominant timeframe
            tf_counts = cluster_df["timeframe"].value_counts()
            dom_tf = tf_counts.index[0] if len(tf_counts) > 0 else "mixed"

            # Build label
            if top_indicators:
                ind_part = "+".join(top_indicators[:2])
                label = f"{ind_part} ({dom_tf})"
            else:
                label = f"Mixed ({dom_tf})"

            # Add behavioral hint
            avg_trades = cluster_df["num_trades"].mean() if "num_trades" in cluster_df else 0
            if avg_trades and avg_trades > 1000:
                label += " [HF]"
            elif avg_trades and avg_trades < 50:
                label += " [LF]"

            labels[cluster_id] = label

        return labels

    def _compute_stats(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Compute per-archetype statistics."""
        from genetic_algorithm.evaluation.surrogate import _INDICATOR_TYPES

        stats = {}
        ind_cols = [f"ind_{t}" for t in _INDICATOR_TYPES]

        for cluster_id in sorted(set(self.labels_)):
            cluster_df = df[df["archetype"] == cluster_id]
            s: Dict[str, Any] = {"count": len(cluster_df)}

            for metric in ["fitness", "profit", "max_drawdown", "win_rate",
                           "sharpe_ratio", "num_trades", "pair_generalization_ratio"]:
                if metric in cluster_df.columns:
                    vals = cluster_df[metric].dropna()
                    if len(vals) > 0:
                        s[f"{metric}_mean"] = float(vals.mean())
                        s[f"{metric}_std"] = float(vals.std())
                        s[f"{metric}_max"] = float(vals.max())
                        s[f"{metric}_min"] = float(vals.min())

            # Top indicators: ranked by enrichment vs global mean (not absolute prevalence)
            # This ensures archetype immigrants carry truly distinctive indicators.
            global_ind_means = df[ind_cols].mean()
            enrichment_ratio = (cluster_ind_means / (global_ind_means + 0.01)).sort_values(ascending=False)
            s["top_indicators"] = [
                col.replace("ind_", "") for col in enrichment_ratio.head(8).index
                if enrichment_ratio[col] > 1.5  # ≥50% more common in this cluster than globally
            ]

            # Dominant timeframe
            if "timeframe" in cluster_df.columns:
                tf_counts = cluster_df["timeframe"].value_counts()
                s["dominant_timeframe"] = tf_counts.index[0] if len(tf_counts) > 0 else ""

            stats[cluster_id] = s

        return stats

    def _nearest_centroid_predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Fallback prediction via nearest centroid distance."""
        if self.labels_ is None or not self._centroids:
            return np.full(len(X_scaled), -1)

        cluster_ids = sorted(self._centroids.keys())
        centroids = np.array([self._centroids[cid] for cid in cluster_ids])
        thresholds = np.array([self._centroid_thresholds[cid] for cid in cluster_ids])

        # Compute distances from each point to each centroid
        # X_scaled: (n, d), centroids: (k, d) -> dists: (n, k)
        dists = np.linalg.norm(X_scaled[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

        nearest_idx = dists.argmin(axis=1)
        nearest_dist = dists[np.arange(len(X_scaled)), nearest_idx]

        labels = np.full(len(X_scaled), -1, dtype=int)
        for i, (idx, dist) in enumerate(zip(nearest_idx, nearest_dist)):
            if dist <= thresholds[idx]:
                labels[i] = cluster_ids[idx]

        return labels
