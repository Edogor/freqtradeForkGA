"""
Multi-Target Strategy Predictors

Trains LightGBM models to predict multiple strategy outcomes from gene structure:
  - Fitness, profit, max drawdown, stability, trade count, generalization ratio

Models are validated with walk-forward splits (train on older waves, test on newer).
Feature importance is extracted per model to understand which gene components
predict each outcome.

Usage:
    from genetic_algorithm.intelligence.predictors import MultiTargetPredictor
    pred = MultiTargetPredictor()
    pred.train(corpus_df)
    pred.report()
    pred.save()
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("genetic_algorithm/ml/models")

# Target definitions: (column_name, display_name, higher_is_better)
TARGET_DEFINITIONS = [
    ("fitness", "Fitness (Composite)", True),
    ("profit", "Profit %", True),
    ("max_drawdown", "Max Drawdown", False),
    ("sharpe_ratio", "Sharpe Ratio", True),
    ("num_trades", "Trade Count", True),
    ("pair_generalization_ratio", "Generalization Ratio", True),
    ("win_rate", "Win Rate", True),
]


def _get_feature_columns() -> List[str]:
    """Return the list of feature column names from the corpus."""
    from genetic_algorithm.intelligence.corpus import SURROGATE_FEATURE_NAMES
    return list(SURROGATE_FEATURE_NAMES)


class MultiTargetPredictor:
    """Trains and manages multiple LightGBM models for strategy outcome prediction."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models: Dict[str, Any] = {}  # target_name -> fitted model
        self.metrics: Dict[str, Dict[str, float]] = {}  # target_name -> {r2, rmse, mae}
        self.feature_importance: Dict[str, pd.DataFrame] = {}  # target_name -> df
        self._feature_columns: List[str] = _get_feature_columns()

    def train(
        self,
        df: pd.DataFrame,
        targets: Optional[List[str]] = None,
        test_waves: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Train models on the corpus DataFrame.

        Args:
            df: Strategy corpus from CorpusBuilder.build()
            targets: List of target columns to train on (default: all TARGET_DEFINITIONS)
            test_waves: Wave IDs to hold out for testing (default: latest 2 waves).
                       If empty or not enough data, falls back to random 80/20 split.

        Returns:
            Dict of target_name -> {r2, rmse, mae} on test set
        """
        import lightgbm as lgb
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        if targets is None:
            targets = [t[0] for t in TARGET_DEFINITIONS]

        # Determine train/test split
        train_df, test_df = self._split_data(df, test_waves)
        logger.info(
            f"[PREDICTOR] Train: {len(train_df)} samples, Test: {len(test_df)} samples"
        )

        X_train = train_df[self._feature_columns].values
        X_test = test_df[self._feature_columns].values

        results = {}
        for target in targets:
            if target not in df.columns:
                logger.warning(f"[PREDICTOR] Target '{target}' not in corpus — skipping")
                continue

            # Drop rows with NaN target
            train_mask = train_df[target].notna()
            test_mask = test_df[target].notna()
            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train < 20:
                logger.warning(
                    f"[PREDICTOR] Only {n_train} samples for '{target}' — skipping (need >=20)"
                )
                continue

            y_train = train_df.loc[train_mask, target].values
            y_test = test_df.loc[test_mask, target].values
            Xt = X_train[train_mask.values]
            Xv = X_test[test_mask.values]

            # Train LightGBM
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(
                Xt, y_train,
                eval_set=[(Xv, y_test)] if n_test >= 5 else None,
                callbacks=[lgb.log_evaluation(period=0)] if n_test >= 5 else None,
            )

            self.models[target] = model

            # Evaluate
            if n_test >= 5:
                y_pred = model.predict(Xv)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
            else:
                # No test data — report training metrics
                y_pred = model.predict(Xt)
                r2 = r2_score(y_train, y_pred)
                rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                mae = mean_absolute_error(y_train, y_pred)
                logger.info(f"[PREDICTOR] '{target}': no test data, reporting train metrics")

            self.metrics[target] = {"r2": r2, "rmse": rmse, "mae": mae, "n_train": n_train, "n_test": n_test}
            results[target] = self.metrics[target]

            # Feature importance
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                "feature": self._feature_columns,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            self.feature_importance[target] = fi_df

            logger.info(
                f"[PREDICTOR] '{target}': R²={r2:.3f}, RMSE={rmse:.4f}, MAE={mae:.4f} "
                f"(train={n_train}, test={n_test})"
            )

        return results

    def predict(self, df: pd.DataFrame, targets: Optional[List[str]] = None) -> pd.DataFrame:
        """Predict outcomes for new strategies.

        Args:
            df: DataFrame with feature columns (from CorpusBuilder or extract_features).
            targets: Which targets to predict (default: all trained).

        Returns:
            DataFrame with prediction columns named 'pred_{target}'.
        """
        if targets is None:
            targets = list(self.models.keys())

        X = df[self._feature_columns].values
        result = df.copy()
        for target in targets:
            if target in self.models:
                result[f"pred_{target}"] = self.models[target].predict(X)
        return result

    def report(self) -> str:
        """Generate a human-readable summary of model performance."""
        lines = ["=" * 70, "  MULTI-TARGET PREDICTOR REPORT", "=" * 70, ""]

        if not self.metrics:
            return "\n".join(lines + ["  No models trained yet."])

        # Summary table
        lines.append(f"  {'Target':<30s} {'R²':>8s} {'RMSE':>10s} {'MAE':>10s} {'Train':>7s} {'Test':>7s}")
        lines.append("  " + "-" * 72)
        for target, m in sorted(self.metrics.items(), key=lambda x: -x[1]["r2"]):
            quality = "***" if m["r2"] > 0.5 else "**" if m["r2"] > 0.3 else "*" if m["r2"] > 0.1 else ""
            lines.append(
                f"  {target:<30s} {m['r2']:>7.3f}{quality} {m['rmse']:>10.4f} {m['mae']:>10.4f} "
                f"{int(m['n_train']):>7d} {int(m['n_test']):>7d}"
            )

        lines.append("")
        lines.append("  Quality: *** R²>0.5 (good), ** R²>0.3 (moderate), * R²>0.1 (weak)")
        lines.append("")

        # Top features per model
        for target, fi_df in self.feature_importance.items():
            top5 = fi_df.head(5)
            lines.append(f"  Top features for '{target}':")
            for _, row in top5.iterrows():
                lines.append(f"    {row['feature']:<35s} {int(row['importance']):>6d}")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: Optional[Path] = None):
        """Save all trained models and metadata to disk."""
        out_dir = Path(path) if path else self.models_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for target, model in self.models.items():
            model_path = out_dir / f"predictor_{target}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Save metadata
        meta = {
            "targets": list(self.models.keys()),
            "metrics": {k: {mk: float(mv) for mk, mv in v.items()} for k, v in self.metrics.items()},
            "feature_columns": self._feature_columns,
        }
        with open(out_dir / "predictor_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"[PREDICTOR] Saved {len(self.models)} models to {out_dir}")

    def load(self, path: Optional[Path] = None):
        """Load trained models from disk."""
        load_dir = Path(path) if path else self.models_dir
        meta_path = load_dir / "predictor_meta.json"
        if not meta_path.exists():
            logger.warning(f"[PREDICTOR] No models found at {load_dir}")
            return

        with open(meta_path) as f:
            meta = json.load(f)

        self._feature_columns = meta.get("feature_columns", _get_feature_columns())
        self.metrics = meta.get("metrics", {})

        for target in meta.get("targets", []):
            model_path = load_dir / f"predictor_{target}.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.models[target] = pickle.load(f)  # noqa: S301

        logger.info(f"[PREDICTOR] Loaded {len(self.models)} models from {load_dir}")

    # ── Internal helpers ──────────────────────────────────────────

    def _split_data(
        self, df: pd.DataFrame, test_waves: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split corpus into train/test sets.

        Prefers wave-based splitting (train on older waves, test on newer).
        Falls back to random 80/20 if wave info insufficient.
        """
        if test_waves:
            test_mask = df["wave"].isin(test_waves)
            if test_mask.sum() >= 10:
                return df[~test_mask].copy(), df[test_mask].copy()
            logger.info("[PREDICTOR] Insufficient test wave data, falling back to random split")

        # Try auto-detecting latest waves
        waves = df["wave"].dropna().unique()
        wave_nums = sorted(
            [(w, int(w.replace("wave", ""))) for w in waves if w.startswith("wave")],
            key=lambda x: x[1],
        )
        if len(wave_nums) >= 3:
            test_wave_ids = [w[0] for w in wave_nums[-2:]]  # Last 2 waves
            test_mask = df["wave"].isin(test_wave_ids)
            if test_mask.sum() >= 10:
                logger.info(f"[PREDICTOR] Auto-detected test waves: {test_wave_ids}")
                return df[~test_mask].copy(), df[test_mask].copy()

        # Fallback: random 80/20 stratified by wave
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        return train_df, test_df
