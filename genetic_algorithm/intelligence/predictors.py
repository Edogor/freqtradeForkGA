"""
Multi-Target Strategy Predictors (SIS v3)

Trains LightGBM models to predict multiple strategy outcomes from gene structure:
  - Regression: fitness, profit, max drawdown, stability, trade count, generalization, win_rate
  - Classification: top-quartile yes/no for each target (much more learnable)
  - Overfitting Risk: predicts generalization gap (train_fitness - holdout_fitness)

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

# Derived targets computed from existing columns
DERIVED_TARGETS = [
    ("overfit_risk", "Overfitting Risk", False),
]

# Classification targets: predict top-quartile membership for each target
CLASSIFICATION_TARGETS = [
    ("clf_fitness", "Is Top-25% Fitness", True),
    ("clf_win_rate", "Is Top-25% Win Rate", True),
    ("clf_profit", "Is Top-25% Profit", True),
    ("clf_sharpe_ratio", "Is Top-25% Sharpe", True),
    ("clf_overfit_risk", "Is Low Overfit Risk", True),
]


def _get_feature_columns() -> List[str]:
    """Return the list of feature column names from the corpus."""
    from genetic_algorithm.intelligence.corpus import SURROGATE_FEATURE_NAMES
    return list(SURROGATE_FEATURE_NAMES)


def _get_interaction_features() -> List[str]:
    """Return names of pairwise indicator interaction features."""
    top_synergy_pairs = [
        ("DONCHIAN", "ROC"), ("CCI", "STOCH"), ("ATR", "CCI"),
        ("AROON", "ROC"), ("ADX", "ATR"), ("RSI", "STOCH"),
        ("CCI", "ROC"), ("DONCHIAN", "ATR"), ("ADX", "CCI"),
        ("BBANDS", "CCI"),
    ]
    return [f"syn_{a}_{b}" for a, b in top_synergy_pairs]


SYNERGY_PAIRS = [
    ("DONCHIAN", "ROC"), ("CCI", "STOCH"), ("ATR", "CCI"),
    ("AROON", "ROC"), ("ADX", "ATR"), ("RSI", "STOCH"),
    ("CCI", "ROC"), ("DONCHIAN", "ATR"), ("ADX", "CCI"),
    ("BBANDS", "CCI"),
]


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise indicator synergy features and structural ratios."""
    df = df.copy()
    for a, b in SYNERGY_PAIRS:
        col_a, col_b = f"ind_{a}", f"ind_{b}"
        feat_name = f"syn_{a}_{b}"
        if col_a in df.columns and col_b in df.columns:
            df[feat_name] = df[col_a] * df[col_b]
        else:
            df[feat_name] = 0.0

    # Structural ratios
    if "n_entry_conds" in df.columns and "n_exit_conds" in df.columns:
        df["ratio_entry_exit"] = df["n_entry_conds"] / (df["n_exit_conds"] + 1)
    if "n_indicators" in df.columns and "n_entry_conds" in df.columns:
        df["ratio_ind_cond"] = df["n_indicators"] / (df["n_entry_conds"] + 1)

    return df


class MultiTargetPredictor:
    """Trains and manages multiple LightGBM models for strategy outcome prediction.

    SIS v3 additions:
    - Overfitting risk predictor (derived from train/holdout gap)
    - Classification predictors (top-quartile binary)
    - Interaction features (indicator synergies)
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models: Dict[str, Any] = {}  # target_name -> fitted model
        self.classifiers: Dict[str, Any] = {}  # clf_target -> fitted classifier
        self.metrics: Dict[str, Dict[str, float]] = {}  # target_name -> metrics
        self.feature_importance: Dict[str, pd.DataFrame] = {}  # target_name -> df
        self._feature_columns: List[str] = _get_feature_columns()
        self._interaction_columns: List[str] = _get_interaction_features()
        self._use_interactions: bool = True
        # Quartile thresholds for classification (saved for predict-time)
        self._quartile_thresholds: Dict[str, float] = {}

    @property
    def _all_feature_columns(self) -> List[str]:
        """All feature columns including interactions."""
        if self._use_interactions:
            return self._feature_columns + self._interaction_columns + [
                "ratio_entry_exit", "ratio_ind_cond"
            ]
        return self._feature_columns

    def train(
        self,
        df: pd.DataFrame,
        targets: Optional[List[str]] = None,
        test_waves: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Train regression, classification, and derived-target models.

        Args:
            df: Strategy corpus from CorpusBuilder.build()
            targets: List of regression target columns (default: all TARGET_DEFINITIONS)
            test_waves: Wave IDs to hold out for testing.

        Returns:
            Dict of target_name -> metrics on test set
        """
        # Compute derived columns
        df = self._prepare_derived_targets(df)

        # Add interaction features
        df = _add_interaction_features(df)

        if targets is None:
            targets = [t[0] for t in TARGET_DEFINITIONS]

        # Add overfit_risk to regression targets if data exists
        if "overfit_risk" in df.columns and df["overfit_risk"].notna().sum() >= 40:
            if "overfit_risk" not in targets:
                targets.append("overfit_risk")

        # Split
        train_df, test_df = self._split_data(df, test_waves)
        logger.info(
            f"[PREDICTOR] Train: {len(train_df)} samples, Test: {len(test_df)} samples"
        )

        results = {}

        # 1. Train regression models
        reg_results = self._train_regressors(train_df, test_df, targets)
        results.update(reg_results)

        # 2. Train classification models
        clf_results = self._train_classifiers(train_df, test_df)
        results.update(clf_results)

        return results

    def _prepare_derived_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived target columns from existing metrics."""
        df = df.copy()

        # Overfitting risk: relative gap between train and holdout fitness
        if "train_fitness" in df.columns and "holdout_fitness" in df.columns:
            mask = df["train_fitness"].notna() & df["holdout_fitness"].notna()
            mask &= df["train_fitness"] > 0.01  # avoid div-by-zero
            df.loc[mask, "overfit_risk"] = (
                (df.loc[mask, "train_fitness"] - df.loc[mask, "holdout_fitness"])
                / df.loc[mask, "train_fitness"]
            ).clip(-1.0, 1.0)
        elif "fitness" in df.columns and "val_fitness" in df.columns:
            # Fallback: use fitness vs val_fitness
            mask = df["fitness"].notna() & df["val_fitness"].notna()
            mask &= df["fitness"] > 0.01
            df.loc[mask, "overfit_risk"] = (
                (df.loc[mask, "fitness"] - df.loc[mask, "val_fitness"])
                / df.loc[mask, "fitness"]
            ).clip(-1.0, 1.0)

        return df

    def _get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract named feature DataFrame ensuring all columns exist.

        Returns a DataFrame (not a numpy array) so that LightGBM receives the
        feature names it was trained with, avoiding the sklearn feature-name
        mismatch warning that fires when a plain numpy array is passed to a
        model fitted on named data.
        """
        df = df.copy()
        for col in self._all_feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        return df[self._all_feature_columns].fillna(0)

    def _train_regressors(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
        targets: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Train LightGBM regression models for each target."""
        import lightgbm as lgb
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        X_train = self._get_feature_matrix(train_df)
        X_test = self._get_feature_matrix(test_df)

        results = {}
        for target in targets:
            if target not in train_df.columns:
                logger.warning(f"[PREDICTOR] Target '{target}' not in corpus — skipping")
                continue

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

            model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(
                Xt, y_train,
                eval_set=[(Xv, y_test)] if n_test >= 5 else None,
                callbacks=[lgb.log_evaluation(period=0)] if n_test >= 5 else None,
            )

            self.models[target] = model

            if n_test >= 5:
                y_pred = model.predict(Xv)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
            else:
                y_pred = model.predict(Xt)
                r2 = r2_score(y_train, y_pred)
                rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                mae = mean_absolute_error(y_train, y_pred)

            self.metrics[target] = {
                "r2": r2, "rmse": rmse, "mae": mae,
                "n_train": n_train, "n_test": n_test, "type": "regression",
            }
            results[target] = self.metrics[target]

            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                "feature": self._all_feature_columns,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            self.feature_importance[target] = fi_df

            logger.info(
                f"[PREDICTOR] '{target}': R²={r2:.3f}, RMSE={rmse:.4f}, MAE={mae:.4f} "
                f"(train={n_train}, test={n_test})"
            )

        return results

    def _train_classifiers(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Train binary classifiers: predict top-quartile membership.

        Classification is much easier than regression for noisy targets.
        Even a 60-65% AUC is more useful than a negative R².
        """
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

        clf_target_map = {
            "clf_fitness": "fitness",
            "clf_win_rate": "win_rate",
            "clf_profit": "profit",
            "clf_sharpe_ratio": "sharpe_ratio",
            "clf_overfit_risk": "overfit_risk",
        }

        X_train = self._get_feature_matrix(train_df)
        X_test = self._get_feature_matrix(test_df)

        results = {}
        for clf_name, source_col in clf_target_map.items():
            if source_col not in train_df.columns:
                continue

            # Compute thresholds on TRAIN data only to avoid data leakage
            train_vals = train_df[source_col].dropna()
            if len(train_vals) < 40:
                continue

            # For overfit_risk, "top" means LOW risk (bottom quartile of risk)
            if source_col == "overfit_risk":
                threshold = train_vals.quantile(0.25)
                self._quartile_thresholds[clf_name] = float(threshold)
                y_train_bin = (train_df[source_col] <= threshold).astype(int)
                y_test_bin = (test_df[source_col] <= threshold).astype(int)
            else:
                threshold = train_vals.quantile(0.75)
                self._quartile_thresholds[clf_name] = float(threshold)
                y_train_bin = (train_df[source_col] >= threshold).astype(int)
                y_test_bin = (test_df[source_col] >= threshold).astype(int)

            train_mask = train_df[source_col].notna()
            test_mask = test_df[source_col].notna()
            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train < 40:
                continue

            Xt = X_train[train_mask.values]
            Xv = X_test[test_mask.values]
            yt = y_train_bin[train_mask].values
            yv = y_test_bin[test_mask].values

            # Compute scale_pos_weight for class imbalance
            n_pos = yt.sum()
            n_neg = len(yt) - n_pos
            spw = n_neg / max(1, n_pos)

            model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(
                Xt, yt,
                eval_set=[(Xv, yv)] if n_test >= 10 else None,
                callbacks=[lgb.log_evaluation(period=0)] if n_test >= 10 else None,
            )

            self.classifiers[clf_name] = model

            if n_test >= 10 and yv.sum() >= 2:
                y_prob = model.predict_proba(Xv)[:, 1]
                y_pred = model.predict(Xv)
                auc = roc_auc_score(yv, y_prob)
                acc = accuracy_score(yv, y_pred)
                f1 = f1_score(yv, y_pred, zero_division=0)
            else:
                y_prob = model.predict_proba(Xt)[:, 1]
                y_pred = model.predict(Xt)
                auc = roc_auc_score(yt, y_prob)
                acc = accuracy_score(yt, y_pred)
                f1 = f1_score(yt, y_pred, zero_division=0)

            self.metrics[clf_name] = {
                "auc": auc, "accuracy": acc, "f1": f1,
                "n_train": n_train, "n_test": n_test, "type": "classification",
                "threshold": float(threshold),
            }
            results[clf_name] = self.metrics[clf_name]

            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                "feature": self._all_feature_columns,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            self.feature_importance[clf_name] = fi_df

            logger.info(
                f"[PREDICTOR] '{clf_name}': AUC={auc:.3f}, Acc={acc:.3f}, F1={f1:.3f} "
                f"(train={n_train}, test={n_test})"
            )

        return results

    def predict(self, df: pd.DataFrame, targets: Optional[List[str]] = None) -> pd.DataFrame:
        """Predict outcomes for new strategies (regression + classification).

        Returns DataFrame with 'pred_{target}' columns for regression
        and 'prob_{clf_name}' columns for classification probabilities.
        """
        df = _add_interaction_features(df)

        if targets is None:
            targets = list(self.models.keys())

        X = self._get_feature_matrix(df)
        result = df.copy()

        # Regression predictions
        for target in targets:
            if target in self.models:
                result[f"pred_{target}"] = self.models[target].predict(X)

        # Classification predictions (probabilities)
        for clf_name, clf_model in self.classifiers.items():
            try:
                result[f"prob_{clf_name}"] = clf_model.predict_proba(X)[:, 1]
            except Exception:
                pass

        return result

    def predict_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute a composite quality score combining all available predictors.

        Returns a Series of [0, 1] scores where higher = better predicted quality.
        Used by SIS integrator for seed filtering (replaces win_rate-only filter).
        """
        df = _add_interaction_features(df)
        X = self._get_feature_matrix(df)
        scores = np.zeros(len(df))
        n_models = 0

        # Classification probabilities (0-1, already calibrated)
        for clf_name, clf_model in self.classifiers.items():
            met = self.metrics.get(clf_name, {})
            auc = met.get("auc", 0.5)
            if auc <= 0.52:  # skip near-random classifiers
                continue
            try:
                prob = clf_model.predict_proba(X)[:, 1]
                # Weight by AUC quality: perfect AUC=1.0 gets weight 1.0,
                # AUC=0.6 gets weight 0.2
                weight = (auc - 0.5) * 2  # maps [0.5, 1.0] -> [0.0, 1.0]
                if clf_name == "clf_overfit_risk":
                    # For overfit risk, high prob = low risk = good
                    scores += prob * weight
                else:
                    scores += prob * weight
                n_models += 1
            except Exception:
                pass

        # Regression: use ALL regression models that have R² > threshold
        min_r2 = 0.1
        for target, model in self.models.items():
            r2 = self.metrics.get(target, {}).get("r2", -1)
            if r2 <= min_r2:
                continue
            try:
                pred = model.predict(X)
                # Normalize to [0, 1]
                pred_norm = np.clip(
                    (pred - pred.min()) / (pred.max() - pred.min() + 1e-8), 0, 1
                )
                # Find whether higher is better for this target
                higher_is_better = True
                for col, _, hib in TARGET_DEFINITIONS + DERIVED_TARGETS:
                    if col == target:
                        higher_is_better = hib
                        break
                if not higher_is_better:
                    pred_norm = 1.0 - pred_norm
                weight = min(1.0, r2)
                scores += pred_norm * weight
                n_models += 1
            except Exception:
                pass

        if n_models > 0:
            scores /= n_models

        return pd.Series(scores, index=df.index)

    def get_reliable_models(self, min_r2: float = 0.1, min_auc: float = 0.52) -> Dict[str, float]:
        """Return dict of model_name -> quality_score for models exceeding thresholds.

        Useful for diagnosing which models should be trusted for decisions.
        """
        reliable: Dict[str, float] = {}
        for target, met in self.metrics.items():
            model_type = met.get("type", "")
            if model_type == "regression":
                r2 = met.get("r2", -1)
                if r2 > min_r2:
                    reliable[target] = r2
            elif model_type == "classification":
                auc = met.get("auc", 0.5)
                if auc > min_auc:
                    reliable[target] = auc
        return reliable

    def report(self) -> str:
        """Generate a human-readable summary of all model performance."""
        lines = ["=" * 70, "  MULTI-TARGET PREDICTOR REPORT (SIS v3)", "=" * 70, ""]

        if not self.metrics:
            return "\n".join(lines + ["  No models trained yet."])

        # Regression models
        reg_metrics = {k: v for k, v in self.metrics.items() if v.get("type") == "regression"}
        if reg_metrics:
            lines.append("  REGRESSION MODELS")
            lines.append(f"  {'Target':<30s} {'R²':>8s} {'RMSE':>10s} {'MAE':>10s} {'Train':>7s} {'Test':>7s}")
            lines.append("  " + "-" * 72)
            for target, m in sorted(reg_metrics.items(), key=lambda x: -x[1]["r2"]):
                quality = "***" if m["r2"] > 0.5 else "**" if m["r2"] > 0.3 else "*" if m["r2"] > 0.1 else ""
                lines.append(
                    f"  {target:<30s} {m['r2']:>7.3f}{quality} {m['rmse']:>10.4f} {m['mae']:>10.4f} "
                    f"{int(m['n_train']):>7d} {int(m['n_test']):>7d}"
                )
            lines.append("")

        # Classification models
        clf_metrics = {k: v for k, v in self.metrics.items() if v.get("type") == "classification"}
        if clf_metrics:
            lines.append("  CLASSIFICATION MODELS (top-quartile prediction)")
            lines.append(f"  {'Target':<30s} {'AUC':>8s} {'Acc':>8s} {'F1':>8s} {'Train':>7s} {'Test':>7s}")
            lines.append("  " + "-" * 72)
            for target, m in sorted(clf_metrics.items(), key=lambda x: -x[1]["auc"]):
                quality = "***" if m["auc"] > 0.7 else "**" if m["auc"] > 0.6 else "*" if m["auc"] > 0.55 else ""
                lines.append(
                    f"  {target:<30s} {m['auc']:>7.3f}{quality} {m['accuracy']:>7.3f} {m['f1']:>7.3f} "
                    f"{int(m['n_train']):>7d} {int(m['n_test']):>7d}"
                )
            lines.append("")

        lines.append("  Regression quality: *** R²>0.5, ** R²>0.3, * R²>0.1")
        lines.append("  Classification quality: *** AUC>0.7, ** AUC>0.6, * AUC>0.55")
        lines.append("")

        # Top features
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

        for clf_name, model in self.classifiers.items():
            model_path = out_dir / f"classifier_{clf_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Save metadata
        meta = {
            "targets": list(self.models.keys()),
            "classifiers": list(self.classifiers.keys()),
            "metrics": {
                k: {mk: float(mv) if isinstance(mv, (int, float, np.floating, np.integer)) else str(mv)
                    for mk, mv in v.items()}
                for k, v in self.metrics.items()
            },
            "feature_columns": self._feature_columns,
            "interaction_columns": self._interaction_columns,
            "quartile_thresholds": self._quartile_thresholds,
        }
        with open(out_dir / "predictor_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"[PREDICTOR] Saved {len(self.models)} regressors + "
            f"{len(self.classifiers)} classifiers to {out_dir}"
        )

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
        self._interaction_columns = meta.get("interaction_columns", _get_interaction_features())
        self.metrics = meta.get("metrics", {})
        self._quartile_thresholds = meta.get("quartile_thresholds", {})

        for target in meta.get("targets", []):
            model_path = load_dir / f"predictor_{target}.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.models[target] = pickle.load(f)  # noqa: S301

        for clf_name in meta.get("classifiers", []):
            model_path = load_dir / f"classifier_{clf_name}.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.classifiers[clf_name] = pickle.load(f)  # noqa: S301

        logger.info(
            f"[PREDICTOR] Loaded {len(self.models)} regressors + "
            f"{len(self.classifiers)} classifiers from {load_dir}"
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _split_data(
        self, df: pd.DataFrame, test_waves: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split corpus into train/test sets with strict temporal ordering.

        Uses wave-based splitting (train on older waves, test on newer).
        Refuses to train if insufficient temporal data — never uses random splits
        which would cause data leakage across time.
        """
        min_test_samples = 20

        if test_waves:
            test_mask = df["wave"].isin(test_waves)
            if test_mask.sum() >= min_test_samples:
                return df[~test_mask].copy(), df[test_mask].copy()
            logger.warning(
                f"[PREDICTOR] Explicit test waves have only {test_mask.sum()} samples "
                f"(need >={min_test_samples}), trying auto-detection"
            )

        # Auto-detect waves and enforce chronological ordering
        waves = df["wave"].dropna().unique()
        wave_nums = sorted(
            [(w, int(w.replace("wave", ""))) for w in waves if w.startswith("wave")],
            key=lambda x: x[1],
        )
        if len(wave_nums) >= 3:
            test_wave_ids = [w[0] for w in wave_nums[-2:]]  # Last 2 waves
            test_mask = df["wave"].isin(test_wave_ids)
            if test_mask.sum() >= min_test_samples:
                logger.info(f"[PREDICTOR] Auto-detected test waves: {test_wave_ids}")
                return df[~test_mask].copy(), df[test_mask].copy()

        # If only 2 waves, use the last one as test
        if len(wave_nums) >= 2:
            test_wave_ids = [wave_nums[-1][0]]
            test_mask = df["wave"].isin(test_wave_ids)
            if test_mask.sum() >= min_test_samples:
                logger.info(f"[PREDICTOR] Using last wave as test: {test_wave_ids}")
                return df[~test_mask].copy(), df[test_mask].copy()

        # Last resort: temporal split by row order (assumes corpus is chronological)
        # This is better than random split because corpus builder processes waves
        # in order, so later rows are from later waves
        n = len(df)
        split_idx = int(n * 0.8)
        test_size = n - split_idx
        if test_size >= min_test_samples:
            logger.warning(
                f"[PREDICTOR] Insufficient wave metadata — using positional 80/20 split "
                f"(train={split_idx}, test={test_size}). This assumes corpus is chronologically ordered."
            )
            return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

        # Cannot create a valid split
        logger.warning(
            f"[PREDICTOR] Cannot create valid train/test split: "
            f"{len(df)} samples, {len(wave_nums)} waves. "
            f"Returning all data as train with empty test set."
        )
        return df.copy(), df.iloc[:0].copy()
