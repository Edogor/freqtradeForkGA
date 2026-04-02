"""
Strategy Corpus Builder

Scans all historical evolution data (hall-of-fame files, generation snapshots)
and builds a unified DataFrame/Parquet dataset for analysis and ML training.

Each row is a unique evaluated strategy with:
  - Structural features (65-dim from surrogate feature extraction)
  - Performance metrics (fitness, profit, drawdown, win_rate, etc.)
  - Behavioral features (monthly profits, per-pair profits, trade frequency)
  - Metadata (run_id, generation, wave, timeframe, source)

Usage:
    from genetic_algorithm.intelligence.corpus import CorpusBuilder
    builder = CorpusBuilder()
    df = builder.build()
    builder.save()  # -> genetic_algorithm/data/strategy_corpus.parquet
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from genetic_algorithm.evaluation.surrogate import extract_features, _INDICATOR_TYPES, _OPERATOR_TYPES
from genetic_algorithm.core.strategy_gene import StrategyGene

logger = logging.getLogger(__name__)

DATA_DIR = Path("genetic_algorithm/data")
CORPUS_PATH = DATA_DIR / "strategy_corpus.parquet"

# Feature column names matching surrogate.extract_features() order
SURROGATE_FEATURE_NAMES = (
    [f"ind_{t}" for t in _INDICATOR_TYPES]           # 26: indicator one-hot
    + ["n_indicators", "n_entry_conds", "n_exit_conds", "n_short_conds"]  # 4: counts
    + [f"entry_op_{op}" for op in _OPERATOR_TYPES]    # 8: entry operator dist
    + [f"exit_op_{op}" for op in _OPERATOR_TYPES]     # 8: exit operator dist
    + ["period_mean", "period_max", "period_min",
       "weight_mean", "weight_max", "weight_min"]     # 6: param stats
    + ["stoploss", "roi_max", "roi_min",
       "max_open_trades", "trailing_stop"]            # 5: risk params
    + ["threshold_mean", "threshold_max",
       "threshold_min", "threshold_unique"]            # 4: threshold stats
    + ["logic_and_count", "logic_or_count"]            # 2: logic
    + ["n_informative_tf", "can_short"]                # 2: MTF
)
assert len(SURROGATE_FEATURE_NAMES) == 65

# Metric columns we extract from individual.metrics
METRIC_COLUMNS = [
    "fitness", "raw_fitness", "profit", "sharpe_ratio", "sortino_ratio",
    "profit_factor", "max_drawdown", "win_rate", "num_trades",
    "train_fitness", "val_fitness", "pair_generalization_ratio",
    "val_profit", "val_sharpe", "val_trades", "val_max_drawdown", "val_win_rate",
    "holdout_fitness", "holdout_degradation", "holdout_profit", "holdout_trades",
    "complexity", "monthly_return_std", "positive_months_ratio",
    "pair_profit_std", "max_consecutive_losses", "max_drawdown_duration_days",
]

# Metadata columns
META_COLUMNS = [
    "fingerprint", "run_id", "generation", "individual_id",
    "source", "timeframe", "origin", "training_pairs", "validation_pairs",
]


class CorpusBuilder:
    """Builds a unified strategy corpus from all historical GA data."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._records: List[Dict[str, Any]] = []
        self._seen_fingerprints: set = set()

    def build(self) -> pd.DataFrame:
        """Scan all data sources and build the corpus DataFrame.

        Returns:
            DataFrame with one row per unique evaluated strategy.
        """
        self._records.clear()
        self._seen_fingerprints.clear()

        # 1. Scan all hall_of_fame files
        hof_count = self._scan_hall_of_fame_files()
        logger.info(f"[CORPUS] Collected {hof_count} strategies from hall-of-fame files")

        # 2. Scan all generation snapshot files
        gen_count = self._scan_generation_snapshots()
        logger.info(f"[CORPUS] Collected {gen_count} strategies from generation snapshots")

        logger.info(
            f"[CORPUS] Total unique strategies: {len(self._records)} "
            f"(from {hof_count} HoF + {gen_count} gen entries, "
            f"{hof_count + gen_count - len(self._records)} duplicates removed)"
        )

        if not self._records:
            logger.warning("[CORPUS] No strategies found!")
            return pd.DataFrame()

        df = pd.DataFrame(self._records)

        # Ensure correct dtypes
        for col in SURROGATE_FEATURE_NAMES + METRIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def save(self, df: Optional[pd.DataFrame] = None, path: Optional[Path] = None):
        """Build (if needed) and save the corpus to Parquet.

        Args:
            df: Pre-built DataFrame, or None to call build().
            path: Output path, defaults to CORPUS_PATH.
        """
        if df is None:
            df = self.build()
        out = Path(path) if path else CORPUS_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, engine="pyarrow")
        logger.info(f"[CORPUS] Saved {len(df)} strategies to {out} ({out.stat().st_size / 1024:.0f} KB)")
        return df

    # ── Data source scanners ──────────────────────────────────────

    def _scan_hall_of_fame_files(self) -> int:
        """Scan all hall_of_fame.json files under data_dir."""
        count = 0
        hof_files = sorted(self.data_dir.rglob("hall_of_fame.json"))
        for hof_path in hof_files:
            try:
                with open(hof_path) as f:
                    data = json.load(f)
                entries = data.get("entries", [])
                # Infer run_id from directory name
                run_id = self._infer_run_id(hof_path)
                for entry in entries:
                    if self._add_hof_entry(entry, run_id):
                        count += 1
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"[CORPUS] Skipping {hof_path}: {e}")
        return count

    def _scan_generation_snapshots(self) -> int:
        """Scan all gen_NNNN.json files under data_dir/runs/."""
        count = 0
        runs_dir = self.data_dir / "runs"
        if not runs_dir.exists():
            return 0
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            for gen_file in sorted(run_dir.glob("gen_*.json")):
                try:
                    with open(gen_file) as f:
                        data = json.load(f)
                    generation = data.get("generation", 0)
                    for ind_data in data.get("individuals", []):
                        if self._add_individual(ind_data, run_id, generation, source="gen_snapshot"):
                            count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"[CORPUS] Skipping {gen_file}: {e}")
        return count

    # ── Record extraction ─────────────────────────────────────────

    def _add_hof_entry(self, entry: Dict, run_id: str) -> bool:
        """Extract a record from a hall-of-fame entry. Returns True if added."""
        gene_dict = entry.get("strategy_gene", {})
        metrics = entry.get("metrics", {})

        fp = self._gene_fingerprint(gene_dict)
        if fp in self._seen_fingerprints:
            return False
        self._seen_fingerprints.add(fp)

        record = self._build_record(
            gene_dict=gene_dict,
            fitness=entry.get("fitness"),
            raw_fitness=entry.get("fitness"),  # HoF doesn't store raw_fitness separately
            metrics=metrics,
            run_id=run_id or entry.get("run_id", "unknown"),
            generation=entry.get("generation_found", 0),
            individual_id=entry.get("individual_id", 0),
            source="hall_of_fame",
        )
        if record:
            self._records.append(record)
            return True
        return False

    def _add_individual(self, ind_data: Dict, run_id: str, generation: int, source: str) -> bool:
        """Extract a record from a gen-snapshot individual. Returns True if added."""
        gene_dict = ind_data.get("strategy_gene", {})
        metrics = ind_data.get("metrics", {})

        fp = self._gene_fingerprint(gene_dict)
        if fp in self._seen_fingerprints:
            return False
        self._seen_fingerprints.add(fp)

        # Skip unevaluated or errored individuals
        if not ind_data.get("evaluated", False):
            return False
        if metrics.get("error"):
            return False

        record = self._build_record(
            gene_dict=gene_dict,
            fitness=ind_data.get("fitness"),
            raw_fitness=ind_data.get("raw_fitness"),
            metrics=metrics,
            run_id=run_id,
            generation=generation,
            individual_id=ind_data.get("id", ""),
            source=source,
        )
        if record:
            self._records.append(record)
            return True
        return False

    def _build_record(
        self,
        gene_dict: Dict,
        fitness: Optional[float],
        raw_fitness: Optional[float],
        metrics: Dict,
        run_id: str,
        generation: int,
        individual_id: Any,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        """Build a flat record dict from a strategy gene + metrics.

        Returns None if the gene can't be parsed.
        """
        try:
            gene = StrategyGene.from_dict(gene_dict)
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"[CORPUS] Could not parse gene: {e}")
            # Still try to extract features from dict directly
            try:
                features = self._extract_features_from_dict(gene_dict)
            except Exception:
                return None
        else:
            features = extract_features(gene)

        record: Dict[str, Any] = {}

        # Surrogate structural features (65 columns)
        for name, val in zip(SURROGATE_FEATURE_NAMES, features):
            record[name] = val

        # Performance metrics
        record["fitness"] = fitness
        record["raw_fitness"] = raw_fitness or fitness
        for col in METRIC_COLUMNS:
            if col not in ("fitness", "raw_fitness"):
                record[col] = metrics.get(col)

        # Behavioral features (variable-length → summary stats)
        monthly = metrics.get("monthly_profits", [])
        if monthly and isinstance(monthly, list):
            record["monthly_profit_mean"] = np.mean(monthly)
            record["monthly_profit_std"] = np.std(monthly)
            record["monthly_profit_min"] = np.min(monthly)
            record["monthly_profit_max"] = np.max(monthly)
            record["n_positive_months"] = sum(1 for m in monthly if m > 0)
            record["n_negative_months"] = sum(1 for m in monthly if m < 0)
            record["n_months_total"] = len(monthly)
        else:
            for k in ["monthly_profit_mean", "monthly_profit_std",
                       "monthly_profit_min", "monthly_profit_max",
                       "n_positive_months", "n_negative_months", "n_months_total"]:
                record[k] = None

        per_pair = metrics.get("per_pair_profit", {})
        if per_pair and isinstance(per_pair, dict):
            pair_vals = list(per_pair.values())
            record["per_pair_profit_mean"] = np.mean(pair_vals)
            record["per_pair_profit_std"] = np.std(pair_vals)
            record["per_pair_profit_min"] = np.min(pair_vals)
            record["per_pair_profit_max"] = np.max(pair_vals)
            record["n_profitable_pairs"] = sum(1 for v in pair_vals if v > 0)
            record["n_pairs"] = len(pair_vals)
        else:
            for k in ["per_pair_profit_mean", "per_pair_profit_std",
                       "per_pair_profit_min", "per_pair_profit_max",
                       "n_profitable_pairs", "n_pairs"]:
                record[k] = None

        # Metadata
        record["fingerprint"] = self._gene_fingerprint(gene_dict)
        record["run_id"] = run_id
        record["generation"] = generation
        record["individual_id"] = str(individual_id)
        record["source"] = source
        record["timeframe"] = gene_dict.get("timeframe", "5m")
        record["origin"] = metrics.get("origin", "unknown")
        record["training_pairs"] = metrics.get("training_pairs", "")
        record["validation_pairs"] = metrics.get("validation_pairs", "")

        # Derive wave from run_id
        record["wave"] = self._extract_wave(run_id)

        return record

    # ── Helpers ────────────────────────────────────────────────────

    def _extract_features_from_dict(self, gene_dict: Dict) -> List[float]:
        """Fallback feature extraction directly from gene dict when from_dict() fails."""
        features: List[float] = []
        indicators = gene_dict.get("indicators", [])
        entry_conds = gene_dict.get("entry_conditions", [])
        exit_conds = gene_dict.get("exit_conditions", [])
        short_entry = gene_dict.get("short_entry_conditions", [])
        short_exit = gene_dict.get("short_exit_conditions", [])

        ind_types = {i.get("type", "") for i in indicators}
        for t in _INDICATOR_TYPES:
            features.append(1.0 if t in ind_types else 0.0)

        features.append(float(len(indicators)))
        features.append(float(len(entry_conds)))
        features.append(float(len(exit_conds)))
        features.append(float(len(short_entry) + len(short_exit)))

        entry_ops = [c.get("operator", "") for c in entry_conds]
        for op in _OPERATOR_TYPES:
            features.append(float(entry_ops.count(op)))

        exit_ops = [c.get("operator", "") for c in exit_conds]
        for op in _OPERATOR_TYPES:
            features.append(float(exit_ops.count(op)))

        periods = []
        for ind in indicators:
            params = ind.get("parameters", {})
            p = params.get("period", params.get("fast_period", 0))
            if isinstance(p, (int, float)):
                periods.append(float(p))
        features.append(sum(periods) / max(1, len(periods)))
        features.append(max(periods) if periods else 0.0)
        features.append(min(periods) if periods else 0.0)

        weights = [ind.get("weight", 1.0) for ind in indicators]
        features.append(sum(weights) / max(1, len(weights)))
        features.append(max(weights) if weights else 0.0)
        features.append(min(weights) if weights else 0.0)

        features.append(float(gene_dict.get("stoploss", -0.1)))
        roi_values = list(gene_dict.get("minimal_roi", {}).values())
        features.append(max(roi_values) if roi_values else 0.0)
        features.append(min(roi_values) if roi_values else 0.0)
        features.append(float(gene_dict.get("max_open_trades", 3)))
        features.append(1.0 if gene_dict.get("trailing_stop", False) else 0.0)

        all_thresholds = [c.get("threshold", 0) for c in entry_conds + exit_conds]
        if all_thresholds:
            features.append(sum(all_thresholds) / len(all_thresholds))
            features.append(max(all_thresholds))
            features.append(min(all_thresholds))
            features.append(float(len(set(all_thresholds))))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        and_count = sum(1 for c in entry_conds if c.get("logic", "AND") == "AND")
        or_count = sum(1 for c in entry_conds if c.get("logic", "AND") == "OR")
        features.append(float(and_count))
        features.append(float(or_count))

        features.append(float(len(gene_dict.get("informative_timeframes", []))))
        features.append(1.0 if gene_dict.get("can_short", False) else 0.0)

        return features

    @staticmethod
    def _gene_fingerprint(gene_dict: Dict) -> str:
        """Compute a stable fingerprint for deduplication.

        Excludes generation/individual_id so identical gene structures get same hash.
        """
        key_fields = {
            "indicators": gene_dict.get("indicators", []),
            "entry_conditions": gene_dict.get("entry_conditions", []),
            "exit_conditions": gene_dict.get("exit_conditions", []),
            "short_entry_conditions": gene_dict.get("short_entry_conditions", []),
            "short_exit_conditions": gene_dict.get("short_exit_conditions", []),
            "timeframe": gene_dict.get("timeframe", "5m"),
            "stoploss": gene_dict.get("stoploss"),
            "minimal_roi": gene_dict.get("minimal_roi"),
            "trailing_stop": gene_dict.get("trailing_stop"),
        }
        raw = json.dumps(key_fields, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _extract_wave(run_id: str) -> str:
        """Extract wave identifier from run_id (e.g., 'wave30_A_rank_5m_ring' -> 'wave30')."""
        if run_id.startswith("wave"):
            parts = run_id.split("_")
            return parts[0]  # 'wave30'
        return ""

    @staticmethod
    def _infer_run_id(hof_path: Path) -> str:
        """Infer run_id from hall_of_fame directory name."""
        parent = hof_path.parent.name
        if parent == "hall_of_fame":
            return "master"
        if parent.startswith("hall_of_fame_"):
            return parent.replace("hall_of_fame_", "")
        return parent
