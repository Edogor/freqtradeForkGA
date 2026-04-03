#!/usr/bin/env python3
"""
Strategy Intelligence System — CLI Entry Point

Runs the full SIS pipeline or individual phases:
  1. corpus   — Build strategy corpus from historical data
  2. predict  — Train multi-target prediction models
  3. cluster  — Discover strategy archetypes
  4. analyze  — Temporal weakness analysis + complementary pairs
  5. patterns — Strategy DNA pattern mining
  6. all      — Run the entire pipeline

Usage:
    python -m genetic_algorithm.intelligence.run_sis corpus
    python -m genetic_algorithm.intelligence.run_sis all
    python -m genetic_algorithm.intelligence.run_sis predict --test-waves wave29 wave30
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger("genetic_algorithm.intelligence")

DATA_DIR = Path("genetic_algorithm/data")
CORPUS_PATH = DATA_DIR / "strategy_corpus.parquet"


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_corpus(args) -> pd.DataFrame:
    """Build the strategy corpus."""
    from genetic_algorithm.intelligence.corpus import CorpusBuilder

    print("\n" + "=" * 70)
    print("  PHASE 1: BUILDING STRATEGY CORPUS")
    print("=" * 70)

    builder = CorpusBuilder(data_dir=Path(args.data_dir) if args.data_dir else None)
    df = builder.build()

    if df.empty:
        print("  ERROR: No strategies found!")
        return df

    # Enrich with monthly/per-pair data from backtest ZIPs
    results_dir_arg = getattr(args, "results_dir", None)
    results_dir = Path(results_dir_arg) if results_dir_arg else None
    skip = getattr(args, "skip_enrichment", False)
    if not skip:
        print("\n  Enriching with backtest trade-level data...")
        before = df["monthly_profit_mean"].notna().sum()
        df = builder.enrich_from_backtest_results(df, results_dir=results_dir)
        after = df["monthly_profit_mean"].notna().sum()
        print(f"  Enriched {after - before} strategies "
              f"({after}/{len(df)} now have monthly/per-pair data)")

    # Summary
    print(f"\n  Strategies collected: {len(df)}")
    print(f"  Features per strategy: {len(df.columns)}")
    print(f"  Waves represented: {sorted(df['wave'].dropna().unique())}")
    print(f"  Sources: {dict(df['source'].value_counts())}")
    print(f"  Timeframes: {dict(df['timeframe'].value_counts())}")

    if "fitness" in df.columns:
        fit = df["fitness"].dropna()
        print(f"\n  Fitness: mean={fit.mean():.3f}, median={fit.median():.3f}, "
              f"max={fit.max():.3f}, min={fit.min():.3f}")

    # Save
    out = Path(args.output) if args.output else CORPUS_PATH
    builder.save(df, out)
    print(f"\n  Saved to: {out}")
    return df


def cmd_predict(args) -> None:
    """Train multi-target prediction models."""
    from genetic_algorithm.intelligence.predictors import MultiTargetPredictor

    print("\n" + "=" * 70)
    print("  PHASE 2: TRAINING MULTI-TARGET PREDICTORS")
    print("=" * 70)

    df = _load_corpus(args)
    if df.empty:
        return

    predictor = MultiTargetPredictor()
    test_waves = args.test_waves if args.test_waves else None
    results = predictor.train(df, test_waves=test_waves)

    print(predictor.report())
    predictor.save()
    print("  Models saved to genetic_algorithm/ml/models/")


def cmd_cluster(args) -> None:
    """Discover strategy archetypes."""
    from genetic_algorithm.intelligence.archetypes import ArchetypeClassifier

    print("\n" + "=" * 70)
    print("  PHASE 3: DISCOVERING STRATEGY ARCHETYPES")
    print("=" * 70)

    df = _load_corpus(args)
    if df.empty:
        return

    classifier = ArchetypeClassifier(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    df = classifier.fit(df)

    print(classifier.report())
    classifier.save()

    # Save enriched corpus with archetype labels
    enriched_path = DATA_DIR / "strategy_corpus_clustered.parquet"
    df.to_parquet(enriched_path, index=False, engine="pyarrow")
    print(f"\n  Enriched corpus saved to: {enriched_path}")


def cmd_analyze(args) -> None:
    """Run temporal weakness analysis."""
    from genetic_algorithm.intelligence.temporal_analysis import TemporalAnalyzer

    print("\n" + "=" * 70)
    print("  PHASE 4: TEMPORAL PERFORMANCE ANALYSIS")
    print("=" * 70)

    df = _load_corpus(args)
    if df.empty:
        return

    analyzer = TemporalAnalyzer(df)
    analyzer.build_monthly_matrix()
    weaknesses = analyzer.analyze_weaknesses(top_n=args.top_n)

    print(analyzer.report())

    # Complementary pairs
    print("\n" + "=" * 70)
    print("  COMPLEMENTARY STRATEGY DISCOVERY")
    print("=" * 70)

    combos = analyzer.find_complementary_pairs(top_n=args.top_n)
    if combos:
        print(f"\n  Top {len(combos)} complementary pairs:")
        for i, c in enumerate(combos[:10], 1):
            print(
                f"    {i}. {c['strategy_a'][:12]} × {c['strategy_b'][:12]}  "
                f"score={c.get('composite_score', c.get('coverage_score', 0)):.3f}  "
                f"fitness=({c.get('fitness_a', 0):.3f}, {c.get('fitness_b', 0):.3f})"
            )
    else:
        print("  No complementary pairs found (insufficient data)")


def cmd_patterns(args) -> None:
    """Run strategy DNA pattern mining."""
    from genetic_algorithm.intelligence.pattern_mining import PatternMiner

    print("\n" + "=" * 70)
    print("  PHASE 5: STRATEGY DNA PATTERN MINING")
    print("=" * 70)

    df = _load_corpus(args)
    if df.empty:
        return

    miner = PatternMiner(df, top_pct=0.2, bottom_pct=0.2)
    summary = miner.mine()

    print(miner.report())
    miner.save()
    print("  Pattern mining results saved to genetic_algorithm/ml/models/pattern_miner.pkl")


def cmd_all(args) -> None:
    """Run the full SIS pipeline."""
    start = time.time()

    print("\n" + "█" * 70)
    print("  STRATEGY INTELLIGENCE SYSTEM — FULL PIPELINE")
    print("█" * 70)

    # Phase 1: Corpus
    df = cmd_corpus(args)
    if df.empty:
        print("\n  ABORTED: No data to analyze.")
        return

    # Phase 2: Predictors (needs corpus)
    args._corpus_df = df  # Pass in-memory to avoid re-loading
    cmd_predict(args)

    # Phase 3: Clustering
    cmd_cluster(args)

    # Phase 4: Temporal analysis
    cmd_analyze(args)

    # Phase 5: Pattern mining
    cmd_patterns(args)

    elapsed = time.time() - start
    print(f"\n{'█' * 70}")
    print(f"  COMPLETE — {elapsed:.1f}s total")
    print(f"{'█' * 70}")


def _load_corpus(args) -> pd.DataFrame:
    """Load the corpus from Parquet or passed DataFrame."""
    # Check if we have an in-memory corpus from `cmd_all`
    if hasattr(args, "_corpus_df") and args._corpus_df is not None:
        return args._corpus_df

    corpus_path = Path(args.output) if args.output else CORPUS_PATH
    if not corpus_path.exists():
        print(f"  ERROR: Corpus not found at {corpus_path}")
        print("  Run 'corpus' phase first: python -m genetic_algorithm.intelligence.run_sis corpus")
        return pd.DataFrame()

    df = pd.read_parquet(corpus_path)
    print(f"  Loaded corpus: {len(df)} strategies from {corpus_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Strategy Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--data-dir", default=None, help="Data directory override")
    parser.add_argument("--output", default=None, help="Corpus output path override")

    subparsers = parser.add_subparsers(dest="command", help="Phase to run")

    # corpus
    sub = subparsers.add_parser("corpus", help="Build strategy corpus")
    sub.add_argument("--results-dir", default=None,
                     help="Backtest results directory (default: user_data/backtest_results)")
    sub.add_argument("--skip-enrichment", action="store_true",
                     help="Skip enrichment from backtest ZIP files")

    # predict
    sub = subparsers.add_parser("predict", help="Train prediction models")
    sub.add_argument("--test-waves", nargs="+", help="Waves to hold out for testing")

    # cluster
    sub = subparsers.add_parser("cluster", help="Discover archetypes")
    sub.add_argument("--min-cluster-size", type=int, default=10)
    sub.add_argument("--min-samples", type=int, default=5)

    # analyze
    sub = subparsers.add_parser("analyze", help="Temporal analysis")
    sub.add_argument("--top-n", type=int, default=20, help="Number of top strategies to analyze")

    # patterns
    sub = subparsers.add_parser("patterns", help="Pattern mining")

    # all
    sub = subparsers.add_parser("all", help="Run full pipeline")
    sub.add_argument("--test-waves", nargs="+", help="Waves to hold out for testing")
    sub.add_argument("--min-cluster-size", type=int, default=10)
    sub.add_argument("--min-samples", type=int, default=5)
    sub.add_argument("--top-n", type=int, default=20)
    sub.add_argument("--results-dir", default=None,
                     help="Backtest results directory (default: user_data/backtest_results)")
    sub.add_argument("--skip-enrichment", action="store_true",
                     help="Skip enrichment from backtest ZIP files")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    commands = {
        "corpus": cmd_corpus,
        "predict": cmd_predict,
        "cluster": cmd_cluster,
        "analyze": cmd_analyze,
        "patterns": cmd_patterns,
        "all": cmd_all,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
