"""
Unified CLI for the Genetic Algorithm system.

Replaces 37+ shell scripts with a single Python entry point::

    python -m genetic_algorithm run config.yaml
    python -m genetic_algorithm monitor
    python -m genetic_algorithm queue add config.yaml --tag batch1
    python -m genetic_algorithm queue start --max-concurrent 5
    python -m genetic_algorithm experiment list
    python -m genetic_algorithm data report
    python -m genetic_algorithm data cleanup --dry-run
    python -m genetic_algorithm serve
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m genetic_algorithm``."""
    parser = argparse.ArgumentParser(
        prog="genetic_algorithm",
        description="Genetic Algorithm for FreqTrade Strategy Evolution",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    p_run = sub.add_parser("run", help="Run a GA evolution")
    p_run.add_argument("config", help="Path to YAML config file or preset name")
    p_run.add_argument("--tag", action="append", default=[], help="Tags for this run (repeatable)")
    p_run.add_argument("--name", help="Experiment name (auto-generated if omitted)")
    p_run.add_argument("--no-monitor", action="store_true", help="Disable terminal monitor")
    p_run.add_argument("--dashboard", action="store_true", help="Enable web dashboard")
    p_run.add_argument("--resume", help="Resume from checkpoint directory")
    p_run.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")

    # --- monitor ---
    p_mon = sub.add_parser("monitor", help="Live monitor for running experiments")
    p_mon.add_argument("--live", action="store_true", help="Show live log tails")
    p_mon.add_argument("--filter", choices=["running", "queued", "completed", "failed", "all"],
                       default="running", help="Filter experiments by status")
    p_mon.add_argument("--tag", action="append", default=[], help="Filter by tag")
    p_mon.add_argument("--once", action="store_true", help="Print once and exit")
    p_mon.add_argument("--interval", type=int, default=5, help="Refresh interval (seconds)")

    # --- queue ---
    p_queue = sub.add_parser("queue", help="Experiment queue management")
    q_sub = p_queue.add_subparsers(dest="queue_cmd")

    q_add = q_sub.add_parser("add", help="Add configs to the queue")
    q_add.add_argument("configs", nargs="*", help="Config files to queue")
    q_add.add_argument("--dir", dest="config_dir", help="Queue all YAML files from a directory")
    q_add.add_argument("--tag", action="append", default=[], help="Tags for queued experiments")
    q_add.add_argument("--priority", type=int, default=50, help="Priority (lower = sooner)")

    q_start = q_sub.add_parser("start", help="Start the queue scheduler daemon")
    q_start.add_argument("--max-concurrent", type=int, default=5, help="Max parallel experiments")
    q_start.add_argument("--persistent", action="store_true", help="Keep watching after queue drains")

    q_sub.add_parser("stop", help="Stop the queue scheduler")
    q_sub.add_parser("status", help="Show queue status")

    # --- experiment ---
    p_exp = sub.add_parser("experiment", help="Experiment management")
    e_sub = p_exp.add_subparsers(dest="exp_cmd")

    e_list = e_sub.add_parser("list", help="List experiments")
    e_list.add_argument("--status", help="Filter by status")
    e_list.add_argument("--tag", action="append", default=[], help="Filter by tag")
    e_list.add_argument("--limit", type=int, default=20)

    e_show = e_sub.add_parser("show", help="Show experiment details")
    e_show.add_argument("experiment_id", help="Experiment ID to show")

    e_compare = e_sub.add_parser("compare", help="Compare experiments")
    e_compare.add_argument("ids", nargs="+", help="Experiment IDs to compare")

    # --- data ---
    p_data = sub.add_parser("data", help="Data management")
    d_sub = p_data.add_subparsers(dest="data_cmd")

    d_sub.add_parser("report", help="Show disk usage report")

    d_clean = d_sub.add_parser("cleanup", help="Clean up old data")
    d_clean.add_argument("--dry-run", action="store_true", help="Preview without deleting")

    d_sub.add_parser("backfill", help="Backfill registry from existing runs/ data")

    # --- config ---
    p_cfg = sub.add_parser("config", help="Config management and validation")
    c_sub = p_cfg.add_subparsers(dest="config_cmd")

    c_validate = c_sub.add_parser("validate", help="Validate a config file")
    c_validate.add_argument("config", help="Path to YAML config file")
    c_validate.add_argument("--strict", action="store_true", help="Treat warnings as errors")

    c_list = c_sub.add_parser("list", help="List available presets and benchmark configs")
    c_list.add_argument("--benchmarks", action="store_true", help="Include benchmark configs")
    c_list.add_argument("--all", action="store_true", help="Include legacy ga_config_* files")

    c_show = c_sub.add_parser("show", help="Show resolved config (defaults + preset + overrides)")
    c_show.add_argument("config", help="Path or preset name")

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start web dashboard server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    # --- hof (T1.1 + T1.6) -----------------------------------------------
    # Replay / stress-test Hall-of-Fame entries with arbitrary cost settings.
    p_hof = sub.add_parser("hof", help="Hall-of-Fame replay, stress, list, show")
    h_sub = p_hof.add_subparsers(dest="hof_cmd")

    h_list = h_sub.add_parser("list", help="List entries in a HoF file")
    h_list.add_argument("hof_file", help="Path to hall_of_fame.json")
    h_list.add_argument("--limit", type=int, default=10)

    h_show = h_sub.add_parser("show", help="Show a single HoF entry by id")
    h_show.add_argument("hof_file", help="Path to hall_of_fame.json")
    h_show.add_argument("entry_id", help="Entry id (e.g. hof_abc123def456)")

    h_replay = h_sub.add_parser("replay", help="Re-backtest HoF entries with custom cost settings")
    h_replay.add_argument("hof_file", help="Path to hall_of_fame.json")
    h_replay.add_argument("--config", required=True, help="Base config (path or preset name) — supplies pairs/timerange defaults")
    h_replay.add_argument("--fee", type=float, default=None, help="Override fee (e.g. 0.0015)")
    h_replay.add_argument("--slippage", type=float, default=None, help="Override slippage")
    h_replay.add_argument("--timerange", default=None, help="Override timerange (YYYYMMDD-YYYYMMDD)")
    h_replay.add_argument("--pair", action="append", default=None, help="Override pair list (repeatable)")
    h_replay.add_argument("--entry-id", action="append", default=None, help="Limit to specific entry id(s)")
    h_replay.add_argument("--top", type=int, default=None, help="Replay only the top-N entries")
    h_replay.add_argument("--min-fitness", type=float, default=None, help="Skip entries below this fitness")
    h_replay.add_argument("--out", default=None, help="Write CSV report to this path")

    h_stress = h_sub.add_parser("stress", help="Cost-sensitivity grid sweep (T1.6)")
    h_stress.add_argument("hof_file", help="Path to hall_of_fame.json")
    h_stress.add_argument("--config", required=True, help="Base config (path or preset name)")
    h_stress.add_argument("--fees", default="0.0005,0.001,0.0015,0.002",
                          help="Comma-separated fee grid")
    h_stress.add_argument("--slippages", default="0.0,0.0005,0.001,0.002",
                          help="Comma-separated slippage grid")
    h_stress.add_argument("--max-open-trades", default=None,
                          help="Comma-separated max_open_trades grid (optional)")
    h_stress.add_argument("--timerange", default=None)
    h_stress.add_argument("--pair", action="append", default=None)
    h_stress.add_argument("--entry-id", action="append", default=None)
    h_stress.add_argument("--top", type=int, default=None)
    h_stress.add_argument("--min-fitness", type=float, default=None)
    h_stress.add_argument("--out", default=None, help="Write CSV report to this path")

    # --- db (T1.3) -------------------------------------------------------
    # Read-only inspector for the experiments SQLite DB.
    p_db = sub.add_parser("db", help="Inspect the experiments database (T1.3)")
    db_sub = p_db.add_subparsers(dest="db_cmd")
    db_runs = db_sub.add_parser("runs", help="List runs")
    db_runs.add_argument("--status", default=None)
    db_runs.add_argument("--limit", type=int, default=25)
    db_show = db_sub.add_parser("show", help="Show run details")
    db_show.add_argument("run_id")
    db_show.add_argument("--top", type=int, default=10, help="Show top-N strategies")
    db_top = db_sub.add_parser("top", help="Top strategies across all runs")
    db_top.add_argument("--limit", type=int, default=25)
    db_path = db_sub.add_parser("path", help="Print absolute DB path")
    db_init = db_sub.add_parser("init", help="Create or upgrade the schema in-place")

    # --- holdout (T1.2) --------------------------------------------------
    p_h = sub.add_parser("holdout", help="Out-of-time holdout tooling (T1.2)")
    h_sub = p_h.add_subparsers(dest="holdout_cmd")
    h_check = h_sub.add_parser("check", help="Show how the lockbox would clip a config")
    h_check.add_argument("config", help="Path to YAML config file")
    h_run = h_sub.add_parser("run", help="Backtest HoF entries on the holdout timerange")
    h_run.add_argument("hof_file", help="Path to hall_of_fame.json")
    h_run.add_argument("--config", required=True, help="YAML config to drive the backtester")
    h_run.add_argument("--timerange", default=None,
                       help="Holdout timerange (YYYYMMDD-YYYYMMDD). Defaults to "
                            "the one computed from holdout.lock_start in the config.")
    h_run.add_argument("--top", type=int, default=10)
    h_run.add_argument("--out", default=None, help="Report output path (default: next to HoF)")
    h_run.add_argument("--csv", default=None, help="Optional CSV companion path")

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch
    try:
        if args.command == "run":
            return _cmd_run(args)
        elif args.command == "monitor":
            return _cmd_monitor(args)
        elif args.command == "queue":
            return _cmd_queue(args)
        elif args.command == "experiment":
            return _cmd_experiment(args)
        elif args.command == "data":
            return _cmd_data(args)
        elif args.command == "config":
            return _cmd_config(args)
        elif args.command == "serve":
            return _cmd_serve(args)
        elif args.command == "hof":
            return _cmd_hof(args)
        elif args.command == "db":
            return _cmd_db(args)
        elif args.command == "holdout":
            return _cmd_holdout(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        logger.error("Command failed: %s", e, exc_info=args.verbose)
        return 1


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _cmd_run(args) -> int:
    """Run a GA evolution using the existing engine (bridge to old code)."""
    from genetic_algorithm.config.schema import load_config
    from genetic_algorithm.orchestration.registry import ExperimentRegistry

    # Resolve config path — check presets dir if not found directly
    config_path = Path(args.config)
    if not config_path.exists():
        preset_path = Path("genetic_algorithm/config/presets") / f"{args.config}.yaml"
        if preset_path.exists():
            config_path = preset_path
        else:
            print(f"Config not found: {args.config}")
            return 1

    config = load_config(config_path)

    # Generate experiment name
    name = args.name
    if not name:
        import hashlib
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = hashlib.md5(str(config_path).encode()).hexdigest()[:4]
        name = f"run_{ts}_{short_hash}"

    # Register in registry
    registry = ExperimentRegistry()
    registry.register(
        name,
        config_path=str(config_path),
        tags=args.tag,
        ga_type=_detect_ga_type(config),
    )

    # Determine which engine to use
    ga_type = _detect_ga_type(config)
    logger.info("Starting experiment '%s' (type=%s)", name, ga_type)

    registry.start(
        name,
        pid=os.getpid(),
        log_path=f"genetic_algorithm/logs/{name}.log",
        data_dir=f"genetic_algorithm/data/runs/{name}",
        generations_total=config["genetic_algorithm"]["generations"],
    )

    try:
        if ga_type == "generic_island":
            from genetic_algorithm.core.generic_island_model import GenericIslandModelEvolution
            evo = GenericIslandModelEvolution(str(config_path))
            results = evo.evolve()
        elif ga_type == "island":
            from genetic_algorithm.core.island_model import IslandModelEvolution
            evo = IslandModelEvolution(str(config_path))
            results = evo.evolve()
        else:
            from genetic_algorithm.core.evolution import GeneticAlgorithm
            ga = GeneticAlgorithm(str(config_path))
            results = ga.evolve(resume_from=args.resume)

        best_fitness = 0.0
        best_profit = 0.0
        if results:
            best = results[0] if isinstance(results, list) else results
            if hasattr(best, "fitness"):
                best_fitness = best.fitness or 0.0
            if hasattr(best, "metrics"):
                best_profit = (best.metrics or {}).get("profit", 0.0)

        registry.complete(name, best_fitness=best_fitness, best_profit=best_profit)
        logger.info("Experiment '%s' completed (fitness=%.4f)", name, best_fitness)
        return 0

    except Exception as e:
        registry.fail(name, error=str(e))
        logger.error("Experiment '%s' failed: %s", name, e)
        raise


def _cmd_monitor(args) -> int:
    """Show live monitor of running experiments."""
    from genetic_algorithm.orchestration.monitor import ExperimentMonitor

    monitor = ExperimentMonitor(
        experiment_id=getattr(args, "experiment", None),
        show_completed=(args.filter in ("all", "completed")),
    )

    if args.once:
        monitor.snapshot()
    else:
        monitor.run(interval=args.interval)

    return 0


def _cmd_queue(args) -> int:
    """Queue management commands."""
    from genetic_algorithm.orchestration.registry import ExperimentRegistry

    if args.queue_cmd == "add":
        registry = ExperimentRegistry()
        config_files = list(args.configs or [])

        # Discover configs from directory
        if args.config_dir:
            dir_path = Path(args.config_dir)
            if not dir_path.is_dir():
                print(f"  ERROR: {args.config_dir} is not a directory")
                return 1
            config_files.extend(str(p) for p in sorted(dir_path.glob("*.yaml")))

        if not config_files:
            print("  No configs specified. Use positional args or --dir.")
            return 1

        for config_path in config_files:
            path = Path(config_path)
            if not path.exists():
                print(f"  WARNING: {config_path} not found, skipping")
                continue
            name = path.stem
            registry.register(name, config_path=str(path), tags=args.tag)
            print(f"  Queued: {name} (tags={args.tag})")
        return 0

    elif args.queue_cmd == "start":
        from genetic_algorithm.orchestration.scheduler import RunScheduler
        scheduler = RunScheduler(
            max_concurrent=args.max_concurrent,
            persistent=args.persistent,
        )
        scheduler.run()
        return 0

    elif args.queue_cmd == "stop":
        from genetic_algorithm.orchestration.scheduler import RunScheduler
        RunScheduler.stop_daemon()
        return 0

    elif args.queue_cmd == "status":
        registry = ExperimentRegistry()
        summary = registry.summary()
        print(f"Queue status: {json.dumps(summary, indent=2)}")
        return 0

    else:
        print("Usage: python -m genetic_algorithm queue {add|start|stop|status}")
        return 1


def _cmd_experiment(args) -> int:
    """Experiment management commands."""
    from genetic_algorithm.orchestration.registry import ExperimentRegistry

    registry = ExperimentRegistry()

    if args.exp_cmd == "list":
        experiments = registry.list(
            status=args.status,
            tags=args.tag if args.tag else None,
            limit=args.limit,
        )
        if not experiments:
            print("No experiments found.")
            return 0

        print(f"{'ID':<30} {'Status':<12} {'Fitness':<10} {'Created':<22} {'Tags'}")
        print("-" * 90)
        for exp in experiments:
            fit = f"{exp['best_fitness']:.4f}" if exp.get("best_fitness") else "-"
            created = exp.get("created_at", "")[:19]
            tags = ", ".join(exp.get("tags", []))
            print(f"{exp['experiment_id']:<30} {exp['status']:<12} "
                  f"{fit:<10} {created:<22} {tags}")
        return 0

    elif args.exp_cmd == "show":
        exp = registry.get(args.experiment_id)
        if not exp:
            print(f"Experiment '{args.experiment_id}' not found.")
            return 1
        print(json.dumps(exp, indent=2))
        return 0

    elif args.exp_cmd == "compare":
        experiments = [registry.get(eid) for eid in args.ids]
        experiments = [e for e in experiments if e]
        if not experiments:
            print("No matching experiments found.")
            return 1
        print(f"{'Field':<25}", end="")
        for exp in experiments:
            print(f" {exp['experiment_id']:<20}", end="")
        print()
        print("-" * (25 + 22 * len(experiments)))
        for key in ["status", "ga_type", "best_fitness", "best_profit",
                     "generation", "generations_total"]:
            print(f"{key:<25}", end="")
            for exp in experiments:
                val = exp.get(key, "-")
                if isinstance(val, float):
                    print(f" {val:<20.4f}", end="")
                else:
                    print(f" {str(val):<20}", end="")
            print()
        return 0

    else:
        print("Usage: python -m genetic_algorithm experiment {list|show|compare}")
        return 1


def _cmd_data(args) -> int:
    """Data management commands."""
    if args.data_cmd == "report":
        from genetic_algorithm.orchestration.lifecycle import DataLifecycle
        lifecycle = DataLifecycle(dry_run=True)
        report = lifecycle.report()

        print("Disk Usage Report")
        print("=" * 50)
        for cat, size in sorted(report["categories"].items()):
            if size > 0:
                from genetic_algorithm.orchestration.lifecycle import _human_bytes
                print(f"  {cat:<20} {_human_bytes(size):>10}")
        print(f"  {'TOTAL':<20} {report['total_human']:>10}")
        print(f"  Files: {report['file_count']}")
        print()
        print("Experiments:")
        for status, count in sorted(report["experiments"].items()):
            if count > 0:
                print(f"  {status:<12} {count}")
        return 0

    elif args.data_cmd == "cleanup":
        from genetic_algorithm.orchestration.lifecycle import DataLifecycle
        lifecycle = DataLifecycle(dry_run=args.dry_run)
        result = lifecycle.run()
        print(f"Archived: {result['archived']}, Deleted: {result['deleted']}"
              f"{' (dry-run)' if result.get('dry_run') else ''}")
        return 0

    elif args.data_cmd == "backfill":
        print("Backfill not yet implemented.")
        return 0

    else:
        print("Usage: python -m genetic_algorithm data {report|cleanup|backfill}")
        return 1


def _cmd_serve(args) -> int:
    """Start the web dashboard server."""
    try:
        import uvicorn
        from genetic_algorithm.web.server import create_app  # noqa: F401
        print(f"Starting web dashboard on {args.host}:{args.port}")
        uvicorn.run("genetic_algorithm.web.server:create_app", host=args.host,
                     port=args.port, factory=True)
        return 0
    except ImportError as e:
        print(f"Web dependencies not installed: {e}")
        print("Install with: pip install uvicorn fastapi")
        return 1


def _cmd_config(args) -> int:
    """Config management commands."""
    if args.config_cmd == "validate":
        from genetic_algorithm.config.schema import load_config, validate_config, resolve_preset, deep_merge, DEFAULTS
        import yaml as _yaml

        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: {args.config} not found")
            return 1

        try:
            with open(config_path) as fh:
                raw = _yaml.safe_load(fh) or {}
        except _yaml.YAMLError as e:
            print(f"ERROR: Invalid YAML: {e}")
            return 1

        # Resolve preset + merge defaults, then validate
        try:
            resolved = resolve_preset(raw.copy())
            config = deep_merge(DEFAULTS, resolved)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1

        errors, warnings = validate_config(config)

        # Also run preflight validator if available
        try:
            from genetic_algorithm.utils.config_validator import validate_ga_config
            extra_errors, extra_warnings = validate_ga_config(config)
            errors.extend(extra_errors)
            warnings.extend(extra_warnings)
        except ImportError:
            pass

        # Report
        if not errors and not warnings:
            print(f"✓ {args.config}: Valid (no errors, no warnings)")
            return 0

        if warnings:
            print(f"⚠ {args.config}: {len(warnings)} warning(s)")
            for w in warnings:
                print(f"  WARN: {w}")

        if errors:
            print(f"✗ {args.config}: {len(errors)} error(s)")
            for e in errors:
                print(f"  ERROR: {e}")
            return 1

        return 1 if args.strict else 0

    elif args.config_cmd == "list":
        presets_dir = Path("genetic_algorithm/config/presets")
        benchmark_dir = Path("genetic_algorithm/config/benchmark")
        legacy_dir = Path("genetic_algorithm/config")

        print("Presets (use with: python -m genetic_algorithm run <name>):")
        for p in sorted(presets_dir.glob("*.yaml")):
            print(f"  {p.stem:<20} {p}")

        if args.benchmarks or args.all:
            print("\nBenchmark configs:")
            for p in sorted(benchmark_dir.glob("*.yaml")):
                print(f"  {p.stem:<40} {p}")

        if args.all:
            print(f"\nLegacy configs ({legacy_dir}):")
            for p in sorted(legacy_dir.glob("ga_config_*.yaml")):
                print(f"  {p.name}")

        return 0

    elif args.config_cmd == "show":
        from genetic_algorithm.config.schema import load_config
        import yaml as _yaml

        config_path = Path(args.config)
        if not config_path.exists():
            # Try as preset name
            preset_path = Path("genetic_algorithm/config/presets") / f"{args.config}.yaml"
            if preset_path.exists():
                config_path = preset_path
            else:
                print(f"Config not found: {args.config}")
                return 1

        config = load_config(config_path)
        print(_yaml.dump(config, default_flow_style=False, sort_keys=False))
        return 0

    else:
        print("Usage: python -m genetic_algorithm config {validate|list|show}")
        return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_ga_type(config: dict) -> str:
    """Detect which GA engine to use from config."""
    if config.get("generic_island_model", {}).get("enabled"):
        return "generic_island"
    if config.get("island_model", {}).get("enabled"):
        return "island"
    return "standard"


def _cmd_hof(args) -> int:
    """T1.1 + T1.6 — Hall-of-Fame replay / stress / list / show."""
    from genetic_algorithm.tools import hof_replay as hr

    hof_path = Path(args.hof_file)

    if args.hof_cmd == "list":
        print(hr.list_hof_entries(hof_path, limit=args.limit))
        return 0

    if args.hof_cmd == "show":
        print(hr.show_hof_entry(hof_path, args.entry_id))
        return 0

    if args.hof_cmd == "replay":
        config = hr.load_base_config(args.config)
        results = hr.replay_hof_file(
            hof_path, config,
            fee=args.fee, slippage=args.slippage,
            timerange=args.timerange, pairs=args.pair,
            entry_ids=args.entry_id, top_n=args.top,
            min_fitness=args.min_fitness,
        )
        print(hr.render_replay_table(results))
        if args.out:
            hr.write_replay_csv(results, Path(args.out))
            print(f"\nCSV written to {args.out}")
        # Exit non-zero if any replay both failed AND the original was successful.
        bad = [r for r in results if not r.success]
        return 0 if not bad else 2

    if args.hof_cmd == "stress":
        config = hr.load_base_config(args.config)
        fees = [float(x) for x in args.fees.split(",") if x.strip()]
        slippages = [float(x) for x in args.slippages.split(",") if x.strip()]
        mot_grid = (
            [int(x) for x in args.max_open_trades.split(",") if x.strip()]
            if args.max_open_trades else None
        )
        reports = hr.stress_hof_file(
            hof_path, config,
            fees=fees, slippages=slippages,
            max_open_trades_grid=mot_grid,
            entry_ids=args.entry_id, top_n=args.top,
            min_fitness=args.min_fitness,
            timerange=args.timerange, pairs=args.pair,
        )
        print(hr.render_stress_table(reports))
        if args.out:
            hr.write_stress_csv(reports, Path(args.out))
            print(f"\nCSV written to {args.out}")
        return 0

    print("Usage: python -m genetic_algorithm hof {list|show|replay|stress} ...")
    return 1


def _cmd_db(args) -> int:
    """T1.3 — Inspect the experiments database."""
    from genetic_algorithm.data.experiment_db import ExperimentDB
    db = ExperimentDB()

    if args.db_cmd == "path":
        print(db.path.resolve())
        return 0

    if args.db_cmd == "init":
        with db.session():
            pass  # schema bootstrap happens inside connect()
        print(f"Initialised schema at {db.path}")
        return 0

    if args.db_cmd == "runs":
        rows = db.list_runs(status=args.status, limit=args.limit)
        if not rows:
            print("No runs found.")
            return 0
        print(f"{'run_id':<32}{'status':<10}{'fitness':>10}{'profit':>10}{'gens':>6}  started_at")
        print("-" * 92)
        for r in rows:
            from datetime import datetime
            ts = datetime.fromtimestamp(r['started_at']).strftime("%Y-%m-%d %H:%M")
            print(f"{(r['run_id'] or '')[:32]:<32}{(r['status'] or '')[:10]:<10}"
                  f"{(r['best_fitness'] or 0):>10.4f}"
                  f"{(r['best_profit'] or 0):>10.2f}"
                  f"{(r['generations_total'] or 0):>6}  {ts}")
        return 0

    if args.db_cmd == "show":
        run = db.get_run(args.run_id)
        if run is None:
            print(f"Run not found: {args.run_id}")
            return 1
        print(json.dumps(run, indent=2, default=str))
        print("\nTop strategies for this run:")
        strats = db.get_strategies_for_run(args.run_id, limit=args.top)
        for s in strats:
            print(f"  rank={s['rank']} fit={s['fitness']:.4f} profit={s['profit']:.2f}"
                  f" sharpe={s['sharpe']:.2f} dd={s['drawdown']:.2f}"
                  f" pinned={'yes' if s['code_pinned'] else 'no'}  id={s['strategy_id']}")
        return 0

    if args.db_cmd == "top":
        rows = db.top_strategies_overall(limit=args.limit)
        print(f"{'strategy_id':<36}{'run':<24}{'fit':>10}{'profit':>10}{'sharpe':>8}{'dd':>8}")
        print("-" * 96)
        for s in rows:
            print(f"{(s['strategy_id'] or '')[:36]:<36}{(s['run_id'] or '')[:24]:<24}"
                  f"{(s['fitness'] or 0):>10.4f}{(s['profit'] or 0):>10.2f}"
                  f"{(s['sharpe'] or 0):>8.2f}{(s['drawdown'] or 0):>8.2f}")
        return 0

    print("Usage: python -m genetic_algorithm db {runs|show|top|path|init}")
    return 1


def _cmd_holdout(args) -> int:
    """T1.2 — Out-of-time holdout tooling."""
    from genetic_algorithm.tools import holdout as holdout_tools
    import yaml as _yaml
    from pathlib import Path as _Path

    if args.holdout_cmd == "check":
        with open(args.config, "r") as f:
            cfg = _yaml.safe_load(f) or {}
        info = holdout_tools.check_lockbox(cfg)
        print(json.dumps(info, indent=2))
        return 0

    if args.holdout_cmd == "run":
        with open(args.config, "r") as f:
            cfg = _yaml.safe_load(f) or {}
        # Resolve the holdout timerange: CLI flag wins; else derive from lockbox.
        holdout_tr = args.timerange
        if not holdout_tr:
            info = holdout_tools.check_lockbox(cfg)
            holdout_tr = info.get("holdout_timerange")
        if not holdout_tr:
            print("ERROR: no --timerange and no holdout.lock_start in config.")
            return 2

        hof_file = _Path(args.hof_file)
        report = holdout_tools.run_holdout(hof_file, holdout_tr, cfg, top_n=args.top)
        out = _Path(args.out) if args.out else holdout_tools.default_report_path(hof_file)
        holdout_tools.write_report(report, out)
        print(f"Wrote holdout report → {out}")
        if args.csv:
            holdout_tools.write_report_csv(report, _Path(args.csv))
            print(f"Wrote CSV companion → {args.csv}")
        print(json.dumps(report.summary, indent=2))
        return 0

    print("Usage: python -m genetic_algorithm holdout {check|run}")
    return 1
