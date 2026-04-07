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
    q_add.add_argument("configs", nargs="+", help="Config files to queue")
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

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start web dashboard server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

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
        elif args.command == "serve":
            return _cmd_serve(args)
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
            ga = GeneticAlgorithm(config, monitor_enabled=not args.no_monitor)
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
        for config_path in args.configs:
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
