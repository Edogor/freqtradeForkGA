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
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m genetic_algorithm``."""
    parser = argparse.ArgumentParser(
        prog="genetic_algorithm",
        description="Genetic Algorithm for FreqTrade Strategy Evolution",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
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
    p_run.add_argument("--state-db", help="Canonical V2 SQLite state path")
    p_run.add_argument("--artifact-root", help="Base directory for immutable attempt artifacts")

    # --- monitor ---
    p_mon = sub.add_parser("monitor", help="Live monitor for running experiments")
    p_mon.add_argument("--live", action="store_true", help="Show live log tails")
    p_mon.add_argument(
        "--filter",
        choices=["running", "queued", "completed", "failed", "all"],
        default="running",
        help="Filter experiments by status",
    )
    p_mon.add_argument("--tag", action="append", default=[], help="Filter by tag")
    p_mon.add_argument("--once", action="store_true", help="Print once and exit")
    p_mon.add_argument("--interval", type=int, default=5, help="Refresh interval (seconds)")
    p_mon.add_argument("--state-db", help="Canonical V2 SQLite state path")
    p_mon.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include imported non-executable legacy history",
    )

    # --- queue ---
    p_queue = sub.add_parser("queue", help="Experiment queue management")
    q_sub = p_queue.add_subparsers(dest="queue_cmd")

    q_add = q_sub.add_parser("add", help="Add configs to the queue")
    q_add.add_argument("configs", nargs="*", help="Config files to queue")
    q_add.add_argument("--dir", dest="config_dir", help="Queue all YAML files from a directory")
    q_add.add_argument("--tag", action="append", default=[], help="Tags for queued experiments")
    q_add.add_argument("--priority", type=int, default=50, help="Priority (higher = sooner)")
    q_add.add_argument("--state-db", help="Canonical V2 SQLite state path")
    q_add.add_argument("--artifact-root", help="Base directory for immutable attempt artifacts")

    q_start = q_sub.add_parser("start", help="Start the queue scheduler daemon")
    q_start.add_argument("--max-concurrent", type=int, default=5, help="Max parallel experiments")
    q_start.add_argument(
        "--persistent", action="store_true", help="Keep watching after queue drains"
    )
    q_start.add_argument("--state-db", help="Canonical V2 SQLite state path")

    q_sub.add_parser("stop", help="Stop the queue scheduler")
    q_status = q_sub.add_parser("status", help="Show queue status")
    q_status.add_argument("--state-db", help="Canonical V2 SQLite state path")

    # --- unattended automation ---
    p_auto = sub.add_parser(
        "automation",
        help="Guarded unattended Generic-Island wave automation",
    )
    a_sub = p_auto.add_subparsers(dest="automation_cmd")
    a_start = a_sub.add_parser(
        "start",
        help="Bootstrap or resume the guarded automation controller",
    )
    a_start.add_argument(
        "config",
        nargs="?",
        default="automation_island_v2",
        help="V2 config path or preset name",
    )
    a_start.add_argument("--state-db", help="Canonical V2 SQLite state path")
    a_start.add_argument("--automation-root", help="Automation artifact directory")
    a_start.add_argument(
        "--once",
        action="store_true",
        help="Run one reconciliation/controller tick and exit",
    )
    a_preflight = a_sub.add_parser(
        "preflight",
        help="Read-only validation of config, data, resources and policy",
    )
    a_preflight.add_argument(
        "config",
        nargs="?",
        default="automation_island_v2",
        help="V2 config path or preset name",
    )
    a_preflight.add_argument("--automation-root", help="Automation artifact directory")
    a_status = a_sub.add_parser("status", help="Show automation lineage status")
    a_status.add_argument("--state-db", help="Canonical V2 SQLite state path")
    a_status.add_argument("--automation-root", help="Automation artifact directory")
    a_stop = a_sub.add_parser("stop", help="Create the persistent kill switch")
    a_stop.add_argument("--automation-root", help="Automation artifact directory")
    a_unit = a_sub.add_parser(
        "service-unit",
        help="Render a systemd --user unit for crash-restart supervision",
    )
    a_unit.add_argument(
        "config",
        nargs="?",
        default="automation_island_v2",
        help="V2 config path or preset name",
    )
    a_unit.add_argument("--output", required=True, help="Destination .service file")
    a_unit.add_argument("--state-db", help="Canonical V2 SQLite state path")
    a_unit.add_argument("--automation-root", help="Automation artifact directory")

    # --- hardcore dual-lane campaign ---
    p_hardcore = sub.add_parser(
        "hardcore-campaign",
        help="Seven-day raw-score six-pair 15m/1h search campaign",
    )
    h_sub = p_hardcore.add_subparsers(dest="hardcore_cmd")
    for command_name, command_help in (
        ("start", "Start or resume the isolated seven-day campaign"),
        ("preflight", "Prove configs, isolation and pushed git commit"),
        ("canary", "Run exactly one reduced 15m and one reduced 1h evolution"),
    ):
        command = h_sub.add_parser(command_name, help=command_help)
        command.add_argument("--campaign-id", required=True)
        command.add_argument("--config-15m")
        command.add_argument("--config-1h")
        command.add_argument("--automation-root")
        command.add_argument("--state-db")
        command.add_argument(
            "--bootstrap-archive",
            help="Hash-covered v4 archive created by strict historical replay",
        )
        if command_name == "start":
            command.add_argument("--once", action="store_true")
        if command_name == "preflight":
            command.add_argument(
                "--canary",
                action="store_true",
                help="Validate the reduced dual-lane canary presets",
            )
    h_status = h_sub.add_parser("status", help="Read the hash-covered live status")
    h_status.add_argument("--automation-root", required=True)
    h_stop = h_sub.add_parser("stop", help="Request a generation-boundary stop")
    h_stop.add_argument("--automation-root", required=True)
    h_unit = h_sub.add_parser("service-unit", help="Render the isolated systemd --user service")
    h_unit.add_argument("--campaign-id", required=True)
    h_unit.add_argument("--config-15m", default="hardcore_multipair_15m_v1")
    h_unit.add_argument("--config-1h", default="hardcore_multipair_1h_v1")
    h_unit.add_argument("--automation-root", required=True)
    h_unit.add_argument("--bootstrap-archive")
    h_unit.add_argument("--output", required=True)
    h_bootstrap = h_sub.add_parser(
        "bootstrap-v3",
        help="Strictly replay historical v3 seeds into a v4 niche archive",
    )
    h_bootstrap.add_argument("--source-root", required=True)
    h_bootstrap.add_argument("--output", required=True)
    h_bootstrap.add_argument("--config-15m", default="hardcore_multipair_15m_v1")
    h_bootstrap.add_argument("--config-1h", default="hardcore_multipair_1h_v1")

    # --- evidence-driven V5 three-lane campaign ---
    p_hardcore_v5 = sub.add_parser(
        "hardcore-campaign-v5",
        help="Seven-day staged raw-score six-pair 15m/1h/4h search campaign",
    )
    h5_sub = p_hardcore_v5.add_subparsers(dest="hardcore_v5_cmd")
    for command_name, command_help in (
        ("start", "Start or resume the isolated V5 campaign"),
        ("canary", "Run exactly one reduced evolution per V5 lane"),
        ("preflight", "Validate the three V5 lane presets and launch inputs"),
    ):
        command = h5_sub.add_parser(command_name, help=command_help)
        command.add_argument("--campaign-id", required=True)
        command.add_argument("--config-15m")
        command.add_argument("--config-1h")
        command.add_argument("--config-4h")
        command.add_argument("--automation-root")
        command.add_argument("--state-db")
        if command_name == "start":
            command.add_argument("--once", action="store_true")
    h5_status = h5_sub.add_parser("status", help="Read V5 campaign state")
    h5_status.add_argument("--automation-root", required=True)
    h5_stop = h5_sub.add_parser("stop", help="Request a V5 generation-boundary stop")
    h5_stop.add_argument("--automation-root", required=True)
    h5_unit = h5_sub.add_parser("service-unit", help="Render the isolated V5 systemd user service")
    h5_unit.add_argument("--campaign-id", required=True)
    h5_unit.add_argument("--config-15m")
    h5_unit.add_argument("--config-1h")
    h5_unit.add_argument("--config-4h")
    h5_unit.add_argument("--automation-root", required=True)
    h5_unit.add_argument("--output", required=True)

    # --- experiment ---
    p_exp = sub.add_parser("experiment", help="Experiment management")
    e_sub = p_exp.add_subparsers(dest="exp_cmd")

    e_list = e_sub.add_parser("list", help="List experiments")
    e_list.add_argument("--status", help="Filter by status")
    e_list.add_argument("--tag", action="append", default=[], help="Filter by tag")
    e_list.add_argument("--limit", type=int, default=20)
    e_list.add_argument("--state-db", help="Canonical V2 SQLite state path")
    e_list.add_argument("--canonical-only", action="store_true")

    e_show = e_sub.add_parser("show", help="Show experiment details")
    e_show.add_argument("experiment_id", help="Experiment ID to show")
    e_show.add_argument("--state-db", help="Canonical V2 SQLite state path")
    e_show.add_argument("--canonical-only", action="store_true")

    e_compare = e_sub.add_parser("compare", help="Compare experiments")
    e_compare.add_argument("ids", nargs="+", help="Experiment IDs to compare")
    e_compare.add_argument("--state-db", help="Canonical V2 SQLite state path")
    e_compare.add_argument("--canonical-only", action="store_true")

    # --- data ---
    p_data = sub.add_parser("data", help="Data management")
    d_sub = p_data.add_subparsers(dest="data_cmd")

    d_report = d_sub.add_parser("report", help="Show disk usage report")
    d_report.add_argument("--state-db", help="Canonical V2 SQLite state path")

    d_clean = d_sub.add_parser("cleanup", help="Clean up old data")
    d_clean.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    d_clean.add_argument("--state-db", help="Canonical V2 SQLite state path")

    d_backfill = d_sub.add_parser(
        "backfill", help="Import legacy registry history into canonical SQLite"
    )
    d_backfill.add_argument(
        "--registry",
        default="genetic_algorithm/data/registry.json",
        help="Legacy registry.json snapshot",
    )
    d_backfill.add_argument("--state-db", help="Canonical V2 SQLite state path")

    d_export = d_sub.add_parser("export", help="Export a read-only V2 catalog snapshot")
    d_export.add_argument("--output", required=True, help="Destination JSON file")
    d_export.add_argument("--state-db", help="Canonical V2 SQLite state path")
    d_export.add_argument("--canonical-only", action="store_true")

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
        elif args.command == "automation":
            return _cmd_automation(args)
        elif args.command == "hardcore-campaign":
            return _cmd_hardcore_campaign(args)
        elif args.command == "hardcore-campaign-v5":
            return _cmd_hardcore_campaign_v5(args)
        elif args.command == "experiment":
            return _cmd_experiment(args)
        elif args.command == "data":
            return _cmd_data(args)
        elif args.command == "config":
            return _cmd_config(args)
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
    """Run one immutable standard evolution attempt through the V2 executor."""
    from genetic_algorithm.orchestration.attempt_state_v2 import (
        AttemptLifecycleStatus,
    )
    from genetic_algorithm.orchestration.runner_v2 import run_standard_attempt

    config_path = Path(args.config)
    if not config_path.exists():
        preset_path = Path("genetic_algorithm/config/presets") / f"{args.config}.yaml"
        if preset_path.exists():
            config_path = preset_path
        else:
            print(f"Config not found: {args.config}")
            return 1

    unsupported = []
    if args.resume:
        unsupported.append("--resume")
    if args.dashboard:
        unsupported.append("--dashboard")
    if args.tag:
        unsupported.append("--tag")
    if unsupported:
        print(
            "Canonical V2 execution does not support "
            + ", ".join(unsupported)
            + "; no legacy fallback was started."
        )
        return 2
    terminal = run_standard_attempt(
        config_path,
        experiment_name=args.name,
        state_path=args.state_db,
        artifact_base=args.artifact_root,
    )
    print(
        f"Attempt {terminal.attempt_id}: {terminal.status.value}\n"
        f"Artifacts: {terminal.artifact_root}"
    )
    return 0 if terminal.status == AttemptLifecycleStatus.SUCCEEDED else 1


def _cmd_monitor(args) -> int:
    """Show live monitor of running experiments."""
    from genetic_algorithm.orchestration.monitor import ExperimentMonitor

    statuses = (
        ["running", "queued", "completed", "failed", "cancelled", "unknown"]
        if args.filter == "all"
        else [args.filter]
    )
    monitor = ExperimentMonitor(
        experiment_id=getattr(args, "experiment", None),
        state_path=args.state_db,
        include_legacy=args.include_legacy,
        statuses=statuses,
        tags=args.tag,
    )

    if args.once:
        monitor.snapshot()
    else:
        monitor.run(interval=args.interval)

    return 0


def _cmd_queue(args) -> int:
    """Queue management commands."""
    from genetic_algorithm.orchestration.attempt_state_v2 import (
        AttemptLifecycleStatus,
        AttemptStateStoreV2,
    )
    from genetic_algorithm.orchestration.runner_v2 import (
        default_state_path,
        prepare_and_queue_standard_attempt,
    )

    if args.queue_cmd == "add":
        if args.tag:
            print("V2 attempt tags are not persisted yet; no queue entries were created.")
            return 2
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

        invalid = 0
        for config_path in config_files:
            path = Path(config_path)
            if not path.exists():
                print(f"  WARNING: {config_path} not found, skipping")
                invalid += 1
                continue
            try:
                prepared = prepare_and_queue_standard_attempt(
                    path,
                    experiment_name=path.stem,
                    state_path=args.state_db,
                    artifact_base=args.artifact_root,
                    priority=args.priority,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"  ERROR: {config_path}: {exc}")
                invalid += 1
                continue
            print(f"  Queued V2 attempt: {prepared.attempt_id}")
        return 1 if invalid else 0

    elif args.queue_cmd == "start":
        from genetic_algorithm.orchestration.scheduler import RunScheduler

        scheduler = RunScheduler(
            max_concurrent=args.max_concurrent,
            persistent=args.persistent,
            state_path=args.state_db,
        )
        scheduler.run()
        return 0

    elif args.queue_cmd == "stop":
        from genetic_algorithm.orchestration.scheduler import RunScheduler

        RunScheduler.stop_daemon()
        return 0

    elif args.queue_cmd == "status":
        store = AttemptStateStoreV2(args.state_db or default_state_path())
        summary = {
            status.value.lower(): store.count_attempts(statuses={status})
            for status in AttemptLifecycleStatus
        }
        summary["state_path"] = str(store.path)
        print(f"Queue status: {json.dumps(summary, indent=2)}")
        return 0

    else:
        print("Usage: python -m genetic_algorithm queue {add|start|stop|status}")
        return 1


def _cmd_automation(args) -> int:
    """Start, inspect, or stop guarded unattended wave search."""

    from genetic_algorithm.config.schema import load_config
    from genetic_algorithm.orchestration.automation_controller_v2 import (
        AutomationBootstrapReceiptV2,
        AutomationControllerV2,
        bootstrap_automation_wave,
        build_automation_preflight,
        default_automation_policy,
        render_systemd_user_unit,
    )
    from genetic_algorithm.orchestration.runner_v2 import (
        default_state_path,
        default_v2_root,
        repository_root,
    )
    from genetic_algorithm.orchestration.wave_state_v2 import WaveStateStoreV2

    repo_root = repository_root()
    automation_root = (
        Path(args.automation_root).resolve()
        if getattr(args, "automation_root", None)
        else (default_v2_root(repo_root) / "automation").resolve()
    )
    if args.automation_cmd == "stop":
        automation_root.mkdir(parents=True, exist_ok=True)
        kill_switch = automation_root / "STOP_AUTOMATION"
        kill_switch.write_text(
            "Guarded automation stopped by CLI operator.\n",
            encoding="utf-8",
        )
        print(f"Kill switch created: {kill_switch}")
        return 0

    if args.automation_cmd == "preflight":
        config_path = Path(args.config)
        if not config_path.exists():
            candidate = (
                repo_root / "genetic_algorithm" / "config" / "presets" / f"{args.config}.yaml"
            )
            if not candidate.is_file():
                print(f"Config not found: {args.config}")
                return 1
            config_path = candidate
        report = build_automation_preflight(
            config_path,
            automation_root=automation_root,
            repo_root=repo_root,
        )
        print(report.model_dump_json(indent=2))
        return 0 if report.ready else 2

    if args.automation_cmd == "service-unit":
        config_path = Path(args.config)
        if not config_path.exists():
            candidate = (
                repo_root / "genetic_algorithm" / "config" / "presets" / f"{args.config}.yaml"
            )
            if not candidate.is_file():
                print(f"Config not found: {args.config}")
                return 1
            config_path = candidate
        state_path = (
            Path(args.state_db).resolve() if args.state_db else default_state_path(repo_root)
        )
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render_systemd_user_unit(
                repo_root=repo_root,
                config_path=config_path,
                state_path=state_path,
                automation_root=automation_root,
            ),
            encoding="utf-8",
        )
        print(f"Systemd user unit written: {output}")
        return 0

    receipt_path = automation_root / "bootstrap_receipt.json"
    database = Path(args.state_db).resolve() if args.state_db else default_state_path()
    store = WaveStateStoreV2(database)

    if args.automation_cmd == "status":
        if not receipt_path.is_file():
            print(f"No automation bootstrap found below {automation_root}")
            return 1
        receipt = AutomationBootstrapReceiptV2.model_validate_json(receipt_path.read_bytes())
        waves = store.list_waves()
        by_id = {item.wave_id: item for item in waves}
        lineage = []
        current = by_id.get(receipt.intent.wave_id)
        while current is not None:
            lineage.append(current)
            current = next(
                (item for item in waves if item.parent_wave_id == current.wave_id),
                None,
            )
        print(
            json.dumps(
                {
                    "root_wave_id": receipt.intent.wave_id,
                    "policy_hash": receipt.intent.automation_policy_hash,
                    "kill_switch_present": (automation_root / "STOP_AUTOMATION").is_file(),
                    "waves": [
                        {
                            "wave_id": item.wave_id,
                            "parent_wave_id": item.parent_wave_id,
                            "status": item.status.value,
                            "updated_at": item.updated_at.isoformat(),
                        }
                        for item in lineage
                    ],
                },
                indent=2,
            )
        )
        return 0

    if args.automation_cmd != "start":
        print(
            "Usage: python -m genetic_algorithm automation "
            "{preflight|service-unit|start|status|stop}"
        )
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        candidate = repo_root / "genetic_algorithm" / "config" / "presets" / f"{args.config}.yaml"
        if not candidate.is_file():
            print(f"Config not found: {args.config}")
            return 1
        config_path = candidate
    config = load_config(config_path)
    policy = default_automation_policy(config, automation_root=automation_root)
    receipt = bootstrap_automation_wave(
        config_path,
        store=store,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
    )
    with AutomationControllerV2(
        store=store,
        root_wave_id=receipt.intent.wave_id,
        policy=policy,
        automation_root=automation_root,
        repo_root=repo_root,
    ) as controller:
        tick = controller.run_once() if args.once else controller.run_forever()
    print(tick.model_dump_json(indent=2))
    return 0 if tick.outcome in {"WAITING", "PROGRESSED"} else 2


def _cmd_hardcore_campaign(args) -> int:
    """Operate the isolated raw-multipair dual-lane search supervisor."""

    import hashlib

    from genetic_algorithm.orchestration.hardcore_backend_v1 import (
        V2HardcoreAttemptBackend,
    )
    from genetic_algorithm.orchestration.hardcore_campaign_v1 import (
        HardcoreCampaignControllerV1,
        HardcoreCampaignPreflightV1,
        build_hardcore_campaign_preflight,
        default_hardcore_campaign_policy,
        render_hardcore_systemd_user_unit,
        require_hardcore_campaign_preflight,
    )
    from genetic_algorithm.orchestration.runner_v2 import repository_root

    repo_root = repository_root()

    def resolve_config(value: str) -> Path:
        direct = Path(value)
        if direct.is_file():
            return direct.resolve()
        preset = repo_root / "genetic_algorithm/config/presets" / f"{value}.yaml"
        if not preset.is_file():
            raise FileNotFoundError(f"Hardcore config not found: {value}")
        return preset.resolve()

    command = args.hardcore_cmd
    if command == "bootstrap-v3":
        from genetic_algorithm.orchestration.hardcore_bootstrap_v4 import (
            build_strict_v4_bootstrap_archive,
        )

        archive = build_strict_v4_bootstrap_archive(
            source_root=args.source_root,
            output_path=args.output,
            repo_root=repo_root,
            config_15m=resolve_config(args.config_15m),
            config_1h=resolve_config(args.config_1h),
        )
        print(archive.model_dump_json(indent=2))
        return 0
    if command == "status":
        root = Path(args.automation_root).resolve()
        path = root / "campaign_status.json"
        checksum = path.with_name(path.name + ".sha256")
        if not path.is_file() or not checksum.is_file():
            print(f"No complete hardcore status found below {root}")
            return 1
        expected = checksum.read_text(encoding="ascii").strip()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            print(f"Hardcore status checksum mismatch: {path}")
            return 2
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
        return 0

    if command == "stop":
        root = Path(args.automation_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "STOP_HARDCORE_CAMPAIGN"
        if not marker.exists():
            marker.write_text(
                "Hardcore campaign stop requested by CLI operator.\n",
                encoding="utf-8",
            )
        print(f"Generation-boundary stop requested: {marker}")
        return 0

    if command not in {
        "start",
        "preflight",
        "canary",
        "service-unit",
        "bootstrap-v3",
    }:
        print(
            "Usage: python -m genetic_algorithm hardcore-campaign "
            "{bootstrap-v3|preflight|canary|service-unit|start|status|stop}"
        )
        return 1

    is_canary = command == "canary" or (
        command == "preflight" and bool(getattr(args, "canary", False))
    )
    default_15m = "hardcore_multipair_canary_15m_v1" if is_canary else "hardcore_multipair_15m_v1"
    default_1h = "hardcore_multipair_canary_1h_v1" if is_canary else "hardcore_multipair_1h_v1"
    config_15m = resolve_config(args.config_15m or default_15m)
    config_1h = resolve_config(args.config_1h or default_1h)
    default_root = (repo_root / "genetic_algorithm/data/v2/hardcore" / args.campaign_id).resolve()
    automation_root = Path(args.automation_root).resolve() if args.automation_root else default_root

    if command == "service-unit":
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render_hardcore_systemd_user_unit(
                repo_root=repo_root,
                automation_root=automation_root,
                config_15m=config_15m,
                config_1h=config_1h,
                campaign_id=args.campaign_id,
                bootstrap_archive_path=args.bootstrap_archive,
            ),
            encoding="utf-8",
        )
        print(f"Systemd user unit written: {output}")
        return 0

    state_path = (
        Path(args.state_db).resolve()
        if args.state_db
        else automation_root / "hardcore_state.sqlite3"
    )
    bootstrap_path = (
        Path(args.bootstrap_archive).resolve() if getattr(args, "bootstrap_archive", None) else None
    )
    bootstrap_hash = None
    if bootstrap_path is not None:
        checksum = bootstrap_path.with_name(bootstrap_path.name + ".sha256")
        if not checksum.is_file():
            raise RuntimeError("bootstrap archive checksum is missing")
        bootstrap_hash = checksum.read_text(encoding="ascii").strip()
    policy = default_hardcore_campaign_policy(
        campaign_id=args.campaign_id,
        automation_root=automation_root,
        config_15m=config_15m,
        config_1h=config_1h,
        canary=is_canary,
        bootstrap_archive_path=bootstrap_path,
        bootstrap_archive_sha256=bootstrap_hash,
    )
    if command == "preflight":
        report = build_hardcore_campaign_preflight(
            policy, repo_root=repo_root, state_path=state_path
        )
        print(report.model_dump_json(indent=2))
        return 0 if report.ready else 2

    report = require_hardcore_campaign_preflight(policy, repo_root=repo_root, state_path=state_path)
    automation_root.mkdir(parents=True, exist_ok=True)
    preflight_path = automation_root / "start_preflight.json"
    preflight_payload = (report.model_dump_json(indent=2) + "\n").encode("utf-8")
    preflight_checksum = preflight_path.with_name(preflight_path.name + ".sha256")
    if preflight_path.exists() or preflight_checksum.exists():
        if not preflight_path.is_file() or not preflight_checksum.is_file():
            raise RuntimeError("existing start preflight is incomplete")
        expected = preflight_checksum.read_text(encoding="ascii").strip()
        if hashlib.sha256(preflight_path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("existing start preflight checksum differs")
        existing = HardcoreCampaignPreflightV1.model_validate_json(preflight_path.read_bytes())
        stable_fields = (
            "ready",
            "campaign_id",
            "policy_hash",
            "repo_root",
            "automation_root",
            "state_path",
            "git_head",
            "git_upstream",
            "git_upstream_head",
            "config_hashes",
        )
        if any(getattr(existing, field) != getattr(report, field) for field in stable_fields):
            raise RuntimeError("existing start preflight differs from pushed launch proof")
    else:
        preflight_path.write_bytes(preflight_payload)
        preflight_checksum.write_text(
            hashlib.sha256(preflight_payload).hexdigest() + "\n",
            encoding="ascii",
        )
    with V2HardcoreAttemptBackend(
        state_path=state_path,
        automation_root=automation_root,
        repo_root=repo_root,
        python_executable=repo_root / ".venv/bin/python",
    ) as backend:
        controller = HardcoreCampaignControllerV1(policy=policy, backend=backend)
        tick = (
            controller.run_once()
            if bool(getattr(args, "once", False))
            else controller.run_forever()
        )
    print(tick.model_dump_json(indent=2))
    # STOPPED is a normal terminal state (deadline, operator stop, resource
    # guard, or the two-run canary contract). Only a blocked campaign is an
    # operational failure requiring intervention.
    return 2 if tick.outcome.value == "BLOCKED" else 0


def _cmd_hardcore_campaign_v5(args) -> int:
    """Operate the isolated V5 campaign without reusing V1 lane semantics."""

    import hashlib
    import subprocess
    import time

    from genetic_algorithm.config.schema import load_config
    from genetic_algorithm.orchestration.hardcore_backend_v5 import V2HardcoreAttemptBackendV5
    from genetic_algorithm.orchestration.hardcore_campaign_v5 import (
        HardcoreCampaignControllerV5,
        default_hardcore_campaign_policy_v5,
    )
    from genetic_algorithm.orchestration.runner_v2 import repository_root

    repo = repository_root()
    command = args.hardcore_v5_cmd
    if command == "status":
        path = Path(args.automation_root).resolve() / "campaign_state_v5.json"
        checksum = path.with_suffix(path.suffix + ".sha256")
        if not path.is_file() or not checksum.is_file():
            return 1
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != checksum.read_text(encoding="ascii").strip()
        ):
            return 2
        print(path.read_text(encoding="utf-8"))
        return 0
    if command == "stop":
        root = Path(args.automation_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / "STOP_HARDCORE_CAMPAIGN_V5").touch(exist_ok=True)
        return 0

    def resolve(value: str | None, default: str) -> Path:
        requested = Path(value or default)
        if requested.is_file():
            return requested.resolve()
        result = repo / "genetic_algorithm/config/presets" / f"{requested}.yaml"
        if not result.is_file():
            raise FileNotFoundError(f"V5 config not found: {requested}")
        return result.resolve()

    configs = {
        "15m": resolve(args.config_15m, "hardcore_multipair_v5_15m"),
        "1h": resolve(args.config_1h, "hardcore_multipair_v5_1h"),
        "4h": resolve(args.config_4h, "hardcore_multipair_v5_4h"),
    }
    configs_valid = all(
        load_config(path)["backtesting"]["timeframe"] == lane
        and load_config(path)["raw_multipair_score"]["policy_version"] == "raw-multipair-score-v5"
        for lane, path in configs.items()
    )
    if command == "service-unit":
        if not configs_valid:
            return 2
        root = Path(args.automation_root).resolve()
        output = Path(args.output).resolve()
        executable = repo / ".venv/bin/python"
        if not executable.is_file():
            raise RuntimeError("V5 systemd Python executable is missing")

        def quote(value: Path | str) -> str:
            text = str(value)
            if any(character in text for character in ("\n", "\r", "\x00")):
                raise ValueError("systemd argument contains a control character")
            return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

        command_line = " ".join(
            [
                quote(executable), "-m", "genetic_algorithm", "hardcore-campaign-v5", "start",
                "--campaign-id", quote(args.campaign_id), "--config-15m", quote(configs["15m"]),
                "--config-1h", quote(configs["1h"]), "--config-4h", quote(configs["4h"]),
                "--automation-root", quote(root),
            ]
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "[Unit]\n"
            "Description=Evidence-driven V5 six-pair GA campaign\n"
            "After=default.target\n\n"
            "[Service]\nType=simple\n"
            f"WorkingDirectory={repo}\nExecStart={command_line}\n"
            "Restart=on-failure\nRestartSec=30s\n"
            "TimeoutStopSec=infinity\nKillMode=mixed\nNoNewPrivileges=true\n\n"
            "[Install]\nWantedBy=default.target\n",
            encoding="utf-8",
        )
        print(f"Systemd user unit written: {output}")
        return 0
    clean = (
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False
        ).stdout.strip()
        == ""
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    )
    upstream = subprocess.run(
        ["git", "rev-parse", "@{u}"], cwd=repo, capture_output=True, text=True, check=False
    )
    pushed = head.returncode == upstream.returncode == 0 and head.stdout == upstream.stdout
    if command == "preflight":
        print(
            json.dumps(
                {
                    "ready": configs_valid and clean and pushed,
                    "configs_valid": configs_valid,
                    "clean": clean,
                    "pushed": pushed,
                },
                indent=2,
            )
        )
        return 0 if configs_valid and clean and pushed else 2
    if command not in {"start", "canary"} or not (configs_valid and clean and pushed):
        return 2
    root = (
        Path(args.automation_root).resolve()
        if args.automation_root
        else repo / "genetic_algorithm/data/v2/hardcore" / args.campaign_id
    )
    policy = default_hardcore_campaign_policy_v5(
        campaign_id=args.campaign_id,
        automation_root=root,
        config_15m=configs["15m"],
        config_1h=configs["1h"],
        config_4h=configs["4h"],
        canary=command == "canary",
    )
    state = Path(args.state_db).resolve() if args.state_db else root / "hardcore_v5_state.sqlite3"
    with V2HardcoreAttemptBackendV5(
        state_path=state,
        automation_root=root,
        repo_root=repo,
        python_executable=repo / ".venv/bin/python",
    ) as backend:
        controller = HardcoreCampaignControllerV5(policy=policy, backend=backend)
        while True:
            current = controller.tick()
            print(json.dumps(current, sort_keys=True))
            if current["lifecycle"] != "RUNNING" or getattr(args, "once", False):
                return 0 if current["lifecycle"] == "COMPLETED" else 2
            time.sleep(5)


def _cmd_experiment(args) -> int:
    """Experiment management commands."""
    if args.exp_cmd is None:
        print("Usage: python -m genetic_algorithm experiment {list|show|compare}")
        return 1

    from genetic_algorithm.orchestration.attempt_state_v2 import AttemptStateStoreV2
    from genetic_algorithm.orchestration.experiment_catalog_v2 import ExperimentCatalogV2
    from genetic_algorithm.orchestration.runner_v2 import default_state_path

    catalog = ExperimentCatalogV2(AttemptStateStoreV2(args.state_db or default_state_path()))
    include_legacy = not args.canonical_only

    if args.exp_cmd == "list":
        experiments = catalog.list_records(
            statuses=[args.status] if args.status else None,
            tags=args.tag if args.tag else None,
            include_legacy=include_legacy,
            limit=args.limit,
        )
        if not experiments:
            print("No experiments found.")
            return 0

        print(f"{'ID':<30} {'Status':<12} {'Fitness':<10} {'Created':<22} {'Tags'}")
        print("-" * 90)
        for exp in experiments:
            fit = f"{exp.best_score:.4f}" if exp.best_score is not None else "-"
            created = exp.created_at.isoformat()[:19] if exp.created_at else "-"
            tags = ", ".join(exp.tags)
            print(f"{exp.experiment_id:<30} {exp.status:<12} {fit:<10} {created:<22} {tags}")
        return 0

    elif args.exp_cmd == "show":
        exp = catalog.get_record(
            args.experiment_id,
            include_legacy=include_legacy,
        )
        if not exp:
            print(f"Experiment '{args.experiment_id}' not found.")
            return 1
        print(exp.model_dump_json(indent=2))
        return 0

    elif args.exp_cmd == "compare":
        experiments = [catalog.get_record(eid, include_legacy=include_legacy) for eid in args.ids]
        experiments = [e for e in experiments if e]
        if not experiments:
            print("No matching experiments found.")
            return 1
        print(f"{'Field':<25}", end="")
        for exp in experiments:
            print(f" {exp.experiment_id:<20}", end="")
        print()
        print("-" * (25 + 22 * len(experiments)))
        for key in [
            "source",
            "status",
            "best_score",
            "best_net_return",
            "best_candidate_id",
            "best_attempt_id",
            "best_net_return_basis",
            "attempt_ids",
            "worker_kinds",
        ]:
            print(f"{key:<25}", end="")
            for exp in experiments:
                val = getattr(exp, key, "-")
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

        lifecycle = DataLifecycle(dry_run=True, state_path=args.state_db)
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

        lifecycle = DataLifecycle(dry_run=args.dry_run, state_path=args.state_db)
        result = lifecycle.run()
        print(
            f"Archived: {result['archived']}, Deleted: {result['deleted']}"
            f"{' (dry-run)' if result.get('dry_run') else ''}"
        )
        return 0

    elif args.data_cmd == "backfill":
        from genetic_algorithm.orchestration.attempt_state_v2 import AttemptStateStoreV2
        from genetic_algorithm.orchestration.experiment_catalog_v2 import ExperimentCatalogV2
        from genetic_algorithm.orchestration.runner_v2 import default_state_path

        catalog = ExperimentCatalogV2(AttemptStateStoreV2(args.state_db or default_state_path()))
        receipt = catalog.import_legacy_registry(args.registry)
        state = "already imported" if receipt.already_imported else "imported"
        print(
            f"Legacy registry {state}: import_id={receipt.import_id}, "
            f"experiments={receipt.experiment_count}, sha256={receipt.source_sha256}"
        )
        return 0

    elif args.data_cmd == "export":
        from genetic_algorithm.orchestration.attempt_state_v2 import AttemptStateStoreV2
        from genetic_algorithm.orchestration.experiment_catalog_v2 import ExperimentCatalogV2
        from genetic_algorithm.orchestration.runner_v2 import default_state_path

        catalog = ExperimentCatalogV2(AttemptStateStoreV2(args.state_db or default_state_path()))
        exported = catalog.export_json(
            args.output,
            include_legacy=not args.canonical_only,
        )
        print(
            f"Exported {len(exported.attempts)} attempts and "
            f"{len(exported.experiments)} experiments to {Path(args.output).resolve()}"
        )
        return 0

    else:
        print("Usage: python -m genetic_algorithm data {report|cleanup|backfill|export}")
        return 1


def _cmd_serve(args) -> int:
    """Start the web dashboard server."""
    try:
        import uvicorn
        from genetic_algorithm.web.server import create_app  # noqa: F401

        print(f"Starting web dashboard on {args.host}:{args.port}")
        uvicorn.run(
            "genetic_algorithm.web.server:create_app", host=args.host, port=args.port, factory=True
        )
        return 0
    except ImportError as e:
        print(f"Web dependencies not installed: {e}")
        print("Install with: pip install uvicorn fastapi")
        return 1


def _cmd_config(args) -> int:
    """Config management commands."""
    if args.config_cmd == "validate":
        from genetic_algorithm.config.schema import resolve_config

        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: {args.config} not found")
            return 1

        try:
            resolution = resolve_config(config_path)
        except (OSError, ValueError) as e:
            print(f"ERROR: {e}")
            return 1

        errors = list(resolution.errors)
        warnings = list(resolution.warnings)

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
