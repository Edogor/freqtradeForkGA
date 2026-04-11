#!/usr/bin/env python3
"""
GA Config Post-Write Validator — PostToolUse Hook

After a YAML config is written to genetic_algorithm/config/, this hook
re-reads the file from disk and checks for structural recommendations:
- Section ordering relative to convention
- Missing recommended keys with safe defaults
- Suspicious combinations (not hard errors, just advisories)

Returns a systemMessage with findings injected into agent context.
Non-blocking: never prevents file writes.

Exit codes:
  0 — OK (always)
"""

import json
import sys
import os

try:
    import yaml
except ImportError:
    sys.exit(0)

# Conventional section order for GA configs
SECTION_ORDER = [
    "experiment",
    "genetic_algorithm",
    "backtesting",
    "parallel_evaluation",
    "holdout_validation",
    "holdout_monitoring",
    "walk_forward",
    "monte_carlo",
    "deflated_sharpe",
    "sis",
    "island_model",
    "generic_island_model",
    "fitness_weights",
    "llm",
]

# Keys that should almost always be present
RECOMMENDED_KEYS = {
    "genetic_algorithm": ["population_size", "generations", "mutation_rate", "elite_size", "tournament_size"],
    "backtesting": ["pairs", "timerange"],
    "parallel_evaluation": ["enabled"],
}

# Suspicious key combinations (description, condition lambda)
ADVISORIES = [
    (
        "parallel_evaluation disabled — consider enabling for 3-5x speedup",
        lambda cfg: cfg.get("parallel_evaluation", {}).get("enabled") is False,
    ),
    (
        "enable_cache not set to true — backtest caching provides 2-5x speedup",
        lambda cfg: cfg.get("backtesting", {}).get("enable_cache") is not True,
    ),
    (
        "SIS enabled but sis_integrator not confirmed — check corpus freshness first",
        lambda cfg: cfg.get("sis", {}).get("enabled") is True,
    ),
    (
        "walk_forward enabled — confirm this is NOT an island model (incompatible)",
        lambda cfg: (
            cfg.get("walk_forward", {}).get("enabled") is True
            and (
                cfg.get("island_model", {}).get("enabled") is True
                or cfg.get("generic_island_model", {}).get("enabled") is True
            )
        ),
    ),
]


def check_config(config: dict) -> list[str]:
    notes = []

    # Check section ordering
    sections_present = [k for k in SECTION_ORDER if k in config]
    sections_actual = [k for k in config if k in SECTION_ORDER]
    if sections_actual != sections_present:
        notes.append(
            f"Section order differs from convention. "
            f"Recommended: {' → '.join(sections_present)}"
        )

    # Check for missing recommended keys
    for section, keys in RECOMMENDED_KEYS.items():
        if section in config:
            missing = [k for k in keys if k not in config[section]]
            if missing:
                notes.append(f"[{section}] missing keys: {', '.join(missing)}")

    # Advisories
    for message, condition in ADVISORIES:
        try:
            if condition(config):
                notes.append(message)
        except Exception:
            pass

    return notes


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = hook_input.get("toolName", "")
    tool_input = hook_input.get("toolInput", {})

    # Only trigger on file write tools
    if tool_name not in ("create_file", "replace_string_in_file", "multi_replace_string_in_file"):
        sys.exit(0)

    file_path = tool_input.get("filePath", "")
    if not file_path:
        # multi_replace_string_in_file has nested replacements
        replacements = tool_input.get("replacements", [])
        if replacements:
            file_path = replacements[0].get("filePath", "")

    if not file_path or "genetic_algorithm/config/" not in file_path or not file_path.endswith(".yaml"):
        sys.exit(0)

    # Read from disk (file already written)
    if not os.path.exists(file_path):
        sys.exit(0)

    try:
        with open(file_path) as f:
            content = f.read()
        config = yaml.safe_load(content)
    except Exception:
        sys.exit(0)

    if not isinstance(config, dict):
        sys.exit(0)

    notes = check_config(config)

    if notes:
        msg = f"[format-yaml] Config review for {os.path.basename(file_path)}:\n"
        msg += "\n".join(f"  • {n}" for n in notes)
        output = {"systemMessage": msg}
        json.dump(output, sys.stdout)

    sys.exit(0)


if __name__ == "__main__":
    main()
