#!/usr/bin/env python3
"""
GA Config Validator Hook — PreToolUse

Validates YAML configs written to genetic_algorithm/config/queue/ against
established GA constraints before the file is created.

Reads JSON from stdin (hook input), checks if the tool is creating/editing
a YAML config in the queue directory, and validates constraints.

Exit codes:
  0 — OK (continue)
  2 — Blocking error (constraint violation)
"""

import json
import sys
import re

import yaml


def validate_ga_config(config: dict) -> list[str]:
    """Return list of constraint violation messages."""
    errors = []
    ga = config.get("genetic_algorithm", {})
    island = config.get("island_model", {})
    wf = config.get("walk_forward", {})
    fitness = config.get("fitness_weights", {})

    # Population size for standard GA
    pop_size = ga.get("population_size", 12)
    island_enabled = island.get("enabled", False)

    if not island_enabled and pop_size > 15:
        errors.append(
            f"population_size={pop_size} > 15 for standard GA. "
            "Causes 59-65% overfitting (E19, E30). Use 10-15."
        )

    # Island model population check
    if island_enabled:
        islands = island.get("islands", [])
        for isle in islands:
            isle_pop = isle.get("population_size", 0)
            if isle_pop < 60:
                errors.append(
                    f"Island '{isle.get('name', '?')}' has population_size={isle_pop} < 60. "
                    "Causes 62-100% overfitting (E24). Use >= 60 per island."
                )

    # Island model + walk-forward incompatibility
    wf_enabled = wf.get("enabled", False)
    if island_enabled and wf_enabled:
        errors.append(
            "Island model + walk-forward are incompatible "
            "(data partitioning conflict). Disable one."
        )

    # Tournament size
    tournament = ga.get("tournament_size", 3)
    if tournament < 3:
        errors.append(
            f"tournament_size={tournament} < 3. "
            "Values < 3 reduce to random search. Use >= 3."
        )
    if tournament > 6:
        errors.append(
            f"tournament_size={tournament} > 6. "
            "Causes premature convergence. Use 3-6."
        )

    # Elite size ratio
    elite = ga.get("elite_size", 1)
    if pop_size > 0 and elite / pop_size > 0.25:
        errors.append(
            f"elite_size={elite} is {elite/pop_size:.0%} of population_size={pop_size}. "
            "Should be ~10%. High elitism kills exploration."
        )

    # NSGA-II + fitness sharing
    mode = ga.get("selection_mode", "tournament")
    sharing = ga.get("fitness_sharing", False)
    if mode == "nsga2" and sharing:
        errors.append(
            "fitness_sharing=true with NSGA-II mode distorts Pareto front. "
            "Disable fitness_sharing when using nsga2."
        )

    return errors


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    # Extract tool info
    tool_name = hook_input.get("toolName", "")
    tool_input = hook_input.get("toolInput", {})

    # Only validate file creation/edit in config queue
    if tool_name not in ("create_file", "replace_string_in_file", "edit_notebook_file"):
        sys.exit(0)

    file_path = tool_input.get("filePath", "")
    if "genetic_algorithm/config/" not in file_path or not file_path.endswith(".yaml"):
        sys.exit(0)

    # For create_file, parse the content
    content = tool_input.get("content", "")
    if not content:
        sys.exit(0)

    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError as e:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Invalid YAML: {e}",
            }
        }
        json.dump(output, sys.stdout)
        sys.exit(0)

    if not isinstance(config, dict):
        sys.exit(0)

    errors = validate_ga_config(config)

    if errors:
        reason = "GA config constraint violations:\n" + "\n".join(f"  - {e}" for e in errors)
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": reason,
            }
        }
        json.dump(output, sys.stdout)
    # If no errors, exit 0 silently (allow)

    sys.exit(0)


if __name__ == "__main__":
    main()
