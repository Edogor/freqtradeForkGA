#!/usr/bin/env python3
"""
GA Config Validator Hook — PreToolUse

Validates YAML configs written to genetic_algorithm/config/ through the same
versioned resolver used by the runtime before the file is created.

Reads JSON from stdin (hook input), checks if the tool is creating/editing
a YAML config in the queue directory, and validates constraints.

Exit codes:
  0 — OK (continue)
  2 — Blocking error (constraint violation)
"""

import json
import sys

import yaml

from genetic_algorithm.config.schema import resolve_config_data


def validate_ga_config(config: dict) -> list[str]:
    """Return canonical runtime constraint violations for an inline mapping."""

    try:
        return list(resolve_config_data(config).errors)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        return [str(exc)]


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
