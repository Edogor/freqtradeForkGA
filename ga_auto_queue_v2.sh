#!/usr/bin/env bash
# Compatibility shim: the queue source of truth is SQLite/WAL, not files/PIDs.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${REPO_DIR}/.venv/bin/python"
MAX_CONCURRENT=5
PERSISTENT=false
MODE="start"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --status) MODE="status" ;;
        --stop) MODE="stop" ;;
        --persistent) PERSISTENT=true ;;
        --max)
            shift
            MAX_CONCURRENT="${1:?--max requires a value}"
            ;;
        --poll)
            shift
            echo "ERROR: --poll is no longer configured by the shell shim." >&2
            exit 2
            ;;
        --wave)
            shift
            echo "ERROR: --wave is obsolete; wave identity comes from immutable V2 manifests." >&2
            exit 2
            ;;
        --help|-h)
            echo "SQLite V2 queue compatibility shim"
            echo "  Add:    .venv/bin/python -m genetic_algorithm queue add CONFIG..."
            echo "  Start:  $0 [--max N] [--persistent]"
            echo "  Status: $0 --status"
            echo "  Stop:   $0 --stop"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

cd "$REPO_DIR"
case "$MODE" in
    status)
        exec "$PYTHON" -m genetic_algorithm queue status
        ;;
    stop)
        exec "$PYTHON" -m genetic_algorithm queue stop
        ;;
    start)
        args=(queue start --max-concurrent "$MAX_CONCURRENT")
        if [[ "$PERSISTENT" == true ]]; then
            args+=(--persistent)
        fi
        echo "Starting canonical SQLite/WAL V2 queue; file/PID slot counting is disabled."
        exec "$PYTHON" -m genetic_algorithm "${args[@]}"
        ;;
esac
