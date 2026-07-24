#!/usr/bin/env bash
# Legacy name retained as a safe redirect to the canonical V2 queue.

set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${REPO_DIR}/ga_auto_queue_v2.sh" "$@"
