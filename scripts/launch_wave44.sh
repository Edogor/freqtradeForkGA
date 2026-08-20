#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="./.venv/bin/python"
QUEUE_DIR="genetic_algorithm/config/queue"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY" >&2
  exit 1
fi

"$PY" scripts/generate_wave44_tracks.py

for f in \
  "$QUEUE_DIR/34_wave44_A_1h_conservative_5of6.yaml" \
  "$QUEUE_DIR/35_wave44_B_1h_balanced_5of6.yaml" \
  "$QUEUE_DIR/36_wave44_C_1h_exploratory_5of6.yaml" \
  "$QUEUE_DIR/37_wave44_D_1h_stressrobust_5of6.yaml"; do
  echo "Validating $f"
  "$PY" genetic_algorithm/run_ga.py --config "$f" --validate-only
 done

echo "Wave44 configs generated + validated."
echo "To start queue daemon: ./ga_auto_queue_v2.sh --wave wave44 --max 4 --persistent"
