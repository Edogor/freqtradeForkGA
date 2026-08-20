#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="./.venv/bin/python"
QUEUE_DIR="genetic_algorithm/config/queue"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY" >&2
  exit 1
fi

"$PY" scripts/generate_wave45_tracks.py

for f in \
  "$QUEUE_DIR/38_wave45_A_1h_conservative_5of6.yaml" \
  "$QUEUE_DIR/39_wave45_B_1h_balanced_5of6.yaml" \
  "$QUEUE_DIR/40_wave45_C_1h_exploratory_5of6.yaml" \
  "$QUEUE_DIR/41_wave45_D_1h_stressrobust_5of6.yaml"; do
  echo "Validating $f"
  "$PY" genetic_algorithm/run_ga.py --config "$f" --validate-only
 done

echo "Wave45 configs generated + validated."
echo "To start queue daemon: ./ga_auto_queue_v2.sh --wave wave45 --max 4 --persistent"
