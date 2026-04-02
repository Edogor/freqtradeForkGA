#!/usr/bin/env bash
# ============================================================
# ga_restart_after_checkpoint.sh
# Watches for a new checkpoint to appear, then kills the old
# GA process and restarts it from that checkpoint with the
# updated surrogate fix.
#
# Usage: bash ga_restart_after_checkpoint.sh &
# ============================================================

set -euo pipefail

GA_PID=131650
CKPT_DIR="/home/periklis/projects/trading/freqtradeForkGA/genetic_algorithm/data/checkpoints_wave28_A1"
KNOWN_CKPT="island_checkpoint_gen8_20260401_124121.json"
CONFIG="genetic_algorithm/config/exploration/wave28/A1_mega_scalp_5m.yaml"
LOG_DIR="/home/periklis/projects/trading/freqtradeForkGA/genetic_algorithm/logs"
WORKDIR="/home/periklis/projects/trading/freqtradeForkGA"

echo "[restart-watcher] Started at $(date). Watching for new checkpoint in ${CKPT_DIR}/"
echo "[restart-watcher] GA PID to kill: ${GA_PID}"

# ── Wait for a checkpoint file that is NOT the known gen8 one ──
NEW_CKPT=""
while true; do
    for f in "${CKPT_DIR}"/island_checkpoint_*.json; do
        basename_f=$(basename "$f")
        if [[ "$basename_f" != "$KNOWN_CKPT" && "$basename_f" != *"manual"* ]]; then
            # Found a new checkpoint. Make sure it's fully written (stable size).
            size1=$(stat -c%s "$f" 2>/dev/null || echo 0)
            sleep 5
            size2=$(stat -c%s "$f" 2>/dev/null || echo 0)
            if [[ "$size1" -eq "$size2" && "$size1" -gt 100000 ]]; then
                NEW_CKPT="$f"
                break 2
            fi
        fi
    done
    sleep 15
done

echo "[restart-watcher] New checkpoint detected: ${NEW_CKPT} ($(stat -c%s "$NEW_CKPT") bytes)"

# ── Kill old GA process ──
if kill -0 "${GA_PID}" 2>/dev/null; then
    echo "[restart-watcher] Killing GA process PID=${GA_PID} ..."
    kill "${GA_PID}" || true
    # Give it 10s to shutdown gracefully, then force-kill
    for i in $(seq 1 10); do
        sleep 1
        kill -0 "${GA_PID}" 2>/dev/null || break
    done
    kill -9 "${GA_PID}" 2>/dev/null || true
    echo "[restart-watcher] GA process stopped."
else
    echo "[restart-watcher] GA PID ${GA_PID} already gone."
fi

# Brief pause to let workers shut down
sleep 5

# ── Launch new GA run from the freshly saved checkpoint ──
cd "${WORKDIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NEW_LOG="${LOG_DIR}/ga_run_${TIMESTAMP}.log"

echo "[restart-watcher] Starting GA from checkpoint: $(basename "${NEW_CKPT}")"
echo "[restart-watcher] New log: ${NEW_LOG}"

# Activate venv if present (same as ga_auto_queue_v2.sh)
VENV_DIR="${WORKDIR}/.venv"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

setsid python genetic_algorithm/run_ga.py \
    --config "${CONFIG}" \
    --resume "${NEW_CKPT}" \
    --no-monitor --yes \
    > "${NEW_LOG}" 2>&1 &

NEW_PID=$!
echo "[restart-watcher] New GA process started with PID=${NEW_PID}"
echo "[restart-watcher] Monitor with: tail -f ${NEW_LOG}"
echo "[restart-watcher] Done at $(date)"
