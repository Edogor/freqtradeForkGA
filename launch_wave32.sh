#!/usr/bin/env bash
# ⚠️  DEPRECATED: Use 'python -m genetic_algorithm queue add config/queue/*.yaml && python -m genetic_algorithm queue start' instead.
# This script is kept for reference only (historic wave32 launch). Do not use for new waves.
# ============================================================================
# Wave32 Launch Script — 8 Scalping Experiments (5 Active + 3 Queue)
# ============================================================================
# Uses ga_auto_queue_v2.sh daemon to manage all 8 experiments.
# First 5 (01-05) launch immediately. Next 3 (06-08) queue until slots open.
#
# Wave32 Key Improvements over Wave31:
#   - 3 training pairs + 3 validation pairs (vs 2+2)
#   - Profit weight: 0.27 → 0.35–0.40 (stronger signal)
#   - Profit floor: -15 → -10 (tighter, fewer candidates rewarded)
#   - pair_validation weight_val: 0.40 → 0.45 (harder generalisation)
#   - Generations: 30 → 35–40
#   - min_trades: 60 → 80
#   - NO NSGA-II runs (bug fixed but too risky until validated)
#   - 2 bugs fixed: exit-127 false-failed, NSGA-II zero-trade exploit
#
# Experiments:
#   ACTIVE (01-05, start immediately):
#     A — Island 15m Rank+Ring       (7×15=105, wave31_A successor, profit=0.35)
#     B — Island 15m Rank+Ring       (7×15=105, max profit pressure, profit=0.40)
#     C — Island 5m  Rank+Ring       (6×15=90,  WR-trap fix: win_rate weight halved)
#     D — Island 15m Rank+Ring       (8×15=120, 2 train + 4 val max generalisation)
#     E — Plain   15m Rank           (pop=30,   baseline control, 3+3 pairs)
#
#   QUEUE (06-08, auto-start as slots open):
#     F — Island 15m Rank+Hierarchical     (6×15=90,  tier diversity topology)
#     G — Island 5m  Ring Ultra Scalp      (6×15=90,  tight ROI [0.002-0.008])
#     H — Island 15m Tournament+Ring       (7×15=105, BTC+ETH train, alt-coin val)
#
# Usage:
#   ./launch_wave32.sh          # Start daemon (launches all 8 over time)
#   ./launch_wave32.sh --status # Show running/queued counts
#   ./launch_wave32.sh --stop   # Stop daemon + all wave32 processes
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${REPO_DIR}/.venv/bin/python"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
QUEUE_DIR="${REPO_DIR}/genetic_algorithm/config/queue"

# Source .env if available
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

cd "$REPO_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────

check_status() {
    echo "=== Wave32 Status ==="
    echo ""
    echo "Running processes:"
    local found=0
    while IFS= read -r line; do
        pid=$(echo "$line" | awk '{print $2}')
        config=$(echo "$line" | grep -oP '(?<=--config )\S+')
        mem=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f", $1/1024}')
        echo "  PID $pid — ${config} — ${mem}MB"
        found=$((found+1))
    done < <(ps aux | grep 'run_ga.py.*wave32' | grep -v grep)
    [[ $found -eq 0 ]] && echo "  (none)"

    echo ""
    echo "Queue:"
    local qcount
    qcount=$(ls "${QUEUE_DIR}"/*wave32*.yaml 2>/dev/null | wc -l)
    if [[ $qcount -gt 0 ]]; then
        ls "${QUEUE_DIR}"/*wave32*.yaml | while read -r f; do echo "  $(basename "$f")"; done
    else
        echo "  (empty — experiments either running or completed)"
    fi

    echo ""
    echo "Auto-queue daemon:"
    local daemon_pid="${LOG_DIR}/auto_queue_daemon.pid"
    if [[ -f "$daemon_pid" ]]; then
        local pid
        pid=$(cat "$daemon_pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  RUNNING (PID $pid)"
        else
            echo "  NOT running (stale PID file)"
        fi
    else
        echo "  NOT running"
    fi

    echo ""
    echo "System memory: $(LANG=C free -m | awk '/Mem:/ {printf "%dMB used / %dMB total", $3, $2}') | swap: $(LANG=C free -m | awk '/Swap:/ {printf "%dMB/%dMB", $3, $2}')"
}

stop_all() {
    echo "Stopping wave32..."
    # Stop daemon gracefully
    ./ga_auto_queue_v2.sh --stop 2>/dev/null || true
    sleep 2
    # Kill any remaining wave32 processes
    local pids
    pids=$(pgrep -f 'run_ga.py.*wave32' 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "Killing wave32 processes: $pids"
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 3
        pids=$(pgrep -f 'run_ga.py.*wave32' 2>/dev/null || true)
        [[ -n "$pids" ]] && echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
    echo "Done."
}

# ── Main ─────────────────────────────────────────────────────────────────

case "${1:-}" in
    --status) check_status; exit 0 ;;
    --stop)   stop_all;     exit 0 ;;
esac

echo "=== Wave32 Pre-Flight ==="

# Warn if wave31 still running
old_pids=$(pgrep -f 'run_ga.py.*wave31' 2>/dev/null || true)
if [[ -n "$old_pids" ]]; then
    echo "NOTE: wave31 processes still running (PIDs: $old_pids)"
    echo "  They share CPU with wave32. This is expected if wave31 D/F/G are finishing."
    echo "  wave32 will queue behind if >5 combined slots are used."
fi

# Verify queue has wave32 configs
qcount=$(ls "${QUEUE_DIR}"/*wave32*.yaml 2>/dev/null | wc -l)
if [[ $qcount -eq 0 ]]; then
    echo "ERROR: No wave32 configs found in queue dir: ${QUEUE_DIR}"
    echo "Run from the repo root after configs have been copied to queue/"
    exit 1
fi

echo "Queue: ${qcount} wave32 experiments ready"
ls "${QUEUE_DIR}"/*wave32*.yaml | while read -r f; do echo "  → $(basename "$f")"; done

# Check memory
available_mb=$(LANG=C free -m | awk '/Mem:/ {print $7}')
echo ""
echo "Available memory: ${available_mb}MB"
if [[ $available_mb -lt 4000 ]]; then
    echo "WARNING: Less than 4GB available — running 5 parallel experiments may cause OOM"
    read -rp "Continue anyway? [y/N] " answer
    [[ "$answer" =~ ^[yY] ]] || exit 1
fi

echo ""
echo "=== Starting Auto-Queue Daemon ==="
echo "  Wave:           wave32"
echo "  Max concurrent: 5 (active slots)"
echo "  Mode:           persistent (auto-starts queue items as slots open)"
echo "  Daemon log:     ${LOG_DIR}/auto_queue.log"
echo ""

mkdir -p "$LOG_DIR"

# Start queue daemon in background — persistent mode, max 5 concurrent
nohup ./ga_auto_queue_v2.sh \
    --wave wave32 \
    --max 5 \
    --persistent \
    >> "${LOG_DIR}/auto_queue.log" 2>&1 &

daemon_pid=$!
echo "Daemon started — PID ${daemon_pid}"
echo "The first 5 experiments (A-E) will launch within ~30 seconds."
echo "Experiments F-H auto-launch as slots free up."
echo ""
echo "Monitor:"
echo "  ./launch_wave32.sh --status"
echo "  ./ga_monitor_v2.sh wave32"
echo "  tail -f ${LOG_DIR}/auto_queue.log"
echo "  ./wave_monitor.sh wave32"
