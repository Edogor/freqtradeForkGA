#!/usr/bin/env bash
# ⚠️  DEPRECATED: Use 'python -m genetic_algorithm queue add config/queue/*.yaml && python -m genetic_algorithm queue start' instead.
# This script is kept for reference only (historic wave31 launch). Do not use for new waves.
# ============================================================================
# Wave31 Launch Script — 10 Scalping Experiments (5 Active + 5 Queue)
# ============================================================================
# Uses ga_auto_queue_v2.sh daemon to manage all 10 experiments.
# First 5 (01-05) launch immediately. Next 5 (06-10) queue until slots open.
#
# Wave31 Key Improvements over Wave30:
#   - Profit weight 0.12-0.14 → 0.27-0.35 (8.6× stronger signal)
#   - Fitness bounds: profit_min=-50 → -15, profit_max=200 → 50
#   - 4 NSGA-II Pareto experiments (C, H, I, J) — Pareto vs weighted-sum
#   - No 3m timeframe (too slow). Using 5m, 15m (majority), 30m
#   - Mix: 6 island model + 4 plain GA
#
# Experiments:
#   ACTIVE (01-05, start immediately):
#     A — Island 15m Rank+Ring       (7 islands × 15 = 105)
#     B — Island 15m Tournament+FC   (8 islands × 15 = 120)
#     C — Plain GA 15m NSGA2 3-obj   (pop=30, Pareto: profit+win_rate+drawdown)
#     D — Island 5m Rank+Ring        (6 islands × 15 = 90, wave30A successor)
#     E — Plain GA 30m Rank          (pop=30, swing-scalp exploration)
#
#   QUEUE (06-10, auto-start as slots open):
#     F — Island 15m Rank+Hierarchical      (6 islands × 15 = 90)
#     G — Island 15m Tournament+Ring+Comp   (7 islands × 15 = 105, component crossover)
#     H — Plain GA 15m NSGA2 2-obj          (pop=35, Pareto: profit+sharpe, strongest P signal)
#     I — Island 30m NSGA2 3-obj            (5 islands × 15 = 75, Pareto: profit+freq+drawdown)
#     J — Plain GA 5m NSGA2 3-obj           (pop=30, Pareto: profit+win_rate+sharpe)
#
# Usage:
#   ./launch_wave31.sh          # Start daemon (launches all 10 over time)
#   ./launch_wave31.sh --status # Show running/queued counts
#   ./launch_wave31.sh --stop   # Stop daemon + all wave31 processes
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
    echo "=== Wave31 Status ==="
    echo ""
    echo "Running processes:"
    local found=0
    while IFS= read -r line; do
        pid=$(echo "$line" | awk '{print $2}')
        config=$(echo "$line" | grep -oP '(?<=--config )\S+')
        mem=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f", $1/1024}')
        echo "  PID $pid — ${config} — ${mem}MB"
        found=$((found+1))
    done < <(ps aux | grep 'run_ga.py.*wave31' | grep -v grep)
    [[ $found -eq 0 ]] && echo "  (none)"

    echo ""
    echo "Queue:"
    local qcount
    qcount=$(ls "${QUEUE_DIR}"/*wave31*.yaml 2>/dev/null | wc -l)
    if [[ $qcount -gt 0 ]]; then
        ls "${QUEUE_DIR}"/*wave31*.yaml | while read -r f; do echo "  $(basename "$f")"; done
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
    echo "Stopping wave31..."
    # Stop daemon gracefully
    ./ga_auto_queue_v2.sh --stop 2>/dev/null || true
    sleep 2
    # Kill any remaining wave31 processes
    local pids
    pids=$(pgrep -f 'run_ga.py.*wave31' 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "Killing wave31 processes: $pids"
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 3
        pids=$(pgrep -f 'run_ga.py.*wave31' 2>/dev/null || true)
        [[ -n "$pids" ]] && echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
    echo "Done."
}

# ── Main ─────────────────────────────────────────────────────────────────

case "${1:-}" in
    --status) check_status; exit 0 ;;
    --stop)   stop_all;     exit 0 ;;
esac

echo "=== Wave31 Pre-Flight ==="

# Ensure no wave30 still running
old_pids=$(pgrep -f 'run_ga.py.*wave30' 2>/dev/null || true)
if [[ -n "$old_pids" ]]; then
    echo "WARNING: wave30 processes still running — killing: $old_pids"
    echo "$old_pids" | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Verify queue has wave31 configs
qcount=$(ls "${QUEUE_DIR}"/*wave31*.yaml 2>/dev/null | wc -l)
if [[ $qcount -eq 0 ]]; then
    echo "ERROR: No wave31 configs found in queue dir: ${QUEUE_DIR}"
    echo "Run from the repo root after configs have been copied to queue/"
    exit 1
fi

echo "Queue: ${qcount} wave31 experiments ready"
ls "${QUEUE_DIR}"/*wave31*.yaml | while read -r f; do echo "  → $(basename "$f")"; done

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
echo "  Wave:       wave31"
echo "  Max concurrent: 5 (active slots)"
echo "  Mode:       persistent (auto-starts queue items as slots open)"
echo "  Daemon log: ${LOG_DIR}/auto_queue.log"
echo ""

mkdir -p "$LOG_DIR"

# Start queue daemon in background — persistent mode, max 5 concurrent
nohup ./ga_auto_queue_v2.sh \
    --wave wave31 \
    --max 5 \
    --persistent \
    >> "${LOG_DIR}/auto_queue.log" 2>&1 &

daemon_pid=$!
echo "Daemon started — PID ${daemon_pid}"
echo "The first 5 experiments (A-E) will launch within ~30 seconds."
echo "Experiments F-J auto-launch as slots free up."
echo ""
echo "Monitor:"
echo "  ./launch_wave31.sh --status"
echo "  ./ga_monitor_v2.sh wave31"
echo "  tail -f ${LOG_DIR}/auto_queue.log"
echo "  ./wave_monitor.sh wave31"
