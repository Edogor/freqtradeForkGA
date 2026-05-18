#!/usr/bin/env bash
# ============================================================================
# SIS Production Launch Script — 4 Experiments (2 Control + 2 SIS)
# ============================================================================
# A/B test to validate SIS system and ga-params-configs parameter discoveries.
#
# Experiment Groups:
#   PERFORMANCE PAIR (E138-inspired, 10 islands × 10 pop):
#     A — CONTROL: A_ctrl_perf_10x10.yaml  (no SIS)
#     C — SIS:     C_sis_perf_10x10.yaml   (SIS enabled, identical otherwise)
#
#   ROBUSTNESS PAIR (R5-inspired, 8 islands × 12 pop):
#     B — CONTROL: B_ctrl_robust_8x12.yaml (no SIS)
#     D — SIS:     D_sis_robust_8x12.yaml  (SIS enabled, identical otherwise)
#
# All 4 experiments use:
#   - random_seed: 42 (reproducible comparison)
#   - 15m timeframe, ring topology, rank selection
#   - Optimized fitness weights (profit 40-42%, no trade_frequency/win_rate)
#   - Corrected ROI/SL ranges (mathematically winnable)
#   - Pair validation anti-overfitting
#
# Usage:
#   ./launch_sis_production.sh                  # Copy configs to queue & start daemon (A-D)
#   ./launch_sis_production.sh --compact        # Launch compact 4h pair E/F directly (nohup)
#   ./launch_sis_production.sh --status         # Show running/queued counts (all 6 experiments)
#   ./launch_sis_production.sh --stop           # Stop daemon + all sis_prod processes
#   ./launch_sis_production.sh --watchdog [MB]  # Background memory guard (default threshold 1500MB free)
#
# Monitoring:
#   # Performance pair (A vs C):
#   python -m genetic_algorithm.intelligence.sis_monitor \
#       --ctrl-log genetic_algorithm/logs/sis_production/A_ctrl_perf_10x10.log \
#       --exp-log genetic_algorithm/logs/sis_production/C_sis_perf_10x10.log
#
#   # Robustness pair (B vs D):
#   python -m genetic_algorithm.intelligence.sis_monitor \
#       --ctrl-log genetic_algorithm/logs/sis_production/B_ctrl_robust_8x12.log \
#       --exp-log genetic_algorithm/logs/sis_production/D_sis_robust_8x12.log
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${REPO_DIR}/.venv/bin/python"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
SIS_LOG_DIR="${LOG_DIR}/sis_production"
QUEUE_DIR="${REPO_DIR}/genetic_algorithm/config/queue"
CONFIG_DIR="${REPO_DIR}/genetic_algorithm/config/sis_production"

# Source .env if available
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

cd "$REPO_DIR"

# Experiment configs (priority-ordered for queue)
CONFIGS=(
    "01_A_ctrl_perf_10x10.yaml:A_ctrl_perf_10x10.yaml"
    "02_B_ctrl_robust_8x12.yaml:B_ctrl_robust_8x12.yaml"
    "03_C_sis_perf_10x10.yaml:C_sis_perf_10x10.yaml"
    "04_D_sis_robust_8x12.yaml:D_sis_robust_8x12.yaml"
)

# ── Helpers ──────────────────────────────────────────────────────────────

mem_watchdog() {
    # Background memory watchdog — kills the largest GA process if available
    # RAM drops below MIN_AVAIL_MB to prevent an OOM server crash.
    local MIN_AVAIL_MB=${1:-1500}
    local CHECK_INTERVAL=30    # seconds between checks
    local PATTERN='run_ga\.py'
    echo "[watchdog] Started — threshold ${MIN_AVAIL_MB}MB free, interval ${CHECK_INTERVAL}s"
    while true; do
        sleep "$CHECK_INTERVAL"
        local avail
        avail=$(LANG=C free -m | awk '/Mem:/ {print $7}')
        if [[ "$avail" -lt "$MIN_AVAIL_MB" ]]; then
            # Find the GA process consuming the most RSS
            local victim_pid victim_rss
            victim_pid=$(ps aux | grep -E "$PATTERN" | grep -v grep \
                | awk '{print $2, $6}' | sort -k2 -n -r | head -1 | awk '{print $1}')
            victim_rss=$(ps aux | grep -E "$PATTERN" | grep -v grep \
                | awk '{print $2, $6}' | sort -k2 -n -r | head -1 | awk '{printf "%.0f", $2/1024}')
            if [[ -n "$victim_pid" ]]; then
                echo "[watchdog] ALERT: only ${avail}MB free — sending SIGTERM to PID ${victim_pid} (${victim_rss}MB)"
                kill -TERM "$victim_pid" 2>/dev/null || true
            else
                echo "[watchdog] ALERT: only ${avail}MB free — no GA processes found to kill"
            fi
        fi
    done
}

check_status() {
    echo "=== SIS Production Status ==="
    echo ""
    echo "Running processes:"
    local found=0
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        pid=$(echo "$line" | awk '{print $2}')
        config=$(echo "$line" | grep -oP '(?<=--config )\S+')
        mem=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f", $1/1024}')
        echo "  PID $pid — ${config} — ${mem}MB"
        found=$((found+1))
    done < <(ps aux | grep -E 'run_ga.py.*(sis_prod|A_ctrl_perf|B_ctrl_robust|C_sis_perf|D_sis_robust|E_ctrl_4h|F_sis_4h)' | grep -v grep)
    [[ $found -eq 0 ]] && echo "  (none)"

    echo ""
    echo "Queue:"
    local qcount
    qcount=$(ls "${QUEUE_DIR}"/*ctrl_perf*.yaml "${QUEUE_DIR}"/*ctrl_robust*.yaml "${QUEUE_DIR}"/*sis_perf*.yaml "${QUEUE_DIR}"/*sis_robust*.yaml 2>/dev/null | wc -l)
    if [[ $qcount -gt 0 ]]; then
        ls "${QUEUE_DIR}"/*ctrl_perf*.yaml "${QUEUE_DIR}"/*ctrl_robust*.yaml "${QUEUE_DIR}"/*sis_perf*.yaml "${QUEUE_DIR}"/*sis_robust*.yaml 2>/dev/null | while read -r f; do echo "  $(basename "$f")"; done
    else
        echo "  (empty — experiments either running or completed)"
    fi

    echo ""
    echo "Log files (sis_production/):"
    for logfile in "${SIS_LOG_DIR}"/*.log; do
        [[ -f "$logfile" ]] || continue
        local lines
        lines=$(wc -l < "$logfile" 2>/dev/null || echo 0)
        local last
        last=$(tail -1 "$logfile" 2>/dev/null | head -c 120)
        echo "  $(basename "$logfile"): ${lines} lines"
        [[ -n "$last" ]] && echo "    └─ ${last}"
    done

    # Also check daemon-redirected logs (auto-queue may place logs elsewhere)
    echo ""
    echo "Log files (daemon-redirected):"
    for pattern in A_ctrl_perf B_ctrl_robust C_sis_perf D_sis_robust E_ctrl_4h F_sis_4h; do
        for logfile in "${LOG_DIR}"/*${pattern}*.log; do
            [[ -f "$logfile" ]] || continue
            local lines
            lines=$(wc -l < "$logfile" 2>/dev/null || echo 0)
            local last
            last=$(tail -1 "$logfile" 2>/dev/null | head -c 120)
            echo "  $(basename "$logfile"): ${lines} lines"
            [[ -n "$last" ]] && echo "    └─ ${last}"
        done
    done

    echo ""
    echo "SIS event logs:"
    for jsonl in "${SIS_LOG_DIR}"/*_sis_events.jsonl; do
        [[ -f "$jsonl" ]] || continue
        local events
        events=$(wc -l < "$jsonl" 2>/dev/null || echo 0)
        echo "  $(basename "$jsonl"): ${events} events"
    done

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
    echo "Stopping SIS production experiments..."
    # Stop daemon gracefully
    ./ga_auto_queue_v2.sh --stop 2>/dev/null || true
    sleep 2
    # Kill any remaining SIS production processes (match config filenames)
    local pids
    pids=$(pgrep -f 'run_ga.py.*(sis_prod|A_ctrl_perf|B_ctrl_robust|C_sis_perf|D_sis_robust|E_ctrl_4h|F_sis_4h)' 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "Killing SIS production processes: $pids"
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 3
        pids=$(pgrep -f 'run_ga.py.*(sis_prod|A_ctrl_perf|B_ctrl_robust|C_sis_perf|D_sis_robust|E_ctrl_4h|F_sis_4h)' 2>/dev/null || true)
        [[ -n "$pids" ]] && echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
    echo "Done."
}

# ── Main ─────────────────────────────────────────────────────────────────

launch_compact() {
    echo "============================================================"
    echo "  SIS Compact 4h Pair — Launch (E ctrl vs F SIS)"
    echo "============================================================"
    echo ""
    echo "Experiments:"
    echo "  COMPACT 4h PAIR (6 islands × 8 pop, ring topology):"
    echo "    E — CONTROL  (no SIS)    → E_ctrl_4h_6x8.yaml"
    echo "    F — SIS      (all hooks) → F_sis_4h_6x8.yaml"
    echo ""

    # Verify configs exist
    for cfg in E_ctrl_4h_6x8.yaml F_sis_4h_6x8.yaml; do
        if [[ ! -f "${CONFIG_DIR}/${cfg}" ]]; then
            echo "ERROR: Missing config: ${CONFIG_DIR}/${cfg}"
            exit 1
        fi
    done
    echo "  [OK] Both compact config files found"

    # Create directories
    mkdir -p "$SIS_LOG_DIR"
    mkdir -p "${REPO_DIR}/genetic_algorithm/data/sis_production"

    # Launch E (ctrl, 4h, 6×8) — direct nohup, bypasses queue daemon
    echo ""
    echo "Launching E (ctrl 4h 6×8)..."
    nohup "${VENV_PYTHON}" genetic_algorithm/run_ga.py \
        --config "${CONFIG_DIR}/E_ctrl_4h_6x8.yaml" \
        --no-monitor --yes \
        >> "${SIS_LOG_DIR}/E_ctrl_4h_6x8_stdout.log" 2>&1 &
    E_PID=$!
    echo "  PID: $E_PID"

    # Brief pause so processes don't stampede on data loading
    sleep 2

    # Launch F (SIS, 4h, 6×8)
    echo "Launching F (SIS 4h 6×8)..."
    nohup "${VENV_PYTHON}" genetic_algorithm/run_ga.py \
        --config "${CONFIG_DIR}/F_sis_4h_6x8.yaml" \
        --no-monitor --yes \
        >> "${SIS_LOG_DIR}/F_sis_4h_6x8_stdout.log" 2>&1 &
    F_PID=$!
    echo "  PID: $F_PID"

    echo ""
    echo "============================================================"
    echo "  Compact pair launched (bypassing queue daemon)"
    echo "  Expected runtime: ~40-60 min each"
    echo "============================================================"
    echo ""
    echo "  Monitoring commands:"
    echo "    tail -f ${SIS_LOG_DIR}/E_ctrl_4h_6x8.log"
    echo "    tail -f ${SIS_LOG_DIR}/F_sis_4h_6x8.log"
    echo ""
    echo "  Compare ctrl vs SIS:"
    echo "    python -m genetic_algorithm.intelligence.sis_monitor \\"
    echo "        --ctrl-log ${SIS_LOG_DIR}/E_ctrl_4h_6x8.log \\"
    echo "        --exp-log  ${SIS_LOG_DIR}/F_sis_4h_6x8.log"
    echo ""
    echo "  SIS events:"
    echo "    cat ${SIS_LOG_DIR}/F_sis_4h_sis_events.jsonl | python -m json.tool"
    echo "============================================================"
}

case "${1:-}" in
    --status)   check_status;      exit 0 ;;
    --stop)     stop_all;          exit 0 ;;
    --compact)  launch_compact;    exit 0 ;;
    --watchdog) mem_watchdog "${2:-1500}"; exit 0 ;;
esac

echo "============================================================"
echo "  SIS Production A/B Test — Launch"
echo "============================================================"
echo ""
echo "Experiments:"
echo "  PERFORMANCE PAIR (10 islands × 10 pop, ring topology):"
echo "    A — CONTROL  (no SIS)    → A_ctrl_perf_10x10.yaml"
echo "    C — SIS      (all hooks) → C_sis_perf_10x10.yaml"
echo ""
echo "  ROBUSTNESS PAIR (8 islands × 12 pop, ring topology):"
echo "    B — CONTROL  (no SIS)    → B_ctrl_robust_8x12.yaml"
echo "    D — SIS      (all hooks) → D_sis_robust_8x12.yaml"
echo ""

# ── Pre-flight checks ───────────────────────────────────────────────────

echo "=== Pre-Flight Checks ==="

# Verify all config files exist
missing=0
for entry in "${CONFIGS[@]}"; do
    src="${entry#*:}"
    if [[ ! -f "${CONFIG_DIR}/${src}" ]]; then
        echo "ERROR: Missing config: ${CONFIG_DIR}/${src}"
        missing=$((missing+1))
    fi
done
if [[ $missing -gt 0 ]]; then
    echo "FATAL: ${missing} config file(s) missing. Aborting."
    exit 1
fi
echo "  [OK] All 4 config files found"

# Verify Python environment
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "ERROR: Python venv not found at $VENV_PYTHON"
    exit 1
fi
echo "  [OK] Python venv found"

# Check if run_ga.py exists
if [[ ! -f "${REPO_DIR}/genetic_algorithm/run_ga.py" ]]; then
    echo "ERROR: run_ga.py not found"
    exit 1
fi
echo "  [OK] run_ga.py found"

# Check if ga_auto_queue_v2.sh exists
if [[ ! -f "${REPO_DIR}/ga_auto_queue_v2.sh" ]]; then
    echo "ERROR: ga_auto_queue_v2.sh not found"
    exit 1
fi
echo "  [OK] ga_auto_queue_v2.sh found"

# Warn if other experiments are running
other_pids=$(pgrep -f 'run_ga.py' 2>/dev/null | head -20 || true)
if [[ -n "$other_pids" ]]; then
    count=$(echo "$other_pids" | wc -l)
    echo "  NOTE: ${count} existing GA process(es) running — they share CPU with SIS production"
fi

# Check memory
available_mb=$(LANG=C free -m | awk '/Mem:/ {print $7}')
echo "  Available memory: ${available_mb}MB"
if [[ $available_mb -lt 4000 ]]; then
    echo "  WARNING: Less than 4GB available — running 4 parallel experiments may cause OOM"
    read -rp "  Continue anyway? [y/N] " answer
    [[ "$answer" =~ ^[yY] ]] || exit 1
fi

echo ""

# ── Create directories ──────────────────────────────────────────────────

mkdir -p "$SIS_LOG_DIR"
mkdir -p "$QUEUE_DIR"
mkdir -p "${REPO_DIR}/genetic_algorithm/data/sis_production"

# ── Copy configs to queue with priority prefixes ────────────────────────

echo "=== Queuing Experiments ==="
for entry in "${CONFIGS[@]}"; do
    queue_name="${entry%%:*}"
    src="${entry#*:}"
    cp "${CONFIG_DIR}/${src}" "${QUEUE_DIR}/${queue_name}"
    echo "  Queued: ${queue_name}"
done

echo ""
echo "Queue ready: 4 experiments"
ls "${QUEUE_DIR}"/*sis_*.yaml 2>/dev/null | while read -r f; do echo "  → $(basename "$f")"; done

echo ""
echo "=== Starting Auto-Queue Daemon ==="
echo "  Max concurrent: 4 (all run simultaneously for fair comparison)"
echo "  Mode:           persistent"
echo "  Daemon log:     ${LOG_DIR}/auto_queue.log"
echo ""

# Stop any existing daemon first to avoid conflicts
local_daemon_pid="${LOG_DIR}/auto_queue_daemon.pid"
if [[ -f "$local_daemon_pid" ]]; then
    old_pid=$(cat "$local_daemon_pid")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "  Stopping existing daemon (PID $old_pid)..."
        ./ga_auto_queue_v2.sh --stop 2>/dev/null || true
        sleep 2
    fi
fi

# Start queue daemon — all 4 run simultaneously
nohup ./ga_auto_queue_v2.sh \
    --wave sis_prod \
    --max 4 \
    --persistent \
    >> "${LOG_DIR}/auto_queue.log" 2>&1 &

daemon_pid=$!
echo "Daemon started — PID ${daemon_pid}"
echo "All 4 experiments will launch within ~30 seconds."
echo ""
echo "============================================================"
echo "  Monitoring Commands"
echo "============================================================"
echo ""
echo "  Status:         ./launch_sis_production.sh --status"
echo "  Stop all:       ./launch_sis_production.sh --stop"
echo "  Daemon log:     tail -f ${LOG_DIR}/auto_queue.log"
echo ""
echo "  Performance pair (A ctrl vs C SIS):"
echo "    python -m genetic_algorithm.intelligence.sis_monitor \\"
echo "        --ctrl-log ${SIS_LOG_DIR}/A_ctrl_perf_10x10.log \\"
echo "        --exp-log ${SIS_LOG_DIR}/C_sis_perf_10x10.log"
echo ""
echo "  Robustness pair (B ctrl vs D SIS):"
echo "    python -m genetic_algorithm.intelligence.sis_monitor \\"
echo "        --ctrl-log ${SIS_LOG_DIR}/B_ctrl_robust_8x12.log \\"
echo "        --exp-log ${SIS_LOG_DIR}/D_sis_robust_8x12.log"
echo ""
echo "  SIS event analysis:"
echo "    cat ${SIS_LOG_DIR}/C_sis_perf_sis_events.jsonl | python -m json.tool"
echo "    cat ${SIS_LOG_DIR}/D_sis_robust_sis_events.jsonl | python -m json.tool"
echo "============================================================"
