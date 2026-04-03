#!/usr/bin/env bash
# ============================================================================
# Wave30 Launch Script — 5 Parallel Scalping Experiments
# ============================================================================
# Launches all 5 wave30 experiments as background processes.
# Each experiment runs sequentially per island (no parallel workers).
# Staggered by 60s to avoid simultaneous data loading spikes.
#
# Usage:
#   ./launch_wave30.sh          # Launch all 5
#   ./launch_wave30.sh A B C    # Launch specific experiments only
#   ./launch_wave30.sh --status # Check running experiments
#   ./launch_wave30.sh --stop   # Stop all wave30 experiments
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${REPO_DIR}/.venv/bin/python"
CONFIG_DIR="${REPO_DIR}/genetic_algorithm/config/exploration/wave30"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"

# Source .env if available (API keys etc.)
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

# Experiment configs (order = launch order)
declare -A CONFIGS=(
    [A]="A_rank_5m_ring.yaml"
    [B]="B_tournament_5m_fc.yaml"
    [C]="C_rank_3m_ring.yaml"
    [D]="D_tournament_15m.yaml"
    [E]="E_rank_5m_multipair.yaml"
)

declare -A DESCRIPTIONS=(
    [A]="Rank 5m Ring (8 islands)"
    [B]="Tournament 5m FC (12 islands)"
    [C]="Rank 3m Ring (6 islands)"
    [D]="Tournament 15m (10 islands)"
    [E]="Rank 5m Multi-Pair (8 islands)"
)

STAGGER_SECONDS=60

# --- Functions ---

check_status() {
    echo "=== Wave30 Running Experiments ==="
    local found=0
    for key in A B C D E; do
        local cfg="${CONFIGS[$key]}"
        local pid
        pid=$(pgrep -f "run_ga.py.*${cfg}" 2>/dev/null | head -1)
        if [[ -n "$pid" ]]; then
            local mem
            mem=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f", $1/1024}')
            echo "  [$key] ${DESCRIPTIONS[$key]} — PID $pid, ${mem}MB RAM"
            found=$((found + 1))
        else
            echo "  [$key] ${DESCRIPTIONS[$key]} — not running"
        fi
    done
    echo ""
    echo "Total running: $found / 5"
    echo "System memory: $(LANG=C free -m | awk '/Mem:/ {printf "%dMB used / %dMB total", $3, $2}') | swap: $(LANG=C free -m | awk '/Swap:/ {printf "%dMB/%dMB", $3, $2}')"
}

stop_all() {
    echo "Stopping all wave30 experiments..."
    for key in A B C D E; do
        local cfg="${CONFIGS[$key]}"
        local pids
        pids=$(pgrep -f "run_ga.py.*${cfg}" 2>/dev/null)
        if [[ -n "$pids" ]]; then
            echo "  Killing [$key] ${DESCRIPTIONS[$key]} (PIDs: $pids)"
            echo "$pids" | xargs kill -TERM 2>/dev/null
        fi
    done
    sleep 2
    # Force-kill any survivors
    for key in A B C D E; do
        local cfg="${CONFIGS[$key]}"
        local pids
        pids=$(pgrep -f "run_ga.py.*${cfg}" 2>/dev/null)
        if [[ -n "$pids" ]]; then
            echo "  Force-killing [$key] (PIDs: $pids)"
            echo "$pids" | xargs kill -9 2>/dev/null
        fi
    done
    echo "Done."
}

launch_experiment() {
    local key="$1"
    local cfg="${CONFIGS[$key]}"
    local config_path="${CONFIG_DIR}/${cfg}"
    local log_file="${LOG_DIR}/wave30_${key}.log"

    if [[ ! -f "$config_path" ]]; then
        echo "  ERROR: Config not found: $config_path"
        return 1
    fi

    # Check if already running
    if pgrep -f "run_ga.py.*${cfg}" > /dev/null 2>&1; then
        echo "  SKIP [$key] — already running"
        return 0
    fi

    mkdir -p "$LOG_DIR"

    cd "$REPO_DIR"
    setsid "$VENV_PYTHON" genetic_algorithm/run_ga.py \
        --config "$config_path" \
        --no-monitor --yes \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "  LAUNCHED [$key] ${DESCRIPTIONS[$key]} → PID $pid (log: $log_file)"
    return 0
}

# --- Main ---

cd "$REPO_DIR"

# Handle flags
if [[ "${1:-}" == "--status" ]]; then
    check_status
    exit 0
fi

if [[ "${1:-}" == "--stop" ]]; then
    stop_all
    exit 0
fi

# Determine which experiments to launch
if [[ $# -gt 0 ]]; then
    LAUNCH_KEYS=("$@")
else
    LAUNCH_KEYS=(A B C D E)
fi

# Pre-flight checks
echo "=== Wave30 Pre-Flight ==="

# Check no wave28/29 still running — kill automatically
    old_pids=$(pgrep -f 'run_ga.py.*wave2[89]' 2>/dev/null)
    if [[ -n "$old_pids" ]]; then
        echo "Killing remaining wave28/29 processes: $old_pids"
        echo "$old_pids" | xargs kill -9 2>/dev/null
        sleep 1
fi

# Check memory
    total_mem_mb=$(LANG=C free -m | awk '/Mem:/ {print $2}')
    available_mb=$(LANG=C free -m | awk '/Mem:/ {print $7}')
    echo "Memory: ${available_mb}MB available / ${total_mem_mb}MB total"
    if [[ -n "$available_mb" && $available_mb -lt 4000 ]]; then
    read -rp "Continue? [y/N] " answer
    [[ "$answer" =~ ^[yY] ]] || exit 1
fi

echo ""
echo "=== Launching ${#LAUNCH_KEYS[@]} Wave30 Experiments ==="

for i in "${!LAUNCH_KEYS[@]}"; do
    key="${LAUNCH_KEYS[$i]}"

    # Validate key
    if [[ -z "${CONFIGS[$key]+x}" ]]; then
        echo "  ERROR: Unknown experiment '$key'. Valid: A B C D E"
        continue
    fi

    launch_experiment "$key"

    # Stagger launches (except after last one)
    if [[ $i -lt $((${#LAUNCH_KEYS[@]} - 1)) ]]; then
        echo "  Waiting ${STAGGER_SECONDS}s before next launch..."
        sleep "$STAGGER_SECONDS"
    fi
done

echo ""
echo "=== All experiments launched ==="
echo "Monitor with: ./launch_wave30.sh --status"
echo "Stop all:     ./launch_wave30.sh --stop"
echo "View logs:    tail -f genetic_algorithm/logs/wave30_*.log"
