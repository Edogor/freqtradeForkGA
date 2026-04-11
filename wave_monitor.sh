#!/usr/bin/env bash
# ⚠️  DEPRECATED: Use 'ga_monitor_v2.sh --wave <N>' or 'python -m genetic_algorithm monitor --tag wave<N>' instead.
# This script is kept for backward compatibility and will be removed in a future release.
# ============================================================================
# Wave Monitor — Live Status Dashboard for Parallel Wave Runs
# ============================================================================
# Polls log files and process status for all experiments in a wave.
# Shows: generation progress, best fitness, memory, errors, alive status.
#
# Usage:
#   ./wave_monitor.sh wave1                  # monitor all experiments
#   ./wave_monitor.sh wave1 --once           # print status once and exit
#   ./wave_monitor.sh wave1 --interval 10    # refresh every 10 seconds
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
WAVE_NAME="${1:-}"
INTERVAL=5
ONCE=false

if [[ -z "$WAVE_NAME" ]]; then
    echo "Usage: $0 <wave_name> [--once] [--interval N]"
    exit 1
fi

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --once) ONCE=true ;;
        --interval) INTERVAL="${2:-5}"; shift ;;
        *) ;;
    esac
    shift
done

LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
OUTPUT_DIR="${REPO_DIR}/genetic_algorithm/output/exploration/${WAVE_NAME}"
CONFIG_DIR="${REPO_DIR}/genetic_algorithm/config/exploration/${WAVE_NAME}"

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

extract_metrics() {
    local log_file="$1"
    local exp_name="$2"

    if [[ ! -f "$log_file" ]]; then
        echo "${exp_name}|—|—|—|—|—|${RED}NO LOG${NC}"
        return
    fi

    local file_size
    file_size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
    if [[ $file_size -eq 0 ]]; then
        echo "${exp_name}|—|—|—|—|—|${YELLOW}EMPTY${NC}"
        return
    fi

    # Read last 500 lines for speed (avoid parsing huge logs)
    local tail_content
    tail_content=$(tail -500 "$log_file" 2>/dev/null)

    # Generation progress
    local current_gen total_gen gen_display
    current_gen=$(echo "$tail_content" | grep -oP 'GENERATION\s+\K\d+' | tail -1)
    total_gen=$(echo "$tail_content" | grep -oP 'GENERATION\s+\d+/\K\d+' | tail -1)
    if [[ -n "$current_gen" ]]; then
        gen_display="${current_gen}/${total_gen:-?}"
    else
        gen_display="init"
    fi

    # Best fitness — island model: "[island_X] NEW BEST: fitness=0.4182" or legacy "Best: 0.42"
    local best_fitness
    best_fitness=$(echo "$tail_content" | grep -oP '(?<=NEW BEST: fitness=)[0-9.]+' | sort -n | tail -1)
    if [[ -z "$best_fitness" ]]; then
        best_fitness=$(echo "$tail_content" | grep -oP '(?<=Best: )[0-9.]+' | tail -1)
    fi
    best_fitness="${best_fitness:-—}"

    # Current avg fitness — island model: "avg=0.XXXX" or legacy "Avg: 0.XX"
    local avg_fitness
    avg_fitness=$(echo "$tail_content" | grep -oP '(?<=avg=)[0-9.]+' \
        | LC_NUMERIC=C awk '{s+=$1;n++} END{if(n>0) printf "%.4f",s/n}')
    if [[ -z "$avg_fitness" ]]; then
        avg_fitness=$(echo "$tail_content" | grep -oP 'Avg: \K[0-9.]+' | tail -1)
    fi
    avg_fitness="${avg_fitness:-—}"

    # Diversity — island model: "diversity=0.XXXX" or legacy "Diversity: 0.XX"
    local diversity
    diversity=$(echo "$tail_content" | grep -oP '(?<=diversity=)[0-9.]+' \
        | LC_NUMERIC=C awk '{s+=$1;n++} END{if(n>0) printf "%.4f",s/n}')
    if [[ -z "$diversity" ]]; then
        diversity=$(echo "$tail_content" | grep -oP '[Dd]iversity[: =]+\K[0-9.]+' | tail -1)
    fi
    diversity="${diversity:-—}"

    # Errors count
    local errors
    errors=$(grep -ciP '- (ERROR|CRITICAL) -|^Traceback' "$log_file" 2>/dev/null) || true
    errors=${errors:-0}

    # Process alive check — PID file, then process list, then log age
    local pid status_display
    local pid_file
    pid_file=$(ls -t "${LOG_DIR}/${WAVE_NAME}_pids_"*.txt 2>/dev/null | head -1)

    if [[ -n "$pid_file" ]]; then
        pid=$(grep "$exp_name" "$pid_file" 2>/dev/null | awk '{print $1}')
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            status_display="${GREEN}RUNNING${NC}"
        elif [[ -n "$pid" ]]; then
            local last_line
            last_line=$(tail -3 "$log_file" 2>/dev/null)
            if echo "$last_line" | grep -qi 'complete\|finished\|converged\|saved'; then
                status_display="${GREEN}DONE${NC}"
            else
                status_display="${RED}DEAD${NC}"
            fi
        else
            status_display="${DIM}UNKNOWN${NC}"
        fi
    else
        # No PID file — check if any run_ga.py process references this config
        local active_proc
        active_proc=$(ps aux 2>/dev/null | grep '[r]un_ga.py' | grep "${WAVE_NAME}/${exp_name}" || true)
        if [[ -n "$active_proc" ]]; then
            status_display="${GREEN}RUNNING${NC}"
        else
            # Fall back to log modification time
            local mod_age
            mod_age=$(( $(date +%s) - $(stat -c%Y "$log_file" 2>/dev/null || echo 0) ))
            if [[ $mod_age -lt 300 ]]; then
                status_display="${GREEN}ACTIVE${NC}"
            elif grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE|ISLAND MODEL EVOLUTION FINISHED' "$log_file" 2>/dev/null; then
                status_display="${GREEN}DONE${NC}"
            elif tail -10 "$log_file" 2>/dev/null | grep -qP 'Traceback|FATAL'; then
                status_display="${RED}CRASHED${NC}"
            else
                status_display="${YELLOW}STALE${NC}"
            fi
        fi
    fi

    echo "${exp_name}|${gen_display}|${best_fitness}|${avg_fitness}|${diversity}|${errors}|${status_display}"
}

print_dashboard() {
    # Clear screen in loop mode
    [[ "$ONCE" == false ]] && clear

    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   WAVE MONITOR — ${WAVE_NAME}                 $(date '+%H:%M:%S')                        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Table header
    printf "  ${BOLD}%-30s  %-8s  %-10s  %-8s  %-8s  %-6s  %-10s${NC}\n" \
        "EXPERIMENT" "GEN" "BEST_FIT" "AVG_FIT" "DIVERS" "ERRS" "STATUS"
    echo "  ────────────────────────────  ────────  ──────────  ────────  ────────  ──────  ──────────"

    # Discover experiments
    local found=false
    for config_file in "${CONFIG_DIR}"/*.yaml; do
        [[ ! -f "$config_file" ]] && continue
        found=true
        local exp_name
        exp_name=$(basename "$config_file" .yaml)
        # Log files are named wave30_A.log (letter-prefix only), not wave30_A_full_name.log
        local exp_prefix
        exp_prefix=$(echo "$exp_name" | cut -d_ -f1)
        local log_file="${LOG_DIR}/${WAVE_NAME}_${exp_prefix}.log"

        local metrics
        metrics=$(extract_metrics "$log_file" "$exp_name")

        IFS='|' read -r name gen best avg div errs status <<< "$metrics"
        printf "  %-30s  %-8s  %-10s  %-8s  %-8s  %-6s  %b\n" \
            "$name" "$gen" "$best" "$avg" "$div" "$errs" "$status"
    done

    if [[ "$found" == false ]]; then
        echo -e "  ${RED}No experiments found in ${CONFIG_DIR}${NC}"
    fi

    echo ""

    # Memory summary
    local total_rss
    total_rss=$(ps aux | grep "run_ga.py" | grep -v grep | awk '{sum+=$6} END {printf "%.0f", sum/1024}')
    local free_mem
    free_mem=$(free -m | awk '/Mem:/ {print $7}')
    local cpu_load
    cpu_load=$(uptime | awk -F'load average: ' '{print $2}' | cut -d, -f1)

    echo -e "  ${BLUE}System:${NC} RSS=${total_rss:-0}MB | Free=${free_mem:-?}MB | Load=${cpu_load:-?}"

    if [[ "$ONCE" == false ]]; then
        echo ""
        echo -e "  ${DIM}Refreshing every ${INTERVAL}s — Ctrl+C to stop monitoring${NC}"
    fi
}

# ── Main loop ──
if [[ "$ONCE" == true ]]; then
    print_dashboard
else
    while true; do
        print_dashboard
        sleep "$INTERVAL"
    done
fi
