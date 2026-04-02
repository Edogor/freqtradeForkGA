#!/usr/bin/env bash
# ============================================================================
# GA Auto-Queue Daemon v2 — Improved Experiment Queue Manager
# ============================================================================
# Monitors running GA processes and launches queued experiments to maintain
# a target number of concurrent runs. Improvements over v1:
#   - Configurable wave name (auto-incremented or manual)
#   - Per-experiment output directories
#   - Persistent daemon mode (keeps watching even when queue empties)
#   - Better log naming consistent with ga_monitor_v2.sh
#   - Queue status integrated into logging
#   - Graceful config validation before launch
#
# Queue directory:  genetic_algorithm/config/queue/
#   - YAML configs named with priority prefix: 01_E19_name.yaml
#   - Lower number = higher priority (launched first)
#
# Done directory:   genetic_algorithm/config/done/{wave}/
#   - Completed configs organized by wave
#
# Usage:
#   ./ga_auto_queue_v2.sh                       # auto-detect wave, run queue
#   ./ga_auto_queue_v2.sh --wave wave16         # explicit wave name
#   ./ga_auto_queue_v2.sh --max 6               # override max concurrent
#   ./ga_auto_queue_v2.sh --persistent          # keep running after queue drains
#   ./ga_auto_queue_v2.sh --status              # show running/queued counts
#   ./ga_auto_queue_v2.sh --stop                # stop daemon gracefully
#   nohup ./ga_auto_queue_v2.sh --persistent &  # background daemon
#
# Monitoring:
#   tail -f genetic_algorithm/logs/auto_queue.log
#   ./ga_monitor_v2.sh                          # live experiment dashboard
# ============================================================================

set -uo pipefail
set +m  # Disable job control

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
QUEUE_DIR="${REPO_DIR}/genetic_algorithm/config/queue"
DONE_BASE="${REPO_DIR}/genetic_algorithm/config/done"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
OUTPUT_BASE="${REPO_DIR}/genetic_algorithm/output/exploration"
LOG_FILE="${LOG_DIR}/auto_queue.log"

# Source .env file for API keys (GROQ_API_KEY needed for LLM experiments)
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
fi

PID_FILE="${LOG_DIR}/auto_queue_daemon.pid"
TRACKED_PIDS_FILE="${LOG_DIR}/auto_queue_tracked.txt"
QUEUE_STATE_FILE="${LOG_DIR}/auto_queue_state.json"

MAX_CONCURRENT=5
POLL_INTERVAL=30
PERSISTENT=false
WAVE_NAME=""

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Parse arguments ──
MODE="run"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)     MODE="status" ;;
        --stop)       MODE="stop" ;;
        --persistent) PERSISTENT=true ;;
        --wave)
            shift
            WAVE_NAME="${1:-}"
            [[ -z "$WAVE_NAME" ]] && echo -e "${RED}ERROR: --wave requires a name${NC}" && exit 1
            ;;
        --max)
            shift
            MAX_CONCURRENT="${1:-5}"
            ;;
        --poll)
            shift
            POLL_INTERVAL="${1:-30}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --wave NAME      Set wave name (default: auto-detect next wave)"
            echo "  --max N          Max concurrent experiments (default: 5)"
            echo "  --persistent     Keep running after queue drains (watch for new configs)"
            echo "  --poll N         Poll interval in seconds (default: 30)"
            echo "  --status         Show queue status and exit"
            echo "  --stop           Stop running daemon"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Logging ──
log() {
    local level="$1"
    shift
    local msg="$*"
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${ts} [${level}] ${msg}" >> "$LOG_FILE"
    if [[ "$level" == "ERROR" ]]; then
        echo -e "${RED}[${ts}] ${msg}${NC}" >&2
    elif [[ "$level" == "WARNING" ]]; then
        echo -e "${YELLOW}[${ts}] ${msg}${NC}"
    else
        echo -e "${BLUE}[${ts}]${NC} ${msg}"
    fi
}

# ── Auto-detect next wave number ──
detect_next_wave() {
    local max_wave=0
    for d in "${OUTPUT_BASE}"/wave*; do
        [[ -d "$d" ]] || continue
        local num="${d##*wave}"
        if [[ "$num" =~ ^[0-9]+$ ]] && (( num > max_wave )); then
            max_wave=$num
        fi
    done
    echo "wave$(( max_wave + 1 ))"
}

# ── Status command ──
if [[ "$MODE" == "status" ]]; then
    running=$(pgrep -cf "run_ga\.py" 2>/dev/null) || running=0
    queued=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
    done_count=$(find "$DONE_BASE" -name '*.yaml' 2>/dev/null | wc -l)

    echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   GA Auto-Queue Status                        ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${BOLD}Running:${NC}    ${GREEN}${running}${NC} / ${MAX_CONCURRENT}"
    echo -e "  ${BOLD}Queued:${NC}     ${YELLOW}${queued}${NC}"
    echo -e "  ${BOLD}Completed:${NC}  ${DIM}${done_count}${NC}"
    echo ""

    if [[ -f "$PID_FILE" ]]; then
        daemon_pid=$(cat "$PID_FILE")
        if kill -0 "$daemon_pid" 2>/dev/null; then
            echo -e "  ${GREEN}● Daemon running${NC} (PID ${daemon_pid})"
        else
            echo -e "  ${YELLOW}○ Daemon not running${NC} (stale PID file)"
        fi
    else
        echo -e "  ${YELLOW}○ Daemon not running${NC}"
    fi

    # Show queue state file info
    if [[ -f "$QUEUE_STATE_FILE" ]]; then
        wave_in_state=$(python3 -c "import json; print(json.load(open('${QUEUE_STATE_FILE}'))['wave'])" 2>/dev/null || echo "?")
        echo -e "  ${BOLD}Wave:${NC}       ${wave_in_state}"
    fi

    # Show queued experiments
    if [[ $queued -gt 0 ]]; then
        echo ""
        echo -e "  ${BOLD}Pending queue (next → first):${NC}"
        find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' 2>/dev/null | sort | head -20 | while read -r f; do
            echo -e "    ${DIM}→${NC} ${f%.yaml}"
        done
        if [[ $queued -gt 20 ]]; then
            echo -e "    ${DIM}... and $(( queued - 20 )) more${NC}"
        fi
    fi

    # Show tracked experiments
    if [[ -f "$TRACKED_PIDS_FILE" ]]; then
        active_count=0
        while IFS=' ' read -r pid name rest; do
            [[ "$pid" == "#"* || -z "$pid" ]] && continue
            if kill -0 "$pid" 2>/dev/null; then
                ((active_count++)) || true
            fi
        done < "$TRACKED_PIDS_FILE"

        if [[ $active_count -gt 0 ]]; then
            echo ""
            echo -e "  ${BOLD}Active experiments:${NC}"
            while IFS=' ' read -r pid name rest; do
                [[ "$pid" == "#"* || -z "$pid" ]] && continue
                if kill -0 "$pid" 2>/dev/null; then
                    # Get brief status from log
                    exp_log="${LOG_DIR}/${name}.log"
                    if [[ ! -f "$exp_log" ]]; then
                        exp_log=$(ls -t "${LOG_DIR}/queue_${name}"*.log "${LOG_DIR}/"*"_${name}.log" 2>/dev/null | head -1)
                    fi
                    gen="?"
                    if [[ -n "$exp_log" && -f "$exp_log" ]]; then
                        gen=$(tail -100 "$exp_log" 2>/dev/null | grep -oP 'GENERATION \K\d+/\d+' | tail -1 || echo "?")
                    fi
                    echo -e "    ${GREEN}●${NC} ${name} (PID ${pid}, gen ${gen})"
                else
                    echo -e "    ${DIM}○${NC} ${name} (PID ${pid}, finished)"
                fi
            done < "$TRACKED_PIDS_FILE"
        fi
    fi
    echo ""
    exit 0
fi

# ── Stop command ──
if [[ "$MODE" == "stop" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        daemon_pid=$(cat "$PID_FILE")
        if kill -0 "$daemon_pid" 2>/dev/null; then
            echo -e "${YELLOW}Sending SIGTERM to daemon (PID ${daemon_pid})...${NC}"
            kill -TERM "$daemon_pid"
            echo -e "${GREEN}Daemon stopped. Running experiments will continue to completion.${NC}"
        else
            echo -e "${YELLOW}Daemon not running (stale PID file). Cleaning up.${NC}"
            rm -f "$PID_FILE"
        fi
    else
        echo -e "${YELLOW}No daemon PID file found.${NC}"
    fi
    exit 0
fi

# ── Pre-flight checks ──
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo -e "${RED}ERROR: Virtual environment not found at ${VENV_DIR}${NC}"
    exit 1
fi

mkdir -p "$QUEUE_DIR" "$DONE_BASE" "$LOG_DIR"

# Check for existing daemon
if [[ -f "$PID_FILE" ]]; then
    existing_pid=$(cat "$PID_FILE")
    if kill -0 "$existing_pid" 2>/dev/null; then
        echo -e "${RED}ERROR: Daemon already running (PID ${existing_pid}). Use --stop first.${NC}"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# ── Determine wave name ──
if [[ -z "$WAVE_NAME" ]]; then
    WAVE_NAME=$(detect_next_wave)
fi

WAVE_OUTPUT="${OUTPUT_BASE}/${WAVE_NAME}"
WAVE_DONE="${DONE_BASE}/${WAVE_NAME}"
mkdir -p "$WAVE_OUTPUT" "$WAVE_DONE"

# ── Activate venv ──
source "${VENV_DIR}/bin/activate"

# ── Track our PID ──
echo $$ > "$PID_FILE"

# ── Initialize tracked PIDs file ──
if [[ ! -f "$TRACKED_PIDS_FILE" ]]; then
    echo "# Auto-queue tracked PIDs" > "$TRACKED_PIDS_FILE"
fi

# ── Save state for monitor ──
save_state() {
    python3 -c "
import json, time
state = {
    'wave': '${WAVE_NAME}',
    'max_concurrent': ${MAX_CONCURRENT},
    'persistent': $([ "$PERSISTENT" = true ] && echo 'True' || echo 'False'),
    'daemon_pid': $$,
    'started_at': '$(date -Iseconds)',
    'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'launched_count': ${LAUNCHED_COUNT:-0},
    'completed_count': ${COMPLETED_COUNT:-0},
    'failed_count': ${FAILED_COUNT:-0}
}
with open('${QUEUE_STATE_FILE}', 'w') as f:
    json.dump(state, f, indent=2)
" 2>/dev/null || true
}

# ── State ──
declare -A TRACKED=()       # pid -> config_basename
declare -A TRACKED_LOG=()   # pid -> log_path
declare -A TRACKED_START=() # pid -> start_timestamp
LAUNCHED_COUNT=0
COMPLETED_COUNT=0
FAILED_COUNT=0

# Load existing tracked PIDs
while IFS=' ' read -r pid name rest; do
    [[ "$pid" == "#"* ]] && continue
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
        TRACKED["$pid"]="$name"
    fi
done < "$TRACKED_PIDS_FILE" 2>/dev/null

# ── Functions ──

count_running_ga() {
    local count
    count=$(pgrep -cf "run_ga\.py" 2>/dev/null) || count=0
    echo "$count"
}

get_next_queued() {
    find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' 2>/dev/null | sort | head -1
}

extract_experiment_name() {
    # From "01_E19_e8_pop20.yaml" extract "E19_e8_pop20"
    local basename="$1"
    basename="${basename%.yaml}"
    # Strip priority prefix (digits followed by underscore)
    echo "$basename" | sed 's/^[0-9]*_//'
}

launch_experiment() {
    local config_basename="$1"
    local config_path="${QUEUE_DIR}/${config_basename}"
    local exp_name
    exp_name=$(extract_experiment_name "$config_basename")
    local full_name="${WAVE_NAME}_${exp_name}"

    # Per-experiment output directory
    local exp_output="${WAVE_OUTPUT}/${exp_name}"
    local exp_log="${LOG_DIR}/${full_name}.log"

    # Check config still exists
    if [[ ! -f "$config_path" ]]; then
        log "WARNING" "Config ${config_basename} no longer in queue — skipping"
        return 1
    fi

    mkdir -p "$exp_output"

    # Validate YAML before launching
    if ! python3 -c "import yaml; yaml.safe_load(open('${config_path}'))" 2>/dev/null; then
        log "ERROR" "YAML parse error in ${config_basename} — moving to done with ERROR tag"
        mv "$config_path" "${WAVE_DONE}/${config_basename%.yaml}_ERROR_$(date '+%Y%m%d_%H%M%S').yaml"
        ((FAILED_COUNT++))
        return 1
    fi

    # Move config to done directory BEFORE launching (prevents duplicate launches)
    local done_path="${WAVE_DONE}/${config_basename}"
    mv "$config_path" "$done_path"

    # Launch with setsid for process isolation
    export GA_OUTPUT_DIR="${exp_output}"

    setsid python genetic_algorithm/run_ga.py \
        --config "$done_path" \
        --no-monitor --yes \
        > "${exp_log}" 2>&1 &

    local pid=$!
    TRACKED["$pid"]="$full_name"
    TRACKED_LOG["$pid"]="$exp_log"
    TRACKED_START["$pid"]="$(date +%s)"

    # Record in tracked file
    echo "${pid} ${full_name}" >> "$TRACKED_PIDS_FILE"

    ((LAUNCHED_COUNT++))
    save_state

    log "INFO" "LAUNCHED: ${full_name} → PID ${pid}"
    log "INFO" "  Config: ${done_path}"
    log "INFO" "  Output: ${exp_output}"
    log "INFO" "  Log:    ${exp_log}"
    return 0
}

check_completed() {
    local any_completed=false
    for pid in "${!TRACKED[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local name="${TRACKED[$pid]}"
            local exp_log="${TRACKED_LOG[$pid]:-}"
            local start_ts="${TRACKED_START[$pid]:-0}"
            local duration="?"

            # Calculate duration
            if [[ "$start_ts" -gt 0 ]]; then
                local elapsed=$(( $(date +%s) - start_ts ))
                local hours=$(( elapsed / 3600 ))
                local mins=$(( (elapsed % 3600) / 60 ))
                duration="${hours}h${mins}m"
            fi

            # Get exit code
            wait "$pid" 2>/dev/null
            local exit_code=$?

            # Extract summary from log
            local best_fitness="?" profit="?" result_tag=""
            if [[ -n "$exp_log" && -f "$exp_log" ]]; then
                best_fitness=$(grep -oP '\[STATS\] Best: \K[0-9.]+' "$exp_log" 2>/dev/null | tail -1 || echo "?")
                profit=$(grep -oP 'total_profit.*?([0-9.-]+)' "$exp_log" 2>/dev/null | tail -1 | grep -oP '[0-9.-]+$' || echo "?")

                local safe warn overfit
                safe=$(grep -oP 'SAFE: \K\d+' "$exp_log" 2>/dev/null | tail -1)
                warn=$(grep -oP 'WARNING: \K\d+' "$exp_log" 2>/dev/null | tail -1)
                overfit=$(grep -oP 'OVERFIT: \K\d+' "$exp_log" 2>/dev/null | tail -1)
                if [[ -n "$safe" || -n "$warn" || -n "$overfit" ]]; then
                    result_tag=" [${safe:-0}S/${warn:-0}W/${overfit:-0}O]"
                fi
            fi

            if [[ $exit_code -eq 0 ]]; then
                log "INFO" "COMPLETED: ${name} (${duration}, fitness=${best_fitness})${result_tag}"
                ((COMPLETED_COUNT++))
            else
                log "WARNING" "FAILED: ${name} (exit ${exit_code}, ${duration})"
                ((FAILED_COUNT++))
            fi

            unset "TRACKED[$pid]"
            unset "TRACKED_LOG[$pid]"
            unset "TRACKED_START[$pid]"
            any_completed=true
        fi
    done

    if [[ "$any_completed" == true ]]; then
        # Rebuild tracked PIDs file
        {
            echo "# Auto-queue tracked PIDs (updated $(date))"
            for pid in "${!TRACKED[@]}"; do
                echo "${pid} ${TRACKED[$pid]}"
            done
        } > "$TRACKED_PIDS_FILE"
        save_state
    fi
}

# ── Graceful shutdown ──
cleanup() {
    echo ""
    log "INFO" "Received shutdown signal — daemon stopping"
    log "INFO" "Running experiments will continue to completion"
    log "INFO" "Stats: launched=${LAUNCHED_COUNT} completed=${COMPLETED_COUNT} failed=${FAILED_COUNT}"
    save_state
    rm -f "$PID_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Header ──
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   GA Auto-Queue Daemon v2                                    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Wave:${NC}            ${BOLD}${WAVE_NAME}${NC}"
echo -e "  ${BLUE}Max concurrent:${NC}  ${MAX_CONCURRENT}"
echo -e "  ${BLUE}Poll interval:${NC}   ${POLL_INTERVAL}s"
echo -e "  ${BLUE}Persistent:${NC}      ${PERSISTENT}"
echo -e "  ${BLUE}Queue dir:${NC}       ${QUEUE_DIR}"
echo -e "  ${BLUE}Done dir:${NC}        ${WAVE_DONE}"
echo -e "  ${BLUE}Output dir:${NC}      ${WAVE_OUTPUT}"
echo -e "  ${BLUE}Log file:${NC}        ${LOG_FILE}"
echo -e "  ${BLUE}Daemon PID:${NC}      $$"
echo ""

queued_count=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
echo -e "  ${BOLD}${queued_count} experiments queued${NC}"
if [[ $queued_count -gt 0 ]]; then
    find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' 2>/dev/null | sort | head -10 | while read -r f; do
        echo -e "    ${DIM}→${NC} ${f%.yaml}"
    done
    if [[ $queued_count -gt 10 ]]; then
        echo -e "    ${DIM}... and $(( queued_count - 10 )) more${NC}"
    fi
fi
echo ""

log "INFO" "Auto-queue daemon v2 started (wave=${WAVE_NAME}, max=${MAX_CONCURRENT}, persistent=${PERSISTENT})"
save_state

# Count existing GA processes
existing_running=$(count_running_ga)
if [[ $existing_running -gt 0 ]]; then
    log "INFO" "Detected ${existing_running} existing GA process(es) from previous runs"
fi

# ── Idle tracking ──
IDLE_SINCE=0
IDLE_NOTIFIED=false

# ── Main loop ──
while true; do
    check_completed

    current_running=$(count_running_ga)
    queued_count=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)

    # Launch new experiments if we have capacity
    slots_available=$((MAX_CONCURRENT - current_running))

    if [[ $slots_available -gt 0 && $queued_count -gt 0 ]]; then
        IDLE_SINCE=0
        IDLE_NOTIFIED=false
        to_launch=$((slots_available < queued_count ? slots_available : queued_count))
        for ((i=0; i<to_launch; i++)); do
            next_config=$(get_next_queued)
            if [[ -n "$next_config" ]]; then
                launch_experiment "$next_config" || true
                sleep 2  # Brief pause between launches
            fi
        done
    fi

    # Check if we're idle (nothing running from our tracking, nothing queued)
    if [[ $queued_count -eq 0 && ${#TRACKED[@]} -eq 0 ]]; then
        if [[ "$PERSISTENT" == true ]]; then
            # In persistent mode, keep watching for new configs
            if [[ $IDLE_SINCE -eq 0 ]]; then
                IDLE_SINCE=$(date +%s)
            fi
            if [[ "$IDLE_NOTIFIED" == false ]]; then
                log "INFO" "Queue empty — watching for new configs (persistent mode)"
                log "INFO" "Drop YAML files into: ${QUEUE_DIR}/"
                IDLE_NOTIFIED=true
            fi
            # Periodic idle reminder every 5 minutes
            now_ts=$(date +%s)
            if (( now_ts - IDLE_SINCE > 300 )) && (( (now_ts - IDLE_SINCE) % 300 < POLL_INTERVAL )); then
                idle_mins=$(( (now_ts - IDLE_SINCE) / 60 ))
                log "INFO" "Still idle (${idle_mins}m) — stats: launched=${LAUNCHED_COUNT} completed=${COMPLETED_COUNT} failed=${FAILED_COUNT}"
            fi
        else
            # Non-persistent: double-check then exit
            sleep 5
            queued_count=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
            if [[ $queued_count -eq 0 && ${#TRACKED[@]} -eq 0 ]]; then
                log "INFO" "All experiments complete — daemon exiting"
                log "INFO" "Stats: launched=${LAUNCHED_COUNT} completed=${COMPLETED_COUNT} failed=${FAILED_COUNT}"
                save_state
                rm -f "$PID_FILE"
                echo ""
                echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
                echo -e "${GREEN}║   All experiments complete!           ║${NC}"
                echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
                echo -e "  Launched:   ${LAUNCHED_COUNT}"
                echo -e "  Completed:  ${COMPLETED_COUNT}"
                echo -e "  Failed:     ${FAILED_COUNT}"
                echo -e "  Output:     ${WAVE_OUTPUT}"
                echo ""
                exit 0
            fi
        fi
    else
        IDLE_SINCE=0
        IDLE_NOTIFIED=false
    fi

    sleep "$POLL_INTERVAL"
done
