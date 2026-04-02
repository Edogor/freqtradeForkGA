#!/usr/bin/env bash
# ============================================================================
# GA Evolution Monitor v3 — Enhanced Live Dashboard
# ============================================================================
# Improvements over v2:
#   - Queue view: shows pending experiments and position
#   - Best profit display from completed experiments
#   - ETA estimation based on generation progress and elapsed time
#   - Overall summary stats (total runs, avg time, best fitness across all)
#   - Daemon status integration (shows auto_queue_v2 state)
#   - Discovers ALL log patterns (wave*, queue_*, ga_run_*, C*_*)
#   - Configurable detail level (compact/normal/detailed)
#
# Usage:
#   ./ga_monitor_v2.sh                       # auto-detect all running
#   ./ga_monitor_v2.sh wave16                # monitor specific wave only
#   ./ga_monitor_v2.sh --all                 # show completed experiments too
#   ./ga_monitor_v2.sh --queue               # show queue contents
#   ./ga_monitor_v2.sh --summary             # show overall summary only
#   ./ga_monitor_v2.sh --once                # print once and exit
#   ./ga_monitor_v2.sh --interval 10         # refresh every 10 seconds
#   ./ga_monitor_v2.sh --detail compact      # compact view (less columns)
#   ./ga_monitor_v2.sh --detail detailed     # show extra metrics
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
QUEUE_DIR="${REPO_DIR}/genetic_algorithm/config/queue"
DONE_BASE="${REPO_DIR}/genetic_algorithm/config/done"
STATE_FILE="${LOG_DIR}/auto_queue_state.json"
WAVE_FILTER=""
INTERVAL=5
ONCE=false
SHOW_ALL=false
SHOW_QUEUE=false
SHOW_SUMMARY_ONLY=false
DETAIL_LEVEL="normal"  # compact, normal, detailed

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)         ONCE=true ;;
        --all)          SHOW_ALL=true ;;
        --queue)        SHOW_QUEUE=true ;;
        --summary)      SHOW_SUMMARY_ONLY=true; ONCE=true ;;
        --detail)       DETAIL_LEVEL="${2:-normal}"; shift ;;
        --interval)     INTERVAL="${2:-5}"; shift ;;
        --help|-h)
            echo "Usage: $0 [wave_name] [OPTIONS]"
            echo ""
            echo "  wave_name           Filter to specific wave (e.g. wave16)"
            echo "  --all               Show completed experiments too"
            echo "  --queue             Show queue contents"
            echo "  --summary           Show overall summary stats only"
            echo "  --detail LEVEL      Detail level: compact, normal, detailed"
            echo "  --once              Print status once and exit"
            echo "  --interval N        Refresh interval in seconds (default: 5)"
            exit 0
            ;;
        -*)         echo "Unknown option: $1"; exit 1 ;;
        *)          WAVE_FILTER="$1" ;;
    esac
    shift
done

# ── Pad a plain string to N chars ──
spad() { printf "%-${2}s" "$1"; }

# ── Pad a string that may contain ANSI codes to visual width N ──
apad() {
    local str="$1" width="$2"
    local plain
    plain=$(printf '%b' "$str" | sed $'s/\x1b\\[[0-9;]*m//g')
    local vlen=${#plain}
    local need=$(( width - vlen ))
    if [[ $need -gt 0 ]]; then
        printf '%b%*s' "$str" "$need" ""
    else
        printf '%b' "$str"
    fi
}

# ── Truncate plain text to N chars with ellipsis ──
clip() {
    local str="$1" max="$2"
    if [[ ${#str} -gt $max ]]; then
        echo "${str:0:$(( max - 2 ))}.." 
    else
        echo "$str"
    fi
}

# ── Colours ──
RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
CYAN='\e[36m'
MAGENTA='\e[35m'
BOLD='\e[1m'
DIM='\e[2m'
NC='\e[0m'

# ── Discover running GA processes ──
declare -A RUNNING_PIDS=()

discover_running() {
    RUNNING_PIDS=()
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local pid cfg
        pid=$(echo "$line" | awk '{print $2}')
        cfg=$(echo "$line" | grep -oP '(?<=--config )\S+' || true)
        [[ -n "$cfg" && -n "$pid" ]] && RUNNING_PIDS["$cfg"]="$pid"
    done < <(ps aux 2>/dev/null | grep '[r]un_ga.py' || true)
}

# ── Discover log files (improved: all patterns) ──
discover_logs() {
    local -a patterns=()

    if [[ -n "$WAVE_FILTER" ]]; then
        patterns+=("${LOG_DIR}/${WAVE_FILTER}_*.log")
        patterns+=("${LOG_DIR}/queue_*${WAVE_FILTER}*.log")
    else
        # Discover all experiment logs
        patterns+=("${LOG_DIR}/wave*_*.log")
        patterns+=("${LOG_DIR}/C*_*.log")
        patterns+=("${LOG_DIR}/queue_*.log")
    fi

    local -a all_logs=()
    for pat in "${patterns[@]}"; do
        while IFS= read -r f; do
            [[ -n "$f" ]] && all_logs+=("$f")
        done < <(ls -1 $pat 2>/dev/null)
    done

    # Deduplicate: each experiment may produce two logs:
    #   GA internal log:  wave25_A1.log          (from Python logging.file config)
    #   Shell redirect:   wave25_A1_console.log  (stdout/stderr capture)
    #   or:               wave26_C1_island_scalp_1m.log
    # Group by canonical experiment ID (wave{N}_{LETTER}{DIGITS}) and keep the
    # GA internal log (shorter name) which has cleaner metrics data.
    local -A best_log=()
    for f in "${all_logs[@]}"; do
        local bn
        bn=$(basename "$f" .log)
        local canonical=""
        if [[ "$bn" =~ ^(wave[0-9]+_[A-Z][0-9]+)(_.+)?$ ]]; then
            canonical="${BASH_REMATCH[1]}"
        fi

        if [[ -n "$canonical" ]]; then
            if [[ -z "${best_log[$canonical]+x}" ]]; then
                best_log["$canonical"]="$f"
            else
                # Prefer shorter basename (GA internal log has cleaner data)
                local existing_bn
                existing_bn=$(basename "${best_log[$canonical]}" .log)
                if [[ ${#bn} -lt ${#existing_bn} ]]; then
                    best_log["$canonical"]="$f"
                fi
            fi
        else
            # Non-wave or non-standard naming — no dedup needed
            best_log["$bn"]="$f"
        fi
    done

    printf '%s\n' "${best_log[@]}" 2>/dev/null | sort
}

# ── Parse wave + experiment name from log path ──
parse_log_name() {
    local bn
    bn=$(basename "$1" .log)
    # Handle different naming patterns:
    # wave16_E125_name -> wave16 | E125_name
    # queue_01_E19_name -> queue | 01_E19_name
    # C3_island_llm -> C | C3_island_llm
    local wave exp
    if [[ "$bn" == wave* ]]; then
        wave="${bn%%_*}"
        exp="${bn#${wave}_}"
    elif [[ "$bn" == queue_* ]]; then
        wave="queue"
        exp="${bn#queue_}"
        # Strip priority number
        exp=$(echo "$exp" | sed 's/^[0-9]*_//')
    elif [[ "$bn" == C[0-9]* ]]; then
        wave="custom"
        exp="$bn"
    else
        wave="other"
        exp="$bn"
    fi
    echo "${wave}|${exp}"
}

# ── Find PID for an experiment ──
find_pid_for() {
    local exp="$1" wave="$2"
    for cfg in "${!RUNNING_PIDS[@]}"; do
        [[ "$cfg" == *"${exp}"* ]] && echo "${RUNNING_PIDS[$cfg]}" && return 0
    done
    # Check PID files
    local pf
    pf=$(ls -t "${LOG_DIR}/${wave}_pids_"*.txt 2>/dev/null | head -1)
    if [[ -n "$pf" && -f "$pf" ]]; then
        local p
        p=$(grep "$exp" "$pf" 2>/dev/null | awk '{print $1}')
        [[ -n "$p" ]] && echo "$p" && return 0
    fi
    # Check tracked PIDs file
    if [[ -f "${LOG_DIR}/auto_queue_tracked.txt" ]]; then
        local p
        p=$(grep "$exp" "${LOG_DIR}/auto_queue_tracked.txt" 2>/dev/null | awk '{print $1}')
        [[ -n "$p" ]] && echo "$p" && return 0
    fi
    return 1
}

# ── Extract metrics from log file ──
get_metrics() {
    local log="$1"
    [[ ! -f "$log" || ! -s "$log" ]] && echo "—|—|—|—|0|—|—|—|—|NO_LOG" && return

    local buf
    buf=$(tail -300 "$log" 2>/dev/null)

    # ── Generation ──
    local gen="init" gen_current=0 gen_total=0
    local gl
    gl=$(echo "$buf" | grep -oP 'GENERATION \d+/\d+' | tail -1 || true)
    if [[ -n "$gl" ]]; then
        gen="${gl#GENERATION }"
        gen_current=$(echo "$gen" | cut -d/ -f1)
        gen_total=$(echo "$gen" | cut -d/ -f2)
    fi

    # ── Eval sub-progress ──
    local evp
    evp=$(echo "$buf" | grep -oP '\[EVAL\] Progress: \K\d+/\d+' | tail -1 || true)

    # ── Best fitness ──
    local best="—"
    local v
    v=$(echo "$buf" | grep -oP '\[STATS\] Best: \K[0-9.]+' | tail -1 || true)
    if [[ -n "$v" ]]; then
        best="$v"
    else
        v=$(echo "$buf" | grep -oP '\[SUMMARY\].*master=\K[0-9.]+' | tail -1 || true)
        if [[ -n "$v" ]]; then
            best="$v"
        else
            v=$(echo "$buf" | grep -oP '\[NEW BEST\].*fitness.?\K[0-9.]+' | tail -1 || true)
            if [[ -n "$v" ]]; then
                best="$v"
            else
                # Island model format: [island_X_name] best=0.XXXX avg=... diversity=...
                # Try buf first (running); fall back to full log for completed runs
                v=$(echo "$buf" | grep -oP '(?<=\] )best=\K[0-9.]+' | sort -n | tail -1 || true)
                [[ -z "$v" ]] && v=$(grep -oP '(?<=\] )best=\K[0-9.]+' "$log" | sort -n | tail -1 || true)
                [[ -n "$v" ]] && best="$v"
            fi
        fi
    fi

    # ── Average fitness ──
    local avg="—"
    v=$(echo "$buf" | grep -oP '\[STATS\].*Avg: \K[0-9.]+' | tail -1 || true)
    if [[ -n "$v" ]]; then
        avg="$v"
    else
        # Island model format: mean of per-island avg= values (last gen block)
        v=$(echo "$buf" | grep -oP '(?<=\] )best=[0-9.]+ avg=\K[0-9.]+' \
            | tail -20 \
            | LC_NUMERIC=C awk '{s+=$1; c++} END {if(c>0) printf "%.4f", s/c}' || true)
        # Fall back to full log for completed runs where tail-300 has no stats
        [[ -z "$v" ]] && v=$(grep -oP '(?<=\] )best=[0-9.]+ avg=\K[0-9.]+' "$log" \
            | tail -20 \
            | LC_NUMERIC=C awk '{s+=$1; c++} END {if(c>0) printf "%.4f", s/c}' || true)
        [[ -n "$v" ]] && avg="$v"
    fi

    # ── Diversity ──
    local div="—"
    v=$(echo "$buf" | grep -oP 'Diversity: \K[0-9.]+' | tail -1 || true)
    if [[ -n "$v" ]]; then
        div="$v"
    else
        # Island model format: mean of per-island diversity= values (last gen block)
        v=$(echo "$buf" | grep -oP 'diversity=\K[0-9.]+' \
            | tail -20 \
            | LC_NUMERIC=C awk '{s+=$1; c++} END {if(c>0) printf "%.4f", s/c}' || true)
        # Fall back to full log for completed runs
        [[ -z "$v" ]] && v=$(grep -oP 'diversity=\K[0-9.]+' "$log" \
            | tail -20 \
            | LC_NUMERIC=C awk '{s+=$1; c++} END {if(c>0) printf "%.4f", s/c}' || true)
        [[ -n "$v" ]] && div="$v"
    fi

    # ── Best Profit ──
    # Use the highest profit from any island NEW BEST line (island model format)
    # e.g. [island_X_name] NEW BEST: fitness=0.XXXX profit=2.43%
    local profit="—"
    v=$(echo "$buf" | grep -oP 'NEW BEST: fitness=[0-9.]+ profit=\K-?[0-9.]+(?=%)' \
        | LC_NUMERIC=C awk 'BEGIN{best=-999} {if($1+0>best+0){best=$1}} END{if(best!=-999) printf "%+.2f%%", best}' || true)
    if [[ -n "$v" ]]; then
        profit="$v"
    else
        # Fall back to full log for completed runs
        v=$(grep -oP 'NEW BEST: fitness=[0-9.]+ profit=\K-?[0-9.]+(?=%)' "$log" \
            | LC_NUMERIC=C awk 'BEGIN{best=-999} {if($1+0>best+0){best=$1}} END{if(best!=-999) printf "%+.2f%%", best}' || true)
        if [[ -n "$v" ]]; then
            profit="$v"
        else
            # Old [STATS] / total_profit patterns
            v=$(echo "$buf" | grep -oP 'profit[_: ]+\K-?[0-9]+\.?[0-9]*%?' | tail -1 || true)
            [[ -z "$v" ]] && v=$(echo "$buf" | grep -oP 'total_profit.*?\K-?[0-9]+\.?[0-9]*' | tail -1 || true)
            [[ -n "$v" ]] && profit="$v"
        fi
    fi

    # ── Errors ──
    local errs
    errs=$(grep -cP '- (ERROR|CRITICAL) -|^Traceback' "$log" 2>/dev/null || echo 0)

    # ── Elapsed ──
    local elapsed="—" elapsed_secs=0
    local t0
    t0=$(head -1 "$log" 2>/dev/null | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
    if [[ -n "$t0" ]]; then
        local e0 e1
        e0=$(date -d "$t0" +%s 2>/dev/null || echo 0)
        if [[ "$e0" -gt 0 ]]; then
            if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE|ISLAND MODEL EVOLUTION FINISHED|\[SHUTDOWN\]|Graceful shutdown' "$log" 2>/dev/null; then
                # Completed/stopped: use last log timestamp for elapsed
                local tN
                tN=$(tac "$log" 2>/dev/null | grep -oP -m1 '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
                e1=$(date -d "${tN:-now}" +%s 2>/dev/null || date +%s)
            else
                # Check if log is stale (not modified recently and no running process)
                local log_age
                log_age=$(( $(date +%s) - $(stat -c%Y "$log" 2>/dev/null || echo 0) ))
                if [[ "$log_age" -gt 600 ]]; then
                    # Not running: use last log timestamp
                    local tN
                    tN=$(tac "$log" 2>/dev/null | grep -oP -m1 '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
                    e1=$(date -d "${tN:-now}" +%s 2>/dev/null || date +%s)
                else
                    e1=$(date +%s)
                fi
            fi
            elapsed_secs=$(( e1 - e0 ))
            [[ $elapsed_secs -lt 0 ]] && elapsed_secs=0
            local m=$(( elapsed_secs / 60 )) s=$(( elapsed_secs % 60 ))
            if [[ $m -ge 60 ]]; then
                elapsed="$(( m / 60 ))h$(( m % 60 ))m"
            else
                elapsed="${m}m${s}s"
            fi
        fi
    fi

    # ── ETA (only meaningful for RUNNING status) ──
    local eta="—"
    if [[ "$gen_current" -gt 1 && "$gen_total" -gt 0 && "$elapsed_secs" -gt 0 && "$gen_current" -lt "$gen_total" ]]; then
        local secs_per_gen=$(( elapsed_secs / gen_current ))
        local remaining_gens=$(( gen_total - gen_current ))
        local eta_secs=$(( secs_per_gen * remaining_gens ))
        local eta_m=$(( eta_secs / 60 ))
        if [[ $eta_m -ge 60 ]]; then
            eta="~$(( eta_m / 60 ))h$(( eta_m % 60 ))m"
        elif [[ $eta_m -gt 0 ]]; then
            eta="~${eta_m}m"
        else
            eta="<1m"
        fi
    elif [[ "$gen_current" -ge "$gen_total" && "$gen_total" -gt 0 ]]; then
        eta="<1m"
    fi

    # ── Completion results ──
    local results="—"
    local _log_for_results="$log"
    if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE|ISLAND MODEL EVOLUTION FINISHED' "$log" 2>/dev/null; then
        if ! grep -q 'SAFE:' "$log" 2>/dev/null; then
            local _bn _exp_name _qlog
            _bn=$(basename "$log" .log)
            _exp_name="${_bn#*_}"
            _qlog=$(ls -t "${LOG_DIR}"/queue_*"${_exp_name}"*.log 2>/dev/null | head -1)
            [[ -n "$_qlog" && -f "$_qlog" ]] && _log_for_results="$_qlog"
        fi
        local sa wa ov sc
        sa=$(grep -oP 'SAFE: \K\d+' "$_log_for_results" 2>/dev/null | tail -1)
        wa=$(grep -oP 'WARNING: \K\d+' "$_log_for_results" 2>/dev/null | tail -1)
        ov=$(grep -oP 'OVERFIT: \K\d+' "$_log_for_results" 2>/dev/null | tail -1)
        sc=$(grep -oP 'Avg composite score: \K[0-9.]+' "$_log_for_results" 2>/dev/null | tail -1)
        if [[ -n "$sa" || -n "$wa" || -n "$ov" ]]; then
            results="${sa:-0}S/${wa:-0}W/${ov:-0}O"
            [[ -n "$sc" ]] && results+=" sc=${sc}"
        fi
    fi

    # ── Status ──
    local status="RUNNING"
    if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE|ISLAND MODEL EVOLUTION FINISHED' "$log" 2>/dev/null; then
        local last_complete last_start
        last_complete=$(grep -nE 'GA RUN COMPLETE|EVOLUTION COMPLETE|ISLAND MODEL EVOLUTION FINISHED' "$log" 2>/dev/null | tail -1 | cut -d: -f1)
        last_start=$(grep -n 'GENETIC ALGORITHM STARTING' "$log" 2>/dev/null | tail -1 | cut -d: -f1)
        if [[ -n "$last_start" && -n "$last_complete" && "$last_start" -gt "$last_complete" ]]; then
            status="RUNNING"
        else
            status="DONE"
        fi
    elif tail -30 "$log" 2>/dev/null | grep -qP 'Traceback|KeyboardInterrupt|FATAL|ERROR DURING EVOLUTION|No space left on device'; then
        status="CRASHED"
    elif tail -30 "$log" 2>/dev/null | grep -qP '\[SHUTDOWN\]|Graceful shutdown'; then
        status="STOPPED"
    elif [[ "$gen_current" -gt 0 && "$gen_total" -gt 0 && "$gen_current" -ge "$gen_total" ]]; then
        status="DONE"
    fi

    # Sub-progress in gen display
    if [[ "$status" == "RUNNING" && -n "$evp" && "$gen" != "init" ]]; then
        gen="${gen} [${evp}]"
    fi

    echo "${gen}|${best}|${avg}|${div}|${errs}|${elapsed}|${eta}|${results}|${profit}|${status}"
}

# ── Print queue section ──
print_queue() {
    local queued
    queued=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)

    if [[ $queued -eq 0 ]]; then
        echo -e "  ${DIM}Queue is empty${NC}"
        return
    fi

    echo -e "  ${BOLD}${YELLOW}${queued} experiments queued:${NC}"
    echo ""

    local i=1
    find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' 2>/dev/null | sort | while read -r f; do
        local name="${f%.yaml}"
        local priority="${name%%_*}"
        local exp="${name#*_}"
        if [[ $i -le 20 ]]; then
            printf "    ${DIM}%2d.${NC} ${CYAN}[P%s]${NC} %s\n" "$i" "$priority" "$exp"
        fi
        ((i++))
    done

    if [[ $queued -gt 20 ]]; then
        echo -e "    ${DIM}... and $(( queued - 20 )) more${NC}"
    fi
    echo ""
}

# ── Print daemon status ──
print_daemon_status() {
    local daemon_running=false
    local daemon_wave="?" daemon_launched=0 daemon_completed=0 daemon_failed=0

    if [[ -f "$STATE_FILE" ]]; then
        daemon_wave=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('wave','?'))" 2>/dev/null || echo "?")
        daemon_launched=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('launched_count',0))" 2>/dev/null || echo 0)
        daemon_completed=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('completed_count',0))" 2>/dev/null || echo 0)
        daemon_failed=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('failed_count',0))" 2>/dev/null || echo 0)
    fi

    local pid_file="${LOG_DIR}/auto_queue_daemon.pid"
    if [[ -f "$pid_file" ]]; then
        local dpid
        dpid=$(cat "$pid_file")
        if kill -0 "$dpid" 2>/dev/null; then
            daemon_running=true
            echo -ne "  ${GREEN}● Queue daemon active${NC}"
        else
            echo -ne "  ${YELLOW}○ Queue daemon stopped${NC}"
        fi
    else
        echo -ne "  ${DIM}○ No queue daemon${NC}"
    fi

    echo -ne " ${DIM}|${NC} "

    if [[ "$daemon_wave" != "?" ]]; then
        echo -ne "wave=${BOLD}${daemon_wave}${NC} "
    fi
    echo -ne "launched=${daemon_launched} done=${daemon_completed}"
    [[ "$daemon_failed" -gt 0 ]] && echo -ne " ${RED}failed=${daemon_failed}${NC}"
    echo ""
}

# ── Main dashboard ──
print_dashboard() {
    [[ "$ONCE" == false ]] && printf '\033[2J\033[H'

    discover_running

    local -a logs=()
    while IFS= read -r f; do
        [[ -n "$f" ]] && logs+=("$f")
    done < <(discover_logs)

    local n_run=0 n_done=0 n_crash=0 n_hidden=0
    local global_best_fitness=0 global_best_exp=""
    local total_elapsed=0 completed_with_time=0

    # ── Title bar ──
    local title="GA EVOLUTION MONITOR"
    [[ -n "$WAVE_FILTER" ]] && title="GA MONITOR — ${WAVE_FILTER}"

    echo ""
    echo -e "  ${CYAN}╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -ne "  ${CYAN}║${NC} ${BOLD}"; apad "$title" 90; echo -e "${NC}${DIM}$(date '+%H:%M:%S')${NC} ${CYAN}║${NC}"
    echo -e "  ${CYAN}╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # ── Daemon status ──
    print_daemon_status
    echo ""

    # ── Queue section ──
    if [[ "$SHOW_QUEUE" == true ]] || [[ "$SHOW_SUMMARY_ONLY" == true ]]; then
        print_queue
    fi

    if [[ "$SHOW_SUMMARY_ONLY" == true ]]; then
        # Just show summary stats
        local total=0
        for lf in "${logs[@]}"; do
            local raw
            raw=$(get_metrics "$lf")
            IFS='|' read -r gen best avg div errs elapsed eta results profit status <<< "$raw"
            ((total++))
            case "$status" in
                RUNNING) ((n_run++)) ;;
                DONE)    ((n_done++)) ;;
                CRASHED) ((n_crash++)) ;;
            esac
            # Track best
            if [[ "$best" != "—" ]]; then
                if awk "BEGIN{exit(!($best > $global_best_fitness))}" 2>/dev/null; then
                    global_best_fitness="$best"
                    global_best_exp=$(basename "$lf" .log)
                fi
            fi
        done

        echo -e "  ${BOLD}Overall Summary:${NC}"
        echo -e "    Total experiments: ${total}"
        echo -e "    Running: ${GREEN}${n_run}${NC}  Done: ${n_done}  Crashed: ${RED}${n_crash}${NC}"
        echo -e "    Best fitness: ${GREEN}${global_best_fitness}${NC} (${global_best_exp})"
        echo ""
        return
    fi

    if [[ ${#logs[@]} -eq 0 ]]; then
        echo -e "  ${YELLOW}No experiment logs found.${NC}"
        [[ -n "$WAVE_FILTER" ]] && echo -e "  ${DIM}Searched: ${LOG_DIR}/${WAVE_FILTER}_*.log${NC}"
        echo -e "  ${DIM}Tip: $0 --all  |  $0 wave16${NC}"
        echo ""

        # Still show queue if not empty
        local queued
        queued=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
        [[ $queued -gt 0 ]] && print_queue
        return
    fi

    # ── Column header ──
    echo -ne "  "
    if [[ "$DETAIL_LEVEL" == "compact" ]]; then
        apad "${BOLD}WAVE${NC}"       6; echo -n " "
        apad "${BOLD}EXPERIMENT${NC}" 29; echo -n " "
        apad "${BOLD}GEN${NC}"        17; echo -n " "
        apad "${BOLD}BEST${NC}"       9; echo -n " "
        apad "${BOLD}TIME${NC}"       9; echo -n " "
        apad "${BOLD}ETA${NC}"        9; echo -n " "
        echo -e "${BOLD}STATUS${NC}"
        echo -e "  ${DIM}───── ───────────────────────────── ───────────────── ───────── ───────── ───────── ──────────${NC}"
    elif [[ "$DETAIL_LEVEL" == "detailed" ]]; then
        apad "${BOLD}WAVE${NC}"       6; echo -n " "
        apad "${BOLD}EXPERIMENT${NC}" 29; echo -n " "
        apad "${BOLD}GENERATION${NC}" 17; echo -n " "
        apad "${BOLD}BEST${NC}"       9; echo -n " "
        apad "${BOLD}AVG${NC}"        9; echo -n " "
        apad "${BOLD}DIV${NC}"        7; echo -n " "
        apad "${BOLD}PROFIT${NC}"     9; echo -n " "
        apad "${BOLD}ERR${NC}"        5; echo -n " "
        apad "${BOLD}TIME${NC}"       9; echo -n " "
        apad "${BOLD}ETA${NC}"        9; echo -n " "
        apad "${BOLD}RESULT${NC}"     22; echo -n " "
        echo -e "${BOLD}STATUS${NC}"
        echo -e "  ${DIM}───── ───────────────────────────── ───────────────── ───────── ───────── ─────── ───────── ───── ───────── ───────── ────────────────────── ──────────${NC}"
    else
        # normal
        apad "${BOLD}WAVE${NC}"       6; echo -n " "
        apad "${BOLD}EXPERIMENT${NC}" 29; echo -n " "
        apad "${BOLD}GENERATION${NC}" 17; echo -n " "
        apad "${BOLD}BEST${NC}"       9; echo -n " "
        apad "${BOLD}AVG${NC}"        9; echo -n " "
        apad "${BOLD}DIV${NC}"        7; echo -n " "
        apad "${BOLD}PROFIT${NC}"     9; echo -n " "
        apad "${BOLD}ERR${NC}"        5; echo -n " "
        apad "${BOLD}TIME${NC}"       9; echo -n " "
        apad "${BOLD}ETA${NC}"        9; echo -n " "
        apad "${BOLD}RESULT${NC}"     22; echo -n " "
        echo -e "${BOLD}STATUS${NC}"
        echo -e "  ${DIM}───── ───────────────────────────── ───────────────── ───────── ───────── ─────── ───────── ───── ───────── ───────── ────────────────────── ──────────${NC}"
    fi

    local prev_wave=""

    for lf in "${logs[@]}"; do
        local parsed wave exp
        parsed=$(parse_log_name "$lf")
        wave="${parsed%%|*}"
        exp="${parsed#*|}"

        local raw
        raw=$(get_metrics "$lf")
        IFS='|' read -r gen best avg div errs elapsed eta results profit status <<< "$raw"

        # Verify RUNNING with actual PID
        if [[ "$status" == "RUNNING" ]]; then
            local pid
            pid=$(find_pid_for "$exp" "$wave" 2>/dev/null || true)
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                : # confirmed alive
            elif [[ -n "$pid" ]]; then
                # PID was tracked but process is dead — not running
                status="STALE"
            else
                # No PID found — use log age as fallback
                local age
                age=$(( $(date +%s) - $(stat -c%Y "$lf" 2>/dev/null || echo 0) ))
                [[ $age -gt 300 ]] && status="STALE"
            fi
        fi

        # Clear ETA for non-running experiments
        if [[ "$status" == "STALE" || "$status" == "STOPPED" || "$status" == "DONE" ]]; then
            [[ "$status" != "DONE" ]] && eta="—"
        fi

        # Track global stats
        if [[ "$best" != "—" ]]; then
            if awk "BEGIN{exit(!($best > $global_best_fitness))}" 2>/dev/null; then
                global_best_fitness="$best"
                global_best_exp="$exp"
            fi
        fi

        # Visibility filter
        if [[ "$SHOW_ALL" == false && "$status" == "DONE" ]]; then
            if [[ -n "$WAVE_FILTER" ]]; then
                : # show DONE for requested wave
            else
                ((n_done++)); ((n_hidden++)); continue
            fi
        fi

        case "$status" in
            DONE)    ((n_done++)) ;;
            RUNNING) ((n_run++)) ;;
            CRASHED) ((n_crash++)) ;;
        esac

        # Wave separator
        if [[ "$wave" != "$prev_wave" && -n "$prev_wave" ]]; then
            if [[ "$DETAIL_LEVEL" == "compact" ]]; then
                echo -e "  ${DIM}───── ───────────────────────────── ───────────────── ───────── ───────── ───────── ──────────${NC}"
            elif [[ "$DETAIL_LEVEL" == "detailed" ]]; then
                echo -e "  ${DIM}───── ───────────────────────────── ───────────────── ───────── ───────── ─────── ───────── ───── ───────── ───────── ────────────────────── ──────────${NC}"
            else
                echo -e "  ${DIM}───── ───────────────────────────── ───────────────── ───────── ───────── ─────── ───────── ───── ───────── ───────── ────────────────────── ──────────${NC}"
            fi
        fi
        prev_wave="$wave"

        # ── Colorize ──
        local c_best c_err c_results c_status c_eta c_profit

        if [[ "$best" == "—" ]]; then
            c_best="${DIM}—${NC}"
        elif awk "BEGIN{exit(!($best >= 0.30))}" 2>/dev/null; then
            c_best="${GREEN}${best}${NC}"
        elif awk "BEGIN{exit(!($best >= 0.15))}" 2>/dev/null; then
            c_best="${YELLOW}${best}${NC}"
        else
            c_best="${RED}${best}${NC}"
        fi

        if [[ "${errs:-0}" -gt 0 ]] 2>/dev/null; then
            c_err="${RED}${errs}${NC}"
        else
            c_err="${DIM}0${NC}"
        fi

        if [[ "$eta" == "—" ]]; then
            c_eta="${DIM}—${NC}"
        else
            c_eta="${MAGENTA}${eta}${NC}"
        fi

        if [[ "$profit" == "—" ]]; then
            c_profit="${DIM}—${NC}"
        elif [[ "$profit" == -* ]]; then
            c_profit="${RED}${profit}${NC}"
        else
            c_profit="${GREEN}${profit}${NC}"
        fi

        case "$status" in
            RUNNING) c_status="${GREEN}● RUN${NC}" ;;
            DONE)    c_status="${GREEN}✓ DONE${NC}" ;;
            CRASHED) c_status="${RED}✗ CRASH${NC}" ;;
            STALE)   c_status="${YELLOW}? STALE${NC}" ;;
            STOPPED) c_status="${MAGENTA}■ STOP${NC}" ;;
            NO_LOG)  c_status="${DIM}— NOLOG${NC}" ;;
            *)       c_status="${DIM}${status}${NC}" ;;
        esac

        if [[ "$results" == "—" ]]; then
            c_results="${DIM}—${NC}"
        else
            local sc_val
            sc_val=$(echo "$results" | grep -oP 'sc=\K[0-9.]+' || true)
            if [[ -n "$sc_val" ]]; then
                if awk "BEGIN{exit(!($sc_val < 0.15))}" 2>/dev/null; then
                    c_results="${GREEN}${results}${NC}"
                elif awk "BEGIN{exit(!($sc_val < 0.25))}" 2>/dev/null; then
                    c_results="${YELLOW}${results}${NC}"
                else
                    c_results="${RED}${results}${NC}"
                fi
            else
                c_results="$results"
            fi
        fi

        # ── Print row ──
        local clipped_exp
        clipped_exp=$(clip "$exp" 28)
        echo -ne "  "
        if [[ "$DETAIL_LEVEL" == "compact" ]]; then
            apad "$wave"       5; echo -n "  "
            apad "$clipped_exp" 28; echo -n "  "
            apad "$gen"        16; echo -n "  "
            apad "$c_best"     8; echo -n "  "
            apad "${elapsed}"  8; echo -n "  "
            apad "$c_eta"      8; echo -n "  "
            echo -e "$c_status"
        elif [[ "$DETAIL_LEVEL" == "detailed" ]]; then
            apad "$wave"       5; echo -n "  "
            apad "$clipped_exp" 28; echo -n "  "
            apad "$gen"        16; echo -n "  "
            apad "$c_best"     8; echo -n "  "
            apad "${avg}"      8; echo -n "  "
            apad "${div}"      6; echo -n "  "
            apad "$c_profit"   8; echo -n "  "
            apad "$c_err"      4; echo -n "  "
            apad "${elapsed}"  8; echo -n "  "
            apad "$c_eta"      8; echo -n "  "
            apad "$c_results"  21; echo -n "  "
            echo -e "$c_status"
        else
            apad "$wave"       5; echo -n "  "
            apad "$clipped_exp" 28; echo -n "  "
            apad "$gen"        16; echo -n "  "
            apad "$c_best"     8; echo -n "  "
            apad "${avg}"      8; echo -n "  "
            apad "${div}"      6; echo -n "  "
            apad "$c_profit"   8; echo -n "  "
            apad "$c_err"      4; echo -n "  "
            apad "${elapsed}"  8; echo -n "  "
            apad "$c_eta"      8; echo -n "  "
            apad "$c_results"  21; echo -n "  "
            echo -e "$c_status"
        fi
    done

    echo ""

    # ── Queue peek ──
    if [[ "$SHOW_QUEUE" != true ]]; then
        local queued
        queued=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
        if [[ $queued -gt 0 ]]; then
            echo -ne "  ${YELLOW}Queue:${NC} ${queued} pending → "
            local next3
            next3=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' 2>/dev/null | sort | head -3 | sed 's/.yaml//g' | tr '\n' ', ' | sed 's/,$//')
            echo -e "${DIM}${next3}${NC}"
        fi
    fi

    # ── System bar ──
    local rss free load cores
    rss=$(ps aux 2>/dev/null | grep '[r]un_ga.py' | awk '{s+=$6} END{printf "%.0f",s/1024}' || echo 0)
    free=$(awk '/MemAvailable/{printf "%.0f",$2/1024}' /proc/meminfo 2>/dev/null || echo "?")
    load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo "?")
    cores=$(nproc 2>/dev/null || echo "?")

    echo -ne "  ${BLUE}System${NC}: "
    [[ $n_run -gt 0 ]] && echo -ne "${GREEN}${n_run} running${NC} · "
    [[ $n_done -gt 0 ]] && echo -ne "${DIM}${n_done} done${NC} · "
    [[ $n_crash -gt 0 ]] && echo -ne "${RED}${n_crash} crashed${NC} · "
    echo -e "RSS ${rss}MB · Free ${free}MB · Load ${load}/${cores}"

    # ── Global best highlight ──
    if [[ "$global_best_fitness" != "0" ]]; then
        echo -e "  ${BOLD}Best overall: ${GREEN}${global_best_fitness}${NC} ${DIM}(${global_best_exp})${NC}"
    fi

    [[ $n_hidden -gt 0 ]] && echo -e "  ${DIM}${n_hidden} completed experiments hidden (use --all to show)${NC}"

    if [[ "$ONCE" == false ]]; then
        echo -e "  ${DIM}Refreshing ${INTERVAL}s · Ctrl+C to stop · --queue to see pending · --detail compact|detailed${NC}"
    fi
    echo ""
}

# ── Entrypoint ──
if [[ "$ONCE" == true ]]; then
    print_dashboard
else
    trap 'echo ""; echo -e "  ${DIM}Monitor stopped.${NC}"; exit 0' INT TERM
    while true; do
        print_dashboard
        sleep "$INTERVAL"
    done
fi
