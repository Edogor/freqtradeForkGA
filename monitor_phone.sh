#!/usr/bin/env bash
# ============================================================================
# monitor_phone.sh — GA Evolution Monitor (Phone / NetHunter Edition)
# ============================================================================
# Optimized for small terminal screens (Kali NetHunter, SSH on mobile).
# Adapts layout to terminal width automatically.
#
# Default: prints all experiments as vertical cards once and exits.
#
# Usage:
#   ./monitor_phone.sh                   # card view, print once
#   ./monitor_phone.sh -t                # table mode (compact, width-adaptive)
#   ./monitor_phone.sh -a                # include DONE experiments
#   ./monitor_phone.sh -q                # show queue contents
#   ./monitor_phone.sh -s 10            # set refresh interval (used with -live)
#   ./monitor_phone.sh -live            # live refresh (card view, auto)
#   ./monitor_phone.sh -live -w 26      # live: wave 26 only
#   ./monitor_phone.sh -live -e A1      # live: one experiment detail + log tail
#   ./monitor_phone.sh -live -i C1      # live: island experiments (C-prefix)
#   ./monitor_phone.sh -live -l         # live: colored log tail of active runs
#   ./monitor_phone.sh -h               # help
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
QUEUE_DIR="${REPO_DIR}/genetic_algorithm/config/queue"
STATE_FILE="${LOG_DIR}/auto_queue_state.json"

# ── Runtime flags ──
TABLE_MODE=false
LIVE_MODE=false
SHOW_ALL=false
SHOW_QUEUE=false
INTERVAL=5
WAVE_FILTER=""
EXP_FILTER=""
ISLAND_FILTER=""
LOG_TAIL_MODE=false

# ── Arg parsing ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--table)     TABLE_MODE=true ;;
        -a|--all)       SHOW_ALL=true ;;
        -q|--queue)     SHOW_QUEUE=true ;;
        -s)             INTERVAL="${2:-5}"; shift ;;
        -live|--live)
            LIVE_MODE=true
            # Peek at next arg for sub-target
            while [[ $# -gt 1 ]]; do
                case "$2" in
                    -w) WAVE_FILTER="${3:-}"; shift 2 ;;
                    -e) EXP_FILTER="${3:-}"; shift 2 ;;
                    -i) ISLAND_FILTER="${3:-}"; shift 2 ;;
                    -l) LOG_TAIL_MODE=true; shift ;;
                    --interval|-s) INTERVAL="${3:-5}"; shift 2 ;;
                    *) break ;;
                esac
            done
            ;;
        -h|--help)
            cat <<'EOF'

  ┌──────────────────────────────────────────────────────────┐
  │  monitor_phone.sh — GA Monitor (Phone / NetHunter)       │
  └──────────────────────────────────────────────────────────┘

  BASIC USAGE
    ./monitor_phone.sh                  Card view, print once & exit
    ./monitor_phone.sh -t               Table mode (compact, width-adaptive)
    ./monitor_phone.sh -a               Include DONE experiments
    ./monitor_phone.sh -q               Show queue contents
    ./monitor_phone.sh -s N             Set refresh interval in seconds (default: 5)

  LIVE MODE  (-live)
    ./monitor_phone.sh -live            Live refresh, card view, all experiments
    ./monitor_phone.sh -live -t         Live refresh, table mode

    SUB-TARGETS (combine with -live):
    -w <waveNum>        Show only wave N (e.g. -live -w 26)
    -e <expName>        Detailed view for one experiment + live log tail
                        (e.g. -live -e A1  or  -live -e wave26_B2)
    -i <islandID>       Show island experiments matching C<N> prefix
                        (e.g. -live -i C1  or  -live -i C)
    -l                  Live colored log tail of all active experiments

  FORMATTING
    Terminal width is auto-detected via tput cols:
      ≤ 60 cols  → ultra-compact one-liner per experiment
      61-90 cols → medium card with abbreviated labels
      > 90 cols  → full card with all metrics

  OTHER
    -h | --help         Show this help

  EXAMPLES
    ./monitor_phone.sh -live -w 26 -s 3
    ./monitor_phone.sh -live -e wave26_C1 -s 10
    ./monitor_phone.sh -live -i C -s 5
    ./monitor_phone.sh -live -l
    ./monitor_phone.sh -t -a

  STATUS CODES
    ● RUN   = running (verified by PID)
    ✓ DONE  = evolution completed
    ✗ CRASH = crashed/traceback detected
    ? STALE = log not updated in 5+ min, PID dead
    — NOLOG = no log file found

  RESULT FORMAT (S/W/O)
    S = SAFE   W = WARNING   O = OVERFIT   sc = composite score (lower = better)

EOF
            exit 0
            ;;
        -*)  echo "Unknown option: $1  (use -h for help)"; exit 1 ;;
        *)   WAVE_FILTER="$1" ;;  # positional wave filter
    esac
    shift
done

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

# ── Utility functions ──
spad() { printf "%-${2}s" "$1"; }

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

clip() {
    local str="$1" max="$2"
    if [[ ${#str} -gt $max ]]; then
        echo "${str:0:$(( max - 2 ))}.."
    else
        echo "$str"
    fi
}

get_term_width() {
    local w
    w=$(tput cols 2>/dev/null || echo 80)
    echo "$w"
}

# ── Process discovery ──
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

# ── Log discovery ──
discover_logs() {
    local -a patterns=()

    if [[ -n "$WAVE_FILTER" ]]; then
        patterns+=("${LOG_DIR}/wave${WAVE_FILTER}_*.log")
        patterns+=("${LOG_DIR}/wave${WAVE_FILTER}.log")
        patterns+=("${LOG_DIR}/queue_*wave${WAVE_FILTER}*.log")
        # Also accept bare filter like "wave26"
        patterns+=("${LOG_DIR}/${WAVE_FILTER}_*.log")
    elif [[ -n "$ISLAND_FILTER" ]]; then
        patterns+=("${LOG_DIR}/wave*_${ISLAND_FILTER}*.log")
        patterns+=("${LOG_DIR}/wave*_${ISLAND_FILTER}_*.log")
    elif [[ -n "$EXP_FILTER" ]]; then
        patterns+=("${LOG_DIR}/*${EXP_FILTER}*.log")
    else
        patterns+=("${LOG_DIR}/wave*_*.log")
        patterns+=("${LOG_DIR}/C*_*.log")
        patterns+=("${LOG_DIR}/queue_*.log")
    fi

    local -a all_logs=()
    for pat in "${patterns[@]}"; do
        # shellcheck disable=SC2086
        while IFS= read -r f; do
            [[ -n "$f" && "$f" != *.archived && "$f" != *.bak ]] && all_logs+=("$f")
        done < <(ls -1 $pat 2>/dev/null)
    done

    # Deduplicate: prefer shorter canonical name (GA internal log)
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
                local existing_bn
                existing_bn=$(basename "${best_log[$canonical]}" .log)
                if [[ ${#bn} -lt ${#existing_bn} ]]; then
                    best_log["$canonical"]="$f"
                fi
            fi
        else
            best_log["$bn"]="$f"
        fi
    done

    printf '%s\n' "${best_log[@]}" 2>/dev/null | sort
}

parse_log_name() {
    local bn
    bn=$(basename "$1" .log)
    local wave exp
    if [[ "$bn" == wave* ]]; then
        wave="${bn%%_*}"
        exp="${bn#${wave}_}"
    elif [[ "$bn" == queue_* ]]; then
        wave="queue"
        exp="${bn#queue_}"
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

find_pid_for() {
    local exp="$1" wave="$2"
    for cfg in "${!RUNNING_PIDS[@]}"; do
        [[ "$cfg" == *"${exp}"* ]] && echo "${RUNNING_PIDS[$cfg]}" && return 0
    done
    local pf
    pf=$(ls -t "${LOG_DIR}/${wave}_pids_"*.txt 2>/dev/null | head -1)
    if [[ -n "$pf" && -f "$pf" ]]; then
        local p
        p=$(grep "$exp" "$pf" 2>/dev/null | awk '{print $1}')
        [[ -n "$p" ]] && echo "$p" && return 0
    fi
    return 1
}

# ── Metrics extraction ──
get_metrics() {
    local log="$1"
    [[ ! -f "$log" || ! -s "$log" ]] && echo "—|—|—|—|0|—|—|—|—|NO_LOG" && return

    local buf
    buf=$(tail -300 "$log" 2>/dev/null)

    # Generation
    local gen="init"
    local gl
    gl=$(echo "$buf" | grep -oP 'GENERATION \d+/\d+' | tail -1 || true)
    [[ -n "$gl" ]] && gen="${gl#GENERATION }"

    # Eval sub-progress
    local evp
    evp=$(echo "$buf" | grep -oP '\[EVAL\] Progress: \K\d+/\d+' | tail -1 || true)

    # Best fitness
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
            [[ -n "$v" ]] && best="$v"
        fi
    fi

    # Avg fitness
    local avg="—"
    v=$(echo "$buf" | grep -oP '\[STATS\].*Avg: \K[0-9.]+' | tail -1 || true)
    [[ -n "$v" ]] && avg="$v"

    # Diversity
    local div="—"
    v=$(echo "$buf" | grep -oP 'Diversity: \K[0-9.]+' | tail -1 || true)
    [[ -n "$v" ]] && div="$v"

    # Errors
    local errs
    errs=$(grep -cP '- (ERROR|CRITICAL) -|^Traceback' "$log" 2>/dev/null || echo 0)

    # Elapsed
    local elapsed="—"
    local t0
    t0=$(head -1 "$log" 2>/dev/null | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
    if [[ -n "$t0" ]]; then
        local e0 e1
        e0=$(date -d "$t0" +%s 2>/dev/null || echo 0)
        if [[ "$e0" -gt 0 ]]; then
            if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$log" 2>/dev/null; then
                local tN
                tN=$(tac "$log" 2>/dev/null | grep -oP -m1 '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
                e1=$(date -d "${tN:-now}" +%s 2>/dev/null || date +%s)
            else
                e1=$(date +%s)
            fi
            local ds=$(( e1 - e0 ))
            [[ $ds -lt 0 ]] && ds=0
            local m=$(( ds / 60 )) s=$(( ds % 60 ))
            if [[ $m -ge 60 ]]; then
                elapsed="$(( m / 60 ))h$(( m % 60 ))m"
            else
                elapsed="${m}m${s}s"
            fi
        fi
    fi

    # ETA
    local eta="—"
    if [[ "$gen" != "init" && "$gen" != "—" ]]; then
        local cur_gen tot_gen
        cur_gen=$(echo "$gen" | grep -oP '^\d+' || true)
        tot_gen=$(echo "$gen" | grep -oP '/\K\d+' || true)
        if [[ -n "$cur_gen" && -n "$tot_gen" && "$cur_gen" -gt 0 && "$tot_gen" -gt 0 && "$elapsed" != "—" ]]; then
            local elapsed_s
            if [[ "$elapsed" =~ ([0-9]+)h([0-9]+)m ]]; then
                elapsed_s=$(( BASH_REMATCH[1]*3600 + BASH_REMATCH[2]*60 ))
            elif [[ "$elapsed" =~ ([0-9]+)m([0-9]+)s ]]; then
                elapsed_s=$(( BASH_REMATCH[1]*60 + BASH_REMATCH[2] ))
            fi
            if [[ -n "${elapsed_s:-}" && $elapsed_s -gt 0 ]]; then
                local per_gen=$(( elapsed_s / cur_gen ))
                local rem=$(( (tot_gen - cur_gen) * per_gen ))
                if [[ $rem -ge 3600 ]]; then
                    eta="$(( rem/3600 ))h$(( (rem%3600)/60 ))m"
                elif [[ $rem -ge 60 ]]; then
                    eta="$(( rem/60 ))m"
                else
                    eta="${rem}s"
                fi
            fi
        fi
    fi

    # Profit (from backtest/result)
    local profit="—"
    v=$(echo "$buf" | grep -oP '\[RESULT\].*profit[= ]+\K[-0-9.]+' | tail -1 || true)
    [[ -n "$v" ]] && profit="${v}%"

    # Completion results (S/W/O)
    local results="—"
    if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$log" 2>/dev/null; then
        local sa wa ov sc
        sa=$(grep -oP 'SAFE: \K\d+' "$log" 2>/dev/null | tail -1)
        wa=$(grep -oP 'WARNING: \K\d+' "$log" 2>/dev/null | tail -1)
        ov=$(grep -oP 'OVERFIT: \K\d+' "$log" 2>/dev/null | tail -1)
        sc=$(grep -oP 'Avg composite score: \K[0-9.]+' "$log" 2>/dev/null | tail -1)
        if [[ -n "$sa" || -n "$wa" || -n "$ov" ]]; then
            results="${sa:-0}S/${wa:-0}W/${ov:-0}O sc=${sc:-?}"
        fi
    fi

    # Status
    local status="RUNNING"
    if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$log" 2>/dev/null; then
        local last_complete last_start
        last_complete=$(grep -nE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$log" 2>/dev/null | tail -1 | cut -d: -f1)
        last_start=$(grep -n 'GENETIC ALGORITHM STARTING' "$log" 2>/dev/null | tail -1 | cut -d: -f1)
        if [[ -n "$last_start" && -n "$last_complete" && "$last_start" -gt "$last_complete" ]]; then
            status="RUNNING"
        else
            status="DONE"
        fi
    elif tail -30 "$log" 2>/dev/null | grep -qP 'Traceback|KeyboardInterrupt|FATAL'; then
        status="CRASHED"
    fi

    # Append eval sub-progress to gen display
    if [[ "$status" == "RUNNING" && -n "$evp" && "$gen" != "init" ]]; then
        gen="${gen} [${evp}]"
    fi

    echo "${gen}|${best}|${avg}|${div}|${errs}|${elapsed}|${eta}|${results}|${profit}|${status}"
}

# ── Status verification ──
verify_status() {
    local status="$1" exp="$2" wave="$3" log="$4"
    if [[ "$status" == "RUNNING" ]]; then
        local pid
        pid=$(find_pid_for "$exp" "$wave" 2>/dev/null || true)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "RUNNING"
        elif [[ -n "$pid" ]]; then
            echo "STALE"
        else
            local age
            age=$(( $(date +%s) - $(stat -c%Y "$log" 2>/dev/null || echo 0) ))
            [[ $age -gt 300 ]] && echo "STALE" || echo "RUNNING"
        fi
    else
        echo "$status"
    fi
}

# ── Color helpers ──
color_fitness() {
    local v="$1"
    if [[ "$v" == "—" ]]; then echo "${DIM}—${NC}"; return; fi
    if awk "BEGIN{exit(!($v >= 0.30))}" 2>/dev/null; then
        echo "${GREEN}${v}${NC}"
    elif awk "BEGIN{exit(!($v >= 0.15))}" 2>/dev/null; then
        echo "${YELLOW}${v}${NC}"
    else
        echo "${RED}${v}${NC}"
    fi
}

color_status() {
    case "$1" in
        RUNNING) echo "${GREEN}● RUN${NC}" ;;
        DONE)    echo "${GREEN}✓ DONE${NC}" ;;
        CRASHED) echo "${RED}✗ CRASH${NC}" ;;
        STALE)   echo "${YELLOW}? STALE${NC}" ;;
        STOPPED) echo "${MAGENTA}■ STOP${NC}" ;;
        NO_LOG)  echo "${DIM}— NOLOG${NC}" ;;
        *)       echo "${DIM}${1}${NC}" ;;
    esac
}

color_errors() {
    local e="$1"
    if [[ "${e:-0}" -gt 0 ]] 2>/dev/null; then
        echo "${RED}${e}${NC}"
    else
        echo "${DIM}0${NC}"
    fi
}

color_results() {
    local r="$1"
    [[ "$r" == "—" ]] && echo "${DIM}—${NC}" && return
    local sc_val
    sc_val=$(echo "$r" | grep -oP 'sc=\K[0-9.]+' || true)
    if [[ -n "$sc_val" ]]; then
        if awk "BEGIN{exit(!($sc_val < 0.15))}" 2>/dev/null; then
            echo "${GREEN}${r}${NC}"
        elif awk "BEGIN{exit(!($sc_val < 0.25))}" 2>/dev/null; then
            echo "${YELLOW}${r}${NC}"
        else
            echo "${RED}${r}${NC}"
        fi
    else
        echo "$r"
    fi
}

# ── Header banner ──
print_banner() {
    local width
    width=$(get_term_width)
    local title="${1:-GA EVOLUTION MONITOR}"
    local ts
    ts=$(date '+%H:%M:%S')
    local inner=$(( width - 4 ))

    if [[ $width -le 60 ]]; then
        echo -e "${CYAN}━━━ ${BOLD}${title}${NC}${CYAN} ━━━ ${DIM}${ts}${NC}"
    else
        # Bordered banner
        local border
        border=$(printf '═%.0s' $(seq 1 $(( width - 2 ))))
        echo -e "${CYAN}╔${border}╗${NC}"
        local padded
        padded=$(printf "%-$(( inner - ${#ts} - 1 ))s %s" "$title" "$ts")
        echo -e "${CYAN}║${NC} ${BOLD}${padded}${NC} ${CYAN}║${NC}"
        echo -e "${CYAN}╚${border}╝${NC}"
    fi
}

# ── Daemon / system line ──
print_system_line() {
    local rss free load cores
    rss=$(ps aux 2>/dev/null | { grep '[r]un_ga.py' || true; } | awk '{s+=$6} END{printf "%.0f",s/1024}')
    [[ -z "$rss" ]] && rss=0
    free=$(awk '/MemAvailable/{printf "%.0f",$2/1024}' /proc/meminfo 2>/dev/null || echo "?")
    load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo "?")
    cores=$(nproc 2>/dev/null || echo "?")

    echo -ne "${BLUE}Sys${NC}: RSS ${rss}MB · Free ${free}MB · Load ${load}/${cores}"
    echo ""
}

print_daemon_line() {
    local pid_file="${LOG_DIR}/auto_queue_daemon.pid"
    if [[ -f "$pid_file" ]]; then
        local dpid
        dpid=$(cat "$pid_file")
        if kill -0 "$dpid" 2>/dev/null; then
            echo -ne "${GREEN}● Daemon${NC}"
        else
            echo -ne "${YELLOW}○ Daemon${NC}"
        fi
    else
        echo -ne "${DIM}○ No daemon${NC}"
    fi

    if [[ -f "$STATE_FILE" ]]; then
        local dwave dlaunched ddone dfailed
        dwave=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('wave','?'))" 2>/dev/null || echo "?")
        dlaunched=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('launched_count',0))" 2>/dev/null || echo 0)
        ddone=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('completed_count',0))" 2>/dev/null || echo 0)
        dfailed=$(python3 -c "import json; d=json.load(open('${STATE_FILE}')); print(d.get('failed_count',0))" 2>/dev/null || echo 0)
        echo -ne " ${DIM}|${NC} wave=${BOLD}${dwave}${NC} launched=${dlaunched} done=${ddone}"
        [[ "$dfailed" -gt 0 ]] && echo -ne " ${RED}failed=${dfailed}${NC}"
    fi
    echo ""
}

# ── Queue section ──
print_queue() {
    local queued
    queued=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
    local width
    width=$(get_term_width)

    if [[ $queued -eq 0 ]]; then
        echo -e "${DIM}Queue: empty${NC}"
        return
    fi

    echo -e "${BOLD}${YELLOW}Queue: ${queued} pending${NC}"
    local i=1
    find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' 2>/dev/null | sort | while read -r f; do
        local name="${f%.yaml}"
        local priority="${name%%_*}"
        local exp="${name#*_}"
        if [[ $i -le 15 ]]; then
            local clipped_exp
            clipped_exp=$(clip "$exp" $(( width - 14 )))
            printf "  ${DIM}%2d.${NC} ${CYAN}[P%s]${NC} %s\n" "$i" "$priority" "$clipped_exp"
        fi
        ((i++))
    done
    [[ $queued -gt 15 ]] && echo -e "  ${DIM}...and $(( queued - 15 )) more${NC}"
    echo ""
}

# ═════════════════════════════════════════════════════════════════════════════
# CARD VIEW — one card per experiment (vertical layout, phone-friendly)
# ═════════════════════════════════════════════════════════════════════════════

print_card_narrow() {
    # Ultra-compact one-liner for ≤60 cols: WAVE EXP GEN BEST STATUS
    local wave="$1" exp="$2" gen="$3" best="$4" elapsed="$5" status="$6"
    local c_best c_status
    c_best=$(color_fitness "$best")
    c_status=$(color_status "$status")
    local short_exp
    short_exp=$(clip "$exp" 14)
    printf "  ${DIM}%-6s${NC} %-14s %-9s %b  %b\n" \
        "$wave" "$short_exp" "$gen" "$c_best" "$c_status"
}

print_card_medium() {
    # Medium card for 61-90 cols
    # Body rows omit the right border to avoid ANSI-padding misalignment.
    local wave="$1" exp="$2" gen="$3" best="$4" avg="$5" div="$6" \
          errs="$7" elapsed="$8" eta="$9" results="${10}" profit="${11}" status="${12}"
    local width
    width=$(get_term_width)
    local inner=$(( width - 4 ))
    local border
    border=$(printf '─%.0s' $(seq 1 $inner))

    local c_best c_status c_err c_results
    c_best=$(color_fitness "$best")
    c_status=$(color_status "$status")
    c_err=$(color_errors "$errs")
    c_results=$(color_results "$results")

    echo -e "  ${CYAN}┌${border}┐${NC}"
    local title="${wave} › ${exp}"
    local title_clipped
    title_clipped=$(clip "$title" $(( inner - 2 )))
    printf "  ${CYAN}│${NC} ${BOLD}%-$(( inner - 2 ))s${NC} ${CYAN}│${NC}\n" "$title_clipped"
    echo -e "  ${CYAN}├${border}┤${NC}"
    echo -ne "  ${CYAN}│${NC}  "; printf "%-8s" "Gen:";    echo -n " "; printf "%-17s" "$gen";     echo -ne "  "; printf "%-8s" "Best:";   echo -n " "; printf '%b' "$c_best";   echo ""
    echo -ne "  ${CYAN}│${NC}  "; printf "%-8s" "Time:";   echo -n " "; printf "%-17s" "$elapsed"; echo -ne "  "; printf "%-8s" "ETA:";    echo -n " "; echo -e "$eta"
    echo -ne "  ${CYAN}│${NC}  "; printf "%-8s" "Status:"; echo -n " "; printf '%b' "$c_status";   echo -ne "    ";                   printf "%-8s" "Errors:"; echo -n " "; printf '%b' "$c_err"; echo ""
    [[ "$results" != "—" ]] && {
        echo -ne "  ${CYAN}│${NC}  "; printf "%-8s" "Result:"; echo -n " "; printf '%b' "$c_results"; echo ""
    }
    echo -e "  ${CYAN}└${border}┘${NC}"
    echo ""
}

print_card_wide() {
    # Full card for >90 cols
    local wave="$1" exp="$2" gen="$3" best="$4" avg="$5" div="$6" \
          errs="$7" elapsed="$8" eta="$9" results="${10}" profit="${11}" status="${12}"
    local width
    width=$(get_term_width)
    local inner=$(( width - 4 ))
    local border
    border=$(printf '─%.0s' $(seq 1 $inner))

    local c_best c_status c_err c_results c_profit
    c_best=$(color_fitness "$best")
    c_status=$(color_status "$status")
    c_err=$(color_errors "$errs")
    c_results=$(color_results "$results")
    if [[ "$profit" == "—" ]]; then c_profit="${DIM}—${NC}"; elif [[ "$profit" == -* ]]; then c_profit="${RED}${profit}${NC}"; else c_profit="${GREEN}${profit}${NC}"; fi

    echo -e "  ${CYAN}┌${border}┐${NC}"
    local title="${wave}  ›  ${exp}"
    printf "  ${CYAN}│${NC} ${BOLD}%-$(( inner - 1 ))s${NC}${CYAN}│${NC}\n" "$title"
    echo -e "  ${CYAN}├${border}┤${NC}"

    # Two-column metric rows
    _card_row "$inner" "Generation" "$gen"              "Best fit."   "$c_best"
    _card_row "$inner" "Avg fit."   "$avg"              "Diversity"   "$div"
    _card_row "$inner" "Elapsed"    "$elapsed"          "ETA"         "$eta"
    _card_row "$inner" "Status"     "$c_status"         "Errors"      "$c_err"
    [[ "$profit" != "—" ]] && \
    _card_row "$inner" "Profit"     "$c_profit"         "Results"     "$c_results"
    [[ "$results" != "—" && "$profit" == "—" ]] && \
    _card_row "$inner" "Results"    "$c_results"        ""            ""

    echo -e "  ${CYAN}└${border}┘${NC}"
    echo ""
}

# Helper: two-column row inside a card
_card_row() {
    local inner="$1" label1="$2" val1="$3" label2="$4" val2="$5"
    local half=$(( inner / 2 - 1 ))
    local plain1 plain2 vlen1 vlen2 pad1 pad2
    plain1=$(printf '%b' "$val1" | sed $'s/\x1b\\[[0-9;]*m//g')
    plain2=$(printf '%b' "$val2" | sed $'s/\x1b\\[[0-9;]*m//g')
    vlen1=${#plain1}; vlen2=${#plain2}

    printf "  ${CYAN}│${NC}  ${DIM}%-12s${NC}" "$label1"
    printf '%b' "$val1"
    pad1=$(( half - 14 - vlen1 ))
    [[ $pad1 -gt 0 ]] && printf '%*s' "$pad1" ""

    if [[ -n "$label2" ]]; then
        printf "  ${DIM}%-12s${NC}" "$label2"
        printf '%b' "$val2"
        pad2=$(( inner - half - 14 - vlen2 - 3 ))
        [[ $pad2 -gt 0 ]] && printf '%*s' "$pad2" ""
    fi
    printf "  ${CYAN}│${NC}\n"
}

# ── Dispatch to the correct card variant ──
print_card() {
    local wave="$1" exp="$2" gen="$3" best="$4" avg="$5" div="$6" \
          errs="$7" elapsed="$8" eta="$9" results="${10}" profit="${11}" status="${12}"
    local width
    width=$(get_term_width)
    if [[ $width -le 60 ]]; then
        print_card_narrow "$wave" "$exp" "$gen" "$best" "$elapsed" "$status"
    elif [[ $width -le 90 ]]; then
        print_card_medium "$wave" "$exp" "$gen" "$best" "$avg" "$div" \
            "$errs" "$elapsed" "$eta" "$results" "$profit" "$status"
    else
        print_card_wide "$wave" "$exp" "$gen" "$best" "$avg" "$div" \
            "$errs" "$elapsed" "$eta" "$results" "$profit" "$status"
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# TABLE VIEW — one compact line per experiment
# ═════════════════════════════════════════════════════════════════════════════

print_table_header() {
    local width
    width=$(get_term_width)
    echo ""
    if [[ $width -le 60 ]]; then
        printf "  ${BOLD}%-6s %-12s %-8s %-7s %-8s${NC}\n" \
            "WAVE" "EXP" "GEN" "BEST" "STATUS"
        echo -e "  ${DIM}────── ──────────── ──────── ─────── ────────${NC}"
    elif [[ $width -le 90 ]]; then
        printf "  ${BOLD}%-6s %-18s %-12s %-7s %-7s %-5s %-7s %-9s${NC}\n" \
            "WAVE" "EXP" "GEN" "BEST" "TIME" "ERR" "ETA" "STATUS"
        echo -e "  ${DIM}────── ────────────────── ──────────── ─────── ─────── ───── ─────── ─────────${NC}"
    else
        printf "  ${BOLD}%-6s %-24s %-16s %-8s %-8s %-7s %-5s %-8s %-7s %-22s %-9s${NC}\n" \
            "WAVE" "EXPERIMENT" "GENERATION" "BEST" "AVG" "DIV" "ERR" "TIME" "ETA" "RESULT" "STATUS"
        echo -e "  ${DIM}────── ──────────────────────── ──────────────── ──────── ──────── ─────── ───── ──────── ─────── ────────────────────── ─────────${NC}"
    fi
}

print_table_row() {
    local wave="$1" exp="$2" gen="$3" best="$4" avg="$5" div="$6" \
          errs="$7" elapsed="$8" eta="$9" results="${10}" profit="${11}" status="${12}"
    local width
    width=$(get_term_width)

    local c_best c_status c_err c_results
    c_best=$(color_fitness "$best")
    c_status=$(color_status "$status")
    c_err=$(color_errors "$errs")
    c_results=$(color_results "$results")

    local short_exp short_gen
    if [[ $width -le 60 ]]; then
        short_exp=$(clip "$exp" 12)
        short_gen=$(clip "$gen" 8)
        printf "  %-6s %-12s %-8s " "$wave" "$short_exp" "$short_gen"
        apad "$c_best" 7; echo -n " "
        apad "$c_status" 8; echo ""
    elif [[ $width -le 90 ]]; then
        short_exp=$(clip "$exp" 18)
        short_gen=$(clip "$gen" 12)
        printf "  %-6s %-18s %-12s " "$wave" "$short_exp" "$short_gen"
        apad "$c_best" 7; echo -n " "
        printf "%-7s " "$elapsed"
        apad "$c_err" 5; echo -n " "
        printf "%-7s " "$eta"
        apad "$c_status" 9; echo ""
    else
        short_exp=$(clip "$exp" 24)
        short_gen=$(clip "$gen" 16)
        printf "  %-6s %-24s %-16s " "$wave" "$short_exp" "$short_gen"
        apad "$c_best" 8; echo -n " "
        printf "%-8s %-7s " "$avg" "$div"
        apad "$c_err" 5; echo -n " "
        printf "%-8s %-7s " "$elapsed" "$eta"
        apad "$c_results" 22; echo -n " "
        apad "$c_status" 9; echo ""
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DETAIL VIEW  (-live -e)
# ═════════════════════════════════════════════════════════════════════════════

print_experiment_detail() {
    local log_file="$1" wave="$2" exp="$3"
    local width
    width=$(get_term_width)

    local raw
    raw=$(get_metrics "$log_file")
    IFS='|' read -r gen best avg div errs elapsed eta results profit status <<< "$raw"
    status=$(verify_status "$status" "$exp" "$wave" "$log_file")

    # Always use wide card in detail mode
    print_card_wide "$wave" "$exp" "$gen" "$best" "$avg" "$div" \
        "$errs" "$elapsed" "$eta" "$results" "$profit" "$status"

    # Log tail section
    local inner=$(( width - 4 ))
    local border
    border=$(printf '─%.0s' $(seq 1 $inner))
    echo -e "  ${CYAN}┌${border}┐${NC}"
    printf "  ${CYAN}│${NC} ${BOLD}%-$(( inner - 1 ))s${NC}${CYAN}│${NC}\n" "Recent Log Output"
    echo -e "  ${CYAN}├${border}┤${NC}"

    local log_lines
    log_lines=$(tail -20 "$log_file" 2>/dev/null)
    while IFS= read -r line; do
        # Colorize log levels
        local colored_line="$line"
        if echo "$line" | grep -qP '\[ERROR\]|\- ERROR \-|CRITICAL'; then
            colored_line="${RED}${line}${NC}"
        elif echo "$line" | grep -qP '\[WARNING\]|\- WARNING \-'; then
            colored_line="${YELLOW}${line}${NC}"
        elif echo "$line" | grep -qP '\[NEW BEST\]|GENERATION|EVOLUTION COMPLETE|GA RUN COMPLETE'; then
            colored_line="${GREEN}${line}${NC}"
        elif echo "$line" | grep -qP '\[STATS\]|\[EVAL\]'; then
            colored_line="${CYAN}${line}${NC}"
        else
            colored_line="${DIM}${line}${NC}"
        fi
        # Clip to terminal width
        local plain_line
        plain_line=$(printf '%b' "$colored_line" | sed $'s/\x1b\\[[0-9;]*m//g')
        local clip_len=$(( inner - 4 ))
        if [[ ${#plain_line} -gt $clip_len ]]; then
            colored_line="${colored_line:0:$clip_len}.."
        fi
        printf "  ${CYAN}│${NC} %b\n" "$colored_line"
    done <<< "$log_lines"

    echo -e "  ${CYAN}└${border}┘${NC}"
    echo ""
}

# ═════════════════════════════════════════════════════════════════════════════
# LOG TAIL VIEW  (-live -l)
# ═════════════════════════════════════════════════════════════════════════════

print_log_tail() {
    local width
    width=$(get_term_width)
    local clip_len=$(( width - 3 ))

    # Find active logs
    local -a active_logs=()
    while IFS= read -r lf; do
        [[ -z "$lf" ]] && continue
        if grep -qE 'GENETIC ALGORITHM STARTING' "$lf" 2>/dev/null && \
           ! grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$lf" 2>/dev/null; then
            active_logs+=("$lf")
        fi
    done < <(discover_logs)

    if [[ ${#active_logs[@]} -eq 0 ]]; then
        echo -e "${DIM}No active experiments found.${NC}"
        return
    fi

    for lf in "${active_logs[@]}"; do
        local parsed wave exp
        parsed=$(parse_log_name "$lf")
        wave="${parsed%%|*}"; exp="${parsed#*|}"

        local inner=$(( width - 4 ))
        local border
        border=$(printf '─%.0s' $(seq 1 $inner))
        echo -e "${CYAN}┌${border}┐${NC}"
        printf "${CYAN}│${NC} ${BOLD}%-$(( inner - 1 ))s${NC}${CYAN}│${NC}\n" "${wave} › ${exp}"
        echo -e "${CYAN}├${border}┤${NC}"

        tail -10 "$lf" 2>/dev/null | while IFS= read -r line; do
            local colored="${DIM}${line}${NC}"
            echo "$line" | grep -qP '\[ERROR\]|CRITICAL|Traceback' && colored="${RED}${line}${NC}"
            echo "$line" | grep -qP '\[WARNING\]'                  && colored="${YELLOW}${line}${NC}"
            echo "$line" | grep -qP '\[NEW BEST\]|GEN|COMPLETE'   && colored="${GREEN}${line}${NC}"
            echo "$line" | grep -qP '\[STATS\]|\[EVAL\]'          && colored="${CYAN}${line}${NC}"
            local plain
            plain=$(printf '%b' "$colored" | sed $'s/\x1b\\[[0-9;]*m//g')
            [[ ${#plain} -gt $clip_len ]] && colored="${colored:0:$clip_len}.."
            printf "${CYAN}│${NC} %b\n" "$colored"
        done

        echo -e "${CYAN}└${border}┘${NC}"
        echo ""
    done
}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD — collects experiments and dispatches to card/table
# ═════════════════════════════════════════════════════════════════════════════

print_dashboard() {
    [[ "$LIVE_MODE" == true ]] && printf '\033[2J\033[H'

    discover_running

    # Route to specialized views first
    if [[ "$LIVE_MODE" == true && "$LOG_TAIL_MODE" == true ]]; then
        print_banner "GA MONITOR › LOG TAIL"
        echo ""
        print_log_tail
        print_system_line
        return
    fi

    # Collect logs
    local -a logs=()
    while IFS= read -r f; do
        [[ -n "$f" ]] && logs+=("$f")
    done < <(discover_logs)

    local n_run=0 n_done=0 n_crash=0 n_hidden=0
    local global_best_fitness=0 global_best_exp=""

    # ── Banner ──
    local banner_title="GA EVOLUTION MONITOR"
    [[ -n "$WAVE_FILTER" ]] && banner_title="GA MONITOR › wave${WAVE_FILTER}"
    [[ -n "$EXP_FILTER" ]] && banner_title="GA MONITOR › ${EXP_FILTER}"
    [[ -n "$ISLAND_FILTER" ]] && banner_title="GA MONITOR › islands ${ISLAND_FILTER}"
    print_banner "$banner_title"
    echo ""

    # ── Daemon line ──
    print_daemon_line

    # ── Queue ──
    if [[ "$SHOW_QUEUE" == true ]]; then
        echo ""
        print_queue
    fi
    echo ""

    # ── Experiment detail mode (-live -e) ──
    if [[ "$LIVE_MODE" == true && -n "$EXP_FILTER" ]]; then
        local found_log=""
        for lf in "${logs[@]}"; do
            if [[ "$lf" == *"${EXP_FILTER}"* ]]; then
                found_log="$lf"
                break
            fi
        done
        if [[ -n "$found_log" ]]; then
            local parsed wave exp
            parsed=$(parse_log_name "$found_log")
            wave="${parsed%%|*}"; exp="${parsed#*|}"
            print_experiment_detail "$found_log" "$wave" "$exp"
        else
            echo -e "${YELLOW}No log found matching '${EXP_FILTER}'${NC}"
        fi
        print_system_line
        return
    fi

    # ── No logs found ──
    if [[ ${#logs[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No experiment logs found.${NC}"
        [[ -n "$WAVE_FILTER" ]] && echo -e "${DIM}Searched for: wave${WAVE_FILTER}${NC}"
        echo -e "${DIM}Try: $0 --all  or  $0 -live -w <wave>${NC}"
        echo ""
        print_system_line
        return
    fi

    # ── Table header (if table mode) ──
    [[ "$TABLE_MODE" == true ]] && print_table_header

    # ── Narrow header (if narrow + card mode) ──
    local width
    width=$(get_term_width)
    if [[ $width -le 60 && "$TABLE_MODE" == false ]]; then
        printf "  ${BOLD}%-6s %-14s %-9s %-7s %-8s${NC}\n" \
            "WAVE" "EXP" "GEN" "BEST" "STATUS"
        echo -e "  ${DIM}────── ────────────── ───────── ─────── ────────${NC}"
    fi

    local prev_wave=""

    for lf in "${logs[@]}"; do
        local parsed wave exp
        parsed=$(parse_log_name "$lf")
        wave="${parsed%%|*}"; exp="${parsed#*|}"

        local raw
        raw=$(get_metrics "$lf")
        IFS='|' read -r gen best avg div errs elapsed eta results profit status <<< "$raw"
        status=$(verify_status "$status" "$exp" "$wave" "$lf")

        # Visibility filter
        if [[ "$SHOW_ALL" == false && "$status" == "DONE" ]]; then
            if [[ -n "$WAVE_FILTER" || -n "$EXP_FILTER" || -n "$ISLAND_FILTER" ]]; then
                : # show DONE when filtered
            else
                ((n_done++)); ((n_hidden++)); continue
            fi
        fi

        case "$status" in
            DONE)    ((n_done++)) ;;
            RUNNING) ((n_run++)) ;;
            CRASHED) ((n_crash++)) ;;
        esac

        # Track global best
        if [[ "$best" != "—" ]]; then
            if awk "BEGIN{exit(!($best > $global_best_fitness))}" 2>/dev/null; then
                global_best_fitness="$best"
                global_best_exp="${wave}/${exp}"
            fi
        fi

        # Wave separator (table mode only)
        if [[ "$TABLE_MODE" == true && "$wave" != "$prev_wave" && -n "$prev_wave" ]]; then
            local sep_width
            sep_width=$(get_term_width)
            echo -e "  ${DIM}$(printf '╌%.0s' $(seq 1 $(( sep_width - 4 ))))${NC}"
        fi
        prev_wave="$wave"

        # Print
        if [[ "$TABLE_MODE" == true ]]; then
            print_table_row "$wave" "$exp" "$gen" "$best" "$avg" "$div" \
                "$errs" "$elapsed" "$eta" "$results" "$profit" "$status"
        else
            print_card "$wave" "$exp" "$gen" "$best" "$avg" "$div" \
                "$errs" "$elapsed" "$eta" "$results" "$profit" "$status"
        fi
    done

    echo ""

    # ── Queue peek ──
    if [[ "$SHOW_QUEUE" == false ]]; then
        local queued
        queued=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
        if [[ $queued -gt 0 ]]; then
            local next3
            next3=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' 2>/dev/null | \
                sort | head -3 | sed 's/.yaml//g' | tr '\n' ', ' | sed 's/,$//')
            echo -e "${YELLOW}Queue:${NC} ${queued} pending → ${DIM}${next3}${NC}"
        fi
    fi

    # ── System summary ──
    print_system_line

    # ── Global best ──
    if [[ "$global_best_fitness" != "0" && -n "$global_best_exp" ]]; then
        echo -e "${BOLD}Best overall: ${GREEN}${global_best_fitness}${NC} ${DIM}(${global_best_exp})${NC}"
    fi

    # ── Summary counts ──
    echo -ne "${DIM}"
    [[ $n_run -gt 0 ]] && echo -ne "${NC}${GREEN}${n_run} running${NC}${DIM}"
    [[ $n_done -gt 0 ]] && echo -ne " · ${n_done} done"
    [[ $n_crash -gt 0 ]] && echo -ne "${NC} · ${RED}${n_crash} crashed${NC}${DIM}"
    [[ $n_hidden -gt 0 ]] && echo -ne " · ${n_hidden} hidden (use -a)"
    echo -e "${NC}"

    # ── Live footer ──
    if [[ "$LIVE_MODE" == true ]]; then
        echo -e "${DIM}Refreshing every ${INTERVAL}s · Ctrl+C to stop · -h for help${NC}"
    fi
    echo ""
}

# ═════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

if [[ "$LIVE_MODE" == true ]]; then
    trap 'echo ""; echo -e "${DIM}Monitor stopped.${NC}"; exit 0' INT TERM
    while true; do
        print_dashboard
        sleep "$INTERVAL"
    done
else
    print_dashboard
fi
