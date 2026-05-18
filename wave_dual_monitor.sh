#!/usr/bin/env bash
# ============================================================================
# Wave Dual Monitor — Live Dashboard for 2 Parallel Running Waves
# ============================================================================
# Shows progress, time, best/avg fitness, profit for waves running in parallel.
#
# Two display modes:
#   --mode wave    (default) — one row per wave, aggregated info only
#   --mode island  — one row per island across both waves
#
# Usage:
#   ./wave_dual_monitor.sh                             # auto-detect 2 newest waves, wave mode
#   ./wave_dual_monitor.sh wave28 wave29               # explicit waves, wave mode
#   ./wave_dual_monitor.sh --mode island               # island detail mode
#   ./wave_dual_monitor.sh wave28 wave29 --mode island
#   ./wave_dual_monitor.sh --once                      # print once and exit
#   ./wave_dual_monitor.sh --interval 10               # refresh every 10 seconds
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"

# ── Args ──
WAVE1=""
WAVE2=""
MODE="wave"       # wave | island
INTERVAL=10
ONCE=false

_positional=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="${2:-wave}"; shift ;;
        --once)        ONCE=true ;;
        --interval)    INTERVAL="${2:-10}"; shift ;;
        --help|-h)
            echo "Usage: $0 [wave_a] [wave_b] [--mode wave|island] [--once] [--interval N]"
            echo ""
            echo "  wave_a / wave_b   Names of two waves to monitor (e.g. wave28 wave29)"
            echo "                    If omitted, the two most recently active waves are used."
            echo "  --mode wave       Show one summary row per wave (default)"
            echo "  --mode island     Show one row per island inside each wave"
            echo "  --once            Print status once and exit"
            echo "  --interval N      Refresh interval in seconds (default: 10)"
            exit 0
            ;;
        -*)  echo "Unknown option: $1" >&2; exit 1 ;;
        *)   _positional+=("$1") ;;
    esac
    shift
done

[[ ${#_positional[@]} -ge 1 ]] && WAVE1="${_positional[0]}"
[[ ${#_positional[@]} -ge 2 ]] && WAVE2="${_positional[1]}"

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

# ── Visual-width aware pad (strips ANSI escapes for length calc) ──
apad() {
    local str="$1" width="$2"
    local plain
    plain=$(printf '%b' "$str" | sed $'s/\x1b\\[[0-9;]*m//g')
    local need=$(( width - ${#plain} ))
    if [[ $need -gt 0 ]]; then
        printf '%b%*s' "$str" "$need" ""
    else
        printf '%b' "$str"
    fi
}

# Clip plain string to N chars with ellipsis
clip() {
    local s="$1" n="$2"
    if [[ ${#s} -gt $n ]]; then echo "${s:0:$(( n - 2 ))}.."; else echo "$s"; fi
}

# ── Auto-detect the two most recently written wave logs ──
auto_detect_waves() {
    # Find unique wave names from log files, pick the 2 most recently modified
    local -A seen=()
    local -a ordered=()
    while IFS= read -r f; do
        local bn wn
        bn=$(basename "$f" .log)
        # Accept wave{N}_* naming (e.g. wave28_A1) — extract the wave prefix
        if [[ "$bn" =~ ^(wave[0-9]+)_ ]]; then
            wn="${BASH_REMATCH[1]}"
        elif [[ "$bn" =~ ^(wave[0-9]+)$ ]]; then
            wn="$bn"
        else
            continue
        fi
        if [[ -z "${seen[$wn]+x}" ]]; then
            seen[$wn]=1
            ordered+=("$wn")
        fi
    done < <(ls -t "${LOG_DIR}"/wave*.log 2>/dev/null)

    WAVE1="${ordered[0]:-}"
    WAVE2="${ordered[1]:-}"
}

# ── Collect all log files for a wave ──
wave_logs() {
    local wave="$1"
    # Primary: wave{N}_*.log but NOT *_console.log (console logs duplicate data)
    local -a logs=()
    while IFS= read -r f; do
        [[ "$f" == *_console.log ]] && continue
        logs+=("$f")
    done < <(ls -1 "${LOG_DIR}/${wave}_"*.log 2>/dev/null | sort)
    printf '%s\n' "${logs[@]}"
}

# ── Extract current generation ──
#    Returns: "CUR/TOTAL" string or "init" or "N/?" if total unknown
extract_gen() {
    local buf="$1"
    local g
    g=$(printf '%s' "$buf" | grep -oP 'GENERATION \K\d+/\d+' | tail -1 || true)
    if [[ -n "$g" ]]; then echo "$g"; return; fi
    # Fallback: checkpoint line "resume from generation N/T"
    g=$(printf '%s' "$buf" | grep -oP 'resume from generation \K\d+' | tail -1 || true)
    local t
    t=$(printf '%s' "$buf" | grep -oP 'Generations:\s+\K\d+' | tail -1 || true)
    if [[ -n "$g" ]]; then echo "${g}/${t:-?}"; return; fi
    # Step fallback: "[STEP] Creating generation N"
    g=$(printf '%s' "$buf" | grep -oP '\[STEP\] Creating generation \K\d+' | tail -1 || true)
    if [[ -n "$g" ]]; then echo "${g}/${t:-?}"; return; fi
    echo "init"
}

# ── Parse gen_current and gen_total from extract_gen output ──
parse_gen() {
    local gen_str="$1"
    if [[ "$gen_str" == "init" ]]; then echo "0 0"; return; fi
    local cur tot
    cur="${gen_str%%/*}"
    tot="${gen_str#*/}"
    [[ "$tot" == "?" ]] && tot=0
    echo "$cur $tot"
}

# ── Compute elapsed time from a log buffer ──
elapsed_from_buf() {
    local log="$1"
    local t0 e0 e1 elapsed_secs
    t0=$(head -1 "$log" 2>/dev/null | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
    [[ -z "$t0" ]] && echo "—" && return
    e0=$(date -d "$t0" +%s 2>/dev/null || echo 0)
    [[ "$e0" -eq 0 ]] && echo "—" && return
    e1=$(date +%s)
    elapsed_secs=$(( e1 - e0 ))
    [[ $elapsed_secs -lt 0 ]] && elapsed_secs=0
    local m=$(( elapsed_secs / 60 )) s=$(( elapsed_secs % 60 ))
    if [[ $m -ge 60 ]]; then
        echo "$(( m / 60 ))h$(( m % 60 ))m"
    else
        echo "${m}m${s}s"
    fi
    echo "$elapsed_secs" >&3
}

# ── Compute ETA ──
compute_eta() {
    local elapsed_secs="$1" gen_cur="$2" gen_tot="$3"
    if [[ "$gen_cur" -le 1 || "$gen_tot" -le 0 || "$elapsed_secs" -le 0 ]]; then echo "—"; return; fi
    if [[ "$gen_cur" -ge "$gen_tot" ]]; then echo "<1m"; return; fi
    local spc=$(( elapsed_secs / gen_cur ))
    local rem=$(( gen_tot - gen_cur ))
    local eta_secs=$(( spc * rem ))
    local eta_m=$(( eta_secs / 60 ))
    if [[ $eta_m -ge 60 ]]; then echo "~$(( eta_m / 60 ))h$(( eta_m % 60 ))m"
    elif [[ $eta_m -gt 0 ]]; then echo "~${eta_m}m"
    else echo "<1m"
    fi
}

# ── Colorize fitness value ──
color_fitness() {
    local v="$1"
    [[ "$v" == "—" ]] && echo "${DIM}—${NC}" && return
    if awk "BEGIN{exit(!($v + 0 >= 0.40))}" 2>/dev/null; then echo "${GREEN}${v}${NC}"
    elif awk "BEGIN{exit(!($v + 0 >= 0.20))}" 2>/dev/null; then echo "${YELLOW}${v}${NC}"
    else echo "${RED}${v}${NC}"; fi
}

# ── Colorize profit ──
color_profit() {
    local v="$1"
    [[ "$v" == "—" ]] && echo "${DIM}—${NC}" && return
    if [[ "$v" == -* ]]; then echo "${RED}${v}${NC}"
    else echo "${GREEN}${v}${NC}"; fi
}

# ─────────────────────────────────────────────────────────
#  PER-ISLAND METRICS
#  Returns pipe-delimited: island_name|gen_cur|gen_tot|best|avg|diversity|best_profit
#
#  Islands that have been defined in the header but not yet reported stats
#  (still evaluating their first/current generation) are shown as "pending".
# ─────────────────────────────────────────────────────────
island_metrics_from_log() {
    local log="$1"
    local -n _out_arr="$2"   # nameref to output array

    [[ ! -f "$log" || ! -s "$log" ]] && return

    # Extract total generations from header (first 60 lines)
    local gen_total
    gen_total=$(head -60 "$log" | grep -oP 'Generations:\s+\K\d+' | tail -1 || true)
    gen_total="${gen_total:-0}"

    # Collect all defined island names from header lines:
    #   "  Island island_X_name           : pop=..."
    local -a all_islands=()
    while IFS= read -r iname; do
        iname="${iname// /}"   # trim trailing spaces
        [[ -n "$iname" ]] && all_islands+=("$iname")
    done < <(grep -oP 'Island \Kisland_\w+' "$log" 2>/dev/null | head -200 | sort -t_ -k2,2 -n -u || true)

    # Per-island stats: read from the LAST occurrence of each island's stats line
    # in the whole log (grep entire file, not just tail).
    # Also COUNT stats lines per island to derive actual per-island generation
    # (avoids using the global GENERATION counter which jumps ahead of stats).
    local -A island_best=() island_avg=() island_div=() island_profit=() island_gen=()
    local island_name v line

    # Stats lines: [island_X_name           ] best=0.XXXX avg=0.XXXX diversity=0.XXXX
    while IFS= read -r line; do
        island_name=$(echo "$line" | grep -oP '(?<=\[)island_\w+' || true)
        island_name="${island_name// /}"
        [[ -z "$island_name" ]] && continue
        v=$(echo "$line" | grep -oP '(?<=best=)[0-9.]+' || true)
        [[ -n "$v" ]] && island_best["$island_name"]="$v"
        v=$(echo "$line" | grep -oP '(?<=avg=)[0-9.]+' || true)
        [[ -n "$v" ]] && island_avg["$island_name"]="$v"
        v=$(echo "$line" | grep -oP '(?<=diversity=)[0-9.]+' || true)
        [[ -n "$v" ]] && island_div["$island_name"]="$v"
        # Count stats lines → per-island generation progress
        island_gen["$island_name"]=$(( ${island_gen["$island_name"]:-0} + 1 ))
    done < <(grep -P '\[island_\w+\s*\] best=' "$log" 2>/dev/null || true)

    # NEW BEST lines: highest profit per island across the whole log
    while IFS= read -r line; do
        island_name=$(echo "$line" | grep -oP '(?<=\[)island_\w+' || true)
        island_name="${island_name// /}"
        [[ -z "$island_name" ]] && continue
        v=$(echo "$line" | grep -oP '(?<=profit=)-?[0-9.]+(?=%)' || true)
        if [[ -n "$v" ]]; then
            local existing="${island_profit[$island_name]:-}"
            if [[ -z "$existing" ]] || awk "BEGIN{exit(!($v + 0 > $existing + 0))}" 2>/dev/null; then
                island_profit["$island_name"]="$v"
            fi
        fi
    done < <(grep -P '\[island_\w+\] NEW BEST:' "$log" 2>/dev/null || true)

    # Build output rows: all defined islands, pending ones shown with dashes
    # If no islands found in header, fall back to only those with stats
    if [[ ${#all_islands[@]} -eq 0 ]]; then
        all_islands=( $(printf '%s\n' "${!island_best[@]}" | sort -t_ -k2,2 -n) )
    fi

    for island_name in "${all_islands[@]}"; do
        local best="${island_best[$island_name]:-—}"
        local avg="${island_avg[$island_name]:-—}"
        local div="${island_div[$island_name]:-—}"
        local profit="${island_profit[$island_name]:-—}"
        [[ "$profit" != "—" ]] && profit="${profit}%"
        # Use per-island generation count (number of completed stats reports)
        # instead of the global GENERATION counter which jumps ahead.
        local ig="${island_gen[$island_name]:-0}"
        _out_arr+=("${island_name}|${ig}|${gen_total}|${best}|${avg}|${div}|${profit}")
    done
}

# ─────────────────────────────────────────────────────────
#  WAVE-LEVEL METRICS  (aggregate over all island logs)
#  Output vars written via nameref or echoed
# ─────────────────────────────────────────────────────────
wave_metrics() {
    local wave="$1"
    local -n _wm="$2"   # assoc array: gen, gen_cur, gen_tot, best, avg, diversity, profit, elapsed, eta, status, n_islands

    # Collect all non-console logs for this wave
    local -a logs=()
    while IFS= read -r f; do
        [[ -n "$f" ]] && logs+=("$f")
    done < <(wave_logs "$wave")

    if [[ ${#logs[@]} -eq 0 ]]; then
        _wm[gen]="—"; _wm[elapsed]="—"; _wm[eta]="—"
        _wm[best]="—"; _wm[avg]="—"; _wm[diversity]="—"
        _wm[profit]="—"; _wm[status]="NO_LOG"; _wm[n_islands]=0
        return
    fi

    # Use the largest log (most data) as the primary log for gen/time parsing
    local primary_log
    primary_log=$(ls -S "${logs[@]}" 2>/dev/null | head -1)

    local prim_buf
    prim_buf=$(tail -800 "$primary_log" 2>/dev/null)

    # ── Generation (from primary log) ──
    local gen_str
    gen_str=$(extract_gen "$prim_buf")
    local gen_cur gen_tot
    read -r gen_cur gen_tot <<< "$(parse_gen "$gen_str")"
    _wm[gen]="$gen_str"
    _wm[gen_cur]="$gen_cur"
    _wm[gen_tot]="$gen_tot"

    # ── Elapsed (from earliest timestamp in primary log) ──
    local t0 e0 elapsed_secs=0
    t0=$(grep -oP -m1 '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' "$primary_log" 2>/dev/null || true)
    if [[ -n "$t0" ]]; then
        e0=$(date -d "$t0" +%s 2>/dev/null || echo 0)
        if [[ "$e0" -gt 0 ]]; then
            elapsed_secs=$(( $(date +%s) - e0 ))
            [[ $elapsed_secs -lt 0 ]] && elapsed_secs=0
        fi
    fi
    local em=$(( elapsed_secs / 60 )) es=$(( elapsed_secs % 60 ))
    if [[ $em -ge 60 ]]; then
        _wm[elapsed]="$(( em / 60 ))h$(( em % 60 ))m"
    else
        _wm[elapsed]="${em}m${es}s"
    fi

    # ── ETA ──
    _wm[eta]=$(compute_eta "$elapsed_secs" "$gen_cur" "$gen_tot")

    # ── Aggregate best/avg/div/profit across all islands ──
    local global_best="-1" sum_avg=0 sum_div=0 cnt_avg=0 cnt_div=0 global_profit="-9999"

    for log in "${logs[@]}"; do
        # Use per-island stats from the ENTIRE log (last occurrence per island wins).
        # Previous approach: tail -400 + tail -1 gave only the LAST island's value.
        # Now: extract the last stats line per island, compute max(best) and mean(avg).
        local -A _lbest=() _lavg=() _ldiv=()
        local _iname _val _line
        while IFS= read -r _line; do
            _iname=$(echo "$_line" | grep -oP '(?<=\[)island_\w+' || true)
            _iname="${_iname// /}"
            [[ -z "$_iname" ]] && continue
            _val=$(echo "$_line" | grep -oP '(?<=best=)[0-9.]+' || true)
            [[ -n "$_val" ]] && _lbest["$_iname"]="$_val"
            _val=$(echo "$_line" | grep -oP '(?<=avg=)[0-9.]+' || true)
            [[ -n "$_val" ]] && _lavg["$_iname"]="$_val"
            _val=$(echo "$_line" | grep -oP 'diversity=\K[0-9.]+' || true)
            [[ -n "$_val" ]] && _ldiv["$_iname"]="$_val"
        done < <(grep -P '\[island_\w+\s*\] best=' "$log" 2>/dev/null || true)

        # Best fitness = max across all islands' latest best
        for _val in "${_lbest[@]}"; do
            if [[ "$global_best" == "-1" ]] || awk "BEGIN{exit(!($_val + 0 > $global_best + 0))}" 2>/dev/null; then
                global_best="$_val"
            fi
        done
        # Average fitness = mean of all islands' latest avg
        for _val in "${_lavg[@]}"; do
            sum_avg=$(LC_NUMERIC=C awk "BEGIN{printf \"%.6f\",$sum_avg + $_val}")
            (( cnt_avg++ ))
        done
        # Diversity = mean of all islands' latest diversity
        for _val in "${_ldiv[@]}"; do
            sum_div=$(LC_NUMERIC=C awk "BEGIN{printf \"%.6f\",$sum_div + $_val}")
            (( cnt_div++ ))
        done

        # Best profit
        local lp
        lp=$(grep -oP 'NEW BEST: fitness=[0-9.]+ profit=\K-?[0-9.]+(?=%)' "$log" \
            | LC_NUMERIC=C awk 'BEGIN{b=-9999} {if($1+0>b+0)b=$1} END{if(b>-9999) print b}' 2>/dev/null || true)
        if [[ -n "$lp" ]]; then
            if awk "BEGIN{exit(!($lp + 0 > $global_profit + 0))}" 2>/dev/null; then
                global_profit="$lp"
            fi
        fi
    done

    _wm[best]="${global_best//-1/—}"
    if [[ $cnt_avg -gt 0 ]]; then
        _wm[avg]=$(LC_NUMERIC=C awk "BEGIN{printf \"%.4f\", $sum_avg / $cnt_avg}")
    else
        _wm[avg]="—"
    fi
    if [[ $cnt_div -gt 0 ]]; then
        _wm[diversity]=$(LC_NUMERIC=C awk "BEGIN{printf \"%.4f\", $sum_div / $cnt_div}")
    else
        _wm[diversity]="—"
    fi
    if [[ "$global_profit" != "-9999" ]]; then
        _wm[profit]="${global_profit}%"
    else
        _wm[profit]="—"
    fi

    # ── Number of islands ──
    local n_islands=0
    n_islands=$(head -30 "$primary_log" | grep -oP 'Islands:\s+\K\d+' | tail -1 || echo 0)
    # Count unique island names if header not found
    if [[ "$n_islands" -eq 0 ]]; then
        n_islands=$(grep -oP '(?<=\[)island_[a-z0-9_]+(?=\s*\])' "$primary_log" \
            | sed 's/ *$//' | sort -u | wc -l || echo 0)
    fi
    _wm[n_islands]="$n_islands"

    # ── Status ──
    local status="RUNNING"
    if grep -qE 'ISLAND MODEL EVOLUTION FINISHED|GENERIC ISLAND MODEL EVOLUTION FINISHED|GA RUN COMPLETE|EVOLUTION COMPLETE' "$primary_log" 2>/dev/null; then
        status="DONE"
    elif tail -30 "$primary_log" 2>/dev/null | grep -qP 'Traceback|KeyboardInterrupt|FATAL|ERROR DURING EVOLUTION'; then
        status="CRASHED"
    elif tail -20 "$primary_log" 2>/dev/null | grep -qP '\[SHUTDOWN\]|Graceful shutdown'; then
        status="STOPPED"
    else
        # Stale detection: log not modified in past 5 minutes and no running process
        local log_age
        log_age=$(( $(date +%s) - $(stat -c%Y "$primary_log" 2>/dev/null || echo 0) ))
        if [[ $log_age -gt 300 ]]; then
            local pid_alive=false
            if ps aux 2>/dev/null | grep '[r]un_ga.py' | grep -q "$wave"; then
                pid_alive=true
            fi
            $pid_alive || status="STALE"
        fi
    fi
    _wm[status]="$status"
}

# ─────────────────────────────────────────────────────────
#   WAVE MODE: one row per wave
# ─────────────────────────────────────────────────────────
print_wave_mode() {
    local -a waves=("$@")

    echo ""
    echo -e "  ${CYAN}╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -ne "  ${CYAN}║${NC} ${BOLD}"
    apad "WAVE DUAL MONITOR — wave mode" 62
    echo -e "${DIM}$(date '+%a %d.%m %H:%M:%S')${NC}   ${CYAN}║${NC}"
    echo -e "  ${CYAN}╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Header
    echo -ne "  "
    apad "${BOLD}WAVE${NC}"         9
    echo -n " "
    apad "${BOLD}ISLANDS${NC}"      8
    echo -n " "
    apad "${BOLD}GENERATION${NC}"   14
    echo -n " "
    apad "${BOLD}ETA${NC}"          9
    echo -n " "
    apad "${BOLD}ELAPSED${NC}"      9
    echo -n " "
    apad "${BOLD}BEST FIT${NC}"     10
    echo -n " "
    apad "${BOLD}AVG FIT${NC}"      10
    echo -n " "
    apad "${BOLD}DIVERSITY${NC}"    10
    echo -n " "
    apad "${BOLD}BEST PROFIT${NC}"  12
    echo -n " "
    echo -e "${BOLD}STATUS${NC}"
    echo -e "  ${DIM}─────────  ────────  ──────────────  ─────────  ─────────  ──────────  ──────────  ──────────  ────────────  ──────────${NC}"

    for wave in "${waves[@]}"; do
        [[ -z "$wave" ]] && continue

        declare -A wm=()
        wave_metrics "$wave" wm

        # Status color
        local c_status
        case "${wm[status]}" in
            RUNNING) c_status="${GREEN}● RUNNING${NC}" ;;
            DONE)    c_status="${GREEN}✓  DONE${NC}" ;;
            CRASHED) c_status="${RED}✗  CRASH${NC}" ;;
            STALE)   c_status="${YELLOW}?  STALE${NC}" ;;
            STOPPED) c_status="${MAGENTA}■  STOP${NC}" ;;
            NO_LOG)  c_status="${DIM}—  NOLOG${NC}" ;;
            *)       c_status="${DIM}${wm[status]}${NC}" ;;
        esac

        local c_best c_avg c_profit c_eta c_div
        c_best=$(color_fitness "${wm[best]}")
        c_avg=$(color_fitness "${wm[avg]}")
        c_profit=$(color_profit "${wm[profit]}")

        if [[ "${wm[eta]}" == "—" ]]; then c_eta="${DIM}—${NC}"; else c_eta="${MAGENTA}${wm[eta]}${NC}"; fi
        if [[ "${wm[diversity]}" == "—" ]]; then c_div="${DIM}—${NC}"; else c_div="${wm[diversity]}"; fi

        echo -ne "  "
        apad "${BOLD}${wave}${NC}"        9
        echo -n " "
        apad "${wm[n_islands]}"           8
        echo -n " "
        apad "${wm[gen]}"                 14
        echo -n " "
        apad "$c_eta"                     9
        echo -n " "
        apad "${wm[elapsed]}"             9
        echo -n " "
        apad "$c_best"                    10
        echo -n " "
        apad "$c_avg"                     10
        echo -n " "
        apad "$c_div"                     10
        echo -n " "
        apad "$c_profit"                  12
        echo -n " "
        echo -e "$c_status"
    done

    echo ""
    _print_sysbar
}

# ─────────────────────────────────────────────────────────
#   ISLAND MODE: one row per island, grouped by wave
# ─────────────────────────────────────────────────────────
print_island_mode() {
    local -a waves=("$@")

    echo ""
    echo -e "  ${CYAN}╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -ne "  ${CYAN}║${NC} ${BOLD}"
    apad "WAVE DUAL MONITOR — island mode" 62
    echo -e "${DIM}$(date '+%a %d.%m %H:%M:%S')${NC}   ${CYAN}║${NC}"
    echo -e "  ${CYAN}╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Header
    echo -ne "  "
    apad "${BOLD}WAVE${NC}"          8
    echo -n " "
    apad "${BOLD}ISLAND${NC}"        28
    echo -n " "
    apad "${BOLD}GEN${NC}"           8
    echo -n " "
    apad "${BOLD}BEST FIT${NC}"      10
    echo -n " "
    apad "${BOLD}AVG FIT${NC}"       10
    echo -n " "
    apad "${BOLD}DIVERSITY${NC}"     10
    echo -n " "
    echo -e "${BOLD}BEST PROFIT${NC}"
    echo -e "  ${DIM}────────  ────────────────────────────  ────────  ──────────  ──────────  ──────────  ────────────${NC}"

    for wave in "${waves[@]}"; do
        [[ -z "$wave" ]] && continue

        # Wave summary line
        declare -A wm=()
        wave_metrics "$wave" wm

        local c_status
        case "${wm[status]}" in
            RUNNING) c_status="${GREEN}● RUNNING${NC}" ;;
            DONE)    c_status="${GREEN}✓  DONE${NC}" ;;
            CRASHED) c_status="${RED}✗  CRASH${NC}" ;;
            STALE)   c_status="${YELLOW}?  STALE${NC}" ;;
            STOPPED) c_status="${MAGENTA}■  STOP${NC}" ;;
            *)       c_status="${DIM}${wm[status]}${NC}" ;;
        esac

        # Wave header row
        echo -ne "  "
        apad "${BOLD}${CYAN}${wave}${NC}" 8
        echo -n " "
        apad "${DIM}gen ${wm[gen]}  elapsed ${wm[elapsed]}  eta ${wm[eta]}${NC}" 60
        echo -e "$c_status"
        echo -e "  ${DIM}────────  ────────────────────────────  ────────  ──────────  ──────────  ──────────  ────────────${NC}"

        # Per-island rows from all logs of this wave
        local -a wave_island_rows=()
        local -a logs=()
        while IFS= read -r f; do
            [[ -n "$f" ]] && logs+=("$f")
        done < <(wave_logs "$wave")

        for log in "${logs[@]}"; do
            local -a rows_for_log=()
            island_metrics_from_log "$log" rows_for_log
            wave_island_rows+=("${rows_for_log[@]+"${rows_for_log[@]}"}")
        done

        # Deduplicate islands (multiple logs for same wave → same island names)
        local -A shown_islands=()
        for row in "${wave_island_rows[@]+"${wave_island_rows[@]}"}"; do
            local iname irow_parts
            iname="${row%%|*}"
            [[ -n "${shown_islands[$iname]+x}" ]] && continue
            shown_islands["$iname"]=1

            IFS='|' read -r iname_ gen_cur gen_tot best avg div profit <<< "$row"
            local gen_disp="${gen_cur}/${gen_tot}"
            [[ "$gen_tot" == "0" || -z "$gen_tot" ]] && gen_disp="${gen_cur}/?"

            local c_best c_avg c_profit c_div
            c_best=$(color_fitness "$best")
            c_avg=$(color_fitness "$avg")
            c_profit=$(color_profit "$profit")
            [[ "$div" == "—" ]] && c_div="${DIM}—${NC}" || c_div="$div"

            local short_name
            short_name=$(clip "$iname" 27)

            # Detect pending island (no stats yet this run)
            local is_pending=false
            [[ "$best" == "—" && "$avg" == "—" ]] && is_pending=true

            echo -ne "  "
            apad "" 8
            echo -n " "
            if $is_pending; then
                apad "  ${DIM}${short_name}${NC}" 28
                echo -n " "
                apad "${DIM}${gen_disp}${NC}" 8
                echo -n " "
                apad "${DIM}evaluating...${NC}" 10
                echo -n " "
                apad "" 10
                echo -n " "
                apad "" 10
                echo -n " "
                echo -e "${DIM}—${NC}"
            else
                apad "  ${short_name}" 28
                echo -n " "
                apad "$gen_disp" 8
                echo -n " "
                apad "$c_best" 10
                echo -n " "
                apad "$c_avg" 10
                echo -n " "
                apad "$c_div" 10
                echo -n " "
                echo -e "$c_profit"
            fi
        done

        if [[ ${#wave_island_rows[@]} -eq 0 ]]; then
            echo -e "  ${DIM}         (no island data yet — still initializing)${NC}"
        fi

        echo ""
    done

    _print_sysbar
}

# ── System bar ──
_print_sysbar() {
    local free load
    free=$(awk '/MemAvailable/{printf "%.0f",$2/1024}' /proc/meminfo 2>/dev/null || echo "?")
    load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo "?")
    echo -e "  ${BLUE}System${NC}: Free=${free}MB  Load=${load}  $(date '+%H:%M:%S')"
    if [[ "$ONCE" == false ]]; then
        echo -e "  ${DIM}Refreshing every ${INTERVAL}s — Ctrl+C to stop  |  --mode wave|island  |  --once${NC}"
    fi
    echo ""
}

# ─────────────────────────────────────────────────────────
#   MAIN LOOP
# ─────────────────────────────────────────────────────────

# Detect waves if not provided
if [[ -z "$WAVE1" && -z "$WAVE2" ]]; then
    auto_detect_waves
fi

if [[ -z "$WAVE1" ]]; then
    echo -e "${RED}No wave logs found in ${LOG_DIR}${NC}" >&2
    exit 1
fi

render() {
    printf '\033[2J\033[H'
    if [[ "$MODE" == "island" ]]; then
        if [[ -n "$WAVE2" ]]; then
            print_island_mode "$WAVE1" "$WAVE2"
        else
            print_island_mode "$WAVE1"
        fi
    else
        if [[ -n "$WAVE2" ]]; then
            print_wave_mode "$WAVE1" "$WAVE2"
        else
            print_wave_mode "$WAVE1"
        fi
    fi
}

if [[ "$ONCE" == true ]]; then
    render
else
    trap 'echo ""; echo -e "  ${DIM}Monitor stopped.${NC}"; exit 0' INT TERM
    while true; do
        render
        sleep "$INTERVAL"
    done
fi
