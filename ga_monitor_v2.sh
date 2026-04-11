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
#   - --live / -l  : show only running experiments + filtered log tail
#   - HoF block    : when --wave WAVENAME given, shows best-strategy per-exp:
#                    fitness | T.profit | V.profit | T/V gap | pair_gen ratio
#
# Usage:
#   ./ga_monitor_v2.sh                       # auto-detect all running
#   ./ga_monitor_v2.sh wave32                # monitor specific wave only
#   ./ga_monitor_v2.sh wave32 --live         # live log tails for running exps
#   ./ga_monitor_v2.sh wave32 --all          # show completed experiments too
#   ./ga_monitor_v2.sh --queue               # show queue contents
#   ./ga_monitor_v2.sh --summary             # show overall summary only
#   ./ga_monitor_v2.sh --once                # print once and exit
#   ./ga_monitor_v2.sh --interval 10         # refresh every 10 seconds
#   ./ga_monitor_v2.sh --detail compact      # compact view (fewer columns)
#   ./ga_monitor_v2.sh --detail detailed     # show extra metrics
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
QUEUE_DIR="${REPO_DIR}/genetic_algorithm/config/queue"
DONE_BASE="${REPO_DIR}/genetic_algorithm/config/done"
STATE_FILE="${LOG_DIR}/auto_queue_state.json"
HOF_BASE="${REPO_DIR}/genetic_algorithm/data"
WAVE_FILTER=""
INTERVAL=5
ONCE=false
SHOW_ALL=false
SHOW_QUEUE=false
SHOW_SUMMARY_ONLY=false
DETAIL_LEVEL="normal"  # compact, normal, detailed
LIVE_MODE=false        # --live: only running experiments + log tails

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)         ONCE=true ;;
        --all)          SHOW_ALL=true ;;
        --queue)        SHOW_QUEUE=true ;;
        --summary)      SHOW_SUMMARY_ONLY=true; ONCE=true ;;
        --detail)       DETAIL_LEVEL="${2:-normal}"; shift ;;
        --interval)     INTERVAL="${2:-5}"; shift ;;
        --live|-l)      LIVE_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [wave_name] [OPTIONS]"
            echo ""
            echo "  wave_name           Filter to specific wave (e.g. wave32)"
            echo "  --all               Show completed experiments too"
            echo "  --live, -l          Live mode: only running experiments + log tail"
            echo "  --queue             Show queue contents"
            echo "  --summary           Show overall summary stats only"
            echo "  --detail LEVEL      Detail level: compact, normal, detailed"
            echo "  --once              Print status once and exit"
            echo "  --interval N        Refresh interval in seconds (default: 5)"
            echo ""
            echo "  When a wave filter is given, completed experiments show HoF data:"
            echo "    fitness | train profit | val profit | T/V gap | pair_gen ratio"
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

# ── Read best-strategy data from Hall-of-Fame JSON ──
# Args: wave_name exp_suffix (e.g. read_hof_data "wave32" "A")
# Outputs pipe-delimited: fitness|train_profit|val_profit|tv_gap_pct|pair_gen|hof_count
read_hof_data() {
    local wave="$1" exp="$2"
    local hof_json="${HOF_BASE}/hall_of_fame_${wave}_${exp}/hall_of_fame.json"
    [[ ! -f "$hof_json" ]] && echo "—|—|—|—|—|0" && return

    python3 - "$hof_json" 2>/dev/null <<'PYEOF' || echo "—|—|—|—|—|0"
import json, sys
data = json.load(open(sys.argv[1]))
entries = data.get('entries', [])
n = len(entries)
if not entries:
    print("—|—|—|—|—|0"); sys.exit(0)
best = max(entries, key=lambda e: e.get('fitness', 0))
m = best.get('metrics', {})
fit  = best.get('fitness', 0)
tp   = m.get('profit', 0)          # already in % (e.g. 1.79)
vp   = m.get('val_profit', 0)      # already in %
gap  = m.get('train_val_gap', 0) * 100  # fraction → %
pgr  = m.get('pair_generalization_ratio', 0)
print(f"{fit:.4f}|{tp:+.2f}%|{vp:+.2f}%|{gap:+.1f}%|{pgr:.3f}|{n}")
PYEOF
}

# ── Print last N key log lines for a running experiment (live tail) ──
# Args: log_path [n_lines=15]
print_live_tail() {
    local logf="$1" nlines="${2:-15}"
    [[ ! -f "$logf" ]] && return

    echo -e "  ${DIM}── live log tail: $(basename "$logf") ──────────────────────────────────────────────${NC}"

    # Filter to informative lines: NEW BEST, GENERATION/SUMMARY, EVAL, ERROR, WARN
    local filtered
    filtered=$(grep -E 'NEW BEST|GENERATION|SUMMARY|EVAL Complete|EVOLUTION COMPLETE|ERROR|FATAL|WARN|island_[0-9]+.*best=' \
        "$logf" 2>/dev/null | tail -"$nlines")

    if [[ -z "$filtered" ]]; then
        # Fallback: just last N raw lines
        tail -"$nlines" "$logf" 2>/dev/null | while IFS= read -r line; do
            echo -e "    ${DIM}${line}${NC}"
        done
    else
        while IFS= read -r line; do
            local colour="$DIM"
            [[ "$line" == *"NEW BEST"* ]]            && colour="$GREEN"
            [[ "$line" == *"ERROR"* || "$line" == *"FATAL"* ]] && colour="$RED"
            [[ "$line" == *"WARN"* ]]                && colour="$YELLOW"
            [[ "$line" == *"EVOLUTION COMPLETE"* ]]  && colour="${BOLD}${GREEN}"
            # Clip to terminal width-ish to avoid wrapping
            local clipped
            clipped=$(echo "$line" | cut -c1-120)
            echo -e "    ${colour}${clipped}${NC}"
        done <<< "$filtered"
    fi
    echo ""
}

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
        # Direct wave logs: wave32_A.log, wave32_A1.log
        patterns+=("${LOG_DIR}/${WAVE_FILTER}_*.log")
        # Nested/detailed island model logs: wave31_wave32_A_island_....log
        patterns+=("${LOG_DIR}/wave*_${WAVE_FILTER}_*.log")
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

    # Deduplicate: each island-model experiment produces two logs:
    #   Short per-island log:  wave32_A.log    (GeneticAlgorithm file handler, no SUMMARY)
    #   Detailed island log:   wave31_wave32_A_island_15m_rank_ring.log  (GenericIslandModel,
    #                          has [SUMMARY] Gen X/Y lines — this is what we want for monitoring)
    #
    # For plain GA there is also a short log (wave32_E.log) with GENERATION/[STATS] AND
    # a redirect main log without them.  Both are from the same process.
    #
    # Canonical key: waveN_LETTER[DIGITS]  e.g. wave32_A, wave25_A1
    # If a log name embeds a nested wave (wave31_wave32_A_...) AND a corresponding short
    # log exists (wave32_A.log), prefer the LONGER/DETAILED log for island model
    # (it has SUMMARY lines) but keep the SHORT log for plain GA (it has STATS lines).
    local -A best_log=()
    for f in "${all_logs[@]}"; do
        local bn
        bn=$(basename "$f" .log)
        local canonical=""

        # Pattern 1: plain short name  wave32_A  or  wave25_A1
        if [[ "$bn" =~ ^(wave[0-9]+_[A-Z][0-9]*)(_.+)?$ ]]; then
            canonical="${BASH_REMATCH[1]}"
        # Pattern 2: nested wave name  wave31_wave32_A_island_...  →  canonical = wave32_A
        elif [[ "$bn" =~ ^wave[0-9]+_(wave[0-9]+_[A-Z][0-9]*)(_.*)?$ ]]; then
            canonical="${BASH_REMATCH[1]}"
        fi

        if [[ -n "$canonical" ]]; then
            if [[ -z "${best_log[$canonical]+x}" ]]; then
                best_log["$canonical"]="$f"
            else
                local existing_bn cur_bn
                existing_bn=$(basename "${best_log[$canonical]}" .log)
                cur_bn="$bn"
                # Prefer the log that matches the "nested wave" pattern (waveM_waveN_X_...)
                # because it is the GenericIslandModel log with [SUMMARY] lines.
                # Fallback: prefer longer name (more context).
                local cur_nested=0 existing_nested=0
                [[ "$cur_bn" =~ ^wave[0-9]+_wave[0-9]+_ ]] && cur_nested=1
                [[ "$existing_bn" =~ ^wave[0-9]+_wave[0-9]+_ ]] && existing_nested=1
                if [[ "$cur_nested" -gt "$existing_nested" ]]; then
                    best_log["$canonical"]="$f"
                elif [[ "$cur_nested" -eq "$existing_nested" && ${#cur_bn} -gt ${#existing_bn} ]]; then
                    best_log["$canonical"]="$f"
                fi
            fi
        else
            # Non-standard naming — keep as-is
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
    # Island model: use [SUMMARY] Gen X/Y which marks a completed gen.
    # Also capture the active GENERATION marker (may be 1 ahead of last SUMMARY).
    local gen="init" gen_current=0 gen_total=0
    local gl summary_gen
    # 1. Try [SUMMARY] Gen X/Y (island model completed-gen marker)
    summary_gen=$(echo "$buf" | grep -oP '\[SUMMARY\] Gen \K\d+/\d+' | tail -1 || true)
    # 2. Try plain GENERATION X/Y marker
    gl=$(echo "$buf" | grep -oP 'GENERATION \K\d+/\d+' | tail -1 || true)
    if [[ -n "$gl" ]]; then
        gen="${gl}"
        gen_current=$(echo "$gl" | cut -d/ -f1)
        gen_total=$(echo "$gl" | cut -d/ -f2)
    elif [[ -n "$summary_gen" ]]; then
        gen="${summary_gen}"
        gen_current=$(echo "$summary_gen" | cut -d/ -f1)
        gen_total=$(echo "$summary_gen" | cut -d/ -f2)
    fi
    # For plain GA experiments, the per-GA file handler log (short log) has GENERATION
    # and [STATS], while the redirect log does not.  If we haven't found gen info yet,
    # try the corresponding short log (e.g. wave32_E.log given wave31_wave32_E_plain...log).
    local _alt_log=""
    if [[ "$gen" == "init" ]]; then
        local _bn _wave_target _letter
        _bn=$(basename "$log" .log)
        _wave_target=$(echo "$_bn" | grep -oP 'wave\d+(?=_[A-Z][0-9]*(?:[_.]|$))' | tail -1 || true)
        _letter=$(echo "$_bn" | grep -oP '(?<=_)[A-Z][0-9]*(?=[_.]|$)' | tail -1 || true)
        if [[ -n "$_wave_target" && -n "$_letter" ]]; then
            _alt_log="${LOG_DIR}/${_wave_target}_${_letter}.log"
            if [[ -f "$_alt_log" && "$_alt_log" != "$log" ]]; then
                local _alt_buf
                _alt_buf=$(tail -300 "$_alt_log" 2>/dev/null)
                gl=$(echo "$_alt_buf" | grep -oP 'GENERATION \K\d+/\d+' | tail -1 || true)
                if [[ -n "$gl" ]]; then
                    gen="${gl}"
                    gen_current=$(echo "$gl" | cut -d/ -f1)
                    gen_total=$(echo "$gl" | cut -d/ -f2)
                    # Re-read buf from the alt log for subsequent metric extraction
                    buf="$_alt_buf"
                fi
            fi
        fi
    fi
    # ── Eval sub-progress ──
    local evp
    evp=$(echo "$buf" | grep -oP '\[EVAL\] Progress: \K\d+/\d+' | tail -1 || true)

    # ── Best fitness ──
    local best="—"
    local v
    # 1. Plain GA: [STATS] Best: X.XXXX
    v=$(echo "$buf" | grep -oP '\[STATS\] Best: \K[0-9.]+' | tail -1 || true)
    if [[ -n "$v" ]]; then
        best="$v"
    else
        # 2. Island model: [SUMMARY] Gen X/Y (...): island_0=V, island_1=V, ... → max value
        local summary_line
        summary_line=$(echo "$buf" | grep '\[SUMMARY\]' | tail -1 || true)
        if [[ -n "$summary_line" ]]; then
            v=$(echo "$summary_line" | grep -oP 'island_[^=]+=\K[0-9.]+' | sort -n | tail -1 || true)
            [[ -n "$v" ]] && best="$v"
        fi
        if [[ "$best" == "—" ]]; then
            # 3. Fallback: [NEW BEST] line
            v=$(echo "$buf" | grep -oP '\[NEW BEST\].*fitness.?\K[0-9.]+' | tail -1 || true)
            [[ -n "$v" ]] && best="$v"
        fi
        if [[ "$best" == "—" ]]; then
            # 4. Island per-island 'best=X' lines: use max from buf first, fall back to full log
            v=$(echo "$buf" | grep -oP '(?<=\] )best=\K[0-9.]+' | sort -n | tail -1 || true)
            [[ -z "$v" ]] && v=$(grep -oP '(?<=\] )best=\K[0-9.]+' "$log" 2>/dev/null | sort -n | tail -1 || true)
            [[ -n "$v" ]] && best="$v"
        fi
    fi

    # ── Average fitness ──
    local avg="—"
    v=$(echo "$buf" | grep -oP '\[STATS\].*Avg: \K[0-9.]+' | tail -1 || true)
    if [[ -n "$v" ]]; then
        avg="$v"
    else
        # Island model format: mean of each island's LATEST avg= value.
        # Extract all per-island avg values, keep last per island via awk, then average.
        v=$(grep -oP '\[island_\w+\s*\] best=[0-9.]+ avg=\K[0-9.]+' "$log" \
            | tail -40 \
            | LC_NUMERIC=C awk '{vals[NR]=$1} END {if(NR>0){s=0;for(i=1;i<=NR;i++)s+=vals[i]; printf "%.4f",s/NR}}' || true)
        [[ -n "$v" ]] && avg="$v"
    fi

    # ── Diversity ──
    local div="—"
    v=$(echo "$buf" | grep -oP 'Diversity: \K[0-9.]+' | tail -1 || true)
    if [[ -n "$v" ]]; then
        div="$v"
    else
        # Island model format: mean of each island's LATEST diversity= value.
        v=$(grep -oP '\[island_\w+\s*\] best=[0-9.]+ avg=[0-9.]+ diversity=\K[0-9.]+' "$log" \
            | tail -40 \
            | LC_NUMERIC=C awk '{vals[NR]=$1} END {if(NR>0){s=0;for(i=1;i<=NR;i++)s+=vals[i]; printf "%.4f",s/NR}}' || true)
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

    # ── Best strategy trades + W/L ratio ──
    # Extract trades and win_rate from the NEW BEST line with the highest fitness
    local best_trades="—" best_wl="—"
    local _best_line
    _best_line=$(grep -oP 'NEW BEST: fitness=\K[0-9.]+ profit=-?[0-9.]+% trades=\d+ win_rate=[0-9.]+' "$log" 2>/dev/null \
        | LC_NUMERIC=C awk -F'[ =]' 'BEGIN{best=-1} {f=$1+0; if(f>best){best=f; line=$0}} END{print line}' || true)
    if [[ -n "$_best_line" ]]; then
        local _t _w
        _t=$(echo "$_best_line" | grep -oP 'trades=\K\d+' || true)
        _w=$(echo "$_best_line" | grep -oP 'win_rate=\K[0-9.]+' || true)
        if [[ -n "$_t" ]]; then
            best_trades="$_t"
        fi
        if [[ -n "$_w" ]]; then
            # Convert win_rate (0.0–1.0) to W/L ratio; avoid division by zero
            best_wl=$(LC_NUMERIC=C awk -v w="$_w" 'BEGIN{
                l=1-w+0;
                if(l<=0){print "∞"}
                else{printf "%.2f", w/l}
            }')
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

    echo "${gen}|${best}|${avg}|${div}|${errs}|${elapsed}|${eta}|${best_trades}|${best_wl}|${profit}|${status}"
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
    local -a live_tail_logs=()

    # ── Title bar ──
    local title="GA EVOLUTION MONITOR"
    [[ -n "$WAVE_FILTER" ]] && title="GA MONITOR — ${WAVE_FILTER}"
    [[ "$LIVE_MODE" == true ]] && title+=' [LIVE]'

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
            IFS='|' read -r gen best avg div errs elapsed eta best_trades best_wl profit status <<< "$raw"
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
        apad "${BOLD}TRADES${NC}"     8; echo -n " "
        apad "${BOLD}W/L${NC}"        7; echo -n " "
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
        apad "${BOLD}TRADES${NC}"     8; echo -n " "
        apad "${BOLD}W/L${NC}"        7; echo -n " "
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
        IFS='|' read -r gen best avg div errs elapsed eta best_trades best_wl profit status <<< "$raw"
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

        # LIVE_MODE: only running experiments get a table row
        if [[ "$LIVE_MODE" == true && "$status" != "RUNNING" ]]; then
            case "$status" in
                DONE)    ((n_done++)) ;;
                CRASHED) ((n_crash++)) ;;
            esac
            continue
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
            RUNNING) ((n_run++)); [[ "$LIVE_MODE" == true ]] && live_tail_logs+=("$lf") ;;
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
        local c_best c_err c_trades c_wl c_status c_eta c_profit

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

        if [[ "$best_trades" == "—" ]]; then
            c_trades="${DIM}—${NC}"
        else
            c_trades="${CYAN}${best_trades}${NC}"
        fi

        if [[ "$best_wl" == "—" ]]; then
            c_wl="${DIM}—${NC}"
        elif awk "BEGIN{exit(!($best_wl >= 1.5))}" 2>/dev/null; then
            c_wl="${GREEN}${best_wl}${NC}"
        elif awk "BEGIN{exit(!($best_wl >= 1.0))}" 2>/dev/null; then
            c_wl="${YELLOW}${best_wl}${NC}"
        else
            c_wl="${RED}${best_wl}${NC}"
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
            apad "$c_trades"   7; echo -n "  "
            apad "$c_wl"       6; echo -n "  "
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
            apad "$c_trades"   7; echo -n "  "
            apad "$c_wl"       6; echo -n "  "
            echo -e "$c_status"
        fi
    done

    echo ""

    # ── Live log tails (only in LIVE_MODE) ──
    if [[ "$LIVE_MODE" == true ]]; then
        if [[ ${#live_tail_logs[@]} -gt 0 ]]; then
            for lf in "${live_tail_logs[@]}"; do
                print_live_tail "$lf" 15
            done
        else
            echo -e "  ${DIM}No experiments currently running.${NC}"
            echo -e "  ${DIM}Use '$0 ${WAVE_FILTER:+${WAVE_FILTER} }--all' to see completed results, or check --queue for pending.${NC}"
            echo ""
        fi
    fi

    # ── HoF Results block (when wave filter given and not in live mode) ──
    if [[ -n "$WAVE_FILTER" && "$LIVE_MODE" == false ]]; then
        local -a hof_rows=()
        for lf in "${logs[@]}"; do
            # Derive canonical key from log basename to handle nested logs correctly:
            #   wave32_A.log                      → canonical = wave32_A
            #   wave31_wave32_A_island_name.log   → canonical = wave32_A
            local h_bn h_canonical
            h_bn=$(basename "$lf" .log)
            if [[ "$h_bn" =~ ^wave[0-9]+_(wave[0-9]+_[A-Z][0-9]*)(_.*)?$ ]]; then
                h_canonical="${BASH_REMATCH[1]}"
            elif [[ "$h_bn" =~ ^(wave[0-9]+_[A-Z][0-9]*)(_.+)?$ ]]; then
                h_canonical="${BASH_REMATCH[1]}"
            else
                continue
            fi
            local hof_wave hof_suffix
            hof_wave="${h_canonical%%_*}"    # e.g. wave32
            hof_suffix="${h_canonical##*_}"  # e.g. A
            [[ "$hof_wave" != "$WAVE_FILTER" ]] && continue
            local hof_json="${HOF_BASE}/hall_of_fame_${hof_wave}_${hof_suffix}/hall_of_fame.json"
            [[ ! -f "$hof_json" ]] && continue
            hof_rows+=("${hof_suffix}|${hof_wave}")
        done

        if [[ ${#hof_rows[@]} -gt 0 ]]; then
            echo -e "  ${CYAN}── HoF Results — ${WAVE_FILTER} ──────────────────────────────────────────────────────────${NC}"
            echo -ne "  "
            apad "${BOLD}EXPERIMENT${NC}"  14; echo -n "  "
            apad "${BOLD}FITNESS${NC}"      8; echo -n "  "
            apad "${BOLD}T.PROFIT${NC}"     9; echo -n "  "
            apad "${BOLD}V.PROFIT${NC}"     9; echo -n "  "
            apad "${BOLD}T/V GAP${NC}"      8; echo -n "  "
            apad "${BOLD}PAIR_GEN${NC}"     8; echo -n "  "
            echo -e "${BOLD}HoF N${NC}"
            echo -e "  ${DIM}──────────────  ────────  ─────────  ─────────  ────────  ────────  ─────${NC}"

            for row in "${hof_rows[@]}"; do
                local h_exp2="${row%%|*}"
                local h_wave2="${row#*|}"
                local hof_data
                hof_data=$(read_hof_data "$h_wave2" "$h_exp2")
                IFS='|' read -r hof_fit hof_tp hof_vp hof_gap hof_pgr hof_n <<< "$hof_data"

                local c_fit c_tp c_vp c_gap c_pgr
                # Fitness color
                if [[ "$hof_fit" == "—" ]]; then c_fit="${DIM}—${NC}"
                elif awk "BEGIN{exit(!($hof_fit + 0 >= 0.30))}" 2>/dev/null; then c_fit="${GREEN}${hof_fit}${NC}"
                elif awk "BEGIN{exit(!($hof_fit + 0 >= 0.15))}" 2>/dev/null; then c_fit="${YELLOW}${hof_fit}${NC}"
                else c_fit="${RED}${hof_fit}${NC}"; fi
                # Train profit color (starts with + or -)
                if [[ "$hof_tp" == "—" ]]; then c_tp="${DIM}—${NC}"
                elif [[ "$hof_tp" == -* ]]; then c_tp="${RED}${hof_tp}${NC}"
                else c_tp="${GREEN}${hof_tp}${NC}"; fi
                # Val profit color
                if [[ "$hof_vp" == "—" ]]; then c_vp="${DIM}—${NC}"
                elif [[ "$hof_vp" == -* ]]; then c_vp="${RED}${hof_vp}${NC}"
                else c_vp="${GREEN}${hof_vp}${NC}"; fi
                # T/V gap color (absolute value: >20% red, >10% yellow, else green)
                if [[ "$hof_gap" == "—" ]]; then c_gap="${DIM}—${NC}"
                else
                    local gap_abs="${hof_gap#+}"; gap_abs="${gap_abs#-}"
                    if awk "BEGIN{exit(!(${gap_abs//[!0-9.]/} + 0 > 20))}" 2>/dev/null; then c_gap="${RED}${hof_gap}${NC}"
                    elif awk "BEGIN{exit(!(${gap_abs//[!0-9.]/} + 0 > 10))}" 2>/dev/null; then c_gap="${YELLOW}${hof_gap}${NC}"
                    else c_gap="${GREEN}${hof_gap}${NC}"; fi
                fi
                # Pair gen color
                if [[ "$hof_pgr" == "—" ]]; then c_pgr="${DIM}—${NC}"
                elif awk "BEGIN{exit(!($hof_pgr + 0 >= 0.90))}" 2>/dev/null; then c_pgr="${GREEN}${hof_pgr}${NC}"
                elif awk "BEGIN{exit(!($hof_pgr + 0 >= 0.75))}" 2>/dev/null; then c_pgr="${YELLOW}${hof_pgr}${NC}"
                else c_pgr="${RED}${hof_pgr}${NC}"; fi

                local clipped_h
                clipped_h=$(clip "${h_wave2}_${h_exp2}" 13)
                echo -ne "  "
                apad "$clipped_h"  14; echo -n "  "
                apad "$c_fit"       8; echo -n "  "
                apad "$c_tp"        9; echo -n "  "
                apad "$c_vp"        9; echo -n "  "
                apad "$c_gap"       8; echo -n "  "
                apad "$c_pgr"       8; echo -n "  "
                echo -e "${DIM}${hof_n}${NC}"
            done
            echo ""
        fi
    fi

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
        echo -e "  ${DIM}Refreshing ${INTERVAL}s · Ctrl+C to stop · --live for log tails · --queue for pending · --detail compact|detailed${NC}"
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
