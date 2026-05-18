#!/usr/bin/env bash
# ============================================================================
# ga_current.sh — Live Monitor: CURRENTLY RUNNING GA Experiments Only
# ============================================================================
# Discovers running experiments from ps + auto_queue_tracked.txt.
# Shows rich detail for both island model and plain GA runs.
#
#   ISLAND MODEL:  per-island fitness table, current island, migrations, ETA
#   PLAIN GA:      gen/best/avg/div, adaptive mutation, surrogate filter, ETA
#
# Usage:
#   ./ga_current.sh                 # live, refresh every 5s
#   ./ga_current.sh --once          # print once and exit
#   ./ga_current.sh --interval 10   # refresh every 10s
#   ./ga_current.sh --tail 15       # show last N filtered log lines per exp
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
TRACKED_FILE="${LOG_DIR}/auto_queue_tracked.txt"

INTERVAL=5
ONCE=false
TAIL_LINES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)     ONCE=true ;;
        --interval) INTERVAL="${2:-5}"; shift ;;
        --tail)     TAIL_LINES="${2:-10}"; shift ;;
        --help|-h)
            echo "Usage: $0 [--once] [--interval N] [--tail N]"
            echo "  --once         Print once and exit"
            echo "  --interval N   Refresh every N seconds (default: 5)"
            echo "  --tail N       Show last N filtered log lines per exp (default: off)"
            exit 0 ;;
        *) ;;
    esac
    shift
done

# ── Colours ──────────────────────────────────────────────────────────────────
R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; M=$'\033[35m'; C=$'\033[36m'
BOLD=$'\033[1m'; DIM=$'\033[2m'; NC=$'\033[0m'

# ── Format seconds → "2h35m" / "45m12s" / "8s" ──────────────────────────────
fmt_dur() {
    local s="${1:-0}"
    local m=$(( s / 60 )) ss=$(( s % 60 ))
    if   [[ $m -ge 60 ]]; then printf "%dh%02dm" $(( m / 60 )) $(( m % 60 ))
    elif [[ $m -gt 0 ]];  then printf "%dm%02ds" "$m" "$ss"
    else                       printf "%ds" "$ss"
    fi
}

# ── Colorize a fitness value ──────────────────────────────────────────────────
cfitness() {
    local v="$1"
    if [[ "$v" == "—" ]]; then printf '%s' "${DIM}—${NC}"
    elif LC_NUMERIC=C awk "BEGIN{exit(!($v >= 0.28))}" 2>/dev/null; then printf '%s' "${G}${v}${NC}"
    elif LC_NUMERIC=C awk "BEGIN{exit(!($v >= 0.18))}" 2>/dev/null; then printf '%s' "${Y}${v}${NC}"
    else printf '%s' "${R}${v}${NC}"
    fi
}

# ── Discover running GA processes ─────────────────────────────────────────────
# Output: "PID|EXP_NAME" one per line, sorted by EXP_NAME
discover_running() {
    declare -A pid_exp=()

    # 1. auto_queue_tracked.txt (most reliable — has exact exp_name used for log file)
    if [[ -f "$TRACKED_FILE" ]]; then
        while read -r pid exp_name; do
            [[ "$pid" =~ ^[0-9]+$ ]] || continue
            [[ -z "$exp_name" ]] && continue
            kill -0 "$pid" 2>/dev/null || continue
            pid_exp["$pid"]="$exp_name"
        done < <(grep -v '^#' "$TRACKED_FILE" 2>/dev/null || true)
    fi

    # 2. ps scan (catches experiments not in tracked file)
    while IFS= read -r line; do
        local pid cfg
        pid=$(echo "$line" | awk '{print $2}')
        cfg=$(echo "$line" | grep -oP '(?<=--config )\S+' || true)
        [[ -z "$cfg" || -z "$pid" ]] && continue
        [[ -n "${pid_exp[$pid]+x}" ]] && continue
        # Derive exp_name from config path: {parent_dir}_{basename_no_priority}
        local dir base
        dir=$(basename "$(dirname "$cfg")")
        base=$(basename "$cfg" .yaml)
        base="${base#[0-9][0-9]*_}"   # strip  01_ / 02_ etc.
        base="${base#[0-9]_}"          # strip  1_  etc.
        pid_exp["$pid"]="${dir}_${base}"
    done < <(ps aux 2>/dev/null | grep '[r]un_ga.py' || true)

    for pid in "${!pid_exp[@]}"; do
        echo "${pid}|${pid_exp[$pid]}"
    done | sort -t'|' -k2
}

# ── Find main (detailed) log for an experiment ────────────────────────────────
# For island model: the long log that has [SUMMARY] Gen X/Y lines.
# exp_name e.g. "wave31_wave32_A_island_15m_rank_ring" → logs/wave31_wave32_A...log
find_main_log() {
    local exp_name="$1"
    echo "${LOG_DIR}/${exp_name}.log"
}

# ── Find per-island short log (individual GA eval progress) ───────────────────
# e.g. exp "wave31_wave32_A_island..." → wave32_A.log
find_short_log() {
    local exp_name="$1"
    # Extract the "target wave" and experiment letter from nested name
    # Pattern: waveM_waveN_LETTER_... → waveN_LETTER.log
    local wave_target letter
    wave_target=$(echo "$exp_name" | grep -oP 'wave\d+(?=_[A-Z](?:[_.]|$))' | tail -1 || true)
    letter=$(echo "$exp_name" | grep -oP '(?<=_)[A-Z](?=[_.]|$)' | tail -1 || true)
    if [[ -n "$wave_target" && -n "$letter" ]]; then
        local candidate="${LOG_DIR}/${wave_target}_${letter}.log"
        [[ -f "$candidate" ]] && echo "$candidate" && return
    fi
    # Also check alphanumeric (e.g. A1, B2)
    local wave_target2 letter2
    wave_target2=$(echo "$exp_name" | grep -oP 'wave\d+(?=_[A-Z][0-9])' | tail -1 || true)
    letter2=$(echo "$exp_name" | grep -oP '(?<=_)[A-Z][0-9]+(?=[_.]|$)' | tail -1 || true)
    if [[ -n "$wave_target2" && -n "$letter2" ]]; then
        local candidate2="${LOG_DIR}/${wave_target2}_${letter2}.log"
        [[ -f "$candidate2" ]] && echo "$candidate2" && return
    fi
}

# ── Detect island model log ───────────────────────────────────────────────────
is_island_log() {
    grep -qm1 'GenericIslandModel\|GenericIsland' "$1" 2>/dev/null
}

# ── Parse island model state ──────────────────────────────────────────────────
# Output: gen_cur|gen_total|elapsed_s|eta_s|best_fit|best_island|best_profit|
#         migrations|num_islands|current_island|eval_progress|island_fitnesses
parse_island() {
    local log="$1"
    local short_log="${2:-}"
    local buf
    buf=$(tail -600 "$log" 2>/dev/null)

    # ── Generation from last [SUMMARY] line ──
    local gen_cur=0 gen_total=0 migrations=0
    local summary
    summary=$(echo "$buf" | grep '\[SUMMARY\]' | tail -1 || true)
    if [[ -n "$summary" ]]; then
        gen_cur=$(echo "$summary" | grep -oP '(?<=Gen )\d+')
        gen_total=$(echo "$summary" | grep -oP 'Gen \d+/\K\d+')
        migrations=$(echo "$summary" | grep -oP 'migrations=\K\d+' || echo 0)
    fi
    # Fallback: count GENERATION markers
    if [[ "$gen_cur" -eq 0 ]]; then
        local gm
        gm=$(echo "$buf" | grep -oP 'GENERATION \K\d+/\d+' | tail -1 || true)
        if [[ -n "$gm" ]]; then
            gen_cur="${gm%/*}"; gen_total="${gm#*/}"
        fi
    fi

    # ── Number of islands from startup ──
    local num_islands
    num_islands=$(grep -m1 -oP 'Islands: \K\d+' "$log" 2>/dev/null || echo 0)

    # ── Current generation marker (start-of-gen, may be ahead of last SUMMARY) ──
    local gen_active
    gen_active=$(echo "$buf" | grep -oP 'GENERATION \K\d+(?=/\d+)' | tail -1 || echo "$gen_cur")
    [[ -z "$gen_active" ]] && gen_active="$gen_cur"

    # ── Per-island fitnesses from last SUMMARY ──
    local island_fitnesses="" best_fit=0 best_island="—"
    if [[ -n "$summary" ]]; then
        # Format: "... Gen X/Y (Ns): island_0_trend=0.2129, island_1_trend=0.2672, ... | migrations=N"
        local islands_part
        islands_part=$(echo "$summary" | grep -oP ': \K[^|]+')
        while IFS='=' read -r iname ival; do
            iname="${iname// /}"; ival="${ival//[, ]/}"
            [[ -z "$iname" || -z "$ival" ]] && continue
            [[ "$iname" != island_* ]] && continue
            # Short form for display: strip "island_N_" prefix → type only
            local ishort
            ishort=$(echo "$iname" | grep -oP '(?<=island_[0-9]_)\w+' || echo "$iname")
            island_fitnesses+="${ishort}:${ival},"
            if LC_NUMERIC=C awk "BEGIN{exit(!($ival+0 > $best_fit+0))}" 2>/dev/null; then
                best_fit="$ival"
                best_island=$(echo "$iname" | grep -oP 'island_\d+_\K\w+' || echo "$iname")
            fi
        done <<< "$(echo "$islands_part" | tr ',' '\n')"
    fi

    # Fallback best_fit: max of any "best=X.XXXX" in buf
    if [[ "$best_fit" == "0" ]]; then
        local bf
        bf=$(echo "$buf" | grep -oP '(?<=best=)[0-9.]+' | sort -n | tail -1 || true)
        [[ -n "$bf" ]] && best_fit="$bf"
    fi

    # ── Best profit ever (from NEW BEST lines) ──
    local best_profit="—"
    local bp
    bp=$(grep -oP 'NEW BEST: fitness=[0-9.]+ profit=\K-?[0-9.]+(?=%)' "$log" 2>/dev/null \
        | LC_NUMERIC=C awk 'BEGIN{b=-9999} {x=$1+0; if(x>b){b=x}} END{if(b>-9999){printf (b>=0?"+%.2f%%":"%.2f%%"),b}}' || true)
    [[ -n "$bp" ]] && best_profit="$bp"

    # ── Current island being evaluated ──
    # Island summary line format: '  [island_0_trend           ] best=X avg=Y diversity=Z'
    local last_island_done
    last_island_done=$(echo "$buf" \
        | grep -oP '\[island_[0-9]+_[a-z_]+\s*\] best=' \
        | grep -oP 'island_[0-9]+' | tail -1 || true)
    local current_island="initialising"
    if [[ -n "$last_island_done" ]]; then
        local done_idx
        done_idx=$(echo "$last_island_done" | grep -oP '\d+')
        local next_idx=$(( done_idx + 1 ))
        [[ "$num_islands" -gt 0 && "$next_idx" -ge "$num_islands" ]] && next_idx=0
        current_island=$(grep -oP "island_${next_idx}_[a-z_]+" "$log" 2>/dev/null | head -1 || echo "island_${next_idx}")
    fi

    # ── Current eval progress (from short per-island log) ──
    local eval_progress="—"
    if [[ -n "$short_log" && -f "$short_log" ]]; then
        eval_progress=$(tail -100 "$short_log" 2>/dev/null \
            | grep -oP '\[EVAL\] Progress: \K\d+/\d+' | tail -1 || echo "—")
    fi

    # ── Elapsed since log start ──
    local t0
    t0=$(head -1 "$log" 2>/dev/null | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
    local elapsed_s=0
    if [[ -n "$t0" ]]; then
        local e0
        e0=$(date -d "$t0" +%s 2>/dev/null || echo 0)
        [[ "$e0" -gt 0 ]] && elapsed_s=$(( $(date +%s) - e0 ))
    fi

    # ── ETA: average of last 5 gen times (embedded in SUMMARY lines as "(Xs)") ──
    local eta_s=0
    if [[ "$gen_cur" -gt 0 && "$gen_total" -gt 0 ]]; then
        local recent_times n_recent sum_recent avg_recent_secs
        recent_times=$(echo "$buf" | grep '\[SUMMARY\]' \
            | grep -oP 'Gen \d+/\d+ \(\K[0-9.]+(?=s\))' | tail -5)
        n_recent=$(echo "$recent_times" | grep -c '[0-9]' || echo 0)
        if [[ "$n_recent" -ge 1 ]]; then
            sum_recent=$(echo "$recent_times" | awk '{sum+=$1} END{printf "%.0f", sum}')
            avg_recent_secs=$(( sum_recent / n_recent ))
            local remaining=$(( gen_total - gen_cur ))
            [[ $remaining -gt 0 ]] && eta_s=$(( avg_recent_secs * remaining ))
        elif [[ "$elapsed_s" -gt 0 ]]; then
            # No SUMMARY gen-time format — fall back to average
            local secs_per_gen=$(( elapsed_s / gen_cur ))
            local remaining=$(( gen_total - gen_cur ))
            [[ $remaining -gt 0 ]] && eta_s=$(( secs_per_gen * remaining ))
        fi
    fi

    # ── Plateau detection: count trailing gens with no best-fitness change ──
    local stall_gens=0
    local _prev_best=""
    while IFS= read -r sline; do
        local _lb
        _lb=$(echo "$sline" | grep -oP 'island_\w+=\K[0-9.]+' | sort -n | tail -1 || echo "")
        if [[ -n "$_lb" ]]; then
            if [[ "$_lb" == "$_prev_best" ]]; then
                (( stall_gens++ )) || true
            else
                stall_gens=0
            fi
            _prev_best="$_lb"
        fi
    done < <(echo "$buf" | grep '\[SUMMARY\]')

    # Output: fixed-field separated by |
    echo "${gen_active}|${gen_total}|${elapsed_s}|${eta_s}|${best_fit}|${best_island}|${best_profit}|${migrations}|${num_islands}|${current_island}|${eval_progress}|${island_fitnesses}|${stall_gens}"
}

# ── Parse plain GA state ──────────────────────────────────────────────────────
# Output: gen_cur|gen_total|elapsed_s|eta_s|best_fit|avg|diversity|
#         best_ever|best_gen|eval_progress|surrogate_skip|mut_rate|rss
parse_plain() {
    local log="$1"
    local buf
    buf=$(tail -400 "$log" 2>/dev/null)

    # ── Generation ──
    local gen_cur=0 gen_total=0
    local gm
    gm=$(echo "$buf" | grep -oP 'GENERATION \K\d+/\d+' | tail -1 || true)
    if [[ -n "$gm" ]]; then
        gen_cur="${gm%/*}"; gen_total="${gm#*/}"
    fi

    # ── Last STATS line ──
    local stats_line
    stats_line=$(echo "$buf" | grep '\[STATS\]' | tail -1 || true)
    local best_fit avg diversity rss
    best_fit=$(echo "$stats_line" | grep -oP 'Best: \K[0-9.]+' || echo "—")
    avg=$(echo "$stats_line" | grep -oP 'Avg: \K[0-9.]+' || echo "—")
    diversity=$(echo "$stats_line" | grep -oP 'Diversity: \K[0-9.]+' || echo "—")
    rss=$(echo "$stats_line" | grep -oP 'RSS: \K[0-9]+' || echo "—")

    # ── Best ever (all-time from NEW BEST lines) ──
    local best_ever best_gen
    local nb_line
    nb_line=$(grep -oP '\[NEW BEST\] \K\S+ with fitness [0-9.]+' "$log" 2>/dev/null | tail -1 || true)
    best_ever=$(echo "$nb_line" | grep -oP 'fitness \K[0-9.]+' || echo "—")
    best_gen=$(echo "$nb_line" | grep -oP 'Gen\K[0-9]+' || echo "—")

    # ── Current eval progress ──
    local eval_progress
    eval_progress=$(echo "$buf" | grep -oP '\[EVAL\] Progress: \K\d+/\d+' | tail -1 || echo "—")

    # ── Surrogate skipped last gen ──
    local surrogate_skip
    surrogate_skip=$(echo "$buf" | grep -oP '\[SURROGATE\] Skipped \K\d+' | tail -1 || echo "0")

    # ── Adaptive mutation rate ──
    local mut_rate
    mut_rate=$(echo "$buf" | grep -oP 'rate (?:increased|decreased) to \K[0-9.]+' | tail -1 || echo "—")

    # ── Elapsed ──
    local t0
    t0=$(head -1 "$log" 2>/dev/null | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
    local elapsed_s=0
    if [[ -n "$t0" ]]; then
        local e0
        e0=$(date -d "$t0" +%s 2>/dev/null || echo 0)
        [[ "$e0" -gt 0 ]] && elapsed_s=$(( $(date +%s) - e0 ))
    fi

    # ── ETA: average of last 5 gen times from GENERATION lines timestamps ──
    local eta_s=0
    if [[ "$gen_cur" -gt 1 && "$gen_total" -gt 0 && "$elapsed_s" -gt 0 ]]; then
        # Plain GA doesn't embed per-gen time; use last-5 GENERATION timestamps to derive it
        local gen_ts_sec
        gen_ts_sec=$(echo "$buf" | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?=.*GENERATION )' \
            | tail -6 | awk '{cmd="date -d \""$0"\" +%s"; cmd | getline t; close(cmd); print t}')
        local n_ts
        n_ts=$(echo "$gen_ts_sec" | grep -c '[0-9]' || echo 0)
        if [[ "$n_ts" -ge 2 ]]; then
            local oldest newest interval
            oldest=$(echo "$gen_ts_sec" | head -1)
            newest=$(echo "$gen_ts_sec" | tail -1)
            interval=$(( (newest - oldest) / (n_ts - 1) ))
            local rem=$(( gen_total - gen_cur ))
            [[ $rem -gt 0 && $interval -gt 0 ]] && eta_s=$(( interval * rem ))
        fi
        if [[ "$eta_s" -eq 0 ]]; then
            # Fallback: simple average
            local spc=$(( elapsed_s / gen_cur ))
            local rem=$(( gen_total - gen_cur ))
            [[ $rem -gt 0 ]] && eta_s=$(( spc * rem ))
        fi
    fi

    # ── Plateau detection: count trailing gens where best_ever hasn't changed ──
    local stall_gens=0
    local _prev_nb=""
    while IFS= read -r nbline; do
        local _nf
        _nf=$(echo "$nbline" | grep -oP 'fitness \K[0-9.]+' || echo "")
        if [[ -n "$_nf" ]]; then
            if [[ "$_nf" == "$_prev_nb" ]]; then
                (( stall_gens++ )) || true
            else
                stall_gens=0
            fi
            _prev_nb="$_nf"
        fi
    done < <(grep '\[NEW BEST\]' "$log" 2>/dev/null)
    # Gens with no NEW BEST at all since last one are also stalled
    local last_nb_gen
    last_nb_gen=$(grep -oP '\[NEW BEST\] Gen\K[0-9]+' "$log" 2>/dev/null | tail -1 || echo "$gen_cur")
    local extra_stall=$(( gen_cur - last_nb_gen ))
    [[ $extra_stall -gt $stall_gens ]] && stall_gens=$extra_stall

    echo "${gen_cur}|${gen_total}|${elapsed_s}|${eta_s}|${best_fit}|${avg}|${diversity}|${best_ever}|${best_gen}|${eval_progress}|${surrogate_skip}|${mut_rate}|${rss}|${stall_gens}"
}

# ── Render island model block ─────────────────────────────────────────────────
render_island() {
    local exp_name="$1" pid="$2" metrics="$3" log="$4"
    IFS='|' read -r gen_cur gen_total elapsed_s eta_s best_fit best_island \
                    best_profit migrations num_islands current_island \
                    eval_progress island_fitnesses stall_gens <<< "$metrics"

    local pct=0
    [[ "$gen_total" -gt 0 ]] && pct=$(( gen_cur * 100 / gen_total ))

    local mem_mb
    mem_mb=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f",$1/1024}' || echo "?")

    local elapsed_str="$(fmt_dur "$elapsed_s")"
    local eta_str
    [[ "$eta_s" -gt 0 ]] && eta_str="ETA ~$(fmt_dur "$eta_s")" || eta_str="ETA —"

    # Stall indicator (shown when best fitness hasn't improved for ≥3 gens)
    local stall_str=""
    if [[ "${stall_gens:-0}" -ge 3 ]]; then
        stall_str="  ${Y}[stalled ${stall_gens}g]${NC}"
    fi

    # Short exp name for display
    local disp_name="${exp_name}"
    # Strip leading "wave{N}_" queue prefix if present (e.g. wave31_wave32_A_... → wave32_A_...)
    disp_name=$(echo "$disp_name" | sed 's/^wave[0-9]*_//')

    # ── Line 1: header ──
    printf '%s● %s%s%s  %s[Island]%s  Gen %d/%d [%d%%]  %s  %s%s%s  %smem=%sMB  PID=%s%s%s\n' \
        "$G" "$BOLD" "$disp_name" "$NC" \
        "$DIM" "$NC" \
        "$gen_cur" "$gen_total" "$pct" "$elapsed_str" \
        "$M" "$eta_str" "$NC" \
        "$DIM" "$mem_mb" "$pid" "$NC" \
        "$stall_str"

    # ── Line 2: best fitness + current state ──
    local profit_col
    if   [[ "$best_profit" == -* ]]; then profit_col="${R}${best_profit}${NC}"
    elif [[ "$best_profit" == +* ]]; then profit_col="${G}${best_profit}${NC}"
    else profit_col="${DIM}${best_profit}${NC}"
    fi
    printf '  best: %s  island: %s%s%s  profit_best: %s  migrations: %s%s%s\n' \
        "$(cfitness "$best_fit")" \
        "$DIM" "$best_island" "$NC" \
        "$profit_col" \
        "$DIM" "$migrations" "$NC"

    # ── Line 3: current island + eval progress ──
    local cur_str
    cur_str=$(echo "$current_island" | sed 's/\s\+/ /g; s/ $//')
    [[ "$eval_progress" != "—" ]] && cur_str+="  (eval: ${eval_progress})"
    printf '  %sCurrent:%s %s\n' "$DIM" "$NC" "$cur_str"

    # ── Line 4: per-island fitness table ──
    if [[ -n "$island_fitnesses" ]]; then
        printf '  %sislands:%s ' "$DIM" "$NC"
        IFS=',' read -ra items <<< "${island_fitnesses%,}"
        local i=0
        for item in "${items[@]}"; do
            [[ -z "$item" ]] && continue
            local itype ival
            itype="${item%%:*}"
            ival="${item##*:}"
            local itype_short="${itype:0:6}"
            printf '%s[%d]%s%s:' "$DIM" "$i" "$itype_short" "$NC"
            cfitness "$ival"
            printf '  '
            (( i++ )) || true
        done
        echo ""
    fi

    # ── Optional: recent log tail ──
    if [[ "${TAIL_LINES}" -gt 0 ]]; then
        echo -e "  ${DIM}── recent activity ──────────────────────────────────────────────${NC}"
        tail -200 "$log" 2>/dev/null \
            | grep -E 'NEW BEST|SUMMARY|island.*best=|CHECKPOINT|ERROR|WARNING.*profit|gen_ratio' \
            | tail -n "$TAIL_LINES" \
            | while IFS= read -r line; do
                # strip timestamp + logger prefix, keep message
                local msg
                msg=$(echo "$line" | sed 's/^[0-9-]* [0-9:,]* - [^-]* - [A-Z]* - //')
                printf '    %b%s%b\n' "$DIM" "${msg:0:100}" "$NC"
            done
    fi

    echo ""
}

# ── Render plain GA block ─────────────────────────────────────────────────────
render_plain() {
    local exp_name="$1" pid="$2" metrics="$3" log="$4"
    IFS='|' read -r gen_cur gen_total elapsed_s eta_s best_fit avg diversity \
                    best_ever best_gen eval_progress surrogate_skip mut_rate rss stall_gens <<< "$metrics"

    local pct=0
    [[ "$gen_total" -gt 0 ]] && pct=$(( gen_cur * 100 / gen_total ))

    local mem_mb
    mem_mb=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f",$1/1024}' || echo "${rss}")

    local elapsed_str="$(fmt_dur "$elapsed_s")"
    local eta_str
    [[ "$eta_s" -gt 0 ]] && eta_str="ETA ~$(fmt_dur "$eta_s")" || eta_str="ETA —"

    # Stall indicator
    local stall_str=""
    if [[ "${stall_gens:-0}" -ge 3 ]]; then
        stall_str="  ${Y}[stalled ${stall_gens}g]${NC}"
    fi

    local disp_name="${exp_name}"
    disp_name=$(echo "$disp_name" | sed 's/^wave[0-9]*_//')

    # ── Line 1: header ──
    printf '%s● %s%s%s  %s[Plain]%s  Gen %d/%d [%d%%]  %s  %s%s%s  %smem=%sMB  PID=%s%s%s\n' \
        "$G" "$BOLD" "$disp_name" "$NC" \
        "$DIM" "$NC" \
        "$gen_cur" "$gen_total" "$pct" "$elapsed_str" \
        "$M" "$eta_str" "$NC" \
        "$DIM" "$mem_mb" "$pid" "$NC" \
        "$stall_str"

    # ── Line 2: current gen stats ──
    printf '  gen_best: %s  avg: %s%s%s  div: %s%s%s  eval: %s%s%s  surrogate_skip: %s%s%s\n' \
        "$(cfitness "$best_fit")" \
        "$DIM" "$avg" "$NC" \
        "$DIM" "$diversity" "$NC" \
        "$DIM" "$eval_progress" "$NC" \
        "$DIM" "$surrogate_skip" "$NC"

    # ── Line 3: all-time best + adaptive mutation ──
    local best_ever_str="${best_ever}"
    [[ "$best_ever" != "—" && "$best_gen" != "—" ]] && best_ever_str+="  (Gen${best_gen})"
    printf '  best_ever: %s  avg_profit_last_gen: %s  mut_rate: %s%s%s\n' \
        "$(cfitness "$best_ever")" \
        "$(grep -oP '\[EVAL\] Complete: \d+ succeeded.*avg profit: \K-?[0-9.]+%' "$log" 2>/dev/null | tail -1 || echo '—')" \
        "$DIM" "$mut_rate" "$NC"

    # ── Optional: recent log tail ──
    if [[ "${TAIL_LINES}" -gt 0 ]]; then
        echo -e "  ${DIM}── recent activity ──────────────────────────────────────────────${NC}"
        tail -200 "$log" 2>/dev/null \
            | grep -E '\[NEW BEST\]|\[STATS\]|\[CHECKPOINT\]|\[SURROGATE\]|mutation: rate|ERROR' \
            | tail -n "$TAIL_LINES" \
            | while IFS= read -r line; do
                local msg
                msg=$(echo "$line" | sed 's/^[0-9-]* [0-9:,]* - [^-]* - [A-Z]* - //')
                printf '    %b%s%b\n' "$DIM" "${msg:0:100}" "$NC"
            done
    fi

    echo ""
}

# ── Main dashboard ────────────────────────────────────────────────────────────
print_dashboard() {
    [[ "$ONCE" == false ]] && printf '\033[2J\033[H'

    local -a running=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && running+=("$line")
    done < <(discover_running)

    local n_running="${#running[@]}"
    local now
    now=$(date '+%H:%M:%S')

    echo ""
    printf '%b╔══════════════════════════════════════════════════════════════════════════════╗%b\n' "$C" "$NC"
    printf '%b║%b  %b%-55s%b %b%s%b  %b║%b\n' \
        "$C" "$NC" "$BOLD" "GA  LIVE  MONITOR  —  ${n_running} running" "$NC" \
        "$DIM" "$now" "$NC" "$C" "$NC"
    printf '%b╚══════════════════════════════════════════════════════════════════════════════╝%b\n' "$C" "$NC"
    echo ""

    if [[ $n_running -eq 0 ]]; then
        printf '  %bNo running GA experiments found.%b\n' "$Y" "$NC"
        printf '  %bTip: start with ./launch_wave32.sh or ./ga_auto_queue_v2.sh%b\n\n' "$DIM" "$NC"
        return
    fi

    for entry in "${running[@]}"; do
        local pid exp_name
        pid="${entry%%|*}"
        exp_name="${entry#*|}"

        local main_log
        main_log=$(find_main_log "$exp_name")
        local short_log
        short_log=$(find_short_log "$exp_name" || true)

        if [[ ! -f "$main_log" ]]; then
            printf '  %b● %s  (log not found: %s)%b\n\n' "$Y" "$exp_name" "$main_log" "$NC"
            continue
        fi

        if is_island_log "$main_log"; then
            local metrics
            metrics=$(parse_island "$main_log" "${short_log:-}")
            render_island "$exp_name" "$pid" "$metrics" "$main_log"
        else
            # For plain GA, the useful metrics (GENERATION, [STATS]) are in the
            # per-GA file handler log (short log), not the root redirect (main log).
            local plain_log="${short_log:-$main_log}"
            local metrics
            metrics=$(parse_plain "$plain_log")
            render_plain "$exp_name" "$pid" "$metrics" "$plain_log"
        fi
    done

    # ── System footer ──
    local total_rss free_mem load
    total_rss=$(ps aux 2>/dev/null | grep '[r]un_ga.py' \
        | awk '{s+=$6} END{printf "%.0f",s/1024}' || echo "?")
    free_mem=$(awk '/MemAvailable/{printf "%.0f",$2/1024}' /proc/meminfo 2>/dev/null || echo "?")
    load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo "?")

    printf '  %sSystem:%s  RSS_total=%sMB  free=%sMB  load=%s\n' \
        "$DIM" "$NC" "$total_rss" "$free_mem" "$load"

    # ── Queue daemon status ──
    local pid_file="${LOG_DIR}/auto_queue_daemon.pid"
    if [[ -f "$pid_file" ]]; then
        local dpid
        dpid=$(cat "$pid_file")
        if kill -0 "$dpid" 2>/dev/null; then
            printf '  %sDaemon:%s  PID=%s  RUNNING\n' "$DIM" "$NC" "$dpid"
        else
            printf '  %sDaemon:%s  PID=%s  STOPPED (stale)\n' "$DIM" "$NC" "$dpid"
        fi
    fi

    if [[ "$ONCE" == false ]]; then
        printf '\n  %sRefreshing every %ds — Ctrl+C to stop%s\n' "$DIM" "$INTERVAL" "$NC"
    fi
    echo ""
}

# ── Entry point ───────────────────────────────────────────────────────────────
if [[ "$ONCE" == true ]]; then
    print_dashboard
else
    trap 'echo ""; printf "  %bMonitor stopped.%b\n\n" "${DIM}" "${NC}"; exit 0' INT TERM
    while true; do
        print_dashboard
        sleep "$INTERVAL"
    done
fi
