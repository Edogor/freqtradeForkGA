#!/usr/bin/env bash
# ============================================================================
# GA Experiment Manager — Create, Queue, List, and Compare Experiments
# ============================================================================
# Swiss-army-knife for experiment lifecycle management.
#
# Usage:
#   ./ga_experiment.sh create --name E130_test --template island --pairs BTC/USDT
#   ./ga_experiment.sh queue  01_E130_test.yaml          # add to queue
#   ./ga_experiment.sh queue  --dir exploration/wave16/   # queue all from dir
#   ./ga_experiment.sh list                               # list all experiments
#   ./ga_experiment.sh list --wave wave16                 # list wave experiments
#   ./ga_experiment.sh list --running                     # show only running
#   ./ga_experiment.sh results --wave wave16              # show wave results
#   ./ga_experiment.sh results --best 10                  # top 10 experiments
#   ./ga_experiment.sh compare E125 E126 E127             # compare experiments
#   ./ga_experiment.sh inspect E125                       # detailed experiment info
#   ./ga_experiment.sh clean --dry-run                    # preview cleanup
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

# Activate venv if available (needed for PyYAML)
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
fi

CONFIG_DIR="${REPO_DIR}/genetic_algorithm/config"
QUEUE_DIR="${CONFIG_DIR}/queue"
DONE_BASE="${CONFIG_DIR}/done"
TEMPLATE_DIR="${CONFIG_DIR}/templates"
OUTPUT_BASE="${REPO_DIR}/genetic_algorithm/output"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
EXPLORATION_DIR="${OUTPUT_BASE}/exploration"

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Ensure directories ──
mkdir -p "$QUEUE_DIR" "$DONE_BASE" "$TEMPLATE_DIR"

# ── Help ──
show_help() {
    echo -e "${BOLD}GA Experiment Manager${NC}"
    echo ""
    echo "Commands:"
    echo "  create    Create a new experiment config from template"
    echo "  queue     Add experiment config(s) to the queue"
    echo "  list      List experiments (all, by wave, or by status)"
    echo "  results   Show experiment results and rankings"
    echo "  compare   Compare multiple experiments side-by-side"
    echo "  inspect   Show detailed info for one experiment"
    echo "  templates List available templates"
    echo "  clean     Clean up old logs and output files"
    echo "  next-id   Show next available experiment ID"
    echo ""
    echo "Examples:"
    echo "  $0 create --name E130_new_test --template island_llm"
    echo "  $0 queue 01_E130_new_test.yaml"
    echo "  $0 list --wave wave16 --running"
    echo "  $0 results --best 10"
    echo "  $0 compare E125 E126 E127"
    echo ""
    echo "Run '$0 COMMAND --help' for command-specific help."
}

# ── Next experiment ID ──
cmd_next_id() {
    local max_id=0
    # Search all configs and done dirs for highest E-number
    for dir in "$CONFIG_DIR" "$DONE_BASE" "$QUEUE_DIR" "$CONFIG_DIR"/exploration/*/; do
        [[ -d "$dir" ]] || continue
        while IFS= read -r f; do
            local num
            num=$(basename "$f" | grep -oP 'E\K\d+' | head -1)
            [[ -n "$num" && "$num" =~ ^[0-9]+$ ]] && (( num > max_id )) && max_id=$num
        done < <(find "$dir" -maxdepth 1 -name '*E[0-9]*.yaml' 2>/dev/null)
    done
    echo -e "${BOLD}Next experiment ID:${NC} E$(( max_id + 1 ))"
}

# ── Create experiment ──
cmd_create() {
    local name="" template="" priority="05" pairs="" timerange="" generations=""
    local population="" mutation_rate="" elite="" selection="" llm="" seed=""
    local output_file=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --name)       name="$2"; shift ;;
            --template)   template="$2"; shift ;;
            --priority)   priority="$2"; shift ;;
            --pairs)      pairs="$2"; shift ;;
            --timerange)  timerange="$2"; shift ;;
            --gen)        generations="$2"; shift ;;
            --pop)        population="$2"; shift ;;
            --mut)        mutation_rate="$2"; shift ;;
            --elite)      elite="$2"; shift ;;
            --selection)  selection="$2"; shift ;;
            --llm)        llm="$2"; shift ;;
            --seed)       seed="$2"; shift ;;
            --out)        output_file="$2"; shift ;;
            --help|-h)
                echo "Usage: $0 create [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --name NAME         Experiment name (required, e.g. E130_diversity_test)"
                echo "  --template TYPE     Base template: standard, island, island_llm, nsga2"
                echo "  --priority N        Queue priority (00=highest, 99=lowest, default: 05)"
                echo "  --pairs 'P1,P2'     Trading pairs (default: BTC/USDT,ETH/USDT)"
                echo "  --timerange RANGE   Backtesting range (default: 20210301-20260301)"
                echo "  --gen N             Number of generations"
                echo "  --pop N             Population size"
                echo "  --mut RATE          Mutation rate (e.g. 0.18)"
                echo "  --elite N           Elite size"
                echo "  --selection TYPE     Selection method: tournament, rank"
                echo "  --llm on|off        Enable/disable LLM integration"
                echo "  --seed N            Random seed"
                echo "  --out FILE          Output path (default: queue dir with priority prefix)"
                return 0
                ;;
            *) echo -e "${RED}Unknown option: $1${NC}"; return 1 ;;
        esac
        shift
    done

    if [[ -z "$name" ]]; then
        echo -e "${RED}ERROR: --name is required${NC}"
        return 1
    fi

    # Default template
    if [[ -z "$template" ]]; then
        template="island_llm"
    fi

    # Find template file
    local template_file=""
    local template_candidates=(
        "${TEMPLATE_DIR}/${template}.yaml"
        "${TEMPLATE_DIR}/base_${template}.yaml"
        "${CONFIG_DIR}/ga_config_island_llm_C4.yaml"  # fallback to C4 as base
    )
    for tf in "${template_candidates[@]}"; do
        if [[ -f "$tf" ]]; then
            template_file="$tf"
            break
        fi
    done

    if [[ -z "$template_file" ]]; then
        echo -e "${YELLOW}No template '${template}' found, generating from defaults${NC}"
        template_file=""
    fi

    # Output file
    if [[ -z "$output_file" ]]; then
        output_file="${QUEUE_DIR}/${priority}_${name}.yaml"
    fi

    if [[ -f "$output_file" ]]; then
        echo -e "${RED}ERROR: ${output_file} already exists${NC}"
        return 1
    fi

    if [[ -n "$template_file" ]]; then
        # Copy template and apply overrides
        cp "$template_file" "$output_file"

        # Update header comment
        sed -i "1s|^.*|# Experiment: ${name} (from template: $(basename "$template_file"))|" "$output_file"

        # Apply overrides using Python for safe YAML editing
        python3 -c "
import yaml, sys

with open('${output_file}', 'r') as f:
    config = yaml.safe_load(f)

# Apply overrides
overrides = {
    'pairs': '${pairs}',
    'timerange': '${timerange}',
    'generations': '${generations}',
    'population': '${population}',
    'mutation_rate': '${mutation_rate}',
    'elite': '${elite}',
    'selection': '${selection}',
    'llm': '${llm}',
    'seed': '${seed}'
}

if overrides['pairs']:
    pairs_list = [p.strip() for p in overrides['pairs'].split(',')]
    if 'backtesting' in config:
        config['backtesting']['pairs'] = pairs_list

if overrides['timerange']:
    if 'backtesting' in config:
        config['backtesting']['timerange'] = overrides['timerange']

if overrides['generations']:
    gen = int(overrides['generations'])
    if 'genetic_algorithm' in config:
        config['genetic_algorithm']['generations'] = gen
    if 'generic_island_model' in config and config.get('generic_island_model',{}).get('enabled'):
        config['generic_island_model']['generations'] = gen

if overrides['population']:
    pop = int(overrides['population'])
    if 'genetic_algorithm' in config:
        config['genetic_algorithm']['population_size'] = pop
    if 'generic_island_model' in config and config.get('generic_island_model',{}).get('enabled'):
        config['generic_island_model']['population_per_island'] = pop

if overrides['mutation_rate']:
    mr = float(overrides['mutation_rate'])
    if 'genetic_algorithm' in config:
        config['genetic_algorithm']['mutation_rate'] = mr
    if 'mutation' in config:
        config['mutation']['rate'] = mr

if overrides['elite']:
    el = int(overrides['elite'])
    if 'genetic_algorithm' in config:
        config['genetic_algorithm']['elite_size'] = el

if overrides['selection']:
    sel = overrides['selection']
    if 'genetic_algorithm' in config:
        config['genetic_algorithm']['selection_method'] = sel

if overrides['llm']:
    enabled = overrides['llm'].lower() in ('on', 'true', 'yes', '1')
    if 'advanced' in config:
        config['advanced']['enable_llm'] = enabled
        if 'llm' in config['advanced']:
            config['advanced']['llm']['enabled'] = enabled

if overrides['seed']:
    seed = int(overrides['seed'])
    if 'genetic_algorithm' in config:
        config['genetic_algorithm']['random_seed'] = seed

with open('${output_file}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
" 2>/dev/null

        if [[ $? -ne 0 ]]; then
            echo -e "${RED}ERROR: Failed to apply overrides${NC}"
            rm -f "$output_file"
            return 1
        fi
    else
        # Generate minimal config from scratch
        cat > "$output_file" << YAML
# Experiment: ${name}
# Created: $(date '+%Y-%m-%d %H:%M:%S')
# Template: generated from defaults

genetic_algorithm:
  random_seed: ${seed:-42}
  population_size: ${population:-15}
  generations: ${generations:-12}
  mutation_rate: ${mutation_rate:-0.18}
  crossover_rate: 0.7
  elite_size: ${elite:-1}
  tournament_size: 3
  selection_method: '${selection:-tournament}'
  convergence_patience: 8
  adaptive_mutation: true
  max_adaptation_factor: 2.5
  adaptation_step: 0.1
  fitness_sharing: false
  allow_self_crossover: false
  random_immigrants: 1
  mode: 'single_objective'

fitness_weights:
  profit: 0.25
  sharpe_ratio: 0.20
  sortino_ratio: 0.15
  profit_factor: 0.10
  drawdown: 0.15
  win_rate: 0.08
  trade_frequency: 0.07

fitness_penalties:
  min_trades: 5
  max_drawdown: 0.25
  min_win_rate: 0.30
  complexity_weight: 0.01

backtesting:
  timerange: "${timerange:-20210301-20260301}"
  stake_amount: 100
  pairs:
$(echo "${pairs:-BTC/USDT,ETH/USDT}" | tr ',' '\n' | sed 's/^ *//;s/ *$//' | awk '{print "    - \"" $0 "\""}')
  max_open_trades: 3
  fee: 0.001
  exchange: "binance"
  auto_download_data: true
  enable_cache: true
  timeout: 180

walk_forward:
  enabled: false

strategy_constraints:
  min_trades: 10
  max_drawdown: 0.25
  min_win_rate: 0.30
  timeframes: ["15m"]
  stoploss_range: [-0.15, -0.05]
  roi_range: [0.01, 0.08]
  max_open_trades_range: [1, 5]

multi_timeframe:
  enabled: false

indicators:
  available:
    - 'RSI'
    - 'MACD'
    - 'BBANDS'
    - 'EMA'
    - 'SMA'
    - 'SUPERTREND'
    - 'PSAR'
    - 'DONCHIAN'
    - 'CMF'
    - 'VROC'
  min_per_strategy: 2
  max_per_strategy: 5

mutation:
  rate: ${mutation_rate:-0.18}
  add_indicator_prob: 0.15
  remove_indicator_prob: 0.10
  modify_param_prob: 0.35
  modify_condition_prob: 0.25

crossover:
  rate: 0.7
  indicator_mix_prob: 0.5

advanced:
  parallel_evaluation: false
  max_workers: 1
  enable_llm: false

terminal_monitor:
  enabled: true
YAML
    fi

    echo -e "${GREEN}Created:${NC} ${output_file}"
    echo -e "${DIM}  Name:     ${name}"
    echo -e "  Template: ${template}"
    echo -e "  Priority: ${priority}${NC}"

    # Validate
    if python3 -c "import yaml; yaml.safe_load(open('${output_file}'))" 2>/dev/null; then
        echo -e "  ${GREEN}YAML valid ✓${NC}"
    else
        echo -e "  ${RED}YAML invalid ✗ — please check${NC}"
    fi
}

# ── Queue experiments ──
cmd_queue() {
    local files=()
    local from_dir=""
    local priority=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dir)     from_dir="$2"; shift ;;
            --priority) priority="$2"; shift ;;
            --help|-h)
                echo "Usage: $0 queue [FILES...] [--dir DIR] [--priority N]"
                echo ""
                echo "  FILES              YAML config files to add to queue"
                echo "  --dir DIR          Queue all YAML files from directory"
                echo "  --priority N       Override priority prefix for all files"
                return 0
                ;;
            *) files+=("$1") ;;
        esac
        shift
    done

    if [[ -n "$from_dir" ]]; then
        # Resolve relative to config dir or absolute
        [[ "$from_dir" != /* ]] && from_dir="${CONFIG_DIR}/${from_dir}"
        while IFS= read -r f; do
            files+=("$f")
        done < <(find "$from_dir" -maxdepth 1 -name '*.yaml' 2>/dev/null | sort)
    fi

    if [[ ${#files[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No files to queue. Provide YAML files or --dir.${NC}"
        return 1
    fi

    local queued=0
    for f in "${files[@]}"; do
        # Resolve path
        local src="$f"
        [[ "$src" != /* ]] && src="${REPO_DIR}/${src}"
        if [[ ! -f "$src" ]]; then
            # Try config dir
            src="${CONFIG_DIR}/${f}"
        fi
        if [[ ! -f "$src" ]]; then
            echo -e "${RED}Not found: ${f}${NC}"
            continue
        fi

        local bn
        bn=$(basename "$src")

        # Apply priority prefix if specified
        if [[ -n "$priority" ]]; then
            # Strip existing priority prefix if any
            local stripped
            stripped=$(echo "$bn" | sed 's/^[0-9]*_//')
            bn="${priority}_${stripped}"
        fi

        local dest="${QUEUE_DIR}/${bn}"
        if [[ -f "$dest" ]]; then
            echo -e "${YELLOW}Already queued: ${bn}${NC}"
            continue
        fi

        # Validate YAML
        if ! python3 -c "import yaml; yaml.safe_load(open('${src}'))" 2>/dev/null; then
            echo -e "${RED}Invalid YAML: ${bn}${NC}"
            continue
        fi

        cp "$src" "$dest"
        echo -e "${GREEN}Queued:${NC} ${bn}"
        ((queued++))
    done

    echo ""
    local total
    total=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
    echo -e "${BOLD}${queued} added, ${total} total in queue${NC}"
}

# ── List experiments ──
cmd_list() {
    local wave="" status_filter="" format="table"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --wave)    wave="$2"; shift ;;
            --running) status_filter="running" ;;
            --done)    status_filter="done" ;;
            --format)  format="$2"; shift ;;
            --help|-h)
                echo "Usage: $0 list [--wave NAME] [--running|--done] [--format table|csv]"
                return 0
                ;;
            *) wave="$1" ;;
        esac
        shift
    done

    echo -e "${BOLD}Experiment Inventory${NC}"
    echo ""

    # Count by wave
    echo -e "  ${BOLD}Waves:${NC}"
    for d in "${EXPLORATION_DIR}"/wave*; do
        [[ -d "$d" ]] || continue
        local wname
        wname=$(basename "$d")
        local exp_count
        exp_count=$(ls -d "$d"/E* "$d"/[a-z]* 2>/dev/null | wc -l)
        local log_count
        log_count=$(ls "${LOG_DIR}/${wname}_"*.log 2>/dev/null | wc -l)

        # Count running in this wave
        local running=0
        for lf in "${LOG_DIR}/${wname}_"*.log; do
            [[ -f "$lf" ]] || continue
            if ! grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$lf" 2>/dev/null; then
                if ! tail -10 "$lf" 2>/dev/null | grep -qP 'Traceback|FATAL'; then
                    ((running++)) || true
                fi
            fi
        done

        if [[ -n "$wave" && "$wname" != "$wave" ]]; then
            continue
        fi

        local status_tag=""
        if [[ $running -gt 0 ]]; then
            status_tag=" ${GREEN}(${running} running)${NC}"
        fi

        echo -e "    ${CYAN}${wname}${NC}: ${log_count} experiments${status_tag}"
    done

    # Queue stats
    local queued
    queued=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
    echo ""
    echo -e "  ${BOLD}Queue:${NC} ${queued} pending"

    # Done stats
    local done_count
    done_count=$(find "$DONE_BASE" -name '*.yaml' 2>/dev/null | wc -l)
    echo -e "  ${BOLD}Done:${NC}  ${done_count} completed configs"

    # Running processes
    local running_total
    running_total=$(pgrep -cf "run_ga\.py" 2>/dev/null) || running_total=0
    echo -e "  ${BOLD}Active:${NC} ${GREEN}${running_total} GA processes${NC}"

    # If specific wave requested, list experiments
    if [[ -n "$wave" ]]; then
        echo ""
        echo -e "  ${BOLD}${wave} experiments:${NC}"

        for lf in "${LOG_DIR}/${wave}_"*.log; do
            [[ -f "$lf" ]] || continue
            local bn
            bn=$(basename "$lf" .log)
            local exp="${bn#${wave}_}"

            local status="UNKNOWN"
            if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$lf" 2>/dev/null; then
                status="DONE"
            elif tail -10 "$lf" 2>/dev/null | grep -qP 'Traceback|FATAL'; then
                status="CRASHED"
            else
                # Check if still running
                local age
                age=$(( $(date +%s) - $(stat -c%Y "$lf" 2>/dev/null || echo 0) ))
                if [[ $age -lt 300 ]]; then
                    status="RUNNING"
                else
                    status="STALE"
                fi
            fi

            if [[ -n "$status_filter" ]]; then
                case "$status_filter" in
                    running) [[ "$status" != "RUNNING" ]] && continue ;;
                    done)    [[ "$status" != "DONE" ]] && continue ;;
                esac
            fi

            local c_status
            case "$status" in
                RUNNING) c_status="${GREEN}● RUN${NC}" ;;
                DONE)    c_status="${DIM}✓ DONE${NC}" ;;
                CRASHED) c_status="${RED}✗ CRASH${NC}" ;;
                STALE)   c_status="${YELLOW}? STALE${NC}" ;;
                *)       c_status="${DIM}?${NC}" ;;
            esac

            # Quick fitness extract
            local best="—"
            local v
            v=$(tail -200 "$lf" 2>/dev/null | grep -oP '\[STATS\] Best: \K[0-9.]+' | tail -1 || true)
            [[ -n "$v" ]] && best="$v"

            printf "    %-30s  fitness=%-8s  %b\n" "$exp" "$best" "$c_status"
        done
    fi
    echo ""
}

# ── Results ranking ──
cmd_results() {
    local wave="" top_n=20

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --wave)  wave="$2"; shift ;;
            --best)  top_n="$2"; shift ;;
            --help|-h)
                echo "Usage: $0 results [--wave NAME] [--best N]"
                return 0
                ;;
            *) wave="$1" ;;
        esac
        shift
    done

    echo -e "${BOLD}Experiment Results Ranking${NC}"
    echo ""

    # Collect results from all logs
    local -a results_data=()

    local log_pattern
    if [[ -n "$wave" ]]; then
        log_pattern="${LOG_DIR}/${wave}_*.log"
    else
        log_pattern="${LOG_DIR}/wave*_*.log"
    fi

    for lf in $log_pattern; do
        [[ -f "$lf" ]] || continue
        local bn
        bn=$(basename "$lf" .log)

        # Only completed experiments
        if ! grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$lf" 2>/dev/null; then
            continue
        fi

        local best="0"
        local v
        v=$(tail -300 "$lf" 2>/dev/null | grep -oP '\[STATS\] Best: \K[0-9.]+' | tail -1 || true)
        [[ -z "$v" ]] && v=$(tail -300 "$lf" 2>/dev/null | grep -oP '\[NEW BEST\].*fitness.?\K[0-9.]+' | tail -1 || true)
        [[ -n "$v" ]] && best="$v"

        local avg="0"
        v=$(tail -300 "$lf" 2>/dev/null | grep -oP '\[STATS\].*Avg: \K[0-9.]+' | tail -1 || true)
        [[ -n "$v" ]] && avg="$v"

        local safe=0 warn=0 overfit=0
        safe=$(grep -oP 'SAFE: \K\d+' "$lf" 2>/dev/null | tail -1 || echo 0)
        warn=$(grep -oP 'WARNING: \K\d+' "$lf" 2>/dev/null | tail -1 || echo 0)
        overfit=$(grep -oP 'OVERFIT: \K\d+' "$lf" 2>/dev/null | tail -1 || echo 0)

        results_data+=("${best}|${avg}|${safe:-0}|${warn:-0}|${overfit:-0}|${bn}")
    done

    if [[ ${#results_data[@]} -eq 0 ]]; then
        echo -e "  ${YELLOW}No completed experiments found.${NC}"
        return
    fi

    # Sort by fitness (descending)
    echo -e "  ${BOLD}Rank  Experiment                         Best     Avg      SAFE  WARN  OVERFIT${NC}"
    echo -e "  ${DIM}───── ───────────────────────────────── ──────── ──────── ───── ───── ───────${NC}"

    local rank=1
    printf '%s\n' "${results_data[@]}" | sort -t'|' -k1 -rn | head -"$top_n" | while IFS='|' read -r best avg safe warn overfit name; do
        local c_best
        if awk "BEGIN{exit(!($best >= 0.30))}" 2>/dev/null; then
            c_best="${GREEN}${best}${NC}"
        elif awk "BEGIN{exit(!($best >= 0.15))}" 2>/dev/null; then
            c_best="${YELLOW}${best}${NC}"
        else
            c_best="${RED}${best}${NC}"
        fi

        local c_safe=""
        [[ "${safe:-0}" -gt 0 ]] && c_safe="${GREEN}${safe}${NC}" || c_safe="${DIM}${safe:-0}${NC}"
        local c_overfit=""
        [[ "${overfit:-0}" -gt 0 ]] && c_overfit="${RED}${overfit}${NC}" || c_overfit="${DIM}${overfit:-0}${NC}"

        printf "  %-5s " "${rank}."
        printf "%-35s " "$name"
        printf "%b  " "$(printf '%-8s' "$best")"
        printf "%-8s " "$avg"
        printf "%b  " "$c_safe"
        printf "%-5s " "${warn:-0}"
        printf "%b\n" "$c_overfit"
        ((rank++))
    done

    echo ""
    echo -e "  ${DIM}Total completed: ${#results_data[@]} experiments${NC}"
    echo ""
}

# ── Compare experiments ──
cmd_compare() {
    local exps=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                echo "Usage: $0 compare EXP1 EXP2 [EXP3...]"
                echo "  Compare experiments by name (e.g. E125, E126)"
                return 0
                ;;
            *) exps+=("$1") ;;
        esac
        shift
    done

    if [[ ${#exps[@]} -lt 2 ]]; then
        echo -e "${RED}Need at least 2 experiments to compare${NC}"
        return 1
    fi

    echo -e "${BOLD}Experiment Comparison${NC}"
    echo ""

    # Header
    printf "  ${BOLD}%-20s${NC}" "Metric"
    for exp in "${exps[@]}"; do
        printf " ${BOLD}%-18s${NC}" "$exp"
    done
    echo ""
    printf "  ${DIM}%-20s" "────────────────────"
    for exp in "${exps[@]}"; do
        printf " %-18s" "──────────────────"
    done
    echo -e "${NC}"

    # Find log for each experiment
    local -a log_files=()
    for exp in "${exps[@]}"; do
        local lf
        lf=$(ls -t "${LOG_DIR}"/*"${exp}"*.log 2>/dev/null | head -1)
        log_files+=("${lf:-}")
    done

    # Extract and compare metrics
    local -a metrics=("Best Fitness" "Avg Fitness" "Diversity" "Generations" "Elapsed" "SAFE" "WARNING" "OVERFIT" "Status")
    local -a metric_patterns=(
        '\[STATS\] Best: \K[0-9.]+'
        '\[STATS\].*Avg: \K[0-9.]+'
        'Diversity: \K[0-9.]+'
        'GENERATION \K\d+/\d+'
        'ELAPSED_PLACEHOLDER'
        'SAFE: \K\d+'
        'WARNING: \K\d+'
        'OVERFIT: \K\d+'
        'STATUS_PLACEHOLDER'
    )

    for mi in "${!metrics[@]}"; do
        printf "  %-20s" "${metrics[$mi]}"

        for li in "${!log_files[@]}"; do
            local lf="${log_files[$li]}"
            local val="—"

            if [[ -n "$lf" && -f "$lf" ]]; then
                local buf
                buf=$(tail -300 "$lf" 2>/dev/null)

                case "${metrics[$mi]}" in
                    "Elapsed")
                        local t0
                        t0=$(head -1 "$lf" 2>/dev/null | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
                        if [[ -n "$t0" ]]; then
                            local e0
                            e0=$(date -d "$t0" +%s 2>/dev/null || echo 0)
                            local tN
                            tN=$(tac "$lf" 2>/dev/null | grep -oP -m1 '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
                            local e1
                            e1=$(date -d "${tN:-now}" +%s 2>/dev/null || date +%s)
                            local ds=$(( e1 - e0 ))
                            local m=$(( ds / 60 ))
                            if [[ $m -ge 60 ]]; then
                                val="$(( m / 60 ))h$(( m % 60 ))m"
                            else
                                val="${m}m"
                            fi
                        fi
                        ;;
                    "Status")
                        if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$lf" 2>/dev/null; then
                            val="DONE"
                        elif tail -20 "$lf" 2>/dev/null | grep -qP 'Traceback|FATAL'; then
                            val="CRASHED"
                        else
                            val="RUNNING"
                        fi
                        ;;
                    *)
                        val=$(echo "$buf" | grep -oP "${metric_patterns[$mi]}" | tail -1 || echo "—")
                        [[ -z "$val" ]] && val="—"
                        ;;
                esac
            fi

            printf " %-18s" "$val"
        done
        echo ""
    done
    echo ""
}

# ── Inspect single experiment ──
cmd_inspect() {
    local exp="$1"
    shift 2>/dev/null || true

    if [[ -z "$exp" ]]; then
        echo -e "${RED}Usage: $0 inspect EXPERIMENT_NAME${NC}"
        return 1
    fi

    echo -e "${BOLD}Experiment: ${CYAN}${exp}${NC}"
    echo ""

    # Find log
    local lf
    lf=$(ls -t "${LOG_DIR}"/*"${exp}"*.log 2>/dev/null | head -1)
    if [[ -z "$lf" || ! -f "$lf" ]]; then
        echo -e "  ${RED}No log file found for ${exp}${NC}"
        return 1
    fi

    echo -e "  ${BOLD}Log:${NC} ${lf}"

    # Find config
    local cfg
    cfg=$(find "$CONFIG_DIR" "$DONE_BASE" -name "*${exp}*.yaml" 2>/dev/null | head -1)
    [[ -n "$cfg" ]] && echo -e "  ${BOLD}Config:${NC} ${cfg}"

    # Find output
    local out_dir
    out_dir=$(find "$EXPLORATION_DIR" -type d -name "*${exp}*" 2>/dev/null | head -1)
    [[ -n "$out_dir" ]] && echo -e "  ${BOLD}Output:${NC} ${out_dir}"
    echo ""

    # Status
    local status="UNKNOWN"
    if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$lf" 2>/dev/null; then
        status="${GREEN}DONE${NC}"
    elif tail -20 "$lf" 2>/dev/null | grep -qP 'Traceback|FATAL'; then
        status="${RED}CRASHED${NC}"
    else
        local age
        age=$(( $(date +%s) - $(stat -c%Y "$lf" 2>/dev/null || echo 0) ))
        if [[ $age -lt 300 ]]; then
            status="${GREEN}RUNNING${NC}"
        else
            status="${YELLOW}STALE${NC}"
        fi
    fi
    echo -e "  ${BOLD}Status:${NC} ${status}"

    # Generation progress
    local gen
    gen=$(tail -300 "$lf" 2>/dev/null | grep -oP 'GENERATION \d+/\d+' | tail -1 || echo "—")
    echo -e "  ${BOLD}Generation:${NC} ${gen}"

    # Fitness history
    echo ""
    echo -e "  ${BOLD}Fitness progression:${NC}"
    local prev_best=0
    grep -oP 'GENERATION (\d+).*Best: ([0-9.]+).*Avg: ([0-9.]+)' "$lf" 2>/dev/null | tail -20 | while IFS= read -r line; do
        echo -e "    ${DIM}${line}${NC}"
    done

    # Best fitness
    local best
    best=$(tail -300 "$lf" 2>/dev/null | grep -oP '\[STATS\] Best: \K[0-9.]+' | tail -1 || echo "—")
    echo ""
    echo -e "  ${BOLD}Best fitness:${NC} ${GREEN}${best}${NC}"

    # SAFE/OVERFIT results
    local safe warn overfit
    safe=$(grep -oP 'SAFE: \K\d+' "$lf" 2>/dev/null | tail -1)
    warn=$(grep -oP 'WARNING: \K\d+' "$lf" 2>/dev/null | tail -1)
    overfit=$(grep -oP 'OVERFIT: \K\d+' "$lf" 2>/dev/null | tail -1)
    if [[ -n "$safe" || -n "$warn" || -n "$overfit" ]]; then
        echo -e "  ${BOLD}Results:${NC} SAFE=${GREEN}${safe:-0}${NC} WARNING=${YELLOW}${warn:-0}${NC} OVERFIT=${RED}${overfit:-0}${NC}"
    fi

    # Errors
    local errs
    errs=$(grep -cP '- (ERROR|CRITICAL) -|^Traceback' "$lf" 2>/dev/null || echo 0)
    if [[ "$errs" -gt 0 ]]; then
        echo ""
        echo -e "  ${RED}${errs} errors found. Last error:${NC}"
        grep -P '- (ERROR|CRITICAL) -' "$lf" 2>/dev/null | tail -3 | while IFS= read -r line; do
            echo -e "    ${DIM}${line}${NC}"
        done
    fi

    echo ""
}

# ── List templates ──
cmd_templates() {
    echo -e "${BOLD}Available Templates${NC}"
    echo ""

    if [[ -d "$TEMPLATE_DIR" ]] && ls "$TEMPLATE_DIR"/*.yaml &>/dev/null; then
        for tf in "$TEMPLATE_DIR"/*.yaml; do
            local name
            name=$(basename "$tf" .yaml)
            local desc
            desc=$(head -3 "$tf" 2>/dev/null | grep -oP '(?<=# ).*' | head -1 || echo "No description")
            echo -e "  ${CYAN}${name}${NC}"
            echo -e "    ${DIM}${desc}${NC}"
            echo ""
        done
    else
        echo -e "  ${YELLOW}No templates found in ${TEMPLATE_DIR}${NC}"
        echo ""
        echo -e "  ${DIM}Creating default templates...${NC}"
        create_default_templates
    fi
}

# ── Create default templates ──
create_default_templates() {
    # Standard template
    if [[ ! -f "${TEMPLATE_DIR}/standard.yaml" ]]; then
        cp "${CONFIG_DIR}/ga_config.yaml" "${TEMPLATE_DIR}/standard.yaml" 2>/dev/null || true
        echo -e "  ${GREEN}Created: standard.yaml${NC}"
    fi

    # Island LLM template (from C4)
    if [[ ! -f "${TEMPLATE_DIR}/island_llm.yaml" ]]; then
        cp "${CONFIG_DIR}/ga_config_island_llm_C4.yaml" "${TEMPLATE_DIR}/island_llm.yaml" 2>/dev/null || true
        echo -e "  ${GREEN}Created: island_llm.yaml${NC}"
    fi

    # Island template (no LLM)
    if [[ ! -f "${TEMPLATE_DIR}/island.yaml" ]]; then
        cp "${CONFIG_DIR}/ga_config_island.yaml" "${TEMPLATE_DIR}/island.yaml" 2>/dev/null || true
        echo -e "  ${GREEN}Created: island.yaml${NC}"
    fi

    echo ""
    echo -e "  ${DIM}Templates saved in ${TEMPLATE_DIR}${NC}"
    echo -e "  ${DIM}Customize them or add new ones.${NC}"
}

# ── Clean old files ──
cmd_clean() {
    local dry_run=false days=30

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) dry_run=true ;;
            --days)    days="$2"; shift ;;
            --help|-h)
                echo "Usage: $0 clean [--dry-run] [--days N]"
                echo "  Remove old logs and output files older than N days"
                return 0
                ;;
        esac
        shift
    done

    echo -e "${BOLD}Cleanup Preview${NC} (${days}+ days old)"
    echo ""

    # Old logs
    local old_logs
    old_logs=$(find "$LOG_DIR" -name '*.log' -mtime "+${days}" 2>/dev/null | wc -l)
    echo -e "  Old log files: ${old_logs}"

    # Stale PID files
    local stale_pids
    stale_pids=$(find "$LOG_DIR" -name '*_pids_*.txt' -mtime "+${days}" 2>/dev/null | wc -l)
    echo -e "  Stale PID files: ${stale_pids}"

    if [[ "$dry_run" == true ]]; then
        echo ""
        echo -e "  ${YELLOW}Dry run — no files deleted${NC}"
        echo -e "  ${DIM}Run without --dry-run to actually clean.${NC}"
    else
        echo ""
        echo -e "  ${YELLOW}Deleting...${NC}"
        find "$LOG_DIR" -name '*.log' -mtime "+${days}" -delete 2>/dev/null
        find "$LOG_DIR" -name '*_pids_*.txt' -mtime "+${days}" -delete 2>/dev/null
        echo -e "  ${GREEN}Done.${NC}"
    fi
    echo ""
}

# ── Command dispatch ──
if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

COMMAND="$1"
shift

case "$COMMAND" in
    create)     cmd_create "$@" ;;
    queue)      cmd_queue "$@" ;;
    list)       cmd_list "$@" ;;
    results)    cmd_results "$@" ;;
    compare)    cmd_compare "$@" ;;
    inspect)    cmd_inspect "$@" ;;
    templates)  cmd_templates "$@" ;;
    clean)      cmd_clean "$@" ;;
    next-id)    cmd_next_id ;;
    help|--help|-h) show_help ;;
    *)
        echo -e "${RED}Unknown command: ${COMMAND}${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
