#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# paper_monitor.sh — Monitor all 10 paper trading bots
# Usage: ./paper_monitor.sh [--loop] [--json]
# ═══════════════════════════════════════════════════════════════════
set -u

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_BOTS=10
BASE_PORT=8090
API_USER="paper"
API_PASS="paper"

# Bot metadata
declare -A BOT_NAMES=(
    [1]="PaperBot1_15m_P1R1"
    [2]="PaperBot2_30m_P2R1"
    [3]="PaperBot3_30m_P2R3"
    [4]="PaperBot4_1h_P3R1"
    [5]="PaperBot5_1h_P3R3"
    [6]="PaperBot6_4h_P4R1"
    [7]="PaperBot7_4h_P8R1"
    [8]="PaperBot8_1h_w25A2R1"
    [9]="PaperBot9_30m_w25A2R2"
    [10]="PaperBot10_30m_w25A2R4"
)
declare -A BOT_TF=(
    [1]="15m" [2]="30m" [3]="30m" [4]="1h"
    [5]="1h"  [6]="4h"  [7]="4h"
    [8]="1h"  [9]="30m"  [10]="30m"
)
declare -A BOT_SHARPE=(
    [1]="1.88" [2]="2.88" [3]="2.47" [4]="2.02"
    [5]="2.10" [6]="1.13" [7]="4.60"
    [8]="12.62" [9]="12.22" [10]="13.33"
)

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

LOOP=false
JSON_MODE=false

for arg in "$@"; do
    case "$arg" in
        --loop) LOOP=true ;;
        --json) JSON_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [--loop] [--json]"
            echo "  --loop   Refresh every 30 seconds"
            echo "  --json   Output JSON format"
            exit 0
            ;;
    esac
done

get_bot_data() {
    # Fetch profit + open trade data for a single bot via python3
    # Output: one line of pipe-separated values
    local port=$1
    python3 -c "
import urllib.request, base64, json, sys

port = ${port}
auth = base64.b64encode(b'paper:paper').decode()

def api_get(path):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/api/v1{path}',
        headers={'Authorization': f'Bearer {token}'})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())

try:
    # Login with Basic Auth
    req = urllib.request.Request(f'http://127.0.0.1:{port}/api/v1/token/login',
        method='POST', headers={'Authorization': f'Basic {auth}', 'Content-Length': '0'})
    token = json.loads(urllib.request.urlopen(req, timeout=5).read())['access_token']

    # Get profit summary
    profit = api_get('/profit')
    profit_all = profit.get('profit_all_coin', 0) or 0
    closed_count = profit.get('closed_trade_count', 0) or 0
    closed_profit = profit.get('profit_closed_coin', 0) or 0

    # Get open trades
    status = api_get('/status')
    open_count = len(status) if isinstance(status, list) else 0
    open_profit = 0.0
    open_pairs = []
    if isinstance(status, list):
        for t in status:
            p = t.get('profit_abs', 0) or 0
            open_profit += p
            pair = t.get('pair', '?')
            open_pairs.append(f\"{pair}({p:+.2f})\")

    pairs_str = ','.join(open_pairs[:3]) if open_pairs else '-'
    print(f'{profit_all:+.4f}|{open_count}|{closed_count}|{open_profit:+.4f}|{closed_profit:+.4f}|{pairs_str}')
except Exception as e:
    print(f'ERR|0|0|0|0|-')
" 2>/dev/null || echo "ERR|0|0|0|0|-"
}

print_separator() {
    printf '%*s\n' 115 '' | tr ' ' '─'
}

show_dashboard() {
    clear
    echo ""
    printf "${BOLD}${CYAN}  ╔══════════════════════════════════════════════════════════════╗${NC}\n"
    printf "${BOLD}${CYAN}  ║          PAPER TRADING DASHBOARD — 7 Wave23 Bots            ║${NC}\n"
    printf "${BOLD}${CYAN}  ╚══════════════════════════════════════════════════════════════╝${NC}\n"
    echo ""

    # ── System Resources ──
    local mem_info
    mem_info=$(LC_ALL=C free -m | LC_ALL=C awk '/^Mem:/ {printf "%.1f / %.1f GB (%.0f%% used)", $3/1024, $2/1024, $3/$2*100}')
    local cpu_load
    cpu_load=$(LC_ALL=C uptime | LC_ALL=C awk -F'load average:' '{print $2}' | LC_ALL=C awk '{printf "%.2f / %.2f / %.2f", $1, $2, $3}')

    printf "  ${BOLD}System:${NC} RAM: %s  |  Load: %s\n" "$mem_info" "$cpu_load"
    print_separator

    # ── Bot Status Table ──
    printf "  ${BOLD}%-5s %-22s %-4s %-7s %-10s %-12s %-8s %-8s %-10s %-22s${NC}\n" \
        "Bot#" "Strategy" "TF" "Sharpe" "Status" "Profit" "Open" "Closed" "RAM" "Open Pairs"
    print_separator

    local running=0
    local stopped=0
    local failed=0

    for i in $(seq 1 $NUM_BOTS); do
        local port=$((BASE_PORT + i - 1))
        local name="${BOT_NAMES[$i]}"
        local tf="${BOT_TF[$i]}"
        local sharpe="${BOT_SHARPE[$i]}"

        # Check systemd status
        local svc_status
        svc_status=$(systemctl --user is-active "freqtrade-paper@${i}" 2>/dev/null || echo "inactive")

        # Get memory usage from systemd cgroup
        local mem_usage="—"
        if [[ "$svc_status" == "active" ]]; then
            local mem_bytes
            mem_bytes=$(systemctl --user show "freqtrade-paper@${i}" --property=MemoryCurrent --value 2>/dev/null || echo "0")
            if [[ "$mem_bytes" != "0" && -n "$mem_bytes" && "$mem_bytes" != "[not set]" ]]; then
                mem_usage="$((mem_bytes / 1048576))M"
            fi
        fi

        # Status colour
        local status_col="${RED}"
        local status_text="STOPPED"
        case "$svc_status" in
            active)
                status_col="${GREEN}"
                status_text="RUNNING"
                ((running++))
                ;;
            failed)
                status_col="${RED}"
                status_text="FAILED"
                ((failed++))
                ;;
            *)
                status_col="${DIM}"
                status_text="STOPPED"
                ((stopped++))
                ;;
        esac

        # Fetch trade data from API
        local profit_text="—"
        local open_text="—"
        local closed_text="—"
        local pairs_text="—"

        if [[ "$svc_status" == "active" ]]; then
            local bot_data=""
            bot_data=$(get_bot_data "$port") || bot_data="ERR|0|0|0|0|-"

            local profit_all open_count closed_count open_profit closed_profit pairs_str
            IFS='|' read -r profit_all open_count closed_count open_profit closed_profit pairs_str <<< "$bot_data"

            if [[ "$profit_all" != "ERR" ]]; then
                # Format profit with color
                if [[ "$profit_all" == +0.0000 || "$profit_all" == -0.0000 ]]; then
                    profit_text="${profit_all} USDT"
                else
                    profit_text="${profit_all} USDT"
                fi
                open_text="${open_count}"
                closed_text="${closed_count}"
                pairs_text="${pairs_str}"
            fi
        fi

        # Color the profit
        local profit_col="${NC}"
        if [[ "$profit_text" == +* ]]; then
            profit_col="${GREEN}"
        elif [[ "$profit_text" == -* ]]; then
            profit_col="${RED}"
        fi

        printf "  %-5s %-22s %-4s %-7s ${status_col}%-10s${NC} ${profit_col}%-12s${NC} %-8s %-8s %-10s ${DIM}%-22s${NC}\n" \
            "$i" "$name" "$tf" "$sharpe" "$status_text" "$profit_text" "$open_text" "$closed_text" "$mem_usage" "$pairs_text"
    done

    print_separator
    printf "  ${GREEN}Running: ${running}${NC}  |  ${DIM}Stopped: ${stopped}${NC}  |  ${RED}Failed: ${failed}${NC}"
    printf "  |  Total bots: ${NUM_BOTS}\n"
    echo ""

    # ── Quick Commands ──
    printf "  ${DIM}Commands:${NC}\n"
    printf "  ${DIM}  Start all:   systemctl --user start freqtrade-paper@{1..7}${NC}\n"
    printf "  ${DIM}  Stop all:    systemctl --user stop freqtrade-paper@{1..7}${NC}\n"
    printf "  ${DIM}  Bot N logs:  journalctl --user -u freqtrade-paper@N -f${NC}\n"
    printf "  ${DIM}  FreqUI:      http://localhost:8090-8096  (user: paper / pass: paper)${NC}\n"
    echo ""

    if $LOOP; then
        printf "  ${DIM}Refreshing every 30s... (Ctrl+C to exit)${NC}\n"
    fi
}

# ── Main ──
if $LOOP; then
    while true; do
        show_dashboard
        sleep 30
    done
else
    show_dashboard
fi
