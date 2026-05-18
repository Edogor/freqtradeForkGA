#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# paper_trading.sh — Manage 12 paper trading bots (Wave23 + Wave24 + Wave26)
#
# Usage:
#   ./paper_trading.sh start       Start all 7 bots
#   ./paper_trading.sh stop        Stop all 7 bots
#   ./paper_trading.sh restart     Restart all 7 bots
#   ./paper_trading.sh status      Show systemd status
#   ./paper_trading.sh smoke       Start bot 1 only (smoke test)
#   ./paper_trading.sh enable      Enable auto-start on boot
#   ./paper_trading.sh disable     Disable auto-start on boot
#   ./paper_trading.sh logs N      Follow logs for bot N
#   ./paper_trading.sh download    Download market data for all pairs
#   ./paper_trading.sh check       Pre-flight checks (configs, strategies, data)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_BOTS=12
VENV="${REPO_DIR}/.venv/bin"
CONFIGS_DIR="${REPO_DIR}/user_data/configs"
STRAT_DIR="${REPO_DIR}/user_data/strategies/paper_trading"
DATA_DIR="${REPO_DIR}/user_data/data"

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Strategy mapping
declare -A STRATEGIES=(
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
    [11]="PaperBot11_15m_w26V2R1"
    [12]="PaperBot12_1h_w26V2R2"
)

log_info()  { printf "${GREEN}[INFO]${NC}  %s\n" "$1"; }
log_warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$1"; }
log_error() { printf "${RED}[ERROR]${NC} %s\n" "$1"; }

# ── Pre-flight checks ──
do_check() {
    local ok=true
    echo ""
    printf "${BOLD}${CYAN}═══ Paper Trading Pre-flight Checks ═══${NC}\n"
    echo ""

    # Check freqtrade binary
    if [[ -x "${VENV}/freqtrade" ]]; then
        log_info "freqtrade binary: OK (${VENV}/freqtrade)"
    else
        log_error "freqtrade binary not found at ${VENV}/freqtrade"
        ok=false
    fi

    # Check base config
    if [[ -f "${CONFIGS_DIR}/paper_trading_base.json" ]]; then
        log_info "Base config: OK"
    else
        log_error "Missing: ${CONFIGS_DIR}/paper_trading_base.json"
        ok=false
    fi

    # Check API keys in base config
    local key
    key=$(python3 -c "
import json
with open('${CONFIGS_DIR}/paper_trading_base.json') as f:
    c = json.load(f)
print(c.get('exchange', {}).get('key', ''))
" 2>/dev/null || echo "")
    if [[ -z "$key" || "$key" == "your_exchange_key" ]]; then
        log_warn "Binance API key not set in paper_trading_base.json (dry_run works without keys but may have limited data)"
    else
        log_info "Binance API key: configured"
    fi

    # Check per-bot configs and strategies
    for i in $(seq 1 $NUM_BOTS); do
        local bot_conf="${CONFIGS_DIR}/paper_bot_${i}.json"
        local bot_env="${CONFIGS_DIR}/paper_bot_${i}.env"
        local strat_file="${STRAT_DIR}/${STRATEGIES[$i]}.py"

        if [[ -f "$bot_conf" && -f "$bot_env" && -f "$strat_file" ]]; then
            log_info "Bot ${i} (${STRATEGIES[$i]}): OK"
        else
            [[ ! -f "$bot_conf" ]] && log_error "Missing: $bot_conf"
            [[ ! -f "$bot_env" ]] && log_error "Missing: $bot_env"
            [[ ! -f "$strat_file" ]] && log_error "Missing: $strat_file"
            ok=false
        fi
    done

    # Check systemd service template
    local svc="$HOME/.config/systemd/user/freqtrade-paper@.service"
    if [[ -f "$svc" ]]; then
        log_info "Systemd template: OK"
    else
        log_error "Missing: $svc"
        ok=false
    fi

    # Check systemd user lingering (needed for services after logout)
    if loginctl show-user "$(whoami)" --property=Linger 2>/dev/null | grep -q "yes"; then
        log_info "User lingering: enabled"
    else
        log_warn "User lingering not enabled. Run: sudo loginctl enable-linger $(whoami)"
        log_warn "Without lingering, bots stop when you log out."
    fi

    # Check system resources
    echo ""
    printf "${BOLD}Resource Status:${NC}\n"
    free -h | head -2
    echo ""
    printf "Load: $(uptime | awk -F'load average:' '{print $2}')\n"

    echo ""
    if $ok; then
        printf "${GREEN}${BOLD}✓ All checks passed!${NC}\n"
    else
        printf "${RED}${BOLD}✗ Some checks failed. Fix issues above before starting.${NC}\n"
    fi
    echo ""
}

# ── Download market data ──
do_download() {
    log_info "Downloading market data for 6 pairs × 4 timeframes (60 days)..."
    cd "$REPO_DIR"
    "${VENV}/freqtrade" download-data \
        --config user_data/configs/paper_trading_base.json \
        --pairs BTC/USDT ETH/USDT SOL/USDT BNB/USDT XRP/USDT PEPE/USDT \
        --timeframes 15m 30m 1h 4h \
        --days 60
    log_info "Download complete!"
}

# ── Start bots ──
do_start() {
    local bots="${1:-all}"
    systemctl --user daemon-reload

    if [[ "$bots" == "all" ]]; then
        log_info "Starting all ${NUM_BOTS} paper trading bots (staggered 10s apart)..."
        for i in $(seq 1 $NUM_BOTS); do
            systemctl --user start "freqtrade-paper@${i}" && \
                log_info "Bot ${i} (${STRATEGIES[$i]}): started" || \
                log_error "Bot ${i} (${STRATEGIES[$i]}): FAILED to start"
            # Stagger starts to avoid memory/API pressure from all bots initializing at once
            if [[ $i -lt $NUM_BOTS ]]; then
                sleep 10
            fi
        done
    else
        systemctl --user start "freqtrade-paper@${bots}" && \
            log_info "Bot ${bots} (${STRATEGIES[$bots]}): started" || \
            log_error "Bot ${bots} (${STRATEGIES[$bots]}): FAILED to start"
    fi
}

# ── Stop bots ──
do_stop() {
    local bots="${1:-all}"

    if [[ "$bots" == "all" ]]; then
        log_info "Stopping all paper trading bots..."
        for i in $(seq 1 $NUM_BOTS); do
            systemctl --user stop "freqtrade-paper@${i}" 2>/dev/null && \
                log_info "Bot ${i}: stopped" || true
        done
    else
        systemctl --user stop "freqtrade-paper@${bots}" 2>/dev/null && \
            log_info "Bot ${bots}: stopped" || true
    fi
}

# ── Status ──
do_status() {
    echo ""
    printf "${BOLD}${CYAN}═══ Paper Bot Status ═══${NC}\n"
    echo ""
    for i in $(seq 1 $NUM_BOTS); do
        local state
        state=$(systemctl --user is-active "freqtrade-paper@${i}" 2>/dev/null || echo "inactive")
        local col="${RED}"
        [[ "$state" == "active" ]] && col="${GREEN}"
        printf "  Bot %-2s %-22s [${col}%-8s${NC}]  port %s\n" \
            "$i" "${STRATEGIES[$i]}" "$state" "$((8089 + i))"
    done
    echo ""
    free -h | head -2
    echo ""
}

# ── Main ──
case "${1:-help}" in
    start)
        do_start "${2:-all}"
        ;;
    stop)
        do_stop "${2:-all}"
        ;;
    restart)
        do_stop "${2:-all}"
        sleep 3
        do_start "${2:-all}"
        ;;
    status)
        do_status
        ;;
    smoke)
        log_info "Smoke test: starting Bot 1 only..."
        do_start 1
        log_info "Check logs: journalctl --user -u freqtrade-paper@1 -f"
        ;;
    enable)
        log_info "Enabling auto-start on boot for all bots..."
        for i in $(seq 1 $NUM_BOTS); do
            systemctl --user enable "freqtrade-paper@${i}" 2>/dev/null
        done
        log_info "Done. All bots will start automatically on login/boot."
        ;;
    disable)
        log_info "Disabling auto-start for all bots..."
        for i in $(seq 1 $NUM_BOTS); do
            systemctl --user disable "freqtrade-paper@${i}" 2>/dev/null
        done
        log_info "Done."
        ;;
    logs)
        if [[ -z "${2:-}" ]]; then
            log_error "Usage: $0 logs <bot_number>"
            exit 1
        fi
        journalctl --user -u "freqtrade-paper@${2}" -f --no-pager
        ;;
    download)
        do_download
        ;;
    check)
        do_check
        ;;
    help|--help|-h)
        echo ""
        printf "${BOLD}Paper Trading Bot Manager${NC}\n"
        echo ""
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  check        Pre-flight checks (configs, strategies, data)"
        echo "  download     Download market data for all pairs/timeframes"
        echo "  start [N]    Start all bots or bot N"
        echo "  stop [N]     Stop all bots or bot N"
        echo "  restart [N]  Restart all bots or bot N"
        echo "  status       Show status of all bots"
        echo "  smoke        Start bot 1 only (smoke test)"
        echo "  enable       Enable auto-start on boot"
        echo "  disable      Disable auto-start"
        echo "  logs N       Follow logs for bot N"
        echo ""
        echo "Bots:"
        for i in $(seq 1 $NUM_BOTS); do
            printf "  %d. %-22s  port %d\n" "$i" "${STRATEGIES[$i]}" "$((8089 + i))"
        done
        echo ""
        echo "Monitor: ./paper_monitor.sh [--loop]"
        echo "FreqUI:  http://localhost:8090-8096  (user: paper / pass: paper)"
        echo ""
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage."
        exit 1
        ;;
esac
