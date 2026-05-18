#!/bin/bash
# SIS Comparison Experiment Runner
# Run ctrl and exp in separate terminals

echo "=== SIS Comparison Experiment ==="
echo ""
echo "Terminal 1 (CTRL - no SIS):"
echo "  cd /home/kali/trading/freqtradeForkGA"
echo "  python3 -m genetic_algorithm --config genetic_algorithm/config/sis_comparison/ctrl_no_sis.yaml"
echo ""
echo "Terminal 2 (EXP - with SIS):"
echo "  cd /home/kali/trading/freqtradeForkGA"
echo "  python3 -m genetic_algorithm --config genetic_algorithm/config/sis_comparison/exp_with_sis.yaml"
echo ""
echo "Terminal 3 (Monitor):"
echo "  python3 -m genetic_algorithm.intelligence.sis_monitor"
echo ""
echo "Or run both in background:"
echo "  nohup python3 -m genetic_algorithm --config genetic_algorithm/config/sis_comparison/ctrl_no_sis.yaml > /dev/null 2>&1 &"
echo "  nohup python3 -m genetic_algorithm --config genetic_algorithm/config/sis_comparison/exp_with_sis.yaml > /dev/null 2>&1 &"
