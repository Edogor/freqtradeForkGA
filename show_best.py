import json

exps = [
    ('V1', '01_V1_3m_mtf_momentum'),
    ('V2', '02_V2_5m_mtf_volume'),
    ('V4', '04_V4_3m_mtf_profit'),
    ('V5', '09_V5_5m_mtf_rank'),
    ('V7', '11_V7_3m_wide_mtf_rank'),
]

print("=== TOP STRATEGIES ACROSS ALL EXPERIMENTS ===\n")

for v, name in exps:
    path = f"genetic_algorithm/data/hall_of_fame_wave26_{v}/hall_of_fame.json"
    with open(path) as f:
        hof = json.load(f)
    entries = hof["entries"]
    top = sorted(entries, key=lambda x: x.get("fitness", 0), reverse=True)[0]
    m = top["metrics"]
    g = top["strategy_gene"]

    indicators = []
    for i in g["indicators"]:
        tf = "+" + i["timeframe"] if i.get("timeframe") else ""
        indicators.append(i["type"] + tf)

    entry_conds = [c["indicator"] + " " + c["operator"] + " " + str(c["threshold"]) for c in g["entry_conditions"]]
    exit_conds  = [c["indicator"] + " " + c["operator"] + " " + str(c["threshold"]) for c in g["exit_conditions"]]

    print(f"--- {v} ({name}) ---")
    print(f"  Fitness:       {top['fitness']:.4f}  (train={m.get('train_fitness',0):.4f} / val={m.get('val_fitness',0):.4f})")
    print(f"  Found:         Gen {top['generation_found']}")
    print(f"  Indicators:    {', '.join(indicators)}")
    print(f"  Entry:         {entry_conds}")
    print(f"  Exit:          {exit_conds}")
    print(f"  Timeframe:     {g['timeframe']} + MTF {g['informative_timeframes']}")
    print(f"  Stoploss:      {g['stoploss']*100:.1f}%  |  Trailing: {g['trailing_stop']}")
    print(f"  Max trades:    {g['max_open_trades']}")
    print(f"  Win rate:      train={m['win_rate']*100:.1f}%  /  val={m.get('val_win_rate',0)*100:.1f}%")
    print(f"  Profit:        train={m['profit']:.2f}%  /  val={m.get('val_profit',0):.2f}%")
    print(f"  Sharpe:        train={m['sharpe_ratio']:.3f}  /  val={m.get('val_sharpe',0):.3f}")
    print(f"  Max Drawdown:  train={m['max_drawdown']*100:.2f}%  /  val={m.get('val_max_drawdown',0)*100:.2f}%")
    print(f"  Profit factor: {m.get('profit_factor',0):.3f}")
    print(f"  Train/Val gap: {m.get('train_val_gap',0):.4f}  (lower = less overfit)")
    print(f"  Num trades:    train={m['num_trades']}  /  val={m.get('val_trades',0)}")
    print()

print("=== CRITICAL OBSERVATION ===")
print("All strategies show NEGATIVE raw profit despite high fitness scores.")
print("This is because the fitness metric is NOT raw profit - it is a composite")
print("of risk-adjusted metrics (Sharpe, drawdown, win rate, etc.).")
print("Negative profit means these strategies currently LOSE money overall.")
print("The evolution is optimizing the fitness function shape, not trading profitability.")
