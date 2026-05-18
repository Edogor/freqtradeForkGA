import json
d = json.load(open('genetic_algorithm/data/hall_of_fame/hall_of_fame.json'))
entries = d['entries']
entries.sort(key=lambda x: x.get('fitness', 0), reverse=True)
print(f"Hall of Fame: {len(entries)} entries")
for i, e in enumerate(entries[:10]):
    f = e.get('fitness', 0)
    p = e.get('profit_pct', '?')
    t = e.get('trades', '?')
    g = e.get('generation', '?')
    print(f"  #{i+1}: fitness={f:.4f}  profit={p}  trades={t}  gen={g}")
