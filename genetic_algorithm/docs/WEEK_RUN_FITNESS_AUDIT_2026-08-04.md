# Week-run fitness alignment audit (2026-08-04)

This note records the evidence used for the small `automation_island_v2`
fitness adjustment. It is diagnostic evidence, not a claim of live trading
performance.

## Input and join

- 1,995 valid strict candidate analyses from 107 completed wave reports.
- 617 exact joins between a persisted Hall-of-Fame `strategy_gene`, its
  `evolution_seed.json`, and the matching strict V2 observation reference.
- Spearman rank correlations compare the cheap search metrics with the later
  strict replay summaries. The joined HOF sample is selection-biased, but it is
  the relevant population for checking whether search survivors point in the
  same direction as replay.

## Main observations

The existing cheap fitness was useful but incomplete: rho was `0.538` with
strict robust score, `0.698` with gate alignment, `0.596` with expectancy LCB,
and `0.551` with annual-return LCB. Its top quartile was materially better than
its bottom quartile (median robust score `-0.141` vs `-0.345`, expectancy LCB
`-0.00224` vs `-0.00646`, and drawdown duration `402` vs `721.5` days).

The strongest directional inputs among the persisted cheap metrics were:

| Cheap metric | rho robust | rho expectancy LCB | rho annual LCB |
|---|---:|---:|---:|
| Profit factor | 0.641 | 0.507 | 0.656 |
| Max drawdown | -0.935 | -0.775 | -0.849 |
| Drawdown duration | -0.714 | -0.702 | -0.687 |
| Trade rate | 0.125 | 0.296 | 0.137 |
| Win rate | -0.661 | -0.676 | -0.704 |
| Legacy Sortino | -0.337 | -0.374 | -0.327 |

The negative win-rate relationship matches the observed high-win-rate pattern:
many small wins were paired with rare, large losses and 18-20% stoplosses.
Profit factor and drawdown therefore receive more survival influence; win rate
is retained only as a small supporting signal.

Both pair groups are visible during every search evaluation, so `TRAIN` and
`PAIR_VALIDATION` are not a hidden holdout. Equal group weighting aligned
better than the old 60/40 blend. Approximate profit correlations were:

| Split blend | rho robust | rho expectancy LCB | rho annual LCB |
|---|---:|---:|---:|
| 60/40 mean | 0.062 | 0.099 | 0.173 |
| 50/50 mean | 0.115 | 0.175 | 0.238 |
| Worst split | 0.401 | 0.550 | 0.513 |

This supports a conservative 25% soft worst-split blend. It is not a gate and
does not zero a candidate.

The previous `profit_max: 10` removed profit gradient above +10%. Of 710
persisted HOF split summaries, 26 training summaries exceeded +10% (maximum
+14.16%); no validation summary exceeded +10%. The new -5..+20 range covers
all observed pessimistic split summaries while keeping a gradient above +10%.

Finally, trades-per-active-month could be gamed by activity concentrated into a
few months. Strict replay already requires 12 active months. Search now counts
actual UTC trade-close months and folds `min(rate coverage, month coverage)`
into its existing smooth pair-coverage multiplier. It does not stack a new
penalty or add a gate.

## Changes justified by this audit

- Profit/PF/drawdown/recovery receive more soft survival weight; legacy
  Sortino, raw win rate, and secondary stability weights receive less.
- Drawdown uses `target / (target + drawdown)` around the existing 25% policy,
  preserving a useful gradient without a cliff.
- Pair groups are 50/50 with a 25% soft worst-split blend.
- Profit normalization is -5..+20 instead of clipping at +10.
- Strategy stoploss search is limited to -12..-3%; `max_open_trades` is fixed
  at the declared value 3 because independent single-pair replay cannot observe
  that gene.
- Activity uses actual trade-close months in both search and strict replay.
- The policy label is bumped to `automation-island-v2.1-shadow`, preventing
  silent pooling with old activity semantics.

No promotion threshold was relaxed and no new quality gate was added.
