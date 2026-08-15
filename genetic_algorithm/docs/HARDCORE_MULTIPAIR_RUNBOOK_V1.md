# Hardcore Multi-Pair Campaign Runbook

This campaign is a search system. Its best artifacts are labelled
`SEARCH_CHAMPION`; neither a canary nor the seven-day run makes a strategy
paper- or live-ready.

The active fitness contract is `raw-multipair-score-v4`. For every pair it
builds economic edge `Q` from annualized net return, net expectancy and
trade-count-damped profit factor. Activity `A` is a smooth geometric progress
signal toward 17 trades/month on 15m or 10 trades/month on 1h plus 75% active
months. Only `F = Q × A` enters the frequency term: more profitable trading is
rewarded and more losing trading is penalized. There is no minimum trade or
profit gate, weekly target, or prescribed entry/exit pattern. Risk totals 8%
of the score. Every numeric policy value is config-backed and its SHA-256 is
part of the timeframe-specific score/panel identity.

Each lane keeps twelve distinct phenotypes: four `BALANCED`, four `EDGE` and
four `ACTIVITY`. Nine specialist islands start fresh. Three broad bridge
islands receive two Edge and two Activity seeds and reserve two ordinary
cross-niche offspring per generation. All candidates still backtest every one
of the six fixed pairs independently; no temporal anti-overfitting subsystem
is enabled.

## Launch contract

Both the canary and production launch fail closed unless the worktree is clean
and `HEAD` exactly matches its configured upstream. Commit and push all code,
presets and fixes before either command. The controller uses a campaign-local
SQLite queue, artifact tree and kill switch; do not point it at the historical
`genetic_algorithm/data/v2/automation` root or its state database.

First strictly replay the compatible historical v3 seeds under v4. Incompatible
optional seeds are recorded in the bootstrap quarantine and do not suspend a
lane:

```bash
.venv/bin/python -m genetic_algorithm hardcore-campaign bootstrap-v3 \
  --source-root "$PWD/genetic_algorithm/data/v2/hardcore/<v3-campaign>" \
  --output "$PWD/genetic_algorithm/data/v2/hardcore/<v4-bootstrap>/bootstrap_archive_v4.json"
```

From the repository root, validate the reduced canary with that archive:

```bash
.venv/bin/python -m genetic_algorithm hardcore-campaign preflight \
  --canary --campaign-id hardcore-canary-YYYYMMDD \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-canary-YYYYMMDD" \
  --bootstrap-archive "$PWD/genetic_algorithm/data/v2/hardcore/<v4-bootstrap>/bootstrap_archive_v4.json"
```

Then run it. It executes exactly one 15m evolution followed by exactly one 1h
evolution, writes the final report with reason `CANARY_COMPLETED`, and never
queues a third run:

```bash
.venv/bin/python -m genetic_algorithm hardcore-campaign canary \
  --campaign-id hardcore-canary-YYYYMMDD \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-canary-YYYYMMDD" \
  --bootstrap-archive "$PWD/genetic_algorithm/data/v2/hardcore/<v4-bootstrap>/bootstrap_archive_v4.json"
```

Use a new isolated root for production:

```bash
.venv/bin/python -m genetic_algorithm hardcore-campaign preflight \
  --campaign-id hardcore-production-YYYYMMDD \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-production-YYYYMMDD" \
  --bootstrap-archive "$PWD/genetic_algorithm/data/v2/hardcore/<v4-bootstrap>/bootstrap_archive_v4.json"

.venv/bin/python -m genetic_algorithm hardcore-campaign start \
  --campaign-id hardcore-production-YYYYMMDD \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-production-YYYYMMDD" \
  --bootstrap-archive "$PWD/genetic_algorithm/data/v2/hardcore/<v4-bootstrap>/bootstrap_archive_v4.json"
```

The production controller alternates 15m and 1h until the exact seven-day
deadline. At the deadline or a resource limit it signals only the evolution
coordinator, lets the active generation finish, writes a checkpoint and does
not queue another run.

## Service and operation

Render (but review before installing) a user service:

```bash
.venv/bin/python -m genetic_algorithm hardcore-campaign service-unit \
  --campaign-id hardcore-production-YYYYMMDD \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-production-YYYYMMDD" \
  --bootstrap-archive "$PWD/genetic_algorithm/data/v2/hardcore/<v4-bootstrap>/bootstrap_archive_v4.json" \
  --output /tmp/hardcore-multipair.service
```

Read the hash-verified status or request a generation-boundary stop:

```bash
.venv/bin/python -m genetic_algorithm hardcore-campaign status \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-production-YYYYMMDD"

.venv/bin/python -m genetic_algorithm hardcore-campaign stop \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-production-YYYYMMDD"
```

Important artifacts below the isolated root are:

- `campaign_state.json` and `.sha256`: crash-resumable controller state.
- the referenced `bootstrap_archive_v4.json` and `.sha256`: historical candidates
  newly strict-replayed on all six pairs, with niche assignments/quarantine.
- `campaign_status.json` and `.sha256`: live lane, generation, scores, pair
  evidence, `Q/A/F`, all three niche leaders, plateau and resource status.
- `runs/<run-id>/evolution_outcome.json` and `.sha256`: immutable run outcome.
- `campaign_final_report.json` and `.sha256`: both lane histories and all six
  pair metrics, always with `live_ready: false`.
