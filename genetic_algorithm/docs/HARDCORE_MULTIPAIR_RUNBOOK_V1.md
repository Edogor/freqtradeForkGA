# Hardcore Multi-Pair Campaign Runbook

This campaign is a search system. Its best artifacts are labelled
`SEARCH_CHAMPION`; neither a canary nor the seven-day run makes a strategy
paper- or live-ready.

## Launch contract

Both the canary and production launch fail closed unless the worktree is clean
and `HEAD` exactly matches its configured upstream. Commit and push all code,
presets and fixes before either command. The controller uses a campaign-local
SQLite queue, artifact tree and kill switch; do not point it at the historical
`genetic_algorithm/data/v2/automation` root or its state database.

From the repository root, validate the reduced canary first:

```bash
.venv/bin/python -m genetic_algorithm hardcore-campaign preflight \
  --canary --campaign-id hardcore-canary-YYYYMMDD \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-canary-YYYYMMDD"
```

Then run it. It executes exactly one 15m evolution followed by exactly one 1h
evolution, writes the final report with reason `CANARY_COMPLETED`, and never
queues a third run:

```bash
.venv/bin/python -m genetic_algorithm hardcore-campaign canary \
  --campaign-id hardcore-canary-YYYYMMDD \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-canary-YYYYMMDD"
```

Use a new isolated root for production:

```bash
.venv/bin/python -m genetic_algorithm hardcore-campaign preflight \
  --campaign-id hardcore-production-YYYYMMDD \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-production-YYYYMMDD"

.venv/bin/python -m genetic_algorithm hardcore-campaign start \
  --campaign-id hardcore-production-YYYYMMDD \
  --automation-root "$PWD/genetic_algorithm/data/v2/hardcore/hardcore-production-YYYYMMDD"
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
- `campaign_status.json` and `.sha256`: live lane, generation, scores, pair
  evidence, plateau and resource status.
- `runs/<run-id>/evolution_outcome.json` and `.sha256`: immutable run outcome.
- `campaign_final_report.json` and `.sha256`: both lane histories and all six
  pair metrics, always with `live_ready: false`.
