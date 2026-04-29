# Web Dashboard — Improvement & Extension Roadmap

**Branch:** `feature/web-dashboard-improvements`  
**Created:** April 29, 2026  
**Status:** Planning Phase

---

## Overview

This roadmap documents all planned fixes, improvements, and new features for the GA Web Dashboard. Items are grouped by priority tier and category. Each item includes its current status, the affected files, and acceptance criteria.

---

## Current State (Baseline Audit)

### What exists today

| Area | Status | Notes |
|------|--------|-------|
| FastAPI backend + WebSocket | ✅ Working | `web/server.py`, `web/ws_monitor.py` |
| Run list / run detail pages | ✅ Working | Live WS updates, pause/stop controls |
| Generation page | ✅ Working | Strategy ranking per generation |
| Strategy page | ✅ Working | Gene tree, OHLCV chart, inline backtest launch, dry-run |
| Hall of Fame page | ✅ Working | Sortable table, CSV export, multi-select compare |
| Analytics page | ✅ Working | Feature importance, overfitting detection tabs |
| Backtest result page | ✅ Working | Equity curve, candlestick chart, paginated trade table |
| Config page | ✅ Working | Load/save YAML configs |
| Compare page | ✅ Working | Side-by-side strategy comparison |
| Run compare page | ✅ Working | Cross-run metrics comparison |
| CandlestickChart component | ✅ Working | `lightweight-charts` v4, trade markers, indicator overlays |
| Theme toggle (dark/light) | ✅ Working | |
| WebSocket real-time events | ✅ Working | Generation stats, phase updates, eval progress |

### Known Pain Points (Observed Gaps)

- No interactive strategy testing panel (adjust params → instant re-backtest)
- Chart data must be manually selected per pair; no auto-load from strategy config
- Strategy performance description is raw JSON metrics — no narrative/human-readable summary
- No SIS (Strategy Intelligence System) visualization page
- No Pareto front visualization for NSGA-II multi-objective runs
- No population diversity heatmap or gene-space visualization
- Backtest result page is separate from strategy page — poor UX flow
- Hall of Fame lacks sparkline/mini-chart per entry
- No experiment comparison across waves (cross-wave analytics)
- No "live evolution replay" — cannot scrub back through previous generations
- Config editor is basic text — no schema-aware form editor
- No drag-and-drop queue management UI
- Mobile/responsive layout partially broken in sidebar
- No notifications/alerts for completed runs or detected overfitting
- No per-strategy "Why did this win?" explanation panel
- Dry-run page exists but has no live P&L streaming or open-trade display
- No dark/light theme persistence across page reloads (localStorage)

---

## TIER 1 — Critical Fixes (Must Do First)

### T1-1: Theme Persistence Bug
**Problem:** Dark/light mode resets on page reload.  
**Files:** `frontend/src/components/ThemeToggle.tsx`, `frontend/src/store/useStore.ts`  
**Fix:** Persist theme choice to `localStorage`; read on initial load.  
**Effort:** XS (< 1h)  
**Status:** 🔴 Not started

---

### T1-2: Strategy Page — Auto-Load Chart Pair From Strategy Config
**Problem:** After opening a strategy, users must manually select pair + timeframe from dropdowns before any chart appears. The strategy already knows its pairs and timeframe.  
**Files:** `frontend/src/pages/StrategyPage.tsx`  
**Fix:** On strategy load, automatically set `selectedPair` and `selectedTimeframe` from `strategy.config` / `strategy.gene.timeframe`. Fall through to first available pair if config is missing.  
**Effort:** S (1-2h)  
**Status:** 🔴 Not started

---

### T1-3: Backtest Result — Merge Into Strategy Page
**Problem:** Launching a backtest from the Strategy page navigates away to `/backtest/:id`, breaking context. User loses the strategy view.  
**Files:** `frontend/src/pages/StrategyPage.tsx`, `frontend/src/pages/BacktestResultPage.tsx`  
**Fix:** Render backtest results inline in a collapsible panel below the chart on the Strategy page (already partially wired — `lastBacktestId`, `btResult` state exists). Deprecate standalone backtest route or keep as full-page fallback.  
**Effort:** M (3-5h)  
**Status:** 🔴 Not started

---

### T1-4: WebSocket Reconnection Hardening
**Problem:** If the WS connection drops (server restart, sleep), the dashboard silently goes stale. No user-visible reconnect attempt feedback.  
**Files:** `frontend/src/hooks/useWebSocket.ts`  
**Fix:** Implement exponential backoff reconnect (max 5 attempts), show a "Reconnecting…" banner in the Layout header, restore subscriptions after reconnect.  
**Effort:** S (2h)  
**Status:** 🔴 Not started

---

### T1-5: Mobile / Responsive Sidebar Layout
**Problem:** Sidebar overlaps content on screens < 768px. Navigation is unusable on phone/tablet.  
**Files:** `frontend/src/components/Sidebar.tsx`, `frontend/src/components/Layout.tsx`  
**Fix:** Add hamburger menu toggle, slide-over sidebar drawer for mobile. Use `lg:` breakpoints for desktop layout.  
**Effort:** M (3-4h)  
**Status:** 🔴 Not started

---

## TIER 2 — High-Value Features

### T2-1: Interactive Strategy Testing Panel ⭐ (Core Request)
**Goal:** Let the user tweak strategy parameters (indicator periods, thresholds, entry/exit conditions) in a UI form, submit, and see re-backtest results displayed on the chart — all without leaving the page.

**Components needed:**
- `StrategyParameterEditor` component: Renders gene fields as interactive form inputs (sliders for periods, dropdowns for operators, number fields for thresholds). Reads schema from `strategy.gene` structure.
- `InteractiveTestPanel` component: Hosts the editor + a "Run Test" button + an inline `BacktestResultPanel`.
- Backend: `POST /api/strategies/{id}/test` endpoint — accepts modified gene JSON, generates code, runs backtest, returns result ID.
- Chart integration: When test completes, overlay new trade markers in a different colour vs original backtest.

**Files to create/modify:**
- `frontend/src/components/StrategyParameterEditor.tsx` (new)
- `frontend/src/components/InteractiveTestPanel.tsx` (new)
- `frontend/src/pages/StrategyPage.tsx` (add panel below chart)
- `web/routers/strategies.py` (add `/test` endpoint)
- `web/models/strategy.py` (add `StrategyTestRequest` model)

**Acceptance criteria:**
- [ ] Form renders all gene fields with correct input types
- [ ] "Run Test" triggers backtest; spinner shows during run
- [ ] New trade markers overlay on chart in orange/purple to distinguish from original
- [ ] Summary diff card shows Δprofit, Δsharpe, Δwin_rate vs original
- [ ] Works for both HoF strategies and generation strategies

**Effort:** L (2-3 days)  
**Status:** 🔴 Not started

---

### T2-2: Strategy Performance Description / Narrative Summary ⭐
**Goal:** Show a human-readable "report card" for each strategy: what it does, how it performed, strengths, weaknesses.

**Components needed:**
- `StrategyNarrativeCard` component
- Logic to auto-generate text from metrics (no LLM required — rule-based templates):
  - "This strategy uses [indicators] on [timeframe] with [entry condition type]."
  - "It traded [N] times over [period], winning [win_rate]% of trades."
  - "Peak drawdown was [max_dd]%, Sharpe ratio [sharpe]."
  - Colour-coded verdict: STRONG / AVERAGE / WEAK / RISKY based on metric thresholds.
  - Warnings: "High drawdown", "Low trade count (statistically weak)", "Win rate near 50% (marginal edge)"

**Files to create/modify:**
- `frontend/src/components/StrategyNarrativeCard.tsx` (new)
- `frontend/src/utils/strategyNarrative.ts` (new — narrative generation logic)
- `frontend/src/pages/StrategyPage.tsx` (add card)
- `frontend/src/pages/HallOfFamePage.tsx` (add expandable narrative per row)

**Effort:** M (4-6h)  
**Status:** 🔴 Not started

---

### T2-3: NSGA-II Pareto Front Visualization
**Goal:** For multi-objective runs (`mode: nsga2`), show an interactive 2D/3D scatter plot of the Pareto front. Hovering a point shows the strategy. Clicking navigates to the Strategy page.

**Components needed:**
- `ParetoFrontChart` component using Recharts `ScatterChart` (2D) or a 3D option for 3 objectives.
- Colour points by Pareto rank (rank 1 = gold, rank 2 = silver, etc.)
- Axis selectors (choose which two objectives to plot)

**Files to create/modify:**
- `frontend/src/components/ParetoFrontChart.tsx` (new)
- `frontend/src/pages/GenerationPage.tsx` (add Pareto tab when `mode === 'nsga2'`)
- `frontend/src/pages/RunDetailPage.tsx` (add Pareto front summary card)
- `web/routers/generations.py` (expose `pareto_front` data in generation response)

**Effort:** M (4-6h)  
**Status:** 🔴 Not started

---

### T2-4: Population Diversity Heatmap / Gene-Space Visualization
**Goal:** Show how spread out the population is across generations — which genes are converging vs diversifying. Helps diagnose premature convergence.

**Components needed:**
- `DiversityHeatmap` component: 2D heatmap where rows = indicators, columns = generations, cell colour = usage frequency.
- `GeneSpaceScatter` component: PCA/t-SNE projection of population (backend-computed, sent as pre-computed 2D coords).
- Add to RunDetailPage as an "Advanced" tab.

**Files to create/modify:**
- `frontend/src/components/DiversityHeatmap.tsx` (new)
- `frontend/src/pages/RunDetailPage.tsx` (new "Diversity" tab)
- `web/routers/runs.py` (add `/api/runs/{id}/diversity` endpoint)
- `web/services/data_service.py` (compute diversity data from checkpoint)

**Effort:** L (1-2 days)  
**Status:** 🔴 Not started

---

### T2-5: Evolution Replay / Generation Scrubber
**Goal:** Like a video scrubber — drag a timeline slider to "replay" the evolution. Chart updates to show which strategies were alive and what the population looked like at generation N.

**Components needed:**
- `GenerationScrubber` component: Range slider 0 → max_gen, shows best fitness and population snapshot at selected generation.
- Population snapshot table updates live as scrubber moves.
- Chart replays equity curve for the best strategy at that generation.

**Files to create/modify:**
- `frontend/src/components/GenerationScrubber.tsx` (new)
- `frontend/src/pages/RunDetailPage.tsx` (embed below fitness chart)
- `web/routers/generations.py` (already returns per-generation data — verify completeness)

**Effort:** M (4-8h)  
**Status:** 🔴 Not started

---

### T2-6: SIS Intelligence Dashboard
**Goal:** Visualize the Strategy Intelligence System (SIS) — corpus stats, predictor performance, archetype clusters, indicator enrichment.

**Tabs:**
1. **Corpus** — count of strategies by fitness tier, distribution charts
2. **Predictors** — model accuracy (MAE/R² for win_rate/profit/drawdown predictors)
3. **Archetypes** — cluster visualization (HDBSCAN results), most common genes per archetype
4. **Indicator Enrichment** — bar chart of indicator lift scores (CCI, STOCH etc. with multipliers)
5. **A/B Framework** — results of SIS on/off experiments

**Files to create/modify:**
- `frontend/src/pages/SISPage.tsx` (new)
- `frontend/src/App.tsx` (add `/sis` route)
- `frontend/src/components/Sidebar.tsx` (add SIS nav link)
- `web/routers/sis.py` (already exists — audit and extend endpoints)

**Effort:** L (1-2 days)  
**Status:** 🔴 Not started

---

### T2-7: Queue Management UI
**Goal:** Show the experiment queue visually and allow drag-and-drop reordering, cancellation, and instant launch.

**Components needed:**
- `QueuePanel` component: Drag-and-drop list of YAML configs in `config/queue/`.
- Each card shows: experiment name, estimated duration, priority number.
- Actions: Move up/down priority, delete from queue, launch immediately.
- "Add to Queue" button: Opens a file picker or inline YAML editor.

**Files to create/modify:**
- `frontend/src/pages/QueuePage.tsx` (new)
- `frontend/src/components/QueueCard.tsx` (new)
- `frontend/src/App.tsx` + `Sidebar.tsx` (new route + nav)
- `web/routers/` — new `queue.py` router with list/reorder/delete/launch endpoints

**Effort:** L (2 days)  
**Status:** 🔴 Not started

---

### T2-8: Run Notifications / Alerts System
**Goal:** Show toast notifications + optional browser notifications when:
- A run completes
- A run fails or crashes
- Overfitting is detected (holdout score drops below threshold)
- A new Hall of Fame entry is added
- Evolution stagnates (no improvement for N generations)

**Components needed:**
- Extend `Toast.tsx` to support persistence and severity levels
- Hook into WS event stream for event types: `run_complete`, `run_failed`, `hof_update`, `overfitting_detected`, `stagnation_detected`
- Add notification history panel (bell icon in top bar)

**Files to create/modify:**
- `frontend/src/components/Toast.tsx` (extend)
- `frontend/src/components/NotificationCenter.tsx` (new)
- `frontend/src/hooks/useNotifications.ts` (new)
- `frontend/src/components/Layout.tsx` (add notification bell)
- `web/event_bus.py` (ensure correct event types are emitted)

**Effort:** M (4-6h)  
**Status:** 🔴 Not started

---

### T2-9: Hall of Fame — Mini Equity Curve Sparklines
**Goal:** Add a tiny equity curve preview per Hall of Fame row so users can visually scan strategy shapes without clicking each one.

**Files to create/modify:**
- `frontend/src/components/MiniEquityCurve.tsx` (new — lightweight SVG sparkline)
- `frontend/src/pages/HallOfFamePage.tsx` (add sparkline column)
- `web/routers/strategies.py` (expose equity curve data in HoF list response — may need endpoint augmentation)

**Effort:** M (3-4h)  
**Status:** 🔴 Not started

---

### T2-10: Cross-Wave Analytics Page
**Goal:** Compare results across experiment waves (wave30, wave31, wave32 etc.) — fitness progression, indicator evolution, win rate trends over time.

**Components needed:**
- `WaveComparisonChart` — line chart showing best fitness per wave
- `IndicatorEvolutionChart` — which indicators gained/lost prevalence wave over wave
- `WaveMetricsTable` — sortable table of wave-level aggregate stats

**Files to create/modify:**
- `frontend/src/pages/WaveAnalyticsPage.tsx` (new)
- `web/routers/` — new `waves.py` router aggregating data from `config/done/` directories

**Effort:** L (1-2 days)  
**Status:** 🔴 Not started

---

## TIER 3 — Nice to Have / Future

### T3-1: Schema-Aware Config Form Editor
**Goal:** Replace the raw YAML text editor in `ConfigPage` with a structured form where each field has a label, description, min/max validation, and linked to `config/schema.py`.  
**Effort:** XL (3+ days)  
**Status:** 🔴 Not started

---

### T3-2: Strategy Code Diff Viewer
**Goal:** When comparing two strategies, show a side-by-side diff of their generated Python code (not just metrics). Uses a syntax-highlighted diff component.  
**Effort:** M (3-4h)  
**Status:** 🔴 Not started

---

### T3-3: Live Dry-Run P&L Streaming
**Goal:** The `DryRunPage` currently launches a dry run but has no live display of open trades or running P&L. Add a live table of open positions with real-time mark-to-market values streamed via WebSocket.  
**Effort:** L (2 days) — requires backend dry-run state management  
**Status:** 🔴 Not started

---

### T3-4: LLM Strategy Designer UI Panel
**Goal:** Expose the LLM strategy designer (`llm/designer.py`) via the dashboard. User types a natural language description ("trend-following strategy for BTC 1h with ATR stop loss") and gets a generated strategy they can immediately backtest.  
**Effort:** L (2 days)  
**Status:** 🔴 Not started

---

### T3-5: Multi-Pair Backtest Results
**Goal:** The current backtest result page shows a single pair's chart. For strategies backtested on multiple pairs, add a pair selector with per-pair equity curves and aggregate stats.  
**Files:** `frontend/src/pages/BacktestResultPage.tsx`, `frontend/src/pages/StrategyPage.tsx`  
**Effort:** M (4-6h)  
**Status:** 🔴 Not started

---

### T3-6: Indicator Preview Overlay (Strategy Page)
**Goal:** When viewing a strategy's gene tree, clicking an indicator node should immediately preview that indicator on the chart with the strategy's configured parameters. Currently the user must manually add overlays.  
**Effort:** M (3-4h)  
**Status:** 🔴 Not started

---

### T3-7: Export / Share Strategy Package
**Goal:** One-click export of a strategy as a zip containing: Python strategy file, backtest results JSON, equity curve PNG, and a README summary. For sharing or deployment.  
**Effort:** M (4-6h)  
**Status:** 🔴 Not started

---

### T3-8: Fitness Landscape 3D Visualization
**Goal:** 3D surface plot showing the fitness landscape across two gene parameters (e.g., RSI period × threshold). Helps understand the search space topology.  
**Effort:** XL (3+ days) — requires systematic parameter sweeps on backend  
**Status:** 🔴 Not started

---

### T3-9: Strategy Tagging & Notes System
**Goal:** Allow users to add tags (e.g., "promising", "overfit", "deployed") and personal notes to strategies in the Hall of Fame. Persisted server-side.  
**Effort:** M (4-6h)  
**Status:** 🔴 Not started

---

### T3-10: Dashboard Onboarding Tour
**Goal:** First-visit interactive walkthrough (using a library like `react-joyride`) explaining each page and feature for new users.  
**Effort:** M (4h)  
**Status:** 🔴 Not started

---

## Backend API Gaps (Supporting Multiple Features Above)

| Endpoint Needed | Purpose | Needed For |
|----------------|---------|-----------|
| `POST /api/strategies/{id}/test` | Re-backtest with modified gene params | T2-1 |
| `GET /api/runs/{id}/diversity` | Population gene diversity data | T2-4 |
| `GET /api/sis/corpus` | SIS corpus statistics | T2-6 |
| `GET /api/sis/predictors` | Predictor accuracy metrics | T2-6 |
| `GET /api/sis/archetypes` | Archetype cluster data | T2-6 |
| `GET /api/sis/enrichment` | Indicator enrichment scores | T2-6 |
| `GET /api/queue` | List queued experiments | T2-7 |
| `PUT /api/queue/reorder` | Change queue priority | T2-7 |
| `DELETE /api/queue/{name}` | Remove from queue | T2-7 |
| `POST /api/queue/{name}/launch` | Immediate launch | T2-7 |
| `GET /api/waves` | List completed waves | T2-10 |
| `GET /api/waves/{wave}/stats` | Aggregate wave metrics | T2-10 |
| `GET /api/strategies/{id}/equity` | Equity curve data for HoF sparklines | T2-9 |

---

## Implementation Order (Recommended)

```
Phase A — Quick Wins (1-2 days total)
  T1-1  Theme persistence
  T1-2  Auto-load chart pair
  T1-3  Inline backtest result on strategy page
  T1-4  WS reconnect hardening
  T2-2  Strategy narrative summary (rule-based)
  T2-9  HoF mini sparklines

Phase B — Core Interactive Features (1 week)
  T1-5  Mobile responsive layout
  T2-1  Interactive strategy testing panel ⭐
  T2-3  Pareto front visualization
  T2-8  Notifications system
  T2-5  Evolution replay scrubber

Phase C — Advanced Dashboards (1-2 weeks)
  T2-4  Diversity heatmap
  T2-6  SIS intelligence dashboard
  T2-7  Queue management UI
  T2-10 Cross-wave analytics

Phase D — Polish & Future (ongoing)
  T3-1  Schema-aware config form
  T3-2  Code diff viewer
  T3-3  Live dry-run P&L
  T3-4  LLM designer UI
  T3-5  Multi-pair backtest
  T3-6  Indicator preview
  T3-7  Export package
  T3-8  Fitness landscape 3D
  T3-9  Tagging & notes
  T3-10 Onboarding tour
```

---

## Tech Stack Notes

| Concern | Current | Recommendation |
|---------|---------|---------------|
| Charts | `lightweight-charts` v4 + Recharts | Keep both — `lightweight-charts` for OHLCV, Recharts for metrics |
| 3D charts | None | Add `plotly.js` (react-plotly) for 3D when needed (T3-8) |
| Drag-and-drop | None | Add `@dnd-kit/core` for queue reordering (T2-7) |
| Diff viewer | None | Add `react-diff-viewer-continued` for code diff (T3-2) |
| Notifications | Basic Toast | Extend existing system before adding library |
| State | Zustand | Keep — works well |
| Forms | Raw HTML | Consider `react-hook-form` for the config editor (T3-1) |

---

## Definition of Done (per feature)

- [ ] Feature works end-to-end (backend + frontend)
- [ ] No TypeScript type errors (`npm run typecheck`)
- [ ] No console errors in browser devtools
- [ ] Works in dark mode and light mode
- [ ] Responsive on desktop (1280px+) and tablet (768px)
- [ ] Backend endpoint has basic error handling (4xx responses)
- [ ] Existing tests still pass (`pytest genetic_algorithm/tests/ -q`)

---

*Last updated: April 29, 2026 — initial planning pass*
