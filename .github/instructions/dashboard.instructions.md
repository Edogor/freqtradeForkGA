---
applyTo: "genetic_algorithm/web/**"
description: "GA web dashboard conventions. Use when: editing the FastAPI server, WebSocket monitor, API routers, or frontend components for the GA evolution dashboard."
---

# GA Web Dashboard Conventions

**Stack:** FastAPI backend (`genetic_algorithm/web/`) + React/TypeScript frontend (`genetic_algorithm/web/frontend/`)

**Key files:**
- `server.py` — FastAPI app, CORS, route mounting
- `ws_monitor.py` — WebSocket live updates for evolution progress
- `event_bus.py` — Internal pub/sub for decoupled event dispatch
- `run_manager.py` — Manages experiment lifecycle from the web layer
- `routers/` — API route handlers (REST endpoints)
- `services/` — Business logic layer
- `models/` — Pydantic models for request/response schemas
- `config.py` — Dashboard configuration

**Conventions:**
- All API responses use Pydantic models (type-safe serialization)
- WebSocket messages are JSON with a `type` field for message routing
- Evolution state read from experiment data files, not from the GA process directly
- Dashboard is read-only by default; launch operations require explicit user action
- CORS configured for local dev; restrict in production

**Testing:** Frontend components tested independently; backend endpoints tested via `pytest` with `httpx.AsyncClient`

See [WEB_DASHBOARD_ACHIEVEMENTS.md](../../genetic_algorithm/WEB_DASHBOARD_ACHIEVEMENTS.md) for feature reference.
