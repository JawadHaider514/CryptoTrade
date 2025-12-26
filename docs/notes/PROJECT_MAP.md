# PROJECT MAP — Crypto Trading System

## Purpose
This document maps the major modules, runtime flow, API endpoints, and where the dashboard calls the APIs. It's designed to make the unified advanced run clear and auditable.

---

## Major directories / modules
- `main.py` — Unified application entry point (preferred). Initializes Flask, registers unified API endpoints, manages database path, bot executor and optional SocketIO streaming. Use `python main.py --mode advanced` to run the full advanced system.
- `run.py` — Backwards-compatible wrapper that delegates to `main.py` (kept for backward compatibility).
- `server/` — Legacy server implementations. `server/app.py` and `server/web_server.py` contain previous server flavors. `server/advanced_web_server.py` is the legacy advanced server (quarantined; features will be migrated into `main.py`).
- `core/` — Core trading and dashboard logic (signal generation, enhanced dashboard, ML predictor, bot logic).
- `api/` — API specific modules and integrations (trading integration, signal models moved here, etc.).
- `models/` — Compatibility shim for signal dataclasses (re-exports from `api.signals`).
- `frontend/` — Optional frontend source. Packaged frontend lives under `frontend/dist/` which is preferred when present; otherwise `static/` is served.
- `data/` — Databases: `backtest.db`, `crypto_historical.db`, etc.
- `legacy/` — (Planned) quarantine area for legacy/duplicate files prior to deletion.

---

## API Endpoints (defined in `main.py`)
- `GET /api/signals` — Returns latest signals (parameters: `limit`, `symbol`)
- `GET /api/health` — Health and status
- `GET /api/coins` — Available coins
- `GET /api/statistics` — Performance statistics
- `GET /api/trades` — Trade history export
- `GET /api/chart/<symbol>` — Chart data per symbol
- `GET /api/routes` — (Added) Returns list of registered routes for verification


> Note: All API responses include `source: 'UNIFIED_MAIN.PY'` to indicate they come from the unified server.

---

## Dashboard → API flow
1. Dashboard (served from `frontend/dist/index.html` or `static/index.html`) loads and issues an HTTP GET to `/api/signals`.
2. `main.py` handles the request and calls `_get_signals_from_database()` which reads `data/backtest.db`.
3. The API returns JSON with `signals` and metadata; the dashboard renders accordingly.
4. When `--mode advanced` is used, SocketIO is enabled and a lightweight streaming bot emits `bot_update` events that the frontend can subscribe to for live updates.

---

## Runtime commands
- Start unified advanced server:
  - `python main.py --mode advanced --port 3000`
- Start unified basic server:
  - `python main.py --mode basic`
- Backwards compatible wrapper (delegates to main):
  - `python run.py --mode advanced`

---

## Migration & Cleanup Plan
1. Quarantine legacy code by moving `server/advanced_web_server.py` (and other duplicates) into `legacy/` with a short rationale in `legacy/README.md`.
   - Moved in this change: `legacy/advanced_web_server.py` (quarantined, not deleted)
2. Verify unified advanced run works and dashboard loads correctly.
3. After verification, remove truly unused files and update this document with the final list and reasons.

---

## Cleanup completed (2025-12-24)
- `legacy/advanced_web_server.py` and `legacy/README.md` were archived (`legacy_archive_20251224.zip`) and removed from the repository after verification and approval.
- Archive and removal details are in `FIXES.md` and `ARCHIVE_LOG.md`.

If you want, I can proceed to run the full test suite and additional integration checks now.