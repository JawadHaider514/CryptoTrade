# FIXES.md — Legacy Deletion Plan

Date: 2025-12-24
Archive: `legacy_archive_20251224.zip`

Summary
-------
This document proposes safe deletion of quarantined legacy files located in `legacy/` after confirming parity with the unified implementation in `main.py`.

Files in `legacy/` (candidate deletions)
--------------------------------------
- `legacy/advanced_web_server.py` — legacy advanced server implementation (moved from `server/advanced_web_server.py`).
- `legacy/README.md` — quarantine notes (can be removed after archiving; recommended to keep a short note in repository history).

Rationale
---------
- The unified `main.py` now implements advanced mode (SocketIO, streaming bot, endpoints) and was tested manually in this environment; `legacy/advanced_web_server.py` is redundant.
- Keeping files in `legacy/` only delays cleanup risk and increases maintenance surface. We archive first to allow reversible deletion.

Verification Checklist (must pass before deletion)
--------------------------------------------------
- [ ] Unified advanced server starts with `python main.py --mode advanced` and exposes endpoints: `/api/signals`, `/api/health`, `/api/routes`, `/api/coins`, `/api/statistics`, `/api/trades`, `/api/chart/<symbol>`.
- [ ] Frontend root (`/`) serves `index.html` from `frontend/dist` or `static/` and displays signals (no "No signal" UI state when API returns signals).
- [ ] Streaming bot thread starts and emits `bot_update` messages (if SocketIO available) or equivalent snapshot endpoints exist.
- [ ] All tests (unit/integration) pass locally (optional if you chose to skip tests now).

Rollback Plan
-------------
If deleting causes issues, reverse the change by extracting `legacy_archive_20251224.zip` into the project root (this restores `legacy/` with the previous contents). Steps:
1. Unzip `legacy_archive_20251224.zip` in the repo root.
2. Run integrated tests and manual verifications.

Next Steps (after your approval)
--------------------------------
1. Approve deletion list in this file (reply: `approve delete legacy`).
2. I'll delete the files listed above, run `python -m pytest -q` (or manual checks if you prefer skipping), and commit the change with message: `chore: remove legacy quarantined files (see FIXES.md)`.
3. I will create a short `REMOVALS.md` and update `PROJECT_MAP.md` to note the removal and link to `legacy_archive_20251224.zip` for rollback.

Questions / Notes
-----------------
- If you'd like to keep `legacy/README.md` for audit, I can preserve it and only delete the code file.
- Let me know if you want me to run the test suite now or leave deletion for a later time.
