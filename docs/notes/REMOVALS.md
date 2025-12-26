# REMOVALS.md — Summary of Deleted Legacy Files

Date: 2025-12-24
Archive: `legacy_archive_20251224.zip`

Deleted files:
- `legacy/advanced_web_server.py` — legacy advanced server implementation (redundant; unified into `main.py`).
- `legacy/README.md` — quarantine notes (archived; removal approved).

Reason: Redundant legacy files were quarantined, verified, archived, and removed to reduce maintenance surface area. The unified `main.py` handles advanced mode and replicates required features.

Rollback: Extract `legacy_archive_20251224.zip` at repo root to restore prior files.
