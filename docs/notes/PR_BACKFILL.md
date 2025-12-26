PR: Add targeted backfill tooling and fixes
=========================================

Summary
-------
This PR introduces `scripts/backfill_gaps.py` — a targeted gap backfill tool to repair missing historical candles in `data/crypto_historical.db`.

What changed
------------
- Added `scripts/backfill_gaps.py` (dry-run mode, CSV import, targeted backfill, safety limits).
- Fixed CSV timestamp parsing to treat timestamps as UTC to avoid local-time offsets.
- Added a single-candle fallback: after bulk fetch the script will attempt to fetch individual missing timestamps.
- Minor improvements: better error handling, retries, and rate-limit respectful sleeps.
- Added `docs/BACKFILL.md` describing usage and limitations.

Validation & test plan
----------------------
- Ran dry-run: `python scripts/backfill_gaps.py --dry-run --limit 10` (listed gaps).
- Ran targeted live backfill (ADA/USDT) for a small gap and verified DB insert (`Saved N candles` message).
- Re-ran `scripts/verify_data_quality.py` and re-generated `reports/gaps_*.csv` to confirm progress.

Notes / Caveats
---------------
- Some missing timestamps may not be available from Binance (e.g., pre-listing); the tool logs these and continues.
- For large-scale remediation, a planned job with checkpointing is recommended.

Next steps
----------
- Add automated backfill integration to the watcher (`scripts/wait_for_download_and_verify.py`) as an opt-in step.
- Run targeted backfill on a subset of gaps (QA) and iterate improvements to the fetching algorithm.

Reviewer checklist
------------------
- [ ] Confirm `backfill_gaps.py` behavior and safety options.
- [ ] Validate CSV timezone handling.
- [ ] Approve adding docs and the new script to the repo.