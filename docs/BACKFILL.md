Backfill gaps (scripts/backfill_gaps.py)
========================================

Purpose
-------
Targeted backfill for missing historical candles (small gaps) using Binance via ccxt.

Key features
------------
- Detect gaps in `data/crypto_historical.db` by timeframe (default `1h`).
- Accepts existing gaps CSV (from `scripts/generate_gap_csv.py`) via `--csv` or scans DB directly.
- Supports `--dry-run` to preview actions without writing to DB.
- Safety guards: `--max-days` default 7 to avoid attempting very long windows.
- Single-candle fallback: after bulk fetch, the script attempts individual candle fetches for any remaining missing timestamps.
- Timezone: CSV timestamps are treated as UTC by default (avoids local-time offsets).

Quick usage
-----------
- Dry run (preview):
  python scripts/backfill_gaps.py --dry-run --limit 20

- Backfill for a single gap using CSV (auto-confirm):
  echo y | & ".venv/Scripts/python.exe" scripts/backfill_gaps.py --csv reports/gaps_YYYYMMDD_HHMMSS.csv --limit 1

Safety notes
------------
- The script does not write to DB unless run without `--dry-run` and the user confirms.
- Use `--max-days` to limit attempts (default 7 days). Use `--force` to override with caution.

Limitations & Next improvements
------------------------------
- Some historical timestamps may be unavailable from Binance (symbol not listed yet); the script logs and skips missing candles.
- For very large gaps, a more robust chunked re-download (parallelization and checkpointing) would be useful.
- Consider adding alternate data sources or caching to increase coverage for older listing dates.

Contact
-------
For questions or to run a bulk backfill, open an issue and tag @Jawad.