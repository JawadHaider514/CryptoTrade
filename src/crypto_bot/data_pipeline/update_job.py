#!/usr/bin/env python3
"""
Incremental + Backfill OHLCV data update job (Requirement-compatible).

CLI Usage:
    python -m crypto_bot.data_pipeline.update_job --symbols data/symbols_32.json --timeframes 15m 1h --lookback_days 365

Behavior:
    - If parquet missing: fetch full lookback range.
    - If parquet exists:
        - If it doesn't cover desired lookback range (e.g., 365 days), backfill missing earlier range.
        - Also fetch any newer missing range up to now.
        - If --force_backfill 1: fetch full lookback range again and merge/dedupe.
    - Clean + save parquet.
    - Print summary + sample ranges.
"""

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from crypto_bot.data_pipeline.binance_ohlcv import fetch_klines
from crypto_bot.data_pipeline.storage import save_parquet, load_parquet
from crypto_bot.data_pipeline.cleaning import clean_ohlcv

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Project root / paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "ohlcv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_symbols(symbols_file: str) -> List[str]:
    """Load symbols from JSON. Supports {"symbols":[...]} or [...]"""
    path = Path(symbols_file)
    if not path.exists():
        logger.error(f"Symbols file not found: {symbols_file}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        symbols = data.get("symbols", [])
    elif isinstance(data, list):
        symbols = data
    else:
        symbols = []

    symbols = [str(s).upper() for s in symbols if str(s).strip()]
    logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
    return symbols


def _ensure_utc_dt(dt: datetime) -> datetime:
    """Ensure datetime has UTC tzinfo."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _pdts_to_utc_dt(ts: pd.Timestamp) -> datetime:
    """Convert pandas timestamp to UTC datetime safely."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    """Load parquet; return empty DF if missing or errors."""
    try:
        df = load_parquet(str(path))
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# Core update logic
# -----------------------------------------------------------------------------
def update_symbol_timeframe(
    symbol: str,
    interval: str,
    lookback_days: int,
    force_backfill: bool = False,
) -> Dict[str, Any]:
    """
    Update OHLCV for a single symbol+timeframe.

    - Ensures coverage of desired lookback (e.g., 365 days) even if parquet already exists.
    - Uses backfill for missing earlier range + incremental for newer range.
    - If force_backfill True: fetch full range desired_start..now and merge/dedupe.
    """
    symbol = symbol.upper().strip()
    interval = interval.strip()

    output_dir = DATA_DIR / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{interval}.parquet"

    result: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "status": "pending",
        "rows_before": 0,
        "rows_after": 0,
        "new_rows": 0,
        "error": None,
    }

    try:
        existing_df = _safe_read_parquet(output_path)
        if not existing_df.empty and "timestamp" in existing_df.columns:
            # Ensure timestamp is datetime+UTC
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True, errors="coerce")
            existing_df = existing_df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
            existing_df = existing_df.reset_index(drop=True)
        else:
            existing_df = pd.DataFrame()

        result["rows_before"] = len(existing_df)

        now_utc = datetime.now(timezone.utc)
        desired_start = now_utc - timedelta(days=lookback_days)

        # Decide what to fetch
        new_parts = []

        if existing_df.empty:
            # Full backfill
            start = desired_start
            end = now_utc
            logger.info(f"{symbol} {interval}: Full backfill from {start} to {end}")
            df_full = fetch_klines(symbol=symbol, interval=interval, start=start, end=end)
            if not df_full.empty:
                new_parts.append(df_full)

        else:
            min_ts = existing_df["timestamp"].min()
            max_ts = existing_df["timestamp"].max()

            # Force backfill: fetch full range regardless
            if force_backfill:
                start = desired_start
                end = now_utc
                logger.info(
                    f"{symbol} {interval}: FORCE backfill from {start} to {end} "
                    f"(existing_min={min_ts}, existing_max={max_ts})"
                )
                df_full = fetch_klines(symbol=symbol, interval=interval, start=start, end=end)
                if not df_full.empty:
                    new_parts.append(df_full)

            else:
                # Backfill missing earlier coverage if needed
                if min_ts > pd.Timestamp(desired_start):
                    # fetch desired_start .. (min_ts - 1ms)
                    backfill_end = _pdts_to_utc_dt(min_ts - pd.Timedelta(milliseconds=1))
                    backfill_start = desired_start
                    logger.info(
                        f"{symbol} {interval}: Backfill missing earlier range {backfill_start} -> {backfill_end} "
                        f"(existing_min={min_ts})"
                    )
                    df_back = fetch_klines(symbol=symbol, interval=interval, start=backfill_start, end=backfill_end)
                    if not df_back.empty:
                        new_parts.append(df_back)

                # Incremental update for newer range (max_ts -> now)
                inc_start = _pdts_to_utc_dt(max_ts + pd.Timedelta(milliseconds=1))
                inc_end = now_utc
                logger.info(
                    f"{symbol} {interval}: Incremental update {inc_start} -> {inc_end} "
                    f"(existing_max={max_ts})"
                )
                df_inc = fetch_klines(symbol=symbol, interval=interval, start=inc_start, end=inc_end)
                if not df_inc.empty:
                    new_parts.append(df_inc)

        # If nothing new fetched
        if not new_parts:
            logger.warning(f"No new data fetched for {symbol} {interval}")
            result["status"] = "no_new_data"
            result["rows_after"] = len(existing_df)
            result["new_rows"] = 0
            return result

        # Combine existing + new parts
        combined_df = pd.concat([existing_df] + new_parts, ignore_index=True)

        # Normalize columns + timestamp
        if "timestamp" not in combined_df.columns:
            raise ValueError("Combined OHLCV missing 'timestamp' column after fetch.")

        combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], utc=True, errors="coerce")
        combined_df = combined_df.dropna(subset=["timestamp"])
        combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        # Clean data (dedupe, missing candles fill, UTC)
        cleaned_df = clean_ohlcv(combined_df, symbol, interval)

        # Save parquet
        save_parquet(cleaned_df, str(output_path))

        result["rows_after"] = len(cleaned_df)
        result["new_rows"] = result["rows_after"] - result["rows_before"]
        result["status"] = "success"

        logger.info(
            f"✅ {symbol} {interval}: Before={result['rows_before']}, After={result['rows_after']}, Added={result['new_rows']}"
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"❌ {symbol} {interval}: {e}", exc_info=True)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental + Backfill OHLCV data update job")

    parser.add_argument(
        "--symbols",
        type=str,
        default="config/coins.json",
        help="Path to symbols JSON file (supports {'symbols':[...]} or [...])",
    )

    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["15m", "1h"],
        help="Timeframes to fetch (e.g. 15m 1h)",
    )

    parser.add_argument(
        "--lookback_days",
        type=int,
        default=365,
        help="Days of historical data to ensure (e.g. 365)",
    )

    parser.add_argument(
        "--force_backfill",
        type=int,
        default=0,
        help="If 1: fetch full lookback range even if parquet exists (slower but guarantees coverage)",
    )

    args = parser.parse_args()

    symbols = load_symbols(args.symbols)
    if not symbols:
        logger.error("No symbols loaded. Exiting.")
        return 1

    logger.info(f"Starting update job: {len(symbols)} symbols × {len(args.timeframes)} timeframes")
    logger.info(f"Output directory: {DATA_DIR}")
    logger.info(f"Lookback days: {args.lookback_days}, force_backfill={bool(args.force_backfill)}")

    results: List[Dict[str, Any]] = []
    for sym in symbols:
        for tf in args.timeframes:
            results.append(
                update_symbol_timeframe(
                    symbol=sym,
                    interval=tf,
                    lookback_days=args.lookback_days,
                    force_backfill=bool(args.force_backfill),
                )
            )

    # Summary
    print("\n" + "=" * 80)
    print("UPDATE JOB SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] in ["no_new_data", "pending"])

    total_rows_before = sum(r["rows_before"] for r in results)
    total_rows_after = sum(r["rows_after"] for r in results)
    total_new_rows = sum(r["new_rows"] for r in results)

    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Skipped: {skipped}/{len(results)}\n")
    print(f"Total rows before: {total_rows_before:,}")
    print(f"Total rows after:  {total_rows_after:,}")
    print(f"Total new rows added: {total_new_rows:,}")
    print("=" * 80)

    # Print failed
    failed_results = [r for r in results if r["status"] == "error"]
    if failed_results:
        print("\nFailed updates:")
        for r in failed_results:
            print(f"  {r['symbol']} {r['interval']}: {r['error']}")
        print("=" * 80)

    # Sample successful ranges
    success_results = [r for r in results if r["status"] == "success"]
    if success_results:
        print("\nSample of successful updates (first 5):")
        for r in success_results[:5]:
            p = DATA_DIR / r["symbol"] / f"{r['interval']}.parquet"
            df = _safe_read_parquet(p)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp"])
                print(
                    f"  {r['symbol']} {r['interval']}: rows={len(df):,} from {df['timestamp'].min()} to {df['timestamp'].max()}"
                )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
