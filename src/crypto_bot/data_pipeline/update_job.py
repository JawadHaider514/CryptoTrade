"""
Incremental OHLCV data update job.

CLI Usage:
    python -m crypto_bot.data_pipeline.update_job --symbols config/coins.json --timeframes 15m 1h --lookback_days 365

Behavior:
    - If parquet exists: fetch from (last_timestamp) to now
    - If parquet missing: fetch full lookback range
    - Clean and save
    - Print summary
"""

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from crypto_bot.data_pipeline.binance_ohlcv import fetch_klines
from crypto_bot.data_pipeline.storage import save_parquet, load_parquet
from crypto_bot.data_pipeline.cleaning import clean_ohlcv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parents[4]  # .../crypto_trading_system
DATA_DIR = PROJECT_ROOT / "data" / "ohlcv"


def load_symbols(symbols_file: str) -> List[str]:
    """Load symbols from JSON file."""
    path = Path(symbols_file)
    if not path.exists():
        logger.error(f"Symbols file not found: {symbols_file}")
        return []
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    symbols = data.get('symbols', [])
    logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
    
    return symbols


def update_symbol_timeframe(
    symbol: str,
    interval: str,
    lookback_days: int,
) -> Dict[str, Any]:
    """
    Update data for a single symbol/timeframe combination.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Timeframe (e.g., '15m', '1h')
        lookback_days: How many days back to fetch if file missing
    
    Returns:
        Dict with update results
    """
    symbol = symbol.upper()
    output_dir = DATA_DIR / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{interval}.parquet"
    
    result = {
        'symbol': symbol,
        'interval': interval,
        'status': 'pending',
        'rows_before': 0,
        'rows_after': 0,
        'new_rows': 0,
        'error': None,
    }
    
    try:
        # Load existing data
        existing_df = load_parquet(str(output_path))
        result['rows_before'] = len(existing_df)
        
        # Determine fetch range
        if len(existing_df) > 0:
            # Incremental update: fetch from last timestamp to now
            last_timestamp = existing_df['timestamp'].max()
            start = last_timestamp
            end = datetime.utcnow()
            logger.info(f"{symbol} {interval}: Incremental update from {start} to {end}")
        else:
            # Full backfill: fetch lookback range
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=lookback_days)
            logger.info(f"{symbol} {interval}: Full backfill from {start} to {end}")
        
        # Fetch new data
        new_df = fetch_klines(symbol=symbol, interval=interval, start=start, end=end)
        
        if len(new_df) == 0:
            logger.warning(f"No new data fetched for {symbol} {interval}")
            result['status'] = 'no_new_data'
            return result
        
        # Combine with existing
        if len(existing_df) > 0:
            # Remove any overlapping candles
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        else:
            combined_df = new_df
        
        # Clean data
        cleaned_df = clean_ohlcv(combined_df, symbol, interval)
        
        # Save
        save_parquet(cleaned_df, str(output_path))
        
        result['rows_after'] = len(cleaned_df)
        result['new_rows'] = result['rows_after'] - result['rows_before']
        result['status'] = 'success'
        
        logger.info(
            f"✅ {symbol} {interval}: "
            f"Before={result['rows_before']}, After={result['rows_after']}, "
            f"Added={result['new_rows']}"
        )
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"❌ {symbol} {interval}: {e}", exc_info=True)
    
    return result


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Incremental OHLCV data update job'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default='config/coins.json',
        help='Path to symbols JSON file'
    )
    parser.add_argument(
        '--timeframes',
        type=str,
        nargs='+',
        default=['15m', '1h'],
        help='Timeframes to fetch'
    )
    parser.add_argument(
        '--lookback_days',
        type=int,
        default=365,
        help='Days of historical data to fetch on first run'
    )
    
    args = parser.parse_args()
    
    # Load symbols
    symbols = load_symbols(args.symbols)
    if not symbols:
        logger.error("No symbols loaded. Exiting.")
        return 1
    
    logger.info(f"Starting update job: {len(symbols)} symbols × {len(args.timeframes)} timeframes")
    logger.info(f"Output directory: {DATA_DIR}")
    
    # Process each symbol/timeframe
    results = []
    for symbol in symbols:
        for interval in args.timeframes:
            result = update_symbol_timeframe(
                symbol=symbol,
                interval=interval,
                lookback_days=args.lookback_days,
            )
            results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("UPDATE JOB SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    skipped = sum(1 for r in results if r['status'] in ['no_new_data', 'pending'])
    
    total_rows_before = sum(r['rows_before'] for r in results)
    total_rows_after = sum(r['rows_after'] for r in results)
    total_new_rows = sum(r['new_rows'] for r in results)
    
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Skipped: {skipped}/{len(results)}")
    print()
    print(f"Total rows before: {total_rows_before:,}")
    print(f"Total rows after: {total_rows_after:,}")
    print(f"Total new rows added: {total_new_rows:,}")
    print()
    
    # Print failed results
    failed_results = [r for r in results if r['status'] == 'error']
    if failed_results:
        print("Failed updates:")
        for r in failed_results:
            print(f"  {r['symbol']} {r['interval']}: {r['error']}")
    
    print("="*80)
    
    # Sanity check: print sample of successful updates
    success_results = [r for r in results if r['status'] == 'success']
    if success_results:
        print("\nSample of successful updates (first 5):")
        for r in success_results[:5]:
            df = load_parquet(str(DATA_DIR / r['symbol'] / f"{r['interval']}.parquet"))
            if len(df) > 0:
                start = df['timestamp'].min()
                end = df['timestamp'].max()
                print(f"  {r['symbol']} {r['interval']}: rows={len(df):,} from {start} to {end}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
