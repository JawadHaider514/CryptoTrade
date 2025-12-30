#!/usr/bin/env python3
"""
Rebuild all datasets with fresh OHLCV data (365 days).
Pipeline: Fetch OHLCV → Build Features → Create Labels → Save Parquet

Uses existing dataset_builder.py and labels.py modules.
No duplication - cleans old data first.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from your existing modules
from crypto_bot.features.dataset_builder import build_dataset
from crypto_bot.features.labels import create_labels

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
OHLCV_DIR = DATA_DIR / "ohlcv"


def cleanup_old_data(symbol: str, timeframe: str):
    """Remove old dataset files for a symbol/timeframe"""
    try:
        symbol_dir = DATASETS_DIR / symbol
        
        # Files to remove
        dataset_file = symbol_dir / f"{timeframe}_dataset.parquet"
        meta_file = symbol_dir / f"{timeframe}_meta.json"

        
        removed = False
        
        if dataset_file.exists():
            dataset_file.unlink()
            logger.info(f"[{symbol}] Removed old {dataset_file.name}")
            removed = True
        
        if meta_file.exists() and not any(
            (DATASETS_DIR / s / "meta.json").exists() 
            for s in get_symbols(None) 
            if s != symbol
        ):
            # Only remove meta.json if no other timeframes exist
            meta_file.unlink()
            logger.info(f"[{symbol}] Removed old meta.json")
            removed = True
        
        return removed
    
    except Exception as e:
        logger.warning(f"[{symbol}] Could not cleanup: {e}")
        return False


def get_symbols(symbols_file: Optional[str] = None) -> list:
    """Load symbols from JSON file"""
    if symbols_file is None:
        symbols_file = str(DATA_DIR / "symbols_32.json")
    
    try:
        with open(symbols_file) as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "symbols" in data:
            return data["symbols"]
        else:
            return list(data.keys()) if isinstance(data, dict) else []
    
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return []


def rebuild_single_dataset(
    symbol: str,
    timeframe: str,
    lookback: int = 60,
    horizon: int = 3
) -> dict:
    """
    Rebuild dataset for single symbol/timeframe.
    Uses build_dataset() from dataset_builder.py
    """
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "status": "pending",
        "rows": 0,
        "features": 0,
        "error": None
    }
    
    try:
        logger.info(f"\n[{symbol}_{timeframe}] Starting rebuild...")
        
        # Step 1: Clean old data
        cleanup_old_data(symbol, timeframe)
        
        # Step 2: Use build_dataset from dataset_builder.py
        logger.info(f"[{symbol}_{timeframe}] Building dataset...")
        build_result = build_dataset(
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback,
            horizon=horizon
        )
        
        if build_result['status'] == 'success':
            result['status'] = 'SUCCESS'
            result['rows'] = build_result['dataset_rows']
            result['features'] = len(build_result['features_list'])
            result['classes'] = build_result['class_distribution']
            logger.info(
                f"[{symbol}_{timeframe}] ✅ Complete: "
                f"{result['rows']:,} rows | "
                f"LONG={result['classes']['LONG']} | "
                f"SHORT={result['classes']['SHORT']} | "
                f"NO_TRADE={result['classes']['NO_TRADE']}"
            )
        else:
            result['status'] = 'FAILED'
            result['error'] = build_result.get('error', 'Unknown error')
            logger.error(f"[{symbol}_{timeframe}] ❌ {result['error']}")
    
    except Exception as e:
        result['status'] = 'ERROR'
        result['error'] = str(e)
        logger.exception(f"[{symbol}_{timeframe}] Exception: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild datasets with dataset_builder + labels integration"
    )
    parser.add_argument(
        "--symbols_file",
        default="data/symbols_32.json",
        help="Path to symbols JSON file"
    )
    parser.add_argument(
        "--timeframes",
        default="15m",
        help="Comma-separated timeframes (15m,1h,4h)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback period for features (default 60)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Horizon for label generation (default 3)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Max parallel workers (default 4)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean all old datasets before rebuild"
    )
    
    args = parser.parse_args()
    
    # Load symbols
    symbols = get_symbols(args.symbols_file)
    if not symbols:
        logger.error("❌ No symbols loaded")
        return 1
    
    timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"DATASET REBUILD - dataset_builder + labels integration")
    logger.info(f"{'='*80}")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Workers: {args.max_workers}")
    logger.info(f"Lookback: {args.lookback}")
    logger.info(f"Horizon: {args.horizon}")
    logger.info(f"{'='*80}\n")
    
    # Optional: Clean all old data first
    if args.cleanup:
        logger.info("🧹 Cleaning all old datasets...")
        for symbol in symbols:
            for tf in timeframes:
                cleanup_old_data(symbol, tf)
        logger.info("✅ Cleanup complete\n")
    
    # Build task list
    tasks = [
        (symbol, timeframe)
        for symbol in symbols
        for timeframe in timeframes
    ]
    
    results = {}
    summary = {
        "total_tasks": len(tasks),
        "success": 0,
        "failed": 0,
        "error": 0,
        "total_rows": 0,
        "by_status": {"SUCCESS": [], "FAILED": [], "ERROR": []}
    }
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                rebuild_single_dataset,
                symbol,
                timeframe,
                lookback=args.lookback,
                horizon=args.horizon
            ): (symbol, timeframe)
            for symbol, timeframe in tasks
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            symbol, timeframe = futures[future]
            
            try:
                result = future.result()
                results[f"{symbol}_{timeframe}"] = result
                
                status = result["status"]
                summary["by_status"][status].append(f"{symbol}_{timeframe}")
                
                if status == "SUCCESS":
                    summary["success"] += 1
                    summary["total_rows"] += result.get("rows", 0)
                elif status == "FAILED":
                    summary["failed"] += 1
                else:
                    summary["error"] += 1
                
                logger.info(
                    f"[{completed:2d}/{len(tasks):2d}] {symbol:10s} {timeframe:4s} | {status}"
                )
            
            except Exception as e:
                logger.error(f"[{completed}/{len(tasks)}] {symbol} {timeframe}: {e}")
                results[f"{symbol}_{timeframe}"] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "ERROR",
                    "error": str(e)
                }
                summary["error"] += 1
                summary["by_status"]["ERROR"].append(f"{symbol}_{timeframe}")
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"REBUILD COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total tasks: {summary['total_tasks']}")
    logger.info(f"✅ Success: {summary['success']}")
    logger.info(f"❌ Failed: {summary['failed']}")
    logger.info(f"⚠️  Error: {summary['error']}")
    logger.info(f"📊 Total rows: {summary['total_rows']:,}")
    logger.info(f"{'='*80}")
    
    if summary['success'] > 0:
        logger.info(f"\n✅ Successful ({len(set(summary['by_status']['SUCCESS']))}):")
        for item in sorted(set(summary['by_status']['SUCCESS']))[:10]:
            logger.info(f"   - {item}")
        if len(set(summary['by_status']['SUCCESS'])) > 10:
            logger.info(f"   ... and {len(set(summary['by_status']['SUCCESS']))-10} more")
    
    if summary['failed'] > 0:
        logger.info(f"\n❌ Failed ({len(set(summary['by_status']['FAILED']))}):")
        for item in sorted(set(summary['by_status']['FAILED']))[:10]:
            logger.info(f"   - {item}")
    
    if summary['error'] > 0:
        logger.info(f"\n⚠️  Errors ({len(set(summary['by_status']['ERROR']))}):")
        for item in sorted(set(summary['by_status']['ERROR']))[:10]:
            logger.info(f"   - {item}")
    
    logger.info(f"\n📂 Output: {DATASETS_DIR}")
    logger.info(f"{'='*80}\n")
    
    # Return status
    return 0 if summary['failed'] == 0 and summary['error'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())