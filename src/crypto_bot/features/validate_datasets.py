"""
Dataset validation and statistics printer.

Confirms all datasets exist and prints per-coin statistics.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"


def validate_datasets(
    symbols: Optional[List[str]] = None,
    timeframe: str = '15m',
) -> Dict[str, dict]:
    """
    Validate that datasets exist and print statistics.
    
    Args:
        symbols: List of symbols to check. If None, checks all in DATASET_DIR
        timeframe: Timeframe to check (default '15m')
    
    Returns:
        Dict with per-symbol stats
    """
    # Load symbols if not provided
    if symbols is None:
        # Get all symbol directories
        symbols = [d.name for d in DATASET_DIR.iterdir() if d.is_dir()]
        symbols = sorted(symbols)
    
    results = {}
    
    print("\n" + "="*100)
    print(f"DATASET VALIDATION - Timeframe: {timeframe}")
    print("="*100)
    print(f"{'Symbol':<12} {'Status':<10} {'Samples':<12} {'LONG':<8} {'SHORT':<8} {'NO_TRADE':<12} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-"*100)
    
    for symbol in symbols:
        symbol = symbol.upper()
        dataset_path = DATASET_DIR / symbol / f"{timeframe}_dataset.parquet"
        meta_path = DATASET_DIR / symbol / "meta.json"
        
        result = {
            'symbol': symbol,
            'exists': False,
            'status': 'missing',
            'samples': 0,
            'class_dist': {},
            'splits': {},
        }
        
        # Check if dataset exists
        if not dataset_path.exists():
            print(f"{symbol:<12} {'MISSING':<10} {'N/A':<12} {'N/A':<8} {'N/A':<8} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            results[symbol] = result
            continue
        
        # Check if meta exists
        if not meta_path.exists():
            print(f"{symbol:<12} {'NO_META':<10} {'MISSING':<12} {'FILE':<8} {'':<8} {'':<12} {'':<10} {'':<10} {'':<10}")
            results[symbol] = result
            continue
        
        try:
            # Load dataset
            df = pd.read_parquet(dataset_path)
            result['samples'] = len(df)
            
            # Load metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Get class distribution
            class_dist = meta.get('class_distribution', {})
            result['class_dist'] = class_dist
            
            # Get split info
            time_split = meta.get('time_split', {})
            train_end_idx = time_split.get('train_end_idx', 0)
            val_end_idx = time_split.get('val_end_idx', 0)
            test_end_idx = time_split.get('test_end_idx', len(df))
            
            train_size = train_end_idx
            val_size = val_end_idx - train_end_idx
            test_size = test_end_idx - val_end_idx
            
            result['splits'] = {
                'train': train_size,
                'val': val_size,
                'test': test_size,
            }
            result['status'] = 'valid'
            result['exists'] = True
            
            # Print stats
            long_pct = class_dist.get('LONG', 0) / len(df) * 100 if len(df) > 0 else 0
            short_pct = class_dist.get('SHORT', 0) / len(df) * 100 if len(df) > 0 else 0
            no_trade_pct = class_dist.get('NO_TRADE', 0) / len(df) * 100 if len(df) > 0 else 0
            
            print(
                f"{symbol:<12} {'OK':<10} {len(df):<12,} "
                f"{class_dist.get('LONG', 0):<8} {class_dist.get('SHORT', 0):<8} "
                f"{class_dist.get('NO_TRADE', 0):<12} "
                f"{train_size:<10} {val_size:<10} {test_size:<10}"
            )
            
        except Exception as e:
            print(f"{symbol:<12} {'ERROR':<10} {str(e)[:40]:<12} {'':<8} {'':<8} {'':<12} {'':<10} {'':<10} {'':<10}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        results[symbol] = result
    
    print("="*100)
    
    # Print summary
    valid_count = sum(1 for r in results.values() if r['exists'])
    missing_count = len(results) - valid_count
    
    print(f"\nSummary:")
    print(f"  Valid datasets: {valid_count}/{len(results)}")
    print(f"  Missing: {missing_count}/{len(results)}")
    
    if valid_count > 0:
        total_samples = sum(r['samples'] for r in results.values() if r['exists'])
        print(f"  Total samples (all coins): {total_samples:,}")
        
        total_long = sum(r['class_dist'].get('LONG', 0) for r in results.values() if r['exists'])
        total_short = sum(r['class_dist'].get('SHORT', 0) for r in results.values() if r['exists'])
        total_no_trade = sum(r['class_dist'].get('NO_TRADE', 0) for r in results.values() if r['exists'])
        
        print(f"\n  Class distribution (all coins):")
        print(f"    LONG:     {total_long:,} ({total_long/total_samples*100:5.1f}%)")
        print(f"    SHORT:    {total_short:,} ({total_short/total_samples*100:5.1f}%)")
        print(f"    NO_TRADE: {total_no_trade:,} ({total_no_trade/total_samples*100:5.1f}%)")
        
        # Train/val/test splits
        total_train = sum(r['splits'].get('train', 0) for r in results.values() if r['exists'])
        total_val = sum(r['splits'].get('val', 0) for r in results.values() if r['exists'])
        total_test = sum(r['splits'].get('test', 0) for r in results.values() if r['exists'])
        
        print(f"\n  Time-based split (all coins):")
        print(f"    Train: {total_train:,} ({total_train/total_samples*100:5.1f}%)")
        print(f"    Val:   {total_val:,} ({total_val/total_samples*100:5.1f}%)")
        print(f"    Test:  {total_test:,} ({total_test/total_samples*100:5.1f}%)")
    
    print()
    
    return results


if __name__ == '__main__':
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get optional timeframe argument
    timeframe = '15m'
    if len(sys.argv) > 1:
        timeframe = sys.argv[1]
    
    # Validate
    results = validate_datasets(timeframe=timeframe)
    
    # Exit with error if any datasets missing
    valid_count = sum(1 for r in results.values() if r['exists'])
    if valid_count == 0:
        print("❌ No valid datasets found!")
        sys.exit(1)
    
    sys.exit(0)
