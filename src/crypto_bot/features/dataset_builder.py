"""
Dataset builder for machine learning.

CLI Usage:
    python -m crypto_bot.features.dataset_builder --symbol BTCUSDT --timeframe 15m --lookback 60 --horizon 3

Creates parquet dataset and metadata file.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from crypto_bot.data_pipeline.storage import load_parquet
from crypto_bot.features.feature_engineering import build_features
from crypto_bot.features.labels import create_labels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "ohlcv"
DATASET_DIR = PROJECT_ROOT / "data" / "datasets"

# Label threshold (0.25% = 0.0025)
LABEL_THRESHOLD = 0.0025


def build_dataset(
    symbol: str,
    timeframe: str,
    lookback: int = 60,
    horizon: int = 3,
) -> dict:
    """
    Build a complete ML dataset for a symbol/timeframe.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Candle interval (e.g., '15m', '1h')
        lookback: Minimum lookback days for indicators (default 60)
        horizon: Candles into future for labels (default 3)
    
    Returns:
        Dict with dataset info and statistics
    """
    symbol = symbol.upper()
    
    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'status': 'pending',
        'total_rows': 0,
        'feature_rows': 0,
        'dataset_rows': 0,
        'features_list': [],
        'class_distribution': {},
        'error': None,
    }
    
    try:
        # Load raw OHLCV data
        ohlcv_path = DATA_DIR / symbol / f"{timeframe}.parquet"
        if not ohlcv_path.exists():
            raise FileNotFoundError(f"OHLCV file not found: {ohlcv_path}")
        
        logger.info(f"Loading OHLCV data from {ohlcv_path}")
        df = load_parquet(str(ohlcv_path))
        result['total_rows'] = len(df)
        
        if len(df) == 0:
            raise ValueError(f"Empty OHLCV data for {symbol} {timeframe}")
        
        logger.info(f"Loaded {len(df):,} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Build features
        logger.info("Building features...")
        df = build_features(df)
        
        # Drop NaN rows from feature calculation
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        logger.info(f"Dropped {dropped} rows with NaN features (indicators need warmup)")
        result['feature_rows'] = len(df)
        
        # Create labels
        logger.info(f"Creating labels (horizon={horizon}, threshold={LABEL_THRESHOLD*100:.2f}%)")
        df, label_config = create_labels(df, horizon=horizon, threshold=LABEL_THRESHOLD)
        
        # Drop rows with NaN labels (last horizon rows don't have future data)
        initial_len = len(df)
        df = df.dropna(subset=['label'])
        dropped = initial_len - len(df)
        logger.info(f"Dropped {dropped} rows without future labels (last {horizon} rows)")
        result['dataset_rows'] = len(df)
        
        # Ensure label is int
        df['label'] = df['label'].astype(int)
        
        # Time-based split info (no shuffle - maintained temporal order)
        train_end_idx = int(len(df) * 0.70)
        val_end_idx = int(len(df) * 0.85)
        test_end_idx = len(df)
        
        time_split = {
            'strategy': 'time-based (no shuffle)',
            'train_ratio': 0.70,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'train_end_idx': train_end_idx,
            'val_end_idx': val_end_idx,
            'test_end_idx': test_end_idx,
            'train_samples': train_end_idx,
            'val_samples': val_end_idx - train_end_idx,
            'test_samples': test_end_idx - val_end_idx,
        }
        
        # Get feature columns (excluding OHLCV and label)
        feature_cols = [c for c in df.columns if c not in [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'label'
        ]]
        result['features_list'] = sorted(feature_cols)
        
        # Class distribution
        class_counts = df['label'].value_counts().to_dict()
        result['class_distribution'] = {
            'LONG': int(class_counts.get(1, 0)),
            'SHORT': int(class_counts.get(-1, 0)),
            'NO_TRADE': int(class_counts.get(0, 0)),
        }
        
        # Save dataset
        output_dir = DATASET_DIR / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_path = output_dir / f"{timeframe}_dataset.parquet"
        logger.info(f"Saving dataset to {dataset_path}")
        df.to_parquet(dataset_path, index=False, compression='snappy')
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'total_samples': result['dataset_rows'],
            'features': result['features_list'],
            'num_features': len(result['features_list']),
            'label_config': {
                'horizon': horizon,
                'threshold': LABEL_THRESHOLD,
                'threshold_pct': f"{LABEL_THRESHOLD * 100:.2f}%",
            },
            'class_distribution': result['class_distribution'],
            'time_split': time_split,
            'data_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
            },
            'ohlcv_rows': result['total_rows'],
            'rows_after_feature_engineering': result['feature_rows'],
            'rows_after_labeling': result['dataset_rows'],
        }
        
        meta_path = output_dir / "meta.json"
        logger.info(f"Saving metadata to {meta_path}")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        result['status'] = 'success'
        result['metadata'] = metadata
        
        logger.info(
            f"✅ {symbol} {timeframe}: "
            f"{result['dataset_rows']:,} samples | "
            f"LONG={result['class_distribution']['LONG']} | "
            f"SHORT={result['class_distribution']['SHORT']} | "
            f"NO_TRADE={result['class_distribution']['NO_TRADE']}"
        )
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"❌ {symbol} {timeframe}: {e}", exc_info=True)
    
    return result


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Build ML dataset from OHLCV data with features and labels'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading symbol (e.g., BTCUSDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        required=True,
        help='Candle interval (e.g., 15m, 1h)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=60,
        help='Minimum lookback days (for indicator warmup)'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=3,
        help='Candles into future for labels (default 3)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Building dataset for {args.symbol} {args.timeframe}")
    logger.info(f"Output directory: {DATASET_DIR}")
    
    # Build dataset
    result = build_dataset(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback=args.lookback,
        horizon=args.horizon,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("DATASET BUILDER SUMMARY")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Status: {result['status'].upper()}")
    print()
    
    if result['status'] == 'success':
        print(f"Total OHLCV rows: {result['total_rows']:,}")
        print(f"After features: {result['feature_rows']:,}")
        print(f"Final dataset: {result['dataset_rows']:,}")
        print()
        print("Class Distribution:")
        dist = result['class_distribution']
        total = sum(dist.values())
        print(f"  LONG:     {dist['LONG']:,} ({dist['LONG']/total*100:5.1f}%)")
        print(f"  SHORT:    {dist['SHORT']:,} ({dist['SHORT']/total*100:5.1f}%)")
        print(f"  NO_TRADE: {dist['NO_TRADE']:,} ({dist['NO_TRADE']/total*100:5.1f}%)")
        print()
        print(f"Features ({len(result['features_list'])}):")
        for feat in result['features_list']:
            print(f"  - {feat}")
        print()
        symbol_dir = DATASET_DIR / args.symbol
        print(f"Files created:")
        print(f"  - {symbol_dir / f'{args.timeframe}_dataset.parquet'}")
        print(f"  - {symbol_dir / 'meta.json'}")
    else:
        print(f"Error: {result['error']}")
    
    print("="*80)
    
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
