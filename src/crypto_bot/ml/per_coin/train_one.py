#!/usr/bin/env python3
"""
Train single coin ML model - CLI wrapper
Usage: python -m crypto_bot.ml.per_coin.train_one --symbol BTCUSDT --tf 15m
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train a single coin model"""
    parser = argparse.ArgumentParser(
        description="Train ML model for a single crypto coin"
    )
    parser.add_argument("--symbol", type=str, required=True, help="Symbol (e.g., BTCUSDT)")
    parser.add_argument("--tf", type=str, default="15m", help="Timeframe (1m/5m/15m/1h)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    try:
        from crypto_bot.ml.train.train_cnn_lstm import train_cnn_lstm
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 TRAINING SINGLE COIN MODEL")
        logger.info(f"   Symbol: {args.symbol}")
        logger.info(f"   Timeframe: {args.tf}")
        logger.info(f"   Epochs: {args.epochs}")
        logger.info(f"   Batch size: {args.batch_size}")
        logger.info(f"{'='*70}\n")
        
        # Call training function
        result = train_cnn_lstm(
            symbol=args.symbol,
            timeframe=args.tf,
            epochs=args.epochs,
            lookback=args.lookback,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        
        if result and result.get('status') == 'success':
            logger.info(f"\n✅ Training completed successfully")
            logger.info(f"   Model saved to: {result.get('model_path')}")
            metrics = result.get('metrics', {})
            logger.info(f"   Test accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"   Test F1-score: {metrics.get('f1', 0):.4f}")
            logger.info(f"   Model path: {result.get('model_path')}")
        else:
            logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
