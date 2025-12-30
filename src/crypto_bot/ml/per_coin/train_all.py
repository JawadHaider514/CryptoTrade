#!/usr/bin/env python3
"""
Train all coins ML models in parallel - CLI wrapper
Usage: python -m crypto_bot.ml.per_coin.train_all --symbols_file data/symbols_32.json --tf 15m --resume 1
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def train_symbol(symbol: str, timeframe: str, epochs: int = 30, batch_size: int = 8, 
                  device: str = "cpu", lookback: int = 60) -> dict:
    """Train a single symbol"""
    try:
        from crypto_bot.ml.train.train_cnn_lstm import train_cnn_lstm
        
        logger.info(f"[{symbol}] Starting training on {device}...")
        
        result = train_cnn_lstm(
            symbol=symbol,
            timeframe=timeframe,
            epochs=epochs,
            lookback=lookback,
            batch_size=batch_size,
            learning_rate=0.001,
            device=device
        )
        
        if result and result.get('status') == 'success':
            metrics = result.get('metrics', {})
            logger.info(f"✅ [{symbol}] Training completed - Accuracy: {metrics.get('accuracy', 0):.4f}")
            return {
                "symbol": symbol,
                "status": "success",
                "accuracy": metrics.get('accuracy', 0),
                "f1": metrics.get('f1', 0),
                "model_path": str(result.get("model_path", ""))
            }
        else:
            error = result.get('error', 'Unknown error') if result else 'No result returned'
            logger.error(f"❌ [{symbol}] Training failed: {error}")
            return {
                "symbol": symbol,
                "status": "failed",
                "error": error
            }
    
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "status": "failed",
            "error": str(e)
        }


def main():
    """Train all coins in parallel"""
    parser = argparse.ArgumentParser(
        description="Train ML models for all crypto coins"
    )
    parser.add_argument(
        "--symbols_file",
        type=str,
        default="data/symbols_32.json",
        help="Path to symbols JSON file"
    )
    parser.add_argument("--tf", type=str, default="15m", help="Timeframe (1m/5m/15m/1h)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per symbol")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default 8 for GPU memory)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                       help="Device: cpu or cuda")
    parser.add_argument("--resume", type=int, default=0, help="Resume from existing (1=yes)")
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel workers (1 for CUDA)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Auto-adjust max_workers based on device
    if args.device == "cuda":
        args.max_workers = 1  # Sequential training on GPU
        logger.info("CUDA detected: forcing sequential training (max_workers=1)")
    
    try:
        # Load symbols from file
        symbols_file = Path(args.symbols_file)
        if not symbols_file.is_absolute():
            symbols_file = (Path.cwd() / symbols_file).resolve()

        logger.info(f"Symbols file resolved to: {symbols_file}")
        logger.info(f"Symbols file size: {symbols_file.stat().st_size if symbols_file.exists() else 'MISSING'} bytes")

        if not symbols_file.exists():
            logger.error(f"❌ Symbols file not found: {symbols_file}")
            sys.exit(1)
        
        with open(symbols_file) as f:
            symbols_data = json.load(f)
        
        # Handle both formats: {"symbols": [...]} or [...]
        if isinstance(symbols_data, dict):
            symbols = symbols_data.get("symbols", []) if isinstance(symbols_data.get("symbols"), list) else list(symbols_data.keys())
        else:
            symbols = symbols_data if isinstance(symbols_data, list) else []
        
        if not symbols:
            logger.error(f"❌ No symbols found in {symbols_file}")
            sys.exit(1)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 TRAINING ALL COINS - BATCH MODE")
        logger.info(f"   Symbols: {len(symbols)}")
        logger.info(f"   Timeframe: {args.tf}")
        logger.info(f"   Epochs per coin: {args.epochs}")
        logger.info(f"   Batch size: {args.batch_size}")
        logger.info(f"   Device: {args.device}")
        logger.info(f"   Parallel workers: {args.max_workers}")
        logger.info(f"   Resume: {'Yes' if args.resume else 'No'}")
        logger.info(f"   Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        logger.info(f"{'='*70}\n")
        
        # Train symbols in parallel
        results = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(train_symbol, sym, args.tf, args.epochs, args.batch_size, args.device, 60): sym
                for sym in symbols
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Future error: {e}")
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 TRAINING SUMMARY")
        logger.info(f"{'='*70}")
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        
        logger.info(f"✅ Successful: {len(successful)}/{len(symbols)}")
        logger.info(f"❌ Failed: {len(failed)}/{len(symbols)}")
        
        if successful:
            logger.info(f"\n✅ Trained symbols:")
            for r in successful:
                acc = r.get("accuracy", 0)
                logger.info(f"   • {r['symbol']}: {acc:.2%} accuracy")
        
        if failed:
            logger.info(f"\n❌ Failed symbols:")
            for r in failed:
                logger.info(f"   • {r['symbol']}: {r.get('error', 'Unknown error')}")
        
        logger.info(f"\n{'='*70}")
        
        if len(successful) > 0:
            logger.info(f"✅ Batch training completed: {len(successful)} coins trained")
        else:
            logger.error(f"❌ No coins trained successfully")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
