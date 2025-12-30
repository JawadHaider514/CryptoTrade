#!/usr/bin/env python3
"""
Complete ML pipeline runner:
1. Train all coins
2. Run backtest
3. Calculate thresholds
4. Generate quality report
"""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_command(cmd, description):
    """Run a shell command and return success status"""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        logger.info(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - FAILED with code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Complete ML training + backtest + threshold pipeline")
    parser.add_argument("--symbols_file", default="data/symbols_32.json", help="Symbols file")
    parser.add_argument("--tf", default="15m", help="Timeframe")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per coin")
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--version", default="v1", help="Model version")
    parser.add_argument("--skip_training", action="store_true", help="Skip training (use existing models)")
    parser.add_argument("--skip_backtest", action="store_true", help="Skip backtest")
    parser.add_argument("--min_score", type=int, default=55, help="Min score threshold")
    
    args = parser.parse_args()
    
    pipeline_steps = []
    
    # Step 1: Train all coins
    if not args.skip_training:
        pipeline_steps.append((
            [
                "python", "-m", "crypto_bot.ml.per_coin.train_all",
                "--symbols_file", args.symbols_file,
                "--tf", args.tf,
                "--epochs", str(args.epochs),
                "--max_workers", str(args.max_workers)
            ],
            f"Train all coins ({args.tf}, {args.epochs} epochs)"
        ))
    
    # Step 2: Run backtest
    if not args.skip_backtest:
        pipeline_steps.append((
            [
                "python", "-m", "crypto_bot.backtesting.run_all",
                "--symbols_file", args.symbols_file,
                "--tf", args.tf,
                "--version", args.version
            ],
            f"Run walk-forward backtest ({args.tf})"
        ))
        
        # Step 3: Calculate thresholds
        pipeline_steps.append((
            [
                "python", "-m", "crypto_bot.backtesting.calculate_thresholds",
                "--version", args.version,
                "--tf", args.tf,
                "--min_score", str(args.min_score)
            ],
            "Calculate per-coin quality thresholds"
        ))
    
    # Execute pipeline
    logger.info(f"\n{'='*70}")
    logger.info(f"ML PRODUCTION PIPELINE")
    logger.info(f"Symbols: {args.symbols_file}")
    logger.info(f"Timeframe: {args.tf}")
    logger.info(f"Steps: {len(pipeline_steps)}")
    logger.info(f"{'='*70}")
    
    results = []
    for cmd, description in pipeline_steps:
        success = run_command(cmd, description)
        results.append((description, success))
        if not success:
            logger.error(f"\n❌ Pipeline stopped at: {description}")
            break
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*70}")
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {description}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        logger.info(f"\n{'='*70}")
        logger.info("✅ COMPLETE ML PIPELINE - ALL STEPS SUCCESSFUL")
        logger.info(f"{'='*70}")
        logger.info(f"Outputs:")
        logger.info(f"  Models: models/per_coin/<symbol>/<timeframe>/")
        logger.info(f"  Reports: reports/per_coin/")
        logger.info(f"  Summary: reports/summary_{args.version}.json")
        logger.info(f"  Thresholds: config/per_coin_thresholds.json")
        logger.info(f"{'='*70}")
        return 0
    else:
        logger.error("\n❌ PIPELINE FAILED - Check logs above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
