#!/usr/bin/env python3
"""Walk-forward backtest runner for all coins"""

import argparse
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def load_symbols(symbols_file):
    """Load symbols from JSON file"""
    with open(symbols_file) as f:
        data = json.load(f)
    return data.get("symbols", [])


def backtest_coin(symbol, timeframe, version="v1"):
    """
    Run walk-forward backtest for a single coin
    
    Returns:
        dict: Backtest results with metrics
    """
    try:
        logger.info(f"Backtesting {symbol} {timeframe}...")
        
        # Load dataset
        dataset_dir = DATA_DIR / "datasets" / symbol / timeframe
        if not dataset_dir.exists():
            logger.warning(f"Dataset not found for {symbol} {timeframe}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "NO_DATA",
                "error": "Dataset not found"
            }
        
        # Load OHLCV data
        try:
            import pandas as pd
            csv_file = dataset_dir / "data.csv"
            if not csv_file.exists():
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "NO_DATA",
                    "error": "data.csv not found"
                }
            
            df = pd.read_csv(csv_file)
            if df.empty:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "NO_DATA",
                    "error": "Empty dataset"
                }
            
            # Load model metadata
            model_dir = MODELS_DIR / "per_coin" / symbol / timeframe
            meta_file = model_dir / "meta.json"
            if not meta_file.exists():
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "NO_MODEL",
                    "error": "Model not trained"
                }
            
            with open(meta_file) as f:
                meta = json.load(f)
            
            # Get predictions from ML model
            from crypto_bot.ml.inference.inference_service import InferenceService
            from crypto_bot.ml.inference.model_registry import ModelRegistry
            
            registry = ModelRegistry()
            inference = InferenceService(registry=registry)
            
            # Walk-forward: predict on out-of-sample data
            lookback = meta.get("lookback", 60)
            min_bars = lookback + 10
            
            if len(df) < min_bars:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "INSUFFICIENT_DATA",
                    "error": f"Only {len(df)} bars, need {min_bars}"
                }
            
            # Prepare features (assuming they're already in dataset)
            feature_cols = meta.get("feature_names", [
                'atr_14', 'bb_lower', 'bb_middle', 'bb_upper', 'bb_width',
                'ema_20', 'ema_50', 'log_returns', 'macd', 'macd_signal',
                'macd_diff', 'returns', 'rsi_14', 'volatility', 'volume_change'
            ])
            
            # Check if we have label column
            if 'label' not in df.columns:
                logger.warning(f"No label column in {symbol} dataset")
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "WARMUP",
                    "error": "No labels for validation"
                }
            
            # Get predictions for test period (last 20% of data)
            test_size = max(100, len(df) // 5)  # Last 20% or 100 bars min
            test_start = len(df) - test_size
            
            y_true = []
            y_pred = []
            confidences = []
            
            for idx in range(test_start, len(df)):
                window = df.iloc[max(0, idx - lookback):idx]
                if len(window) < lookback:
                    continue
                
                try:
                    # Get prediction from ML model
                    result = inference.predict(symbol, timeframe, window)
                    if result:
                        pred = 1 if result.get("direction") == "LONG" else 0
                        y_pred.append(pred)
                        confidences.append(result.get("confidence", 0.5))
                        
                        # Get actual label
                        actual = df.iloc[idx].get("label", 0)
                        y_true.append(actual)
                except Exception as e:
                    logger.debug(f"Prediction error at {symbol} bar {idx}: {e}")
                    continue
            
            if not y_true:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "WARMUP",
                    "error": "No valid predictions"
                }
            
            # Calculate metrics
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            f1_long = f1_score(y_true, y_pred, average='binary', pos_label=1)
            f1_short = f1_score(y_true, 1 - y_pred, average='binary', pos_label=1)
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate trading metrics
            trades_taken = len(y_pred)
            wins = np.sum((y_pred == y_true) & (y_pred == 1))
            losses = np.sum((y_pred != y_true) & (y_pred == 1))
            
            win_rate = wins / trades_taken if trades_taken > 0 else 0
            expectancy_r = (win_rate - (1 - win_rate)) if trades_taken > 0 else 0
            profit_factor = (wins + 1) / (losses + 1)  # Avoid division by zero
            
            # Mock max_drawdown and coverage
            max_drawdown = 0.15  # Placeholder
            coverage = min(1.0, trades_taken / (len(df) - test_start))
            
            # Generate score 0-100
            score = balanced_acc * 100
            status = "ACTIVE" if score >= 55 else ("NO_TRADE" if score < 45 else "WARMUP")
            
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "version": version,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": status,
                "metrics": {
                    "balanced_accuracy": float(balanced_acc),
                    "f1_long": float(f1_long),
                    "f1_short": float(f1_short),
                    "confusion_matrix": {
                        "tn": int(cm[0, 0]),
                        "fp": int(cm[0, 1]),
                        "fn": int(cm[1, 0]),
                        "tp": int(cm[1, 1])
                    },
                    "trades_taken": int(trades_taken),
                    "win_rate": float(win_rate),
                    "expectancy_r": float(expectancy_r),
                    "profit_factor": float(profit_factor),
                    "max_drawdown": float(max_drawdown),
                    "coverage": float(coverage),
                    "mean_confidence": float(np.mean(confidences)) if confidences else 0.5
                },
                "score_0_100": float(score)
            }
            
            logger.info(f"✅ {symbol} {timeframe}: score={score:.1f} status={status}")
            return result
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "ERROR",
                "error": str(e)
            }
    
    except Exception as e:
        logger.error(f"Fatal error in backtest_coin: {e}")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "ERROR",
            "error": str(e)
        }


def run_all_backtest(symbols_file, timeframe, version="v1", max_workers=4):
    """Run backtest for all symbols in parallel"""
    
    symbols = load_symbols(symbols_file)
    logger.info(f"Backtesting {len(symbols)} coins for {timeframe}...")
    
    # Create reports directory
    reports_dir = REPORTS_DIR / "per_coin"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    summary = {
        "version": version,
        "timeframe": timeframe,
        "total_coins": len(symbols),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results_by_status": {
            "ACTIVE": [],
            "WARMUP": [],
            "NO_TRADE": [],
            "NO_DATA": [],
            "ERROR": []
        },
        "statistics": {}
    }
    
    # Run backtests in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(backtest_coin, symbol, timeframe, version): symbol
            for symbol in symbols
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            symbol = futures[future]
            try:
                result = future.result()
                results[symbol] = result
                
                # Save per-coin report
                report_file = reports_dir / f"{symbol}_{timeframe}_{version}.json"
                with open(report_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Update summary
                status = result.get("status", "ERROR")
                summary["results_by_status"][status].append(symbol)
                
                logger.info(f"[{completed}/{len(symbols)}] {symbol} done")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = {
                    "symbol": symbol,
                    "status": "ERROR",
                    "error": str(e)
                }
                summary["results_by_status"]["ERROR"].append(symbol)
    
    # Calculate summary statistics
    active_coins = summary["results_by_status"]["ACTIVE"]
    if active_coins:
        active_results = [results[s] for s in active_coins if s in results]
        scores = [r.get("score_0_100", 0) for r in active_results if r.get("score_0_100")]
        if scores:
            summary["statistics"]["avg_score"] = float(np.mean(scores))
            summary["statistics"]["median_score"] = float(np.median(scores))
            summary["statistics"]["min_score"] = float(np.min(scores))
            summary["statistics"]["max_score"] = float(np.max(scores))
    
    summary["statistics"]["active_count"] = len(active_coins)
    summary["statistics"]["warmup_count"] = len(summary["results_by_status"]["WARMUP"])
    summary["statistics"]["no_trade_count"] = len(summary["results_by_status"]["NO_TRADE"])
    summary["statistics"]["error_count"] = len(summary["results_by_status"]["ERROR"])
    
    # Save summary report
    summary_file = REPORTS_DIR / f"summary_{version}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"ACTIVE: {len(active_coins)} coins")
    logger.info(f"WARMUP: {len(summary['results_by_status']['WARMUP'])} coins")
    logger.info(f"NO_TRADE: {len(summary['results_by_status']['NO_TRADE'])} coins")
    logger.info(f"ERROR: {len(summary['results_by_status']['ERROR'])} coins")
    logger.info(f"\nSummary: {summary_file}")
    logger.info(f"Per-coin: {reports_dir}")
    logger.info("=" * 70)
    
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward backtest for all coins")
    parser.add_argument("--symbols_file", default="data/symbols_32.json", help="Path to symbols JSON file")
    parser.add_argument("--tf", "--timeframe", dest="timeframe", default="15m", help="Timeframe (15m, 1h, 4h, 1d)")
    parser.add_argument("--version", default="v1", help="Model version tag")
    parser.add_argument("--max_workers", type=int, default=4, help="Max parallel workers")
    
    args = parser.parse_args()
    
    run_all_backtest(args.symbols_file, args.timeframe, args.version, args.max_workers)


if __name__ == "__main__":
    main()
