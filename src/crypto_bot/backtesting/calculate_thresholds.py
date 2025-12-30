#!/usr/bin/env python3
"""Calculate per-coin quality thresholds from backtest results"""

import json
import logging
from pathlib import Path
import argparse
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"


def calculate_thresholds(summary_file, min_score_default=55, volatility_guard=True):
    """
    Calculate per-coin thresholds from backtest summary
    
    Args:
        summary_file: Path to summary_<version>.json
        min_score_default: Default minimum score threshold (0-100)
        volatility_guard: Enable volatility-based adjustment
    
    Returns:
        dict: Per-coin thresholds
    """
    
    if not Path(summary_file).exists():
        logger.error(f"Summary file not found: {summary_file}")
        return {}
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    version = summary.get("version", "v1")
    timeframe = summary.get("timeframe", "15m")
    
    # Load per-coin backtest results
    per_coin_results = {}
    reports_dir = REPORTS_DIR / "per_coin"
    
    for symbol in summary.get("results_by_status", {}).get("ACTIVE", []):
        report_file = reports_dir / f"{symbol}_{timeframe}_{version}.json"
        if report_file.exists():
            with open(report_file) as f:
                per_coin_results[symbol] = json.load(f)
    
    # Calculate thresholds for each coin
    thresholds = {
        "version": version,
        "timeframe": timeframe,
        "min_score_default": min_score_default,
        "volatility_guard_enabled": volatility_guard,
        "coins": {}
    }
    
    for symbol, result in per_coin_results.items():
        status = result.get("status", "ERROR")
        score = result.get("score_0_100", 0)
        metrics = result.get("metrics", {})
        
        # Base threshold is min_score_default
        min_confidence = 0.55  # Default minimum confidence
        
        # Quality score determines min_confidence
        if score >= 75:
            # Strong coin: relax threshold
            min_confidence = 0.45
            quality_tier = "HIGH"
        elif score >= 60:
            # Good coin: standard threshold
            min_confidence = 0.55
            quality_tier = "MEDIUM"
        elif score >= 50:
            # Marginal coin: raise threshold
            min_confidence = 0.65
            quality_tier = "LOW"
        else:
            # Weak coin: don't trade
            quality_tier = "REJECTED"
            min_confidence = 0.99  # Effectively disable
        
        # Volatility guard: adjust based on win_rate
        volatility_adjustment = 0.0
        if volatility_guard:
            win_rate = metrics.get("win_rate", 0.5)
            if win_rate < 0.40:
                volatility_adjustment = 0.10  # Increase threshold by 10%
                quality_tier = "VOLATILITY_HIGH"
            elif win_rate < 0.45:
                volatility_adjustment = 0.05  # Increase by 5%
        
        min_confidence = min(0.99, min_confidence + volatility_adjustment)
        
        # Determine action
        if quality_tier == "REJECTED" or score < 45:
            action = "NO_TRADE"
            reason = f"Low score ({score:.1f}) - Insufficient quality"
        elif quality_tier == "VOLATILITY_HIGH":
            action = "WARMUP"
            reason = f"High volatility (wr={win_rate:.1%}) - Monitor before trading"
        elif quality_tier == "LOW":
            action = "TRADE_WITH_CAUTION"
            reason = f"Marginal score ({score:.1f}) - Reduced position size"
        else:
            action = "ACTIVE"
            reason = f"Quality tier: {quality_tier}"
        
        thresholds["coins"][symbol] = {
            "status": status,
            "quality_tier": quality_tier,
            "score_0_100": float(score),
            "min_score": min_score_default,
            "min_confidence": float(min_confidence),
            "action": action,
            "reason": reason,
            "metrics_snapshot": {
                "balanced_accuracy": metrics.get("balanced_accuracy", 0.5),
                "f1_long": metrics.get("f1_long", 0.5),
                "win_rate": metrics.get("win_rate", 0.5),
                "trades_taken": metrics.get("trades_taken", 0),
                "profit_factor": metrics.get("profit_factor", 1.0),
                "mean_confidence": metrics.get("mean_confidence", 0.5)
            }
        }
    
    # Summary statistics
    thresholds["summary"] = {
        "total_coins": len(thresholds["coins"]),
        "active_coins": len([c for c in thresholds["coins"].values() if c["action"] == "ACTIVE"]),
        "warmup_coins": len([c for c in thresholds["coins"].values() if c["action"] == "WARMUP"]),
        "caution_coins": len([c for c in thresholds["coins"].values() if c["action"] == "TRADE_WITH_CAUTION"]),
        "no_trade_coins": len([c for c in thresholds["coins"].values() if c["action"] == "NO_TRADE"]),
        "coverage": len([c for c in thresholds["coins"].values() if c["status"] in ["ACTIVE", "WARMUP"]]) / max(1, len(thresholds["coins"]))
    }
    
    # Save thresholds
    config_file = CONFIG_DIR / "per_coin_thresholds.json"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("THRESHOLD CALCULATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"ACTIVE: {thresholds['summary']['active_coins']} coins")
    logger.info(f"WARMUP: {thresholds['summary']['warmup_coins']} coins")
    logger.info(f"CAUTION: {thresholds['summary']['caution_coins']} coins")
    logger.info(f"NO_TRADE: {thresholds['summary']['no_trade_coins']} coins")
    logger.info(f"Coverage: {thresholds['summary']['coverage']:.1%}")
    logger.info(f"\nThresholds saved: {config_file}")
    logger.info("=" * 70)
    
    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Calculate per-coin quality thresholds")
    parser.add_argument("--summary", help="Path to summary_<version>.json")
    parser.add_argument("--version", default="v1", help="Model version tag")
    parser.add_argument("--tf", "--timeframe", dest="timeframe", default="15m", help="Timeframe")
    parser.add_argument("--min_score", type=int, default=55, help="Default minimum score (0-100)")
    parser.add_argument("--volatility_guard", type=int, default=1, help="Enable volatility guard (0/1)")
    
    args = parser.parse_args()
    
    # Infer summary path
    summary_file = args.summary or (REPORTS_DIR / f"summary_{args.version}.json")
    
    calculate_thresholds(
        summary_file,
        min_score_default=args.min_score,
        volatility_guard=bool(args.volatility_guard)
    )


if __name__ == "__main__":
    main()
