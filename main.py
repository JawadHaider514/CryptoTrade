#!/usr/bin/env python3
"""
Crypto Trading System - Main Entry Point
Starts ADVANCED Flask-SocketIO server with ML predictions + threading mode

Usage:
    python main.py                          # Default: server mode
    python main.py --use_ml 1              # Enable ML predictions
    python main.py --use_ml 1 --tf 15m     # ML with 15m timeframe
    python main.py --device cuda            # Use GPU
    python main.py --symbols_file data/symbols_32.json
"""

import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Crypto Trading System with ML predictions"
    )
    parser.add_argument(
        "--use_ml",
        type=int,
        default=0,
        help="Enable ML predictions (0=disabled, 1=enabled)"
    )
    parser.add_argument(
        "--tf",
        type=str,
        default="15m",
        choices=["1m", "5m", "15m", "1h", "4h"],
        help="Default timeframe for ML predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for ML inference (cpu or cuda)"
    )
    parser.add_argument(
        "--symbols_file",
        type=str,
        default="data/symbols_32.json",
        help="Path to symbols file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    try:
        from crypto_bot.server.advanced_web_server import app, socketio
        from crypto_bot.config.settings import APP_CONFIG
        import os
        
        # Set environment variables for ML system
        os.environ["USE_ML_PER_COIN"] = str(args.use_ml)
        os.environ["ML_DEFAULT_TF"] = args.tf
        os.environ["ML_DEVICE"] = args.device
        os.environ["SYMBOLS_FILE"] = args.symbols_file

        host = args.host or APP_CONFIG.get("HOST", "0.0.0.0")
        port = args.port or APP_CONFIG.get("PORT", 5000)
        try:
            port = int(port)
        except (TypeError, ValueError):
            port = 5000

        debug = APP_CONFIG.get("DEBUG", False)
        if isinstance(debug, str):
            debug = debug.strip().lower() in {"1", "true", "yes", "y", "on"}

        print(f"\n{'='*70}")
        print("🚀 Starting Crypto Trading System (ADVANCED)")
        print(f"   Server: http://{host}:{port}")
        print(f"   ML Enabled: {'✅ YES' if args.use_ml else '❌ NO'}")
        if args.use_ml:
            print(f"   ML Timeframe: {args.tf}")
            print(f"   ML Device: {args.device}")
        print(f"   Debug: {debug}")
        print(f"{'='*70}\n")

        # IMPORTANT: use_reloader=False to avoid double threads / double bot loop
        # Force debug=False on Windows to prevent eventlet issues
        socketio.run(
            app,
            host=host,
            port=port,
            debug=False,  # Always False to prevent eventlet reloader conflicts
            use_reloader=False,  # Essential on Windows
            allow_unsafe_werkzeug=True
        )

    except ImportError as e:
        print(f"❌ Import Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
