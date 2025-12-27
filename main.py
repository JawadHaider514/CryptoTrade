#!/usr/bin/env python3
"""
Crypto Trading System - Main Entry Point
Starts ADVANCED Flask-SocketIO server
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

def main() -> None:
    try:
        from crypto_bot.server.advanced_web_server import app, socketio
        from crypto_bot.config.settings import APP_CONFIG

        host = APP_CONFIG.get("HOST", "0.0.0.0")

        port = APP_CONFIG.get("PORT", 5000)
        try:
            port = int(port)
        except (TypeError, ValueError):
            port = 5000

        debug = APP_CONFIG.get("DEBUG", False)
        if isinstance(debug, str):
            debug = debug.strip().lower() in {"1", "true", "yes", "y", "on"}

        print(f"\n{'='*60}")
        print("🚀 Starting Crypto Trading System (ADVANCED)")
        print(f"   Server: http://{host}:{port}")
        print(f"   Debug:  {debug}")
        print(f"{'='*60}\n")

        # IMPORTANT: use_reloader=False to avoid double threads / double bot loop
        socketio.run(app, host=host, port=port, debug=debug, use_reloader=False)

    except ImportError as e:
        print(f"❌ Import Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
