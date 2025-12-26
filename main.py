#!/usr/bin/env python3
"""
Crypto Trading System - Main Entry Point
=========================================

This is a thin wrapper that starts the Flask/SocketIO web server.

Usage:
    python main.py                    # Start server (default)

Configuration is loaded from crypto_bot.config.settings
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import crypto_bot
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    """Start the web server"""
    try:
        from crypto_bot.server.web_server import app
        from crypto_bot.config.settings import APP_CONFIG
        
        # Get server config
        host = APP_CONFIG.get('SERVER_HOST', '0.0.0.0')
        port = APP_CONFIG.get('SERVER_PORT', 5000)
        debug = APP_CONFIG.get('DEBUG', False)
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Crypto Trading System")
        print(f"   Server: http://{host}:{port}")
        print(f"   Debug:  {debug}")
        print(f"{'='*60}\n")
        
        # Run server
        app.run(host=host, port=port, debug=debug)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}", file=sys.stderr)
        print(f"Make sure all dependencies are installed: pip install -e .", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
