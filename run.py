#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 CRYPTO TRADING SYSTEM - MAIN ENTRY POINT
============================================
Run this file to start the trading system.

Usage:
    python run.py                    # Start web server (default)
    python run.py --mode advanced    # Start advanced server with WebSocket
    python run.py --mode dashboard   # Run dashboard only (no web server)
    python run.py --test-timing      # Test signal timing logic
    python run.py --test-bot         # Test demo bot
    python run.py --help             # Show help
"""

import sys
import os
import argparse
import io

# Fix Unicode output for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import configuration
from crypto_bot.config.settings import APP_CONFIG, TRADING_CONFIG


def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🚀 CRYPTO TRADING SYSTEM                                       ║
║   ══════════════════════                                         ║
║                                                                  ║
║   📊 Real-time Signals  │  🤖 ML Predictions  │  📈 Analytics    ║
║                                                                  ║
║   University Final Year Project                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_basic_server():
    """Run the basic Flask web server"""
    print("\n🌐 Starting Basic Web Server...")
    print("="*60)
    
    try:
        from server.web_server import main as server_main
        server_main()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


def run_advanced_server():
    """Run the advanced web server with WebSocket"""
    print("\n🔌 Starting Advanced Web Server with WebSocket...")
    print("="*60)
    
    try:
        from server.advanced_web_server import main as advanced_main
        advanced_main()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


def run_dashboard_only():
    """Run the dashboard without web server"""
    print("\n📊 Starting Dashboard (Console Mode)...")
    print("="*60)
    
    try:
        from core.enhanced_crypto_dashboard import main as dashboard_main
        dashboard_main()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        sys.exit(1)


def run_test_timing():
    """Test the timing logic"""
    print("\n⏱️ Testing Signal Timing Logic...")
    print("="*60)
    
    try:
        import importlib
        dashboard_mod = importlib.import_module('core.enhanced_crypto_dashboard')
        test_improved_timing = getattr(dashboard_mod, 'test_improved_timing', None)
        if callable(test_improved_timing):
            test_improved_timing()
        else:
            print("⚠️  Timing test not available (removed or not installed).")
        return
    except Exception:
        print("⚠️  Timing test not available (removed or not installed).")
        return


def run_test_bot():
    """Test the demo bot"""
    print("\n🤖 Testing Demo Trading Bot...")
    print("="*60)
    
    try:
        from core.enhanced_crypto_dashboard import test_demo_bot
        test_demo_bot()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        sys.exit(1)


def show_status():
    """Show system status"""
    print("\n📋 System Status")
    print("="*60)
    
    # Check imports
    modules = {
        'Flask': 'flask',
        'Flask-CORS': 'flask_cors',
        'Flask-SocketIO': 'flask_socketio',
        'Pandas': 'pandas',
        'NumPy': 'numpy',
        'Requests': 'requests'
    }
    
    print("\n📦 Dependencies:")
    for name, module in modules.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Not installed")
    
    # Check project modules
    print("\n📁 Project Modules:")
    project_modules = [
        ('Core Dashboard', 'core.enhanced_crypto_dashboard'),
        ('Web Server', 'server.web_server'),
        ('Advanced Server', 'server.advanced_web_server'),
        ('Trading Integration', 'api.trading_integration'),
        ('Config', 'config.settings')
    ]
    
    for name, module in project_modules:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ⚠️  {name} - Not found")
    
    print("\n" + "="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='🚀 Crypto Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    Start basic web server
  python run.py --mode advanced    Start advanced server with WebSocket
  python run.py --mode dashboard   Run dashboard only (console)
  python run.py --test-timing      Test signal timing logic
  python run.py --test-bot         Test demo trading bot
  python run.py --status           Show system status
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['basic', 'advanced', 'dashboard'],
        default='basic',
        help='Server mode (default: basic)'
    )
    
    parser.add_argument(
        '--test-timing',
        action='store_true',
        help='Test the signal timing logic'
    )
    
    parser.add_argument(
        '--test-bot',
        action='store_true',
        help='Test the demo trading bot'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status'
    )
    parser.add_argument(
        '--paper-demo',
        action='store_true',
        help='Run a one-off paper trading demo (no real orders)'
    )
    parser.add_argument(
        '--paper-demo',
        action='store_true',
        help='Run a one-off paper trading demo (no real orders)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Server port (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle commands
    if args.status:
        show_status()
        return

    if args.paper_demo:
        print('\n📘 Running paper trading demo...')
        try:
            from scripts.demo_paper_trade import main as demo_main
            demo_main()
        except Exception:
            # Fallback: run module directly
            import runpy
            runpy.run_path('scripts/demo_paper_trade.py', run_name='__main__')
        return
    
    if args.test_timing:
        run_test_timing()
        return
    
    if args.test_bot:
        run_test_bot()
        return
    
    # Run selected mode
    if args.mode == 'basic':
        run_basic_server()
    elif args.mode == 'advanced':
        run_advanced_server()
    elif args.mode == 'dashboard':
        run_dashboard_only()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 System stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
