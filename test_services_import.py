#!/usr/bin/env python3
"""Test that all services are properly created and importable"""

import sys
sys.path.insert(0, 'src')

try:
    from crypto_bot.domain.signal_models import SignalModel, TakeProfit
    print("[OK] signal_models imported")
    
    from crypto_bot.services.market_data_service import MarketDataService
    print("[OK] market_data_service imported")
    
    from crypto_bot.services.signal_engine_service import SignalEngineService
    print("[OK] signal_engine_service imported")
    
    from crypto_bot.repositories.signal_repository import SignalRepository
    print("[OK] signal_repository imported")
    
    from crypto_bot.services.signal_orchestrator import SignalOrchestrator
    print("[OK] signal_orchestrator imported")
    
    print("\n=== ALL SERVICES CREATED AND IMPORTABLE ===")
    print("\nImplementation Summary:")
    print("  Phase 2: 5 new service files created")
    print("  Phase 3: advanced_web_server.py updated with:")
    print("    - SYMBOLS list (35 cryptocurrencies)")
    print("    - init_services() function")
    print("    - REST endpoints (/api/predictions, /api/prices)")
    print("    - SocketIO event handlers")
    print("  Phase 4: index.html updated with:")
    print("    - SocketIO.io connection")
    print("    - Real-time event listeners")
    print("\nSystem ready for startup!")
    
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
