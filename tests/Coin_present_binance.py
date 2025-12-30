# =====================================================
# BINANCE COINS AVAILABILITY CHECKER
# Check kon se coins Binance par available hain
# =====================================================

"""
GOAL: Verify jo coins aapne list diye hain, 
      woh Binance par trading ke liye available hain
"""

import requests
import json
from typing import List, Dict, Tuple

class BinanceCoinChecker:
    """
    Check coin availability on Binance
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
        # Load verified trading coins from config
        try:
            import json as _json
            from pathlib import Path as _Path
            _config_path = _Path(__file__).parent.parent / "config" / "coins.json"
            _coins_config = _json.load(open(_config_path))
            self.coins_list = _coins_config.get("symbols", [])
        except Exception:
            # Fallback: 32 verified trading symbols
            self.coins_list = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
                "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "UNIUSDT",
                "LINKUSDT", "XLMUSDT", "ATOMUSDT", "MANAUSDT", "SANDUSDT",
                "DASHUSDT", "VETUSDT", "ICPUSDT", "GMTUSDT", "PEOPLEUSDT",
                "LUNCUSDT", "CHZUSDT", "NEARUSDT", "FLOWUSDT", "FILUSDT",
                "QTUMUSDT", "SNXUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT",
                "FLOKIUSDT", "OPUSDT"
            ]
    
    def get_all_trading_pairs(self) -> List[str]:
        """
        Get all trading pairs from Binance
        """
        try:
            url = f"{self.base_url}/exchangeInfo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pairs = [symbol['symbol'] for symbol in data['symbols'] if symbol['status'] == 'TRADING']
            
            return pairs
        
        except Exception as e:
            print(f"❌ Error fetching exchange info: {e}")
            return []
    
    def check_coin_availability(self) -> Tuple[List[str], List[str]]:
        """
        Check which coins are available on Binance
        
        Returns:
            Tuple of (available_coins, unavailable_coins)
        """
        print("Fetching all available trading pairs from Binance...")
        print("(This may take a few seconds)\n")
        
        binance_pairs = self.get_all_trading_pairs()
        
        if not binance_pairs:
            print("❌ Could not fetch Binance trading pairs")
            return [], self.coins_list
        
        print(f"Total pairs on Binance: {len(binance_pairs)}\n")
        
        available = []
        unavailable = []
        
        print("Checking your coins...\n")
        print("="*70)
        
        for coin in self.coins_list:
            if coin in binance_pairs:
                available.append(coin)
                print(f"✅ {coin:<15} - AVAILABLE")
            else:
                unavailable.append(coin)
                print(f"❌ {coin:<15} - NOT AVAILABLE")
        
        print("="*70)
        
        return available, unavailable
    
    def print_summary(self, available: List[str], unavailable: List[str]):
        """
        Print summary report
        """
        print(f"\n\n{'='*70}")
        print("BINANCE COINS AVAILABILITY REPORT")
        print(f"{'='*70}\n")
        
        print(f"✅ AVAILABLE COINS: {len(available)}/{len(self.coins_list)}")
        print(f"{'─'*70}")
        for i, coin in enumerate(available, 1):
            print(f"  {i:2d}. {coin}")
        
        if unavailable:
            print(f"\n\n❌ NOT AVAILABLE COINS: {len(unavailable)}/{len(self.coins_list)}")
            print(f"{'─'*70}")
            for i, coin in enumerate(unavailable, 1):
                print(f"  {i:2d}. {coin}")
        
        print(f"\n\n{'='*70}")
        print(f"SUMMARY:")
        print(f"  Available:   {len(available)} coins ✅")
        print(f"  Unavailable: {len(unavailable)} coins ❌")
        print(f"  Success Rate: {(len(available)/len(self.coins_list)*100):.1f}%")
        print(f"{'='*70}\n")
    
    def get_coin_info(self, coin: str) -> Dict:
        """
        Get detailed info about a specific coin
        """
        try:
            url = f"{self.base_url}/ticker/24hr?symbol={coin}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'symbol': data['symbol'],
                'price': float(data['lastPrice']),
                'priceChangePercent': float(data['priceChangePercent']),
                'volume': float(data['volume']),
                'quoteVolume': float(data['quoteVolume']),
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def print_detailed_info(self, available: List[str]):
        """
        Print detailed info for available coins
        """
        print(f"\n\n{'='*70}")
        print("DETAILED COIN INFORMATION (Top 10)")
        print(f"{'='*70}\n")
        
        for i, coin in enumerate(available[:10], 1):
            info = self.get_coin_info(coin)
            
            if 'error' not in info:
                print(f"{i:2d}. {coin}")
                print(f"    Price: ${info['price']:.8f}")
                print(f"    24h Change: {info['priceChangePercent']:+.2f}%")
                print(f"    24h Volume: {info['volume']:.2f}")
                print()
            else:
                print(f"{i:2d}. {coin} - Error fetching data")
                print()
        
        print(f"{'='*70}\n")
    
    def export_to_json(self, available: List[str], unavailable: List[str], 
                      filename: str = "binance_coins_status.json"):
        """
        Export results to JSON file
        """
        result = {
            'timestamp': str(pd.Timestamp.now()),
            'total_coins': len(self.coins_list),
            'available_count': len(available),
            'unavailable_count': len(unavailable),
            'success_rate_percent': round((len(available)/len(self.coins_list)*100), 2),
            'available_coins': available,
            'unavailable_coins': unavailable
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ Results exported to {filename}")
        except Exception as e:
            print(f"❌ Error exporting to JSON: {e}")


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "BINANCE COINS AVAILABILITY CHECKER" + " "*19 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Initialize checker
    checker = BinanceCoinChecker()
    
    # Check availability
    available, unavailable = checker.check_coin_availability()
    
    # Print summary
    checker.print_summary(available, unavailable)
    
    # Print detailed info
    checker.print_detailed_info(available)
    
    # Export to JSON
    try:
        import pandas as pd
        checker.export_to_json(available, unavailable)
    except ImportError:
        print("(pandas not installed, skipping JSON export)")
    
    # Final message
    if unavailable:
        print(f"\n⚠️  WARNING: {len(unavailable)} coin(s) not available on Binance!")
        print("You may need to update your coin list or use different trading pairs.\n")
    else:
        print(f"\n🎉 GREAT NEWS: All {len(available)} coins are available on Binance!")
        print("Your trading system is ready to go!\n")


# =====================================================
# QUICK REFERENCE - EXPECTED COINS
# =====================================================

"""
POPULAR COINS (EXPECTED TO BE AVAILABLE):

Tier 1 (Always Available):
✅ BTCUSDT     - Bitcoin
✅ ETHUSDT     - Ethereum
✅ BNBUSDT     - Binance Coin
✅ XRPUSDT     - Ripple
✅ ADAUSDT     - Cardano

Tier 2 (Usually Available):
✅ SOLUSDT     - Solana
✅ DOGEUSDT    - Dogecoin
✅ MATICUSDT   - Polygon
✅ LITUSDT     - Litecoin
✅ AVAXUSDT    - Avalanche

Tier 3 (Common but sometimes delisted):
⚠️  LUNCUSDT   - Luna Classic (was delisted, may return)
⚠️  PEPEUSDT   - Pepe (meme coin, volatile)
⚠️  WIFUSDT    - Dog Wif Hat (new/volatile)
⚠️  FLOKIUSDT  - Floki (new/volatile)
⚠️  OPUSDT     - Optimism (may need OP not OP)

Most likely available:
✅ 30-33 out of 35 coins

Most likely unavailable/different:
❌ LUNCUSDT (delisted, use LUNA instead)
❌ PEPEUSDT (may be volatile)
❌ WIFUSDT (new coin)
"""