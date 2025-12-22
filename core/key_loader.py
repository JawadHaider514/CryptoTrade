"""
Key Loader & Management
=======================
Secure environment-based API key loading with validation and encryption guidance.
"""
import os
import sys
from typing import Optional, Tuple
from dotenv import load_dotenv


class KeyLoadError(Exception):
    """Raised when key loading fails or keys are missing."""
    pass


class KeyLoader:
    """
    Secure loader for API keys from environment variables.
    
    Supports testnet and live trading modes. Keys are loaded from .env file
    or system environment variables at startup.
    
    Usage:
        loader = KeyLoader(mode='testnet')
        api_key, api_secret = loader.get_keys()
    
    Security notes:
    - Keep .env out of version control (add to .gitignore)
    - Use environment-specific keys (testnet for development, live for production)
    - For production, use encrypted secrets (e.g., AWS Secrets Manager, HashiCorp Vault)
    - Rotate keys regularly and revoke compromised keys immediately
    """
    
    def __init__(self, mode: str = 'testnet', env_file: str = '.env'):
        """
        Initialize KeyLoader.
        
        Args:
            mode: 'testnet' (default, safe) or 'live' (requires explicit confirmation)
            env_file: Path to .env file (default: .env in project root)
        """
        self.mode = mode.lower()
        self.env_file = env_file
        
        # Validate mode
        if self.mode not in ('testnet', 'live'):
            raise KeyLoadError(f"Invalid mode '{mode}'. Use 'testnet' or 'live'.")
        
        # Load .env file
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
        else:
            print(f"⚠️  Warning: {self.env_file} not found. Using system environment variables only.")
        
        # Extra safety check for live mode
        if self.mode == 'live':
            self._confirm_live_mode()
    
    def _confirm_live_mode(self) -> None:
        """Ensure user explicitly intends to use live keys (safety guard)."""
        confirm = os.getenv('TRADING_MODE', 'testnet').lower()
        if confirm != 'live':
            raise KeyLoadError(
                "⚠️  LIVE MODE REQUESTED but TRADING_MODE != 'live' in .env. "
                "Set TRADING_MODE=live in your .env file to enable live trading. "
                "This is a safety guard to prevent accidental live trades."
            )
    
    def get_keys(self) -> Tuple[str, str]:
        """
        Retrieve API key and secret for the configured mode.
        
        Returns:
            Tuple[str, str]: (api_key, api_secret)
        
        Raises:
            KeyLoadError: If keys are missing or invalid
        """
        if self.mode == 'testnet':
            api_key = os.getenv('API_KEY_TESTNET', '').strip()
            api_secret = os.getenv('API_SECRET_TESTNET', '').strip()
            key_type = 'TESTNET'
        else:  # live
            api_key = os.getenv('API_KEY_LIVE', '').strip()
            api_secret = os.getenv('API_SECRET_LIVE', '').strip()
            key_type = 'LIVE'
        
        # Validate keys exist and are non-empty
        if not api_key or not api_secret:
            raise KeyLoadError(
                f"❌ {key_type} API keys not found or empty. "
                f"Set API_KEY_{key_type} and API_SECRET_{key_type} in .env"
            )
        
        # Additional validation
        if len(api_key) < 10 or len(api_secret) < 10:
            raise KeyLoadError(
                f"❌ {key_type} API keys appear invalid (too short). "
                f"Check your .env configuration."
            )
        
        return api_key, api_secret
    
    def validate_keys(self) -> bool:
        """
        Check if keys are valid (exist, non-empty, reasonable length).
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            self.get_keys()
            return True
        except KeyLoadError:
            return False
    
    @staticmethod
    def get_trading_mode() -> str:
        """
        Retrieve configured trading mode from environment.
        
        Returns:
            str: 'testnet' or 'live'
        """
        load_dotenv()
        mode = os.getenv('TRADING_MODE', 'testnet').lower()
        if mode not in ('testnet', 'live'):
            print(f"⚠️  Invalid TRADING_MODE '{mode}'; defaulting to 'testnet'")
            return 'testnet'
        return mode
    
    @staticmethod
    def print_status() -> None:
        """Print current key loading status (useful for debugging)."""
        load_dotenv()
        mode = KeyLoader.get_trading_mode()
        
        print("\n🔐 Key Loader Status:")
        print(f"   Mode: {mode.upper()}")
        
        testnet_ok = bool(os.getenv('API_KEY_TESTNET', '').strip() and 
                          os.getenv('API_SECRET_TESTNET', '').strip())
        live_ok = bool(os.getenv('API_KEY_LIVE', '').strip() and 
                       os.getenv('API_SECRET_LIVE', '').strip())
        
        print(f"   Testnet keys: {'✅ Configured' if testnet_ok else '❌ Missing'}")
        print(f"   Live keys:    {'✅ Configured' if live_ok else '❌ Missing'}")
        print()


if __name__ == '__main__':
    # Diagnostic script
    print("🔐 API Key Loader Diagnostic\n")
    
    try:
        # Try testnet
        print("1. Testing TESTNET loader...")
        testnet_loader = KeyLoader(mode='testnet')
        testnet_key, testnet_secret = testnet_loader.get_keys()
        print(f"   ✅ Testnet keys loaded (key: {testnet_key[:6]}..., secret: {testnet_secret[:6]}...)")
    except KeyLoadError as e:
        print(f"   ❌ {e}")
    
    print()
    
    try:
        # Try live (will fail if not explicitly set)
        print("2. Testing LIVE loader (requires TRADING_MODE=live in .env)...")
        live_loader = KeyLoader(mode='live')
        live_key, live_secret = live_loader.get_keys()
        print(f"   ✅ LIVE keys loaded (key: {live_key[:6]}..., secret: {live_secret[:6]}...)")
    except KeyLoadError as e:
        print(f"   ℹ️  {e}")
    
    print()
    
    # Print status
    KeyLoader.print_status()
