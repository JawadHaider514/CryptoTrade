API Key Management & Security
==============================

Overview
--------
This guide covers secure API key management, environment configuration, and production-safe practices for the CryptoTrade system.

Quick Start
-----------

1. Copy .env.example to .env:
   cp .env.example .env

2. Edit .env with your keys:
   API_KEY_TESTNET=your_testnet_key
   API_SECRET_TESTNET=your_testnet_secret
   TRADING_MODE=testnet

3. Test key loading:
   python core/key_loader.py

4. Use in code:
   from core.key_loader import KeyLoader
   loader = KeyLoader(mode='testnet')
   api_key, api_secret = loader.get_keys()

Security Best Practices
-----------------------

File-Level (.env)
  - .env contains all sensitive keys; NEVER commit to Git.
  - Add to .gitignore: .env, .env.local, *.key
  - Recommended: Use separate .env files per environment (.env.testnet, .env.live)
  - File permissions: chmod 600 .env (on Unix/Mac) or Restricted (Windows)

Key Rotation & Lifecycle
  - Rotate API keys every 90 days (or per exchange policy).
  - When rotating: generate new keys in Binance, update .env, revoke old keys.
  - Keep a rotation log (not in repo) with dates and changed symbols.
  - Upon compromise: immediately revoke and regenerate keys (within minutes).

Environment-Specific Setup
  - Development: testnet keys only (TRADING_MODE=testnet)
  - Staging: testnet keys or isolated live subaccount (TRADING_MODE=testnet)
  - Production: live keys with restricted permissions (TRADING_MODE=live)

Binance API Key Configuration
  - Create restricted keys (do NOT use master account):
    - Enable Spot/Margin trading (for your strategy)
    - Disable Withdrawal
    - Restrict to IP whitelist (your server IP)
    - Set spending limits (e.g., max order size)
  - Example restrictions:
    - IP Whitelist: 203.0.113.0 (your server)
    - Permissions: Spot trading, Read data
    - Disable: Withdraw, Transfer

Production Deployment
---------------------

For production, replace environment files with a secrets manager:

AWS Secrets Manager:
  from boto3 import client as boto_client
  secrets_client = boto_client('secretsmanager')
  secret = secrets_client.get_secret_value(SecretId='crypto-trade/binance-live')
  api_key = secret['SecretString']['API_KEY_LIVE']
  api_secret = secret['SecretString']['API_SECRET_LIVE']

HashiCorp Vault:
  import hvac
  client = hvac.Client(url='https://vault.example.com', token='your_token')
  secret = client.secrets.kv.read_secret_version(path='crypto-trade/binance-live')
  api_key = secret['data']['data']['API_KEY_LIVE']
  api_secret = secret['data']['data']['API_SECRET_LIVE']

Environment Variables (Docker/Kubernetes):
  - Pass secrets as environment variables (K8s Secrets, Docker secrets)
  - Never embed keys in images or config files

Example Docker setup:
  docker run -e API_KEY_LIVE=$BINANCE_KEY -e API_SECRET_LIVE=$BINANCE_SECRET crypto-trade

Encryption (At-Rest)
---------------------

For additional safety, encrypt keys at rest:

Option 1: cryptography library
  from cryptography.fernet import Fernet
  
  # Generate key once and store securely
  key = Fernet.generate_key()  # Store in env or secrets manager
  cipher = Fernet(key)
  
  encrypted_secret = cipher.encrypt(api_secret.encode())
  # Save encrypted_secret to .env or database
  
  # On load:
  api_secret = cipher.decrypt(encrypted_secret).decode()

Option 2: AWS KMS
  import boto3
  kms = boto3.client('kms')
  
  encrypted = kms.encrypt(KeyId='alias/crypto-trade', Plaintext=api_secret)
  # Store encrypted['CiphertextBlob']
  
  decrypted = kms.decrypt(CiphertextBlob=encrypted_blob)
  api_secret = decrypted['Plaintext'].decode()

Monitoring & Auditing
---------------------

- Log all key access (but never log the key itself)
- Monitor for unusual API usage (large orders, new IPs, rapid calls)
- Set up alerts:
  - Unexpected API calls from new locations
  - Rate limit near-breaches
  - Unusual order patterns
- Use API key activity logs (Binance provides these)

Testing & Validation
---------------------

Test key loading locally:
  python core/key_loader.py

Test with PaperTrader (no real funds):
  python scripts/demo_paper_trade.py

Run in testnet before moving to live:
  # Ensure TRADING_MODE=testnet in .env
  python run.py --paper-demo

Troubleshooting
---------------

❌ "KeyLoadError: API_KEY_TESTNET not found or empty"
   → Check .env exists and has keys populated
   → Ensure no whitespace issues in key values

❌ "LIVE MODE REQUESTED but TRADING_MODE != 'live'"
   → This is intentional (safety guard)
   → Set TRADING_MODE=live in .env to enable live trading

❌ "Keys appear invalid (too short)"
   → Check Binance keys (should be ~60 chars for key, ~90 for secret)
   → Regenerate if keys look incomplete

Emergency Procedure
-------------------

If keys are compromised:
  1. Immediately revoke all keys in Binance account settings
  2. Generate new keys (Binance enforces 24h wait for some situations)
  3. Update .env with new keys
  4. Restart all trading processes
  5. Review trade history for unauthorized trades
  6. Document incident (for compliance/audit)

Next Steps
----------

- Integrate KeyLoader into exchange adapter (core/exchange_adapter.py)
- Add automated key rotation alerts
- Implement secrets manager integration for production
- Add comprehensive audit logging for all key access

References
----------

- Binance API Documentation: https://binance-docs.github.io/apidocs/
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework
- OWASP Secret Management: https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html
