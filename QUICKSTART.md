# ⚡ Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Install Dependencies
```bash
cd crypto_trading_system
pip install -r requirements.txt
```

### Step 2: Run the System
```bash
# Option A: Basic web server
python run.py

# Option B: Advanced server with WebSocket
python run.py --mode advanced

# Option C: Dashboard only (console)
python run.py --mode dashboard
```

### Step 3: Open Dashboard
Visit: **http://localhost:5000**

---

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `python run.py` | Start basic server |
| `python run.py --mode advanced` | Start with WebSocket |
| `python run.py --mode dashboard` | Console mode only |
| `python run.py --test-timing` | Test signal timing |
| `python run.py --test-bot` | Test demo bot |
| `python run.py --status` | Show system status |
| `python run.py --help` | Show all options |

---

## 🔧 Quick Configuration

### Edit `config/settings.py`:

```python
# Change trading pairs
TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

# Adjust risk settings
TRADING_CONFIG = {
    'DEFAULT_LEVERAGE': 20,
    'RISK_PERCENTAGE': 2.0,
    'MAX_CONCURRENT_TRADES': 5,
}

# Set minimum signal quality
TRADING_CONFIG['MIN_CONFLUENCE_SCORE'] = 70
```

---

## 📡 API Quick Reference

```bash
# Get all signals
curl http://localhost:5000/api/signals

# Get specific symbol
curl http://localhost:5000/api/signals/BTCUSDT

# Start bot
curl -X POST http://localhost:5000/api/bot/start

# Get statistics
curl http://localhost:5000/api/statistics
```

---

## 📂 Key Files

| File | What It Does |
|------|--------------|
| `run.py` | Main entry point |
| `core/enhanced_crypto_dashboard.py` | Trading engine |
| `server/web_server.py` | REST API server |
| `config/settings.py` | All configuration |
| `models/signals.py` | Signal data models |

---

## 🆘 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "Port already in use"
```bash
# Kill existing process
lsof -i :5000
kill -9 <PID>

# Or use different port
python run.py --port 5001
```

### "Dashboard not loading"
Check that Flask is running:
```bash
curl http://localhost:5000/api/signals
```

---

## 📞 Need Help?

1. Check `README.md` for full documentation
2. Check `ARCHITECTURE.md` for system details
3. Run `python run.py --status` for diagnostics

---

*Happy Trading! 🚀📈*
