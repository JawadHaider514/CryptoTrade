# ⚡ Quick Start Guide

## One-Time Setup (Already Done ✅)

```bash
# Virtual environment created
.venv/

# Package installed
pip install -e .

# IDE configured
.vscode/settings.json

# Tests verified
pytest tests/  # 12 passing ✅
```

---

## Start Development Now

### 1️⃣ Restart VS Code (Required)
- **Ctrl+Shift+P** → "Python: Restart Language Server"
- OR **Ctrl+R** → Reload Window
- OR close/reopen VS Code

### 2️⃣ Run the Server
```bash
python main.py
```

Starts Flask server on http://localhost:5000

### 3️⃣ Run Tests
```bash
pytest tests/
# or with verbose output
pytest tests/ -v
```

---

## Common Commands

| Task | Command |
|------|---------|
| Run server | `python main.py` |
| Run tests | `pytest tests/` |
| Run specific test | `pytest tests/test_config.py -v` |
| Format code | `black src/` |
| Check types | `pyright src/` |
| Install dependency | `pip install package_name` |
| Update requirements | `pip freeze > requirements.txt` |
| List installed packages | `pip list` |

---

## Project Structure Quick Reference

```
crypto_bot/
├── api/              # REST API endpoints & integrations
├── config/           # Settings & configuration
├── core/             # Trading logic & bot executor
├── models/           # Data models & schemas
├── server/           # Flask app & routes
├── utils/            # Helper functions
└── __init__.py       # Package initialization
```

---

## Import Pattern

All imports use the `crypto_bot` namespace:

```python
# ✅ Correct
from crypto_bot.server.web_server import app
from crypto_bot.config.settings import APP_CONFIG
from crypto_bot.core.bot_executor import BotExecutor

# ❌ Avoid
from core.bot_executor import BotExecutor  # Old pattern
```

---

## File Locations

| Type | Location |
|------|----------|
| Source code | `src/crypto_bot/` |
| Tests | `tests/` |
| Scripts | `scripts/` |
| Docs | `docs/` |
| Config | `.vscode/settings.json` |
| Env file | `.venv/` |

---

## Verification

Check everything works:

```bash
# Test imports
python -c "from crypto_bot.server.web_server import app; print('✅')"

# Test config
python -c "from crypto_bot.config.settings import APP_CONFIG; print('✅')"

# Run tests
pytest tests/
```

---

## Need Help?

1. **IDE shows errors?** → Restart Language Server (Ctrl+Shift+P)
2. **Import errors?** → Ensure using .venv Python
3. **Tests fail?** → Check `pytest tests/ -v` output
4. **New dependency?** → `pip install package` then `pip freeze > requirements.txt`

---

**Everything is ready! Just restart VS Code and start coding.** 🚀
