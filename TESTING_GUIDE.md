# Quick Testing Guide - Signal System Verification

## Start the System

1. Open terminal in project root:
```bash
python main.py
```

2. Wait for output showing:
```
✅ EventletPatcher initialized
🚀 Starting Flask-SocketIO server...
Serving Flask app 'advanced_web_server'
...
Running on http://0.0.0.0:5000
```

3. Then in another terminal, start testing:
```bash
# Option A: Open in browser
http://localhost:5000

# Option B: Use curl for testing
curl http://localhost:5000/api/health
```

---

## Test 1: Health Check (Real-Time Signal Count)

```bash
curl http://localhost:5000/api/health
```

**Expected Response:**
```json
{
  "success": true,
  "status": "ok",
  "services_available": true,
  "active_predictions": 34,  // ← Real-time count
  "symbols_count": 34,
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

**What to check:**
- ✅ `active_predictions` should be > 0
- ✅ Should match number of symbols in your config
- ✅ Should update in real-time (keep calling, count should stay high)

---

## Test 2: Debug Repository Count (Fast Status)

```bash
curl http://localhost:5000/api/debug/repo_count
```

**Expected Response:**
```json
{
  "success": true,
  "cache_count": 34,
  "repo_error": null,
  "symbols_expected": 34,
  "message": "Repository has 34 signals cached"
}
```

**What to check:**
- ✅ `success: true` (no errors)
- ✅ `cache_count` should equal number of symbols
- ✅ `repo_error: null` (repository is working)

---

## Test 3: Get All Predictions with Transparency

```bash
curl http://localhost:5000/api/predictions
```

**Expected Response:**
```json
{
  "predictions": [
    {
      "symbol": "BTCUSDT",
      "direction": "LONG",
      "confidence": 65,
      "accuracy": 72,
      "entry_price": 45000.00,
      "timestamp": "2024-01-01T12:00:00.000000"
    }
  ],
  "filtered_predictions": {
    "ETHUSDT": {
      "raw_confidence": 45,
      "raw_accuracy": 50,
      "filtered_out_reason": "confidence 45 < MIN_CONFIDENCE 0",  // Usually won't filter in dev
      "source": "RSI_MA_FALLBACK"
    }
  },
  "errors": [],
  "count": 34,
  "filtered_count": 34,
  "error_count": 0,
  "dev_thresholds": {
    "MIN_CONFIDENCE": 0,
    "MIN_ACCURACY": 0
  }
}
```

**What to check:**
- ✅ `predictions` array is NOT empty
- ✅ `count` should equal number of symbols
- ✅ `filtered_count` should equal `count` (no filtering in dev mode)
- ✅ `error_count` should be 0
- ✅ `dev_thresholds` shows MIN_CONFIDENCE=0, MIN_ACCURACY=0

---

## Test 4: Get Single Symbol Prediction

```bash
curl http://localhost:5000/api/predictions/BTCUSDT
```

**Expected Response:**
```json
{
  "symbol": "BTCUSDT",
  "direction": "LONG",
  "confidence": 65,
  "accuracy": 72,
  "entry_price": 45000.00,
  "source": "PROFESSIONAL_ANALYZER",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

**What to check:**
- ✅ Response is NOT empty/null
- ✅ `direction` is LONG or SHORT (not HOLD)
- ✅ `confidence` and `accuracy` are populated
- ✅ `source` shows which analyzer generated it

---

## Test 5: Monitor Orchestrator Logging

While main.py is running, watch the console output for:

```
[TICK 1] Generated 34 signals, Saved 34/34 signals
[TICK 2] Generated 34 signals, Saved 34/34 signals
[TICK 3] Generated 34 signals, Saved 34/34 signals
```

**What to check:**
- ✅ Should appear every ~30 seconds
- ✅ `Generated` count should match `Saved` count (e.g., 34/34)
- ✅ Tick counter should increment (1, 2, 3...)

---

## Test 6: Monitor Database Persistence

You should also see logging like:
```
💾 CACHE STORED: {'BTCUSDT': {...}, 'ETHUSDT': {...}}
✅ DB STORED: BTCUSDT at 2024-01-01 12:00:00
✅ DB STORED: ETHUSDT at 2024-01-01 12:00:00
```

**What to check:**
- ✅ Signals are being cached
- ✅ Signals are being saved to database
- ✅ All symbols are being processed

---

## Test 7: Check Database Directly

```bash
# Open SQLite in project root
sqlite3 data/signals.db

# Inside sqlite3:
SELECT COUNT(*) FROM signals;
SELECT symbol, direction, confidence, accuracy FROM signals LIMIT 5;
.quit
```

**What to check:**
- ✅ COUNT should return > 0
- ✅ Should see BTCUSDT, ETHUSDT, etc. in results
- ✅ `direction` should be LONG or SHORT
- ✅ `confidence` and `accuracy` should have values

---

## Troubleshooting

### Problem: `/api/health` returns `active_predictions: 0`
- Check orchestrator logs for `[TICK N] Generated 0 signals`
- Check if signal generation is failing (look for errors)
- Try `/api/predictions` to see what errors are returned

### Problem: `/api/debug/repo_count` returns `success: false`
- Check the `repo_error` field for specific error
- Might indicate repository initialization issue
- Check logs for "CACHE READ" messages

### Problem: `/api/predictions` returns empty `predictions` array
- Check `filtered_predictions` - are signals being filtered?
- If yes: raise MIN_CONFIDENCE in config/settings.py
- Check `errors` array - are there processing errors?

### Problem: Orchestrator not logging `[TICK N]` messages
- Orchestrator might not have started
- Check logs for orchestrator startup message
- Might need to restart main.py

### Problem: Database shows 0 signals
- Check logs for "DB STORED" messages
- Check if signal_repository._store_signal() is being called
- Might be a database connection issue

---

## Expected Behavior Sequence

1. **On Startup (0-5 seconds):**
   ```
   ✅ EventletPatcher initialized
   SignalEngineService initialized with:
     MIN_CONFIDENCE: 0
     MIN_ACCURACY: 0
     USE_PRO_ANALYZER: True
   Starting signal orchestrator...
   ```

2. **After ~30 seconds:**
   ```
   [TICK 1] Generated 34 signals, Saved 34/34 signals
   💾 CACHE STORED: {...}
   ✅ DB STORED: BTCUSDT...
   📊 CACHE READ: 34 items
   ```

3. **When calling /api/predictions:**
   ```
   📡 GET PREDICTIONS: Retrieved 34 signals from repo
   Returning 34 predictions (0 filtered, 0 errors)
   ```

4. **When calling /api/health:**
   ```
   Health check: 34 signals in repository
   ```

---

## Performance Baseline

- **Orchestrator cycle:** 30 seconds
- **Signals generated per cycle:** 34 (one per symbol)
- **API response time:** < 100ms
- **Database query time:** < 50ms
- **Startup time:** 5-10 seconds

If any of these are significantly slower, check:
- System CPU/memory usage
- Database connectivity
- Network latency to Binance API
- Professional analyzer complexity

---

## Next Steps

1. ✅ Run all 7 tests above
2. ✅ Verify orchestrator is logging `[TICK N]` messages every 30 seconds
3. ✅ Check dashboard shows predictions
4. ✅ Verify database has > 1000 signals
5. ⚠️  If issues: Check logs for error patterns
6. 📝 When ready for production: Adjust MIN_CONFIDENCE and MIN_ACCURACY in config/settings.py

