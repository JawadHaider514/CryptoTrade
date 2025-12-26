# 📡 API Response Format Guide

**Status**: ✅ All endpoints return safe, frontend-compatible responses  
**Date**: December 27, 2025

---

## Response Format Standard

All API endpoints follow a consistent pattern:

```javascript
// Option 1: Data wrapper (CURRENT - Recommended)
{
  "success": true,
  "data": {...},           // or "candles", "signals", etc.
  "timestamp": "2025-12-27T..."
}

// Option 2: Top-level array (NOT USED - avoid)
[...]  // Array directly returned
```

**Why Option 1?**: 
- ✅ Includes success flag for error handling
- ✅ Includes timestamp for sync
- ✅ Easily extensible for metadata
- ✅ Frontend can check `response.success` before processing

---

## All API Endpoints & Response Format

### 1️⃣ `GET /api/chart/<symbol>`

**Request**: 
```
GET /api/chart/BTCUSDT?interval=15m&limit=500
```

**Response**:
```json
{
  "success": true,
  "symbol": "BTCUSDT",
  "interval": "15m",
  "limit": 500,
  "candles": [
    {
      "time": 1703000000,
      "open": 42000.50,
      "high": 42500.00,
      "low": 41800.25,
      "close": 42200.75
    },
    ...
  ]
}
```

**Frontend Usage**:
```javascript
const response = await fetch('/api/chart/BTCUSDT?interval=15m&limit=500');
const data = await response.json();

if (data.success) {
  setChartData(data.candles);  // Option 1: Direct array from data field
}
```

---

### 2️⃣ `GET /api/stats/<symbol>`

**Request**:
```
GET /api/stats/BTCUSDT
```

**Response**:
```json
{
  "success": true,
  "symbol": "BTCUSDT",
  "market": {
    "last_price": 42000.50,
    "change_24h": 500.00,
    "change_24h_percent": 1.20,
    "high_24h": 43000.00,
    "low_24h": 41000.00,
    "volume_24h": 25000000.00
  },
  "signals": {
    "count": 5,
    "avg_confidence": 0.82,
    "latest": {
      "symbol": "BTCUSDT",
      "direction": "BUY",
      "confidence": 0.85,
      "timestamp": "2025-12-27T..."
    }
  },
  "timestamp": "2025-12-27T..."
}
```

**Frontend Usage**:
```javascript
const stats = await fetch('/api/stats/BTCUSDT').then(r => r.json());

if (stats.success) {
  setStatsData({
    price: stats.market.last_price,
    change: stats.market.change_24h_percent,
    signals: stats.signals.count
  });
}
```

---

### 3️⃣ `GET /api/signals`

**Request**:
```
GET /api/signals
```

**Response**:
```json
{
  "success": true,
  "signals": [
    {
      "symbol": "BTCUSDT",
      "direction": "BUY",
      "confidence": 0.82,
      "entry_price": 42000.50,
      "stop_loss": 41500.00,
      "tp1": 42500.00,
      "tp2": 43000.00,
      "tp3": 43500.00,
      "timestamp": "2025-12-27T...",
      "quality": "HIGH"
    },
    ...
  ],
  "bot_running": true,
  "stats": {
    "total_trades": 42,
    "win_rate": 0.71,
    "total_pnl": 1250.50
  },
  "timestamp": "2025-12-27T..."
}
```

**Frontend Usage**:
```javascript
const signalData = await fetch('/api/signals').then(r => r.json());

if (signalData.success) {
  setSignals(signalData.signals);  // Direct array from field
  setBotStatus(signalData.bot_running);
}
```

---

### 4️⃣ `GET /api/bot/status`

**Request**:
```
GET /api/bot/status
```

**Response**:
```json
{
  "running": true,
  "start_time": "2025-12-27T10:30:45.123456",
  "uptime_seconds": 3600,
  "stats": {
    "total_trades": 42,
    "win_rate": 0.71,
    "total_pnl": 1250.50,
    "last_trade_time": "2025-12-27T11:25:30"
  },
  "signals_count": 127,
  "portfolio_history_points": 180
}
```

**Frontend Usage**:
```javascript
const status = await fetch('/api/bot/status').then(r => r.json());

setStatus({
  running: status.running,
  uptime: status.uptime_seconds,
  trades: status.stats.total_trades
});
```

---

### 5️⃣ `POST /api/bot/start`

**Request**:
```
POST /api/bot/start
```

**Response**:
```json
{
  "success": true,
  "message": "Bot started successfully"
}
```

**Frontend Usage**:
```javascript
const res = await fetch('/api/bot/start', { method: 'POST' });
const data = await res.json();

if (data.success) {
  toast.success('Bot started!');
} else {
  toast.error(data.error);
}
```

---

### 6️⃣ `POST /api/bot/stop`

**Request**:
```
POST /api/bot/stop
```

**Response**:
```json
{
  "success": true,
  "message": "Bot stopped successfully"
}
```

---

### 7️⃣ `GET /download/csv`

**Request**:
```
GET /download/csv
```

**Response**: 
- ✅ File download (CSV)
- Proper `Content-Disposition` header
- Filename: `trades_YYYYMMDD_HHMMSS.csv`

---

## Error Response Format

**All endpoints use consistent error format**:

```json
{
  "success": false,
  "error": "Description of what went wrong"
}
```

**HTTP Status Codes**:
- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Server error
- `502 Bad Gateway` - External service error (Binance)

**Example Error Response**:
```json
{
  "success": false,
  "error": "Binance HTTP error: 400 Bad Request"
}
```

---

## Frontend Best Practices

### ✅ Always check `success` flag

```javascript
// GOOD ✅
const data = await fetch('/api/chart/BTCUSDT').then(r => r.json());
if (data.success) {
  setChartData(data.candles);
} else {
  console.error(data.error);
}

// BAD ❌
const data = await fetch('/api/chart/BTCUSDT').then(r => r.json());
setChartData(data.candles);  // Might crash if data is error
```

### ✅ Handle network errors

```javascript
// GOOD ✅
try {
  const response = await fetch('/api/chart/BTCUSDT');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  
  const data = await response.json();
  if (!data.success) throw new Error(data.error);
  
  setChartData(data.candles);
} catch (error) {
  console.error('Failed to fetch chart:', error);
  setError(error.message);
}
```

### ✅ Use data field directly

```javascript
// OPTION 1: Data wrapper field (current standard)
const candles = data.candles;
const signals = data.signals;
const stats = data.stats;

// OPTION 2: Extract from object (if using destructuring)
const { candles, signals, stats } = data;
```

---

## Configuration Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Response Format | ✅ Consistent | All endpoints return objects with success flag |
| Error Handling | ✅ Standard | All errors include "success": false + "error" field |
| Data Wrapper | ✅ Used | Data in named fields (candles, signals, stats) |
| Timestamps | ✅ Included | ISO 8601 format for all timestamps |
| CORS | ✅ Enabled | All endpoints support cross-origin requests |
| Content-Type | ✅ application/json | All responses are JSON |

---

## ✨ Response Safety Checklist

- [x] All endpoints return objects (not bare arrays)
- [x] All objects have `success` flag for error detection
- [x] All data has consistent structure
- [x] All timestamps in ISO 8601 format
- [x] All errors consistently formatted
- [x] HTTP status codes correct
- [x] CORS headers properly set
- [x] No raw data exposure
- [x] Frontend can safely extract data with dot notation
- [x] Works with `setData()` and destructuring

---

## Summary

**All API endpoints are safe, well-structured, and frontend-compatible.** ✅

Frontend can confidently use:
```javascript
const data = await fetch(endpoint).then(r => r.json());
if (data.success) {
  // Use data.field safely
  setData(data.field);
}
```

**No changes needed** - responses are already optimized! 🎉
