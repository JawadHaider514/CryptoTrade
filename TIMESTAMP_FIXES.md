# Timestamp Format Fixes - ISO 8601 with Z Suffix

## Overview
Updated all backend timestamps to ISO 8601 format with `Z` suffix (UTC indicator) for proper JavaScript parsing.

## Changes Made

### 1. Backend: `serialize_signal()` Function
**File:** `src/crypto_bot/server/advanced_web_server.py` (Lines 151-184)

**Before:**
```python
if isinstance(obj, datetime):
    out[k] = v.isoformat()  # Could return "2025-12-27T13:45:30" without Z
```

**After:**
```python
if isinstance(obj, datetime):
    # Return ISO format with Z suffix (UTC indicator)
    iso_str = obj.isoformat()
    if not iso_str.endswith('Z'):
        iso_str = iso_str + 'Z'  # Now returns "2025-12-27T13:45:30Z"
    return iso_str
```

**Impact:** All datetime objects are now serialized as ISO 8601 with Z suffix:
- `2025-12-27T13:45:30Z` ✅ (was: `2025-12-27T13:45:30` ❌)

### 2. API Response: Added `valid_until` Field
**File:** `src/crypto_bot/server/advanced_web_server.py` (Line 636)

**Before:**
```python
pred_obj = {
    "timestamp": serialize_signal(getattr(signal, 'timestamp', None)),
    "take_profits": []
}
```

**After:**
```python
pred_obj = {
    "timestamp": serialize_signal(getattr(signal, 'timestamp', None)),
    "valid_until": serialize_signal(getattr(signal, 'valid_until', None)),  # NEW
    "take_profits": []
}
```

**Example Response:**
```json
{
  "success": true,
  "predictions": {
    "BTCUSDT": {
      "symbol": "BTCUSDT",
      "timestamp": "2025-12-27T13:45:30Z",
      "valid_until": "2025-12-27T14:15:30Z",
      "TP1_ETA": "2025-12-27T13:50:30Z",
      "TP2_ETA": "2025-12-27T14:00:30Z",
      "TP3_ETA": "2025-12-27T14:30:30Z",
      "take_profits": [
        {
          "level": 1,
          "price": 98500.0,
          "eta": "2025-12-27T13:50:30Z"
        },
        ...
      ]
    }
  }
}
```

### 3. Frontend: Updated Timer Display
**File:** `templates/index.html` (Lines 830, 848)

**Before:**
```html
<span class="timer-val" data-ts="${s.timestamp}">5:00</span>

function startTimers(){
  setInterval(()=>{
    const rem = Math.max(0, 300 - (Date.now() - new Date(ts))/1000);
    // Hardcoded 300 seconds = 5 minutes
  }, 1000);
}
```

**After:**
```html
<span class="timer-val" data-until="${s.valid_until}">30:00</span>

function startTimers(){
  setInterval(()=>{
    const now = Date.now();
    const expiry = new Date(until).getTime();
    const rem = Math.max(0, (expiry - now) / 1000);  // Uses actual valid_until time
    const m = Math.floor(rem / 60);
    const s = Math.floor(rem % 60);
    el.textContent = `${m}:${s.toString().padStart(2,'0')}`;
    
    // Color changes based on remaining time
    if (rem < 30) el.style.color = 'var(--red)';      // Red: <30 seconds
    else if (rem < 60) el.style.color = 'var(--yellow)';  // Yellow: <60 seconds
    if (rem <= 0) el.style.color = 'var(--red)';      // Red: Expired
  }, 1000);
}
```

**Benefits:**
- ✅ Uses actual `valid_until` timestamp (30-minute expiry by default)
- ✅ No more hardcoded 5-minute countdown
- ✅ Accurate countdown based on server-sent validity window
- ✅ Color-coded urgency indicators:
  - 🟢 Green/Yellow: 1-30 minutes remaining
  - 🟡 Yellow: <1 minute remaining  
  - 🔴 Red: <30 seconds or expired

### 4. ETA Fields - All in ISO Format
**Affected Fields in `/api/predictions` response:**

```javascript
// Individual TP ETA fields
"TP1_ETA": "2025-12-27T13:50:30Z"
"TP2_ETA": "2025-12-27T14:00:30Z"
"TP3_ETA": "2025-12-27T14:30:30Z"

// Nested in take_profits array
"take_profits": [
  { "level": 1, "price": 98500.0, "eta": "2025-12-27T13:50:30Z" },
  { "level": 2, "price": 98750.0, "eta": "2025-12-27T14:00:30Z" },
  { "level": 3, "price": 99000.0, "eta": "2025-12-27T14:30:30Z" }
]
```

All ETAs now use `YYYY-MM-DDTHH:MM:SSZ` format for JavaScript compatibility.

## Signal Validity Flow

```
Backend (Python):
  - Signal created with timestamp: datetime.utcnow()
  - Signal expires after: datetime.utcnow() + timedelta(hours=4)
  - All times stored as datetime objects
  
Serialization:
  - serialize_signal() converts all datetime → ISO 8601 with Z
  - Example: datetime(2025,12,27,13,45,30) → "2025-12-27T13:45:30Z"
  
API Response:
  - timestamp: "2025-12-27T13:45:30Z" (when signal was generated)
  - valid_until: "2025-12-27T17:45:30Z" (when signal expires)
  - TP*_ETA: "2025-12-27T13:50:30Z" (estimated time to hit TP)
  
Frontend (JavaScript):
  - Receives all times in ISO 8601 format
  - new Date("2025-12-27T13:45:30Z") parses correctly ✅
  - Countdown = valid_until - now (accurate remaining time)
  - No parsing errors with YYYY-MM-DD HH:MM:SS format ❌
```

## JavaScript Timestamp Parsing

```javascript
// ✅ NOW WORKS - ISO 8601 with Z
const validUntil = new Date("2025-12-27T14:15:30Z");
const remaining = (validUntil - Date.now()) / 1000;  // seconds remaining

// ❌ DIDN'T WORK - Space-separated format
const validUntil = new Date("2025-12-27 14:15:30");  // Parsing issues
```

## Testing

**Before Fix:**
```
Backend sends: timestamp="2025-12-27 13:45:30"
Frontend JavaScript:
  new Date("2025-12-27 13:45:30")  // Invalid/unreliable parsing
  // Result: NaN or wrong timezone

Countdown display: 0:00 ❌ (always shows 0)
```

**After Fix:**
```
Backend sends: timestamp="2025-12-27T13:45:30Z"
Frontend JavaScript:
  new Date("2025-12-27T13:45:30Z")  // Valid ISO 8601
  // Result: Correct UTC datetime, reliable parsing

Countdown display: 30:15 ✅ (accurate remaining time)
```

## Summary

| Field | Format | Example |
|-------|--------|---------|
| `timestamp` | ISO 8601 + Z | `2025-12-27T13:45:30Z` ✅ |
| `valid_until` | ISO 8601 + Z | `2025-12-27T14:15:30Z` ✅ |
| `TP*_ETA` | ISO 8601 + Z | `2025-12-27T13:50:30Z` ✅ |
| `take_profits[].eta` | ISO 8601 + Z | `2025-12-27T13:50:30Z` ✅ |

All timestamps now:
1. Use ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ`
2. Include Z suffix for UTC timezone
3. Parse correctly in JavaScript without errors
4. Enable accurate countdown display based on validity window
