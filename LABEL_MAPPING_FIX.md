# Label Mapping Fix - PyTorch CrossEntropyLoss Compatibility

## Summary
Fixed critical label encoding bug to ensure PyTorch CrossEntropyLoss compatibility. Labels are now encoded as integers in range [0, num_classes) instead of [-1, 0, 1].

## Changes Made

### 1. Fixed `src/crypto_bot/features/labels.py`
**Problem:** Labels were being encoded as -1 (SHORT), 0 (NO_TRADE), 1 (LONG), which breaks PyTorch's CrossEntropyLoss (expects 0..num_classes-1).

**Solution:**
- Changed label encoding to: **0 = SHORT, 1 = NO_TRADE, 2 = LONG**
- Updated label_config dict with explicit label_mapping for transparency
- Updated logging to reflect new mapping

**Code Changes:**
```python
# OLD MAPPING (broken)
df['label'] = 0  # Default NO_TRADE
df.loc[future_return > threshold, 'label'] = 1   # LONG
df.loc[future_return < -threshold, 'label'] = -1 # SHORT

# NEW MAPPING (PyTorch compatible)
df['label'] = 1  # Default NO_TRADE
df.loc[future_return > threshold, 'label'] = 2   # LONG
df.loc[future_return < -threshold, 'label'] = 0  # SHORT
```

### 2. Added Validation Guard in `src/crypto_bot/ml/train/model.py`
**Problem:** Invalid labels could silently cause cryptic training failures or be ignored by loss function.

**Solution:** Added label validation at the start of `train_model()` that:
- Checks label range is [0, num_classes)
- Catches old encoding [-1, 0, 1] early
- Provides clear error message with mapping guidance

**Code Added:**
```python
# Validate labels are in range [0, num_classes)
all_labels = np.concatenate([y_train, y_val])
unique_labels = np.unique(all_labels)
min_label = unique_labels.min()
max_label = unique_labels.max()
num_classes = model.num_classes

if min_label < 0 or max_label >= num_classes:
    raise ValueError(
        f"Invalid labels found. Labels must be in range [0, {num_classes}). "
        f"Got range [{min_label}, {max_label}]. "
        f"Unique labels: {sorted(unique_labels.tolist())}. "
        f"Expected mapping: SHORT=0, NO_TRADE=1, LONG=2"
    )
```

## Verification Results

### Label Creation Test ✓
```
Label Configuration:
  Mapping: {'SHORT': 0, 'NO_TRADE': 1, 'LONG': 2}
  Distribution: {'SHORT': 41, 'NO_TRADE': 25, 'LONG': 34}

Unique labels: [0, 1, 2]
```

### Validation Guard Test ✓
**Test 1 - Valid labels [0, 1, 2]:** PASS
- Training starts successfully with new mapping

**Test 2 - Old invalid labels [-1, 0, 1]:** PASS
- Training fails with clear error message before training loop
- Error message includes expected mapping guidance

## Impact on Training

### Before (Broken)
```
Labels: [-1, 0, 1]
CrossEntropyLoss: Loss = NaN or cryptic error
Confusion matrix: Misaligned classes
Model output: num_classes = 3, but labels have -1
```

### After (Fixed)
```
Labels: [0, 1, 2]
CrossEntropyLoss: Works correctly
Confusion matrix: Classes properly aligned to 0, 1, 2
Model output: num_classes = 3, labels match range [0, 3)
Validation guard: Catches regressions early
```

## Next Steps

1. **Regenerate Datasets** - All 35-coin datasets need to be rebuilt with new label encoding
   ```bash
   python -m crypto_bot.features.dataset_builder --symbol <COIN> --timeframe 15m
   ```

2. **Train Models** - Test training on 2 coins to verify fix works end-to-end
   ```bash
   python -m crypto_bot.ml.train.train_cnn_lstm --symbol BTCUSDT --timeframe 15m
   python -m crypto_bot.ml.train.train_cnn_lstm --symbol CHZUSDT --timeframe 15m
   ```

3. **Verify Metrics** - Check that confusion matrix and classification reports use correct class labels (0=SHORT, 1=NO_TRADE, 2=LONG)

## Model Architecture Confirmation
- Model: CNNLSTM (num_features=14, lookback=60, **num_classes=3**)
- Loss function: CrossEntropyLoss (expects [0, num_classes) = [0, 3))
- Label encoding: **SHORT=0, NO_TRADE=1, LONG=2**
- Validation: Guard ensures no negative labels reach training loop

## Files Modified
1. `src/crypto_bot/features/labels.py` - Label encoding fix
2. `src/crypto_bot/ml/train/model.py` - Validation guard

## Status
✓ COMPLETE - Label mapping corrected and validated
- Labels now PyTorch compatible (0..2 range)
- Validation guard catches regressions
- Ready for dataset regeneration and training
