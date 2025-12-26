#!/usr/bin/env python3
"""
TRAIN ML SIGNAL PREDICTOR
Uses real backtest data to train a model that predicts if signals will win/lose
"""

import sqlite3
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_signal_predictor():
    """Train ML model on real backtest data"""
    
    print("\n" + "="*70)
    print("TRAINING ML SIGNAL PREDICTOR MODEL")
    print("="*70)
    
    # Load data from database
    conn = sqlite3.connect("data/backtest.db")
    c = conn.cursor()
    
    # Get all signals with outcomes
    c.execute("""
        SELECT 
            bs.confluence_score,
            bs.rsi,
            bs.macd,
            bs.volume_ratio,
            bs.ema_9,
            CASE WHEN so.result = 'WIN' THEN 1 ELSE 0 END as is_win
        FROM backtest_signals bs
        LEFT JOIN signal_outcomes so ON bs.id = so.signal_id
    """)
    
    data = np.array(c.fetchall(), dtype=float)
    
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Target (win/loss)
    
    print(f"\n📊 Training data loaded:")
    print(f"   Total samples: {len(X)}")
    print(f"   Win rate: {np.mean(y)*100:.1f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print(f"\n🤖 Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model training complete:")
    print(f"   Training accuracy: {model.score(X_train_scaled, y_train)*100:.1f}%")
    print(f"   Test accuracy: {accuracy*100:.1f}%")
    
    print(f"\n📊 Feature importance:")
    feature_names = ["confluence_score", "rsi", "macd", "volume_ratio", "ema_9"]
    for name, importance in sorted(zip(feature_names, model.feature_importances_), 
                                   key=lambda x: x[1], reverse=True):
        print(f"   {name}: {importance*100:.1f}%")
    
    print(f"\n📈 Detailed classification report:")
    print(classification_report(y_test, y_pred, target_names=["LOSS", "WIN"]))
    
    # Save model
    model_path = "models/signal_predictor.pkl"
    Path("models").mkdir(exist_ok=True)
    
    model_data = {
        "model": model,
        "scaler": scaler,
        "features": feature_names,
        "accuracy": accuracy,
        "timestamp": str(__import__('datetime').datetime.now())
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ MODEL SAVED:")
    print(f"   Path: {model_path}")
    print(f"   Size: {Path(model_path).stat().st_size / 1024:.1f} KB")
    print(f"   Accuracy: {accuracy*100:.1f}%")
    
    conn.close()

if __name__ == "__main__":
    train_signal_predictor()
