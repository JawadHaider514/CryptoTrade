#!/usr/bin/env python3
"""
ML MODEL TRAINER
Train RandomForest model on historical backtesting data
Learns to predict winning signals
"""

import sqlite3
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from core.ml_features import MLFeatureExtractor

logger = logging.getLogger(__name__)

class MLModelTrainer:
    """Train ML model on backtesting data"""
    
    def __init__(self, db_path: str = "data/backtest.db", model_save_path: str = "models/signal_predictor.pkl"):
        """Initialize trainer"""
        self.db_path = db_path
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.extractor = MLFeatureExtractor()
        self.model: Optional[RandomForestClassifier] = None
        self.training_history: Dict = {}
    
    def load_signals_and_outcomes(self, symbol: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Load signals and outcomes from backtesting database
        
        Args:
            symbol: Specific symbol to load (None = all symbols)
        
        Returns:
            (signals, outcomes) tuples
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Load signals
                if symbol:
                    signals_query = f"SELECT * FROM backtest_signals WHERE symbol = ?"
                    signals_df = pd.read_sql_query(signals_query, conn, params=(symbol,))
                else:
                    signals_df = pd.read_sql_query("SELECT * FROM backtest_signals", conn)
                
                # Load outcomes
                if symbol:
                    outcomes_query = f"SELECT * FROM signal_outcomes WHERE symbol = ?"
                    outcomes_df = pd.read_sql_query(outcomes_query, conn, params=(symbol,))
                else:
                    outcomes_df = pd.read_sql_query("SELECT * FROM signal_outcomes", conn)
                
                logger.info(f"Loaded {len(signals_df)} signals and {len(outcomes_df)} outcomes")
                
                # Convert to dictionaries
                signals = signals_df.to_dict('records')
                outcomes = outcomes_df.to_dict('records')
                
                return signals, outcomes
                
        except Exception as e:
            logger.error(f"Failed to load signals: {e}")
            return [], []
    
    def prepare_training_data(self, signals: List[Dict], outcomes: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features and labels for training
        
        Returns:
            (features_df, labels_array)
        """
        # Extract features from signals
        logger.info(f"Extracting features from {len(signals)} signals...")
        features_df = self.extractor.extract_features_batch(signals)
        
        # Create labels
        labels = []
        for signal in signals:
            signal_id = signal.get('id')
            
            # Find matching outcome
            outcome = next((o for o in outcomes if o.get('signal_id') == signal_id), None)
            
            if outcome:
                # WIN = 1, LOSS = 0 (ignore TIMEOUT)
                is_win = outcome.get('result') == 'WIN'
                labels.append(1 if is_win else 0)
            else:
                labels.append(0)  # Default to loss
        
        labels = np.array(labels)
        
        logger.info(f"Prepared data: {len(features_df)} samples, {labels.sum()} wins")
        
        return features_df, labels
    
    def train(self, symbols: Optional[List[str]] = None, test_size: float = 0.2):
        """
        Train RandomForest model on backtesting data
        
        Args:
            symbols: List of symbols to train on (None = all)
            test_size: Fraction for test set
        """
        logger.info("="*70)
        logger.info("TRAINING ML MODEL ON BACKTESTING DATA")
        logger.info("="*70)
        
        # Load data
        all_signals = []
        all_outcomes = []
        
        if symbols:
            for symbol in symbols:
                signals, outcomes = self.load_signals_and_outcomes(symbol)
                all_signals.extend(signals)
                all_outcomes.extend(outcomes)
        else:
            all_signals, all_outcomes = self.load_signals_and_outcomes()
        
        if len(all_signals) < 100:
            logger.error(f"Not enough signals for training: {len(all_signals)}")
            return False
        
        # Prepare data
        X, y = self.prepare_training_data(all_signals, all_outcomes)
        
        if len(X) < 10:
            logger.error("Not enough training samples")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        logger.info(f"\nTraining set: {len(X_train)} samples ({y_train.sum()} wins)")
        logger.info(f"Test set: {len(X_test)} samples ({y_test.sum()} wins)")
        
        # Train model
        logger.info("\nTraining RandomForest (100 estimators)...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        logger.info("\nEvaluating model...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Log results
        logger.info("\n" + "="*70)
        logger.info("MODEL PERFORMANCE")
        logger.info("="*70)
        logger.info(f"Accuracy:  {accuracy:.1%}")
        logger.info(f"Precision: {precision:.1%} (true positives / predicted positives)")
        logger.info(f"Recall:    {recall:.1%} (true positives / actual positives)")
        logger.info(f"F1-Score:  {f1:.3f}")
        logger.info(f"ROC-AUC:   {auc:.3f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {tn}")
        logger.info(f"  False Positives: {fp}")
        logger.info(f"  False Negatives: {fn}")
        logger.info(f"  True Positives:  {tp}")
        
        # Feature importance
        logger.info("\n" + "="*70)
        logger.info("FEATURE IMPORTANCE (Top 10)")
        logger.info("="*70)
        
        feature_importance = self.model.feature_importances_
        feature_names = self.extractor.get_feature_names()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"{row['feature']:30s}: {row['importance']:.4f}")
        
        # Save metrics
        self.training_history = {
            'timestamp': datetime.now().isoformat(),
            'samples_train': len(X_train),
            'samples_test': len(X_test),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
        
        logger.info("\n✅ Model training complete!")
        logger.info("="*70 + "\n")
        
        return True
    
    def save_model(self) -> bool:
        """Save trained model to disk"""
        if self.model is None:
            logger.error("No model trained yet")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.extractor.get_feature_names(),
                'history': self.training_history
            }
            
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✅ Model saved to {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if not self.model_save_path.exists():
                logger.error(f"Model file not found: {self.model_save_path}")
                return False
            
            with open(self.model_save_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.training_history = model_data.get('history', {})
            
            logger.info(f"✅ Model loaded from {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, signal: Dict) -> Tuple[int, float]:
        """
        Predict outcome for a signal
        
        Returns:
            (prediction, probability) - 1 for WIN, 0 for LOSS, probability of WIN
        """
        if self.model is None:
            logger.error("No model loaded")
            return 0, 0.5
        
        try:
            # Extract features
            features = self.extractor.extract_features(signal)
            
            # Convert to array
            X = self.extractor.features_to_array(features, self.extractor.get_feature_names())
            X = X.reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]
            
            return int(prediction), float(probability)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0, 0.5


class ModelEvaluator:
    """Evaluate model performance and generate reports"""
    
    def __init__(self, model_path: str = "models/signal_predictor.pkl"):
        """Initialize evaluator"""
        self.model_path = model_path
        self.trainer = MLModelTrainer()
    
    def evaluate_on_symbol(self, symbol: str) -> Dict:
        """Evaluate model performance on specific symbol"""
        if not self.trainer.load_model():
            return {}
        
        signals, outcomes = self.trainer.load_signals_and_outcomes(symbol)
        X, y = self.trainer.prepare_training_data(signals, outcomes)
        
        if len(X) == 0:
            return {}
        
        model = self.trainer.model
        if model is None:
            return {}
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        return {
            'symbol': symbol,
            'samples': len(X),
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0
        }
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        if not self.trainer.load_model():
            return "Failed to load model"
        
        report = "\n" + "="*70 + "\n"
        report += "ML MODEL EVALUATION REPORT\n"
        report += "="*70 + "\n\n"
        
        # Training history
        if self.trainer.training_history:
            hist = self.trainer.training_history
            report += f"Training Timestamp: {hist.get('timestamp')}\n"
            report += f"Training Samples: {hist.get('samples_train')}\n"
            report += f"Test Samples: {hist.get('samples_test')}\n\n"
            
            report += "Performance Metrics:\n"
            report += f"  Accuracy:  {hist.get('accuracy', 0):.1%}\n"
            report += f"  Precision: {hist.get('precision', 0):.1%}\n"
            report += f"  Recall:    {hist.get('recall', 0):.1%}\n"
            report += f"  F1-Score:  {hist.get('f1_score', 0):.3f}\n"
            report += f"  AUC-ROC:   {hist.get('auc', 0):.3f}\n"
        
        report += "\n" + "="*70 + "\n"
        return report


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Train and test ML model"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Train model
    trainer = MLModelTrainer()
    
    if trainer.train():
        # Save model
        trainer.save_model()
        
        # Print report
        evaluator = ModelEvaluator()
        print(evaluator.generate_report())
    else:
        print("❌ Training failed")
