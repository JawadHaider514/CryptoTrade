"""
CNN-LSTM training script with CLI.

CLI Usage:
    python -m crypto_bot.ml.train.train_cnn_lstm --symbol BTCUSDT --timeframe 15m --epochs 25 --lookback 60
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch

from crypto_bot.ml.train.model import CNNLSTM, load_dataset_with_splits, train_model, evaluate_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Project root
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"


def train_cnn_lstm(
    symbol: str,
    timeframe: str = '15m',
    epochs: int = 25,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    lookback: int = 60,
    early_stopping_patience: int = 5,
) -> dict:
    """
    Train CNN-LSTM model for a symbol/timeframe.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Candle interval (e.g., '15m', '1h')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        lookback: Lookback window size
        early_stopping_patience: Early stopping patience
    
    Returns:
        Dict with results and metrics
    """
    symbol = symbol.upper()
    
    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'status': 'pending',
        'model_path': None,
        'metrics_path': None,
        'metrics': None,
        'error': None,
    }
    
    try:
        # Paths
        dataset_path = DATASET_DIR / symbol / f"{timeframe}_dataset.parquet"
        meta_path = DATASET_DIR / symbol / "meta.json"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        logger.info(f"Loading dataset for {symbol} {timeframe}")
        
        # Load data
        data_dict, meta = load_dataset_with_splits(
            str(dataset_path),
            str(meta_path),
            lookback=lookback,
        )
        
        # Initialize model
        logger.info("Initializing CNN-LSTM model")
        num_features = len(meta.get('features', []))
        model = CNNLSTM(
            num_features=num_features,
            lookback=lookback,
            num_classes=3,
        ).to(DEVICE)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        logger.info(f"Starting training ({epochs} epochs)")
        history = train_model(
            model,
            (data_dict['X_train'], data_dict['y_train']),
            (data_dict['X_val'], data_dict['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
        )
        
        # Evaluate
        logger.info("Evaluating on test set")
        metrics = evaluate_model(
            model,
            data_dict['X_test'],
            data_dict['y_test'],
            batch_size=batch_size,
        )
        
        # Save model
        output_dir = MODELS_DIR / symbol / timeframe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "cnn_lstm.pt"
        logger.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        full_metrics = {
            'symbol': symbol,
            'timeframe': timeframe,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'hyperparameters': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'lookback': lookback,
                'early_stopping_patience': early_stopping_patience,
            },
            'training_history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'train_acc': [float(x) for x in history['train_acc']],
                'val_loss': [float(x) for x in history['val_loss']],
                'val_acc': [float(x) for x in history['val_acc']],
            },
            'test_metrics': metrics,
            'dataset_info': {
                'num_features': num_features,
                'feature_names': meta.get('features', []),
                'train_samples': len(data_dict['X_train']),
                'val_samples': len(data_dict['X_val']),
                'test_samples': len(data_dict['X_test']),
            },
        }
        
        logger.info(f"Saving metrics to {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(full_metrics, f, indent=2)
        
        result['status'] = 'success'
        result['model_path'] = str(model_path)
        result['metrics_path'] = str(metrics_path)
        result['metrics'] = metrics
        
        logger.info(f"✅ Training complete for {symbol} {timeframe}")
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"❌ Training failed for {symbol} {timeframe}: {e}", exc_info=True)
    
    return result


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Train CNN-LSTM model for cryptocurrency predictions'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading symbol (e.g., BTCUSDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        help='Candle interval (default 15m)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=60,
        help='Lookback window size'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=5,
        help='Early stopping patience'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Training CNN-LSTM for {args.symbol} {args.timeframe}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model directory: {MODELS_DIR}")
    
    # Train
    result = train_cnn_lstm(
        symbol=args.symbol,
        timeframe=args.timeframe,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lookback=args.lookback,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Status: {result['status'].upper()}")
    print()
    
    if result['status'] == 'success':
        metrics = result['metrics']
        print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall:    {metrics['recall']:.4f}")
        print()
        print(f"Model saved to: {result['model_path']}")
        print(f"Metrics saved to: {result['metrics_path']}")
    else:
        print(f"Error: {result['error']}")
    
    print("="*80)
    
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
