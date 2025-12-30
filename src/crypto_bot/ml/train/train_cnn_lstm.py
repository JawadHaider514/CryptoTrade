# =====================================================
# CNN-LSTM Training Script - COMPLETE & PRODUCTION READY
# File: crypto_bot/ml/train/train_cnn_lstm.py
# =====================================================

"""
CNN-LSTM training script with CLI - FIXED & COMPLETE.

CLI Usage:
    python -m crypto_bot.ml.train.train_cnn_lstm --symbol BTCUSDT --timeframe 5m --epochs 25 --lookback 60
    python -m crypto_bot.ml.train.train_cnn_lstm --symbol ETHUSDT --timeframe 15m --epochs 50
    python -m crypto_bot.ml.train.train_cnn_lstm --help

Training Process:
    1. Load dataset from parquet file
    2. Scale features using StandardScaler
    3. Create sequences for LSTM
    4. Split into train/val/test
    5. Train CNN-LSTM model
    6. Evaluate and save results
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# =====================================================
# CONFIGURATION & SETUP
# =====================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Device logging is done in train_cnn_lstm() for each call

# Project paths
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_trading_system" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Dataset directory: {DATASET_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")


# =====================================================
# PART 1: DATA LOADING & PREPROCESSING
# =====================================================

class DataProcessor:
    """
    Load, preprocess, and split data for CNN-LSTM training.
    Handles scaling, sequence creation, and train/val/test splits.
    """
    
    def __init__(self, dataset_path: str, meta_path: str, lookback: int = 60):
        """
        Initialize data processor.
        
        Args:
            dataset_path: Path to parquet dataset file
            meta_path: Path to metadata JSON file
            lookback: Lookback window size for sequences
        """
        self.dataset_path = Path(dataset_path)
        self.meta_path = Path(meta_path)
        self.lookback = lookback
        self.scaler: StandardScaler = StandardScaler()
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")
    
    def load_dataset(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Load parquet dataset and JSON metadata.
        
        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        logger.info(f"Loading dataset from {self.dataset_path}")
        df = pd.read_parquet(self.dataset_path)
        
        logger.info(f"Loading metadata from {self.meta_path}")
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Features: {len(meta.get('features', []))} columns")
        
        return df, meta
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping sequences for LSTM input.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
        
        Returns:
            Tuple of (X_sequences, y_sequences) where:
            - X_sequences shape: (n_sequences, lookback, n_features)
            - y_sequences shape: (n_sequences,)
        """
        # Convert to numpy arrays if needed
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float32)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.int64)
        
        X_sequences: list = []
        y_sequences: list = []
        
        for i in range(len(X) - self.lookback):
            X_sequences.append(X[i:i + self.lookback])
            y_sequences.append(y[i + self.lookback])
        
        X_arr = np.array(X_sequences, dtype=np.float32)
        y_arr = np.array(y_sequences, dtype=np.int64)
        
        logger.info(f"Created sequences - X: {X_arr.shape}, y: {y_arr.shape}")
        
        return X_arr, y_arr
    
    def process_and_split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Any]:
        """
        Load, scale, create sequences, and split data.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        
        Returns:
            Dictionary containing:
            - X_train, y_train, X_val, y_val, X_test, y_test (numpy arrays)
            - scaler (StandardScaler object)
            - meta (metadata dict)
            - feature_cols (list of feature names)
        """
        # Load data
        df, meta = self.load_dataset()
        
        # Extract features and labels
        feature_cols = meta.get('features', [])
        label_col = meta.get('label_column', 'label')
        
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in dataset. "
                f"Available columns: {df.columns.tolist()}. "
                f"Check meta.json 'label_column' setting."
            )
        
        X = df[feature_cols].values.astype(np.float32)
        y = np.asarray(df[label_col].values, dtype=np.int64)
        
        logger.info(f"Extracted features: X {X.shape}, y {y.shape}")
        logger.info(f"Label column: '{label_col}'")
        
        # Map labels from -1/0/1 to 0/1/2 if needed
        unique_labels = np.unique(y)
        logger.info(f"Unique labels before mapping: {sorted(unique_labels.tolist())}")
        
        if -1 in unique_labels:
            logger.info("Converting labels from [-1, 0, 1] to [0, 1, 2]")
            label_mapping_old = {-1: 1, 0: 0, 1: 2}  # OLD: -1=SHORT -> NEW: 1=NO_TRADE, etc
            y = np.array([label_mapping_old.get(label, label) for label in y], dtype=np.int64)
            unique_labels = np.unique(y)
            logger.info(f"Unique labels after mapping: {sorted(unique_labels.tolist())}")
        
        # Guard: ensure all labels are in valid range [0, num_classes)
        if np.any(y < 0):
            raise ValueError(
                f"Invalid negative labels found: {sorted(np.unique(y[y < 0]).tolist())}. "
                f"All labels must be >= 0. Expected mapping: SHORT=0, NO_TRADE=1, LONG=2"
            )
        
        if np.any(y >= 3):
            raise ValueError(
                f"Invalid labels >= 3 found: {sorted(np.unique(y[y >= 3]).tolist())}. "
                f"Expected labels in range [0, 3). Expected mapping: SHORT=0, NO_TRADE=1, LONG=2"
            )
        
        # Handle NaN/Inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("Found NaN/Inf values - replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        logger.info("Scaling features with StandardScaler")
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        logger.info("Creating sequences")
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        # Calculate split indices
        n_total = len(X_seq)
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))
        
        # Split data
        X_train = X_seq[:train_end]
        y_train = y_seq[:train_end]
        
        X_val = X_seq[train_end:val_end]
        y_val = y_seq[train_end:val_end]
        
        X_test = X_seq[val_end:]
        y_test = y_seq[val_end:]
        
        logger.info(f"Train set: {X_train.shape}")
        logger.info(f"Val set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': self.scaler,
            'meta': meta,
            'feature_cols': feature_cols,
        }


# =====================================================
# PART 2: CNN-LSTM MODEL ARCHITECTURE
# =====================================================

class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture for time series prediction.
    
    Components:
    - CNN: Extracts local spatial features using Conv1d
    - LSTM: Learns temporal dependencies
    - Attention: Focuses on important timesteps
    - Fusion: Combines CNN and LSTM features
    """
    
    def __init__(
        self,
        num_features: int,
        lookback: int = 60,
        num_classes: int = 3,
        cnn_filters: Tuple[int, int, int] = (64, 128, 256),
        lstm_hidden: int = 100,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize CNN-LSTM model.
        
        Args:
            num_features: Number of input features
            lookback: Sequence length (lookback window)
            num_classes: Number of output classes (3 for UP/DOWN/HOLD)
            cnn_filters: Number of filters for each CNN layer
            lstm_hidden: LSTM hidden state size
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_features = num_features
        self.lookback = lookback
        self.num_classes = num_classes
        
        # ===== CNN Feature Extractor =====
        self.cnn = nn.Sequential(
            # Layer 1: Conv -> ReLU -> BatchNorm -> MaxPool -> Dropout
            nn.Conv1d(num_features, cnn_filters[0], kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(cnn_filters[0]),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            # Layer 2: Conv -> ReLU -> BatchNorm -> MaxPool -> Dropout
            nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(cnn_filters[1]),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            # Layer 3: Conv -> ReLU -> AdaptiveAvgPool
            nn.Conv1d(cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # ===== Bidirectional LSTM =====
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        # ===== Attention Mechanism =====
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=4,
            batch_first=True,
            dropout=dropout
        )
        
        # ===== Fusion & Output Layers =====
        fusion_input_size = cnn_filters[2] + lstm_hidden * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, lookback, num_features)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # CNN path: transpose to (batch, features, lookback)
        x_cnn = self.cnn(x.transpose(1, 2))  # (batch, cnn_filters[-1], 1)
        x_cnn = x_cnn.squeeze(-1)  # (batch, cnn_filters[-1])
        
        # LSTM path: process sequence
        lstm_out, _ = self.lstm(x)  # (batch, lookback, lstm_hidden*2)
        
        # Attention: focus on important timesteps
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, lookback, lstm_hidden*2)
        lstm_final = attn_out[:, -1, :]  # Take last timestep: (batch, lstm_hidden*2)
        
        # Fusion: concatenate and classify
        combined = torch.cat([x_cnn, lstm_final], dim=1)  # (batch, cnn_filters[-1] + lstm_hidden*2)
        output = self.fusion(combined)  # (batch, num_classes)
        
        return output


# =====================================================
# PART 3: TRAINING & EVALUATION FUNCTIONS
# =====================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, float]:
    """
    Train model for one epoch with optional mixed precision.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass with optional mixed precision
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Calculate metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate model on validation set with optional mixed precision.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def evaluate_on_test(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    device: torch.device = DEVICE
) -> Dict[str, Any]:
    """
    Evaluate model on test set with detailed metrics.
    
    Args:
        model: Trained neural network model
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size
        device: Device to run on
    
    Returns:
        Dictionary with accuracy, precision, recall, f1, confusion matrix
    """
    model.eval()
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    # Collect predictions with optional mixed precision
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(y_batch.cpu().numpy().tolist())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Confusion matrix with explicit label order [0=SHORT, 1=NO_TRADE, 2=LONG]
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2])
    
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }


# =====================================================
# PART 4: MAIN TRAINING PIPELINE
# =====================================================

def train_cnn_lstm(
    symbol: str,
    timeframe: str = '5m',
    epochs: int = 25,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    lookback: int = 60,
    early_stopping_patience: int = 5,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Complete training pipeline for CNN-LSTM model with GPU support.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Candle interval (e.g., '5m', '15m')
        epochs: Number of training epochs
        batch_size: Training batch size (default 8 for GPU memory efficiency)
        learning_rate: Optimizer learning rate
        lookback: Sequence length for LSTM
        early_stopping_patience: Patience for early stopping
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with training results, metrics, and paths
    """
    symbol = symbol.upper()
    
    result: Dict[str, Any] = {
        'symbol': symbol,
        'timeframe': timeframe,
        'status': 'pending',
        'model_path': None,
        'metrics_path': None,
        'metrics': None,
        'error': None,
    }
    
    try:
        # ===== SETUP =====
        dataset_path = DATASET_DIR / symbol / f"{timeframe}_dataset.parquet"
        meta_path = DATASET_DIR / symbol / "meta.json"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        logger.info(f"{'='*70}")
        logger.info(f"Training CNN-LSTM: {symbol} {timeframe}")
        logger.info(f"{'='*70}")
        
        # ===== GPU OPTIMIZATIONS =====
        train_device = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {train_device}")
        
        if train_device.type == 'cuda':
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True  # Auto-tune kernels
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')  # Use TF32 for matmul
            logger.info("CUDA optimizations enabled: benchmark=True, TF32 matmul")
        
        # ===== LOAD & PREPROCESS DATA =====
        logger.info("Loading and preprocessing data...")
        processor = DataProcessor(str(dataset_path), str(meta_path), lookback=lookback)
        data_dict = processor.process_and_split()
        
        X_train: np.ndarray = data_dict['X_train']
        y_train: np.ndarray = data_dict['y_train']
        X_val: np.ndarray = data_dict['X_val']
        y_val: np.ndarray = data_dict['y_val']
        X_test: np.ndarray = data_dict['X_test']
        y_test: np.ndarray = data_dict['y_test']
        data_scaler: StandardScaler = data_dict['scaler']
        meta: Dict = data_dict['meta']
        feature_cols = data_dict['feature_cols']
        
        # ===== INITIALIZE MODEL =====
        logger.info("Initializing CNN-LSTM model...")
        num_features = X_train.shape[2]
        model = CNNLSTM(
            num_features=num_features,
            lookback=lookback,
            num_classes=3,
        ).to(train_device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        # ===== TRAINING SETUP =====
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Add class weights to handle imbalance
        # SHORT (class 0): 2.0 - underrepresented
        # NO_TRADE (class 1): 1.0 - majority class
        # LONG (class 2): 2.0 - underrepresented
        class_weights = torch.tensor([2.0, 1.0, 2.0]).to(train_device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Class weights: SHORT={class_weights[0]}, NO_TRADE={class_weights[1]}, LONG={class_weights[2]}")
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Mixed precision scaler for faster training on GPU
        scaler = torch.cuda.amp.GradScaler() if train_device.type == 'cuda' else None
        if scaler:
            logger.info("Mixed precision (AMP) enabled")
        
        # ===== DATA LOADERS =====
        # DataLoader optimizations for GPU
        loader_pin_memory = train_device.type == 'cuda'
        loader_num_workers = 2 if train_device.type == 'cuda' else 0
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=loader_pin_memory,
            num_workers=loader_num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=loader_pin_memory,
            num_workers=loader_num_workers
        )
        
        # ===== TRAINING LOOP =====
        logger.info(f"Starting training ({epochs} epochs, batch_size={batch_size})")
        history: Dict[str, list] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, train_device, scaler)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, train_device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model in per_coin structure
                output_dir = MODELS_DIR / "per_coin" / symbol / timeframe
                output_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = output_dir / "cnn_lstm_best.pth"
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model at epoch {epoch + 1}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch [{epoch+1:3d}/{epochs}] | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
        
        # ===== LOAD BEST MODEL & EVALUATE =====
        # Use per_coin structure for compatibility with registry
        output_dir = MODELS_DIR / "per_coin" / symbol / timeframe
        best_model_path = output_dir / "cnn_lstm_best.pth"
        
        if best_model_path.exists():
            logger.info(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=train_device))
        
        logger.info("Evaluating on test set...")
        metrics = evaluate_on_test(model, X_test, y_test, batch_size=batch_size, device=train_device)
        
        # ===== SAVE MODEL & ARTIFACTS =====
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model with registry-compatible name
        model_path = output_dir / "cnn_lstm_v1.pth"
        logger.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
        
        # Save scaler
        scaler_path = output_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(data_scaler, f)
        logger.info(f"Saving scaler to {scaler_path}")
        
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
                'feature_names': feature_cols,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
            },
        }
        
        logger.info(f"Saving metrics to {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(full_metrics, f, indent=2)
        
        # Save metadata compatible with registry
        meta_path = output_dir / "meta.json"
        registry_meta = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_version": "cnn_lstm_v1",
            "lookback": lookback,
            "num_features": num_features,
            "dataset_info": {
                "num_features": num_features,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
            },
            "test_metrics": metrics,
        }
        with open(meta_path, 'w') as f:
            json.dump(registry_meta, f, indent=2)
        logger.info(f"Saving metadata to {meta_path}")
        
        # ===== RESULTS =====
        result['status'] = 'success'
        result['model_path'] = str(model_path)
        result['metrics_path'] = str(metrics_path)
        result['metrics'] = metrics
        
        logger.info(f"{'='*70}")
        logger.info(f"✅ Training complete for {symbol} {timeframe}")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"❌ Training failed for {symbol} {timeframe}: {e}", exc_info=True)
    
    return result


# =====================================================
# PART 5: CLI INTERFACE
# =====================================================

def main() -> int:
    """
    Main entry point for CLI.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description='Train CNN-LSTM model for cryptocurrency price prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model
  python -m crypto_bot.ml.train.train_cnn_lstm --symbol BTCUSDT --timeframe 5m --epochs 25
  
  # Train with custom parameters
  python -m crypto_bot.ml.train.train_cnn_lstm --symbol ETHUSDT --timeframe 15m --epochs 50 --batch_size 128
  
  # Train multiple models
  for symbol in BTCUSDT ETHUSDT SOLUSDT; do
    python -m crypto_bot.ml.train.train_cnn_lstm --symbol $symbol --timeframe 5m
  done
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading symbol (e.g., BTCUSDT, ETHUSDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='5m',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='Candle interval (default: 5m)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Number of training epochs (default: 25)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for training (default: 256)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Optimizer learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=60,
        help='Lookback window size for sequences (default: 60)'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=5,
        help='Early stopping patience in epochs (default: 5)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Training CNN-LSTM")
    logger.info(f"  Symbol: {args.symbol}")
    logger.info(f"  Timeframe: {args.timeframe}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Lookback: {args.lookback}")
    logger.info(f"  Device: {DEVICE}")
    
    # Run training
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
    print(f"Symbol:   {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Status:   {result['status'].upper()}")
    print()
    
    if result['status'] == 'success':
        metrics = result['metrics']
        print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall:    {metrics['recall']:.4f}")
        print(f"Test F1:        {metrics['f1']:.4f}")
        print()
        print(f"Model saved to:   {result['model_path']}")
        print(f"Metrics saved to: {result['metrics_path']}")
        print()
        print("✅ Training successful!")
    else:
        print(f"❌ Error: {result['error']}")
    
    print("="*80 + "\n")
    
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())