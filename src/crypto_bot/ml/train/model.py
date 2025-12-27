"""
CNN-LSTM model architecture and training.

Model: Convolutional Neural Network + LSTM for time series classification
Input: (batch, lookback, num_features) -> 14 technical indicators
Output: 3-class classification (LONG, SHORT, NO_TRADE)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for time series classification.
    
    Architecture:
    - Conv1d layers for feature extraction
    - LSTM layers for temporal dependencies
    - Dense layers for classification
    """
    
    def __init__(self, num_features: int = 14, lookback: int = 60, num_classes: int = 3):
        """
        Args:
            num_features: Number of input features (14 indicators)
            lookback: Lookback window size (60 by default)
            num_classes: Number of output classes (3: LONG/SHORT/NO_TRADE)
        """
        super(CNNLSTM, self).__init__()
        
        self.num_features = num_features
        self.lookback = lookback
        self.num_classes = num_classes
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layer
        # After 2 maxpools (2x): lookback // 4
        cnn_output_len = lookback // 4
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        
        # Dense layers
        self.dense1 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.dense2 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        
        self.output = nn.Linear(32, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, num_features, lookback)
        
        Returns:
            Output logits (batch, num_classes)
        """
        # CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        x, (h, c) = self.lstm(x)
        # Take last output
        x = x[:, -1, :]
        
        # Dense
        x = self.dense1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        
        x = self.dense2(x)
        x = self.relu4(x)
        
        x = self.output(x)
        
        return x


def load_dataset_with_splits(
    dataset_path: str,
    meta_path: str,
    lookback: int = 60,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Load dataset and split into train/val/test.
    
    Args:
        dataset_path: Path to parquet dataset
        meta_path: Path to meta.json
        lookback: Lookback window size
    
    Returns:
        Tuple of:
        - dict with 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        - metadata dict
    """
    # Load dataset
    df = pd.read_parquet(dataset_path)
    
    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Get split indices
    time_split = meta.get('time_split', {})
    train_end = time_split.get('train_end_idx', int(len(df) * 0.7))
    val_end = time_split.get('val_end_idx', int(len(df) * 0.85))
    
    # Get feature columns
    feature_cols = meta.get('features', [])
    
    # Extract features and labels
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create sequences with lookback window
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    
    X_seq = np.array(X_seq, dtype=np.float32)  # (samples, lookback, features)
    y_seq = np.array(y_seq, dtype=np.int64)
    
    # Split based on time indices
    # Adjust for lookback offset
    adjusted_train_end = train_end - lookback
    adjusted_val_end = val_end - lookback
    
    X_train = X_seq[:adjusted_train_end]
    y_train = y_seq[:adjusted_train_end]
    
    X_val = X_seq[adjusted_train_end:adjusted_val_end]
    y_val = y_seq[adjusted_train_end:adjusted_val_end]
    
    X_test = X_seq[adjusted_val_end:]
    y_test = y_seq[adjusted_val_end:]
    
    logger.info(f"Dataset shapes:")
    logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
    }, meta


def train_model(
    model: nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 25,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 5,
) -> Dict[str, list]:
    """
    Train CNN-LSTM model.
    
    Args:
        model: CNN-LSTM model
        train_data: Tuple of (X_train, y_train)
        val_data: Tuple of (X_val, y_val)
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        early_stopping_patience: Early stopping patience
    
    Returns:
        Dict with training history
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.from_numpy(X_train[i:i+batch_size]).transpose(1, 2).to(DEVICE)
            batch_y = torch.from_numpy(y_train[i:i+batch_size]).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().detach().numpy())
            train_targets.extend(batch_y.cpu().numpy())
        
        train_loss /= (len(X_train) // batch_size + 1)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = torch.from_numpy(X_val[i:i+batch_size]).transpose(1, 2).to(DEVICE)
                batch_y = torch.from_numpy(y_val[i:i+batch_size]).to(DEVICE)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= (len(X_val) // batch_size + 1)
        val_acc = accuracy_score(val_targets, val_preds)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return history


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size
    
    Returns:
        Dict with metrics
    """
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = torch.from_numpy(X_test[i:i+batch_size]).transpose(1, 2).to(DEVICE)
            batch_y = torch.from_numpy(y_test[i:i+batch_size]).to(DEVICE)
            
            outputs = model(batch_X)
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    preds = np.array(preds)
    targets = np.array(targets)
    
    # Metrics
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(targets, preds, labels=[0, 1, -1])
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': cm.tolist(),
        'class_report': classification_report(targets, preds, output_dict=True, zero_division=0),
    }
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    
    return metrics
