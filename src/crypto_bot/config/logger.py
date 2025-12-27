#!/usr/bin/env python3
"""
Logging Configuration
====================
Centralized logging setup for all services
"""

import logging
import sys
from pathlib import Path


def setup_logger(name, level=logging.INFO, log_file=None):
    """
    Setup a configured logger
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to log to
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            log_path = Path(log_file).parent
            log_path.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file {log_file}: {e}")
    
    return logger


def get_logger(name):
    """Get an existing logger by name"""
    return logging.getLogger(name)
