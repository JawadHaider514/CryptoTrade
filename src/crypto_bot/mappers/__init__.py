#!/usr/bin/env python3
"""
Mappers / Adapters
==================
Normalize different analyzer outputs into unified prediction schema.

Available Mappers:
- PredictionMapper: Convert SignalModel, ProAnalyzer, Dashboard to unified schema
"""

from .prediction_mapper import (
    PredictionMapper,
    map_to_prediction
)

__all__ = [
    'PredictionMapper',
    'map_to_prediction',
]
