#!/usr/bin/env python3
"""
Signal Domain Models - Pydantic v2 + Pylance FIXED
=================================================
Fixes:
- No invalid Field(3,3)
- Fixed-length list enforced via Annotated + Field(min_length/max_length)
- Uses Literal for direction (LONG/SHORT)
- ISO datetime serialization via json_encoders + to_dict()
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Literal, Annotated

from pydantic import BaseModel, Field


class TakeProfit(BaseModel):
    """Single Take Profit level with estimated time"""
    level: int = Field(ge=1, le=3)  # 1, 2, or 3
    price: float = Field(gt=0)
    eta: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# Fixed-length list type (exactly 3)
TakeProfits3 = Annotated[List[TakeProfit], Field(min_length=3, max_length=3)]


class SignalModel(BaseModel):
    """Complete trading signal model"""

    # Core identification
    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Direction and entry
    direction: Literal["LONG", "SHORT"]
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)

    # Take profits (EXACTLY 3 levels)
    take_profits: TakeProfits3

    # Quality metrics
    confidence_score: int = Field(ge=0, le=100)
    accuracy_percent: float = Field(ge=0, le=100)

    # Trading parameters
    leverage: int = Field(ge=1, le=125)

    # Validity
    valid_until: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=30))

    # Current market state
    current_price: float = Field(gt=0)

    # Optional metadata
    reasons: Optional[List[str]] = None
    patterns: Optional[List[str]] = None
    market_context: Optional[str] = None
    source: Optional[str] = Field(default="UNKNOWN")  # PRO, DASHBOARD, or FALLBACK

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with ISO timestamps (v2 uses model_dump)"""
        data: Dict[str, Any]
        if hasattr(self, "model_dump"):
            data = self.model_dump()
        else:
            data = self.dict()

        data["timestamp"] = self.timestamp.isoformat()
        data["valid_until"] = self.valid_until.isoformat()
        data["take_profits"] = [
            {"level": tp.level, "price": tp.price, "eta": tp.eta.isoformat()}
            for tp in self.take_profits
        ]
        return data

    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio using TP3"""
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return 0.0
        if not self.take_profits:
            return 0.0
        reward = abs(self.take_profits[-1].price - self.entry_price)
        return reward / risk

    def is_valid(self) -> bool:
        """Check if signal is still valid (not expired)"""
        return datetime.utcnow() < self.valid_until


class PriceSnapshot(BaseModel):
    """Live price snapshot for a symbol"""
    symbol: str = Field(min_length=1)
    price: float = Field(gt=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PredictionUpdate(BaseModel):
    """WebSocket message for prediction update"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    predictions: Dict[str, Any]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PriceUpdate(BaseModel):
    """WebSocket message for price update"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    prices: Dict[str, float]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
