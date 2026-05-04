"""YAML-based strategy configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class IndicatorThresholds(BaseModel):
    """Configurable thresholds for indicator-based signals."""

    rsi_overbought: float = Field(default=70.0, ge=50, le=100)
    rsi_oversold: float = Field(default=30.0, ge=0, le=50)
    rsi_period: int = Field(default=14, ge=2)

    macd_fast: int = Field(default=12, ge=2)
    macd_slow: int = Field(default=26, ge=5)
    macd_signal: int = Field(default=9, ge=2)

    sma_fast: int = Field(default=10, ge=2)
    sma_slow: int = Field(default=30, ge=5)

    ema_fast: int = Field(default=9, ge=2)
    ema_slow: int = Field(default=21, ge=5)

    vwap_enabled: bool = Field(default=True)
    bollinger_period: int = Field(default=20, ge=5)
    bollinger_std: float = Field(default=2.0, ge=0.5)

    atr_period: int = Field(default=14, ge=2)
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5)


class RiskParams(BaseModel):
    """Risk management configuration."""

    max_capital_per_trade: float = Field(default=0.02, ge=0.001, le=0.1)
    stop_loss_pct: float = Field(default=0.03, ge=0.005, le=0.2)
    trailing_stop_pct: float = Field(default=0.02, ge=0.005, le=0.15)
    daily_loss_limit: float = Field(default=0.05, ge=0.01, le=0.2)
    max_consecutive_losses: int = Field(default=3, ge=1)


class OvertradingControls(BaseModel):
    """Controls to prevent overtrading from noisy signals."""

    max_trades_per_day: int = Field(default=5, ge=1)
    cooldown_bars: int = Field(default=3, ge=0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SignalWeights(BaseModel):
    """Weights for combining multiple indicator signals."""

    rsi: float = Field(default=0.25, ge=0.0, le=1.0)
    macd: float = Field(default=0.25, ge=0.0, le=1.0)
    ma_crossover: float = Field(default=0.25, ge=0.0, le=1.0)
    vwap: float = Field(default=0.25, ge=0.0, le=1.0)
    ml_prediction: float = Field(default=0.0, ge=0.0, le=1.0)


class MLConfig(BaseModel):
    """Configuration for ML model integration."""

    enabled: bool = Field(default=False)
    model_path: str = Field(default="")
    sequence_length: int = Field(default=30, ge=5)
    confidence_boost: float = Field(default=0.1, ge=0.0, le=0.5)


class StrategyConfig(BaseModel):
    """Complete strategy configuration."""

    name: str = Field(default="rule_based_v1")
    version: str = Field(default="1.0.0")
    indicators: IndicatorThresholds = Field(default_factory=IndicatorThresholds)
    risk: RiskParams = Field(default_factory=RiskParams)
    overtrading: OvertradingControls = Field(default_factory=OvertradingControls)
    weights: SignalWeights = Field(default_factory=SignalWeights)
    ml: MLConfig = Field(default_factory=MLConfig)


def load_strategy_config(path: str | Path | None = None) -> StrategyConfig:
    """Load strategy config from YAML file. Falls back to defaults if missing."""
    if path is None:
        path = Path("strategy_config.yaml")
    else:
        path = Path(path)

    if not path.exists():
        return StrategyConfig()

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return StrategyConfig()

    return StrategyConfig(**data)
