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
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5, le=10.0)


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
    strategies: dict[str, Any] = Field(default_factory=dict)
    regime_weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    consensus_min_weighted_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


def default_strategy_specs() -> dict[str, Any]:
    """Default multi-strategy registry when YAML omits `strategies:` — backward compatible."""
    return {
        "rule_based_v1": {"enabled": True, "weight": 1.0, "params": {}},
        "momentum_breakout": {"enabled": True, "weight": 0.8, "params": {"lookback": 20, "breakout_pct": 0.005}},
        "mean_reversion": {"enabled": True, "weight": 0.8, "params": {"rsi_low": 35.0, "rsi_high": 65.0}},
        "vwap_reversion": {"enabled": True, "weight": 0.8, "params": {"deviation_pct": 0.008}},
        "ma_crossover": {"enabled": True, "weight": 0.7, "params": {}},
        "fibonacci_confluence": {
            "enabled": True,
            "weight": 0.6,
            "params": {"swing_window": 20, "tolerance_pct": 0.004, "min_confirmations": 2},
        },
    }


def default_regime_weights() -> dict[str, dict[str, float]]:
    return {
        "bullish_trend": {
            "rule_based_v1": 1.0,
            "momentum_breakout": 1.2,
            "mean_reversion": 0.6,
            "vwap_reversion": 0.8,
            "ma_crossover": 1.1,
            "fibonacci_confluence": 0.9,
        },
        "bearish_trend": {
            "rule_based_v1": 1.0,
            "momentum_breakout": 1.1,
            "mean_reversion": 0.7,
            "vwap_reversion": 0.9,
            "ma_crossover": 1.1,
            "fibonacci_confluence": 0.9,
        },
        "sideways": {
            "rule_based_v1": 1.0,
            "momentum_breakout": 0.5,
            "mean_reversion": 1.2,
            "vwap_reversion": 1.2,
            "ma_crossover": 0.6,
            "fibonacci_confluence": 1.0,
        },
        "volatile": {
            "rule_based_v1": 0.9,
            "momentum_breakout": 0.8,
            "mean_reversion": 0.4,
            "vwap_reversion": 0.7,
            "ma_crossover": 0.8,
            "fibonacci_confluence": 0.7,
        },
        "low_liquidity": {
            "rule_based_v1": 0.8,
            "momentum_breakout": 0.5,
            "mean_reversion": 0.8,
            "vwap_reversion": 0.9,
            "ma_crossover": 0.7,
            "fibonacci_confluence": 0.8,
        },
    }
def load_strategy_config(path: str | Path | None = None) -> StrategyConfig:
    """Load strategy config from YAML file. Falls back to defaults if missing."""
    if path is None:
        path = Path("strategy_config.yaml")
    else:
        path = Path(path)

    if not path.exists():
        cfg = StrategyConfig()
        cfg.strategies = default_strategy_specs()
        cfg.regime_weights = default_regime_weights()
        return cfg

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return StrategyConfig()

    cfg = StrategyConfig(**data)
    if not cfg.strategies:
        cfg.strategies = default_strategy_specs()
    if not cfg.regime_weights:
        cfg.regime_weights = default_regime_weights()
    return cfg
