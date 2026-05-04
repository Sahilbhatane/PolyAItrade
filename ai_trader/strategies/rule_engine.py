"""Rule-based strategy engine combining technical indicators into trade signals."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ai_trader.backtesting.strategy import BaseStrategy, Signal, TradeSignal
from ai_trader.strategies.config_loader import StrategyConfig
from ai_trader.strategies.indicators import Indicators
from ai_trader.strategies.risk_controls import RiskController
from ai_trader.logs import get_logger

logger = get_logger(__name__)


class RuleBasedStrategy(BaseStrategy):
    """Configurable rule-based strategy using technical indicators.

    Combines RSI, MACD, moving average crossover, VWAP, and optional ML signals
    with weighted voting. Enforces risk controls and overtrading limits.

    The ML model ASSISTS the strategy — it never decides or executes trades alone.
    """

    def __init__(self, config: StrategyConfig, initial_capital: float = 100_000.0):
        self.config = config
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self.risk_controller = RiskController(config.risk, config.overtrading)

        # Pre-computed indicator series (set during initialize)
        self._rsi: pd.Series | None = None
        self._macd_line: pd.Series | None = None
        self._macd_signal: pd.Series | None = None
        self._macd_hist: pd.Series | None = None
        self._sma_fast: pd.Series | None = None
        self._sma_slow: pd.Series | None = None
        self._vwap: pd.Series | None = None
        self._atr: pd.Series | None = None

        # ML model integration (optional, loaded if configured)
        self._ml_model = None
        self._ml_pipeline = None
        self._ml_features_df: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return self.config.name

    def initialize(self, data: pd.DataFrame) -> None:
        """Pre-compute all indicators on the full dataset."""
        close = data["close"]
        ind = self.config.indicators

        self._rsi = Indicators.rsi(close, ind.rsi_period)
        self._macd_line, self._macd_signal, self._macd_hist = Indicators.macd(
            close, ind.macd_fast, ind.macd_slow, ind.macd_signal
        )
        self._sma_fast = Indicators.sma(close, ind.sma_fast)
        self._sma_slow = Indicators.sma(close, ind.sma_slow)
        self._atr = Indicators.atr(data, ind.atr_period)

        if ind.vwap_enabled:
            self._vwap = Indicators.vwap(data)

        self._init_ml_model(data)

        self.risk_controller.reset()
        self._capital = self._initial_capital

    def _init_ml_model(self, data: pd.DataFrame) -> None:
        """Load ML model if configured. Failures are non-fatal — strategy degrades gracefully."""
        if not self.config.ml.enabled or not self.config.ml.model_path:
            return

        try:
            from ai_trader.models.lstm import LSTMPredictor
            from ai_trader.models.features import FeaturePipeline, FeatureConfig

            model = LSTMPredictor()
            model.load(self.config.ml.model_path)
            self._ml_model = model

            f_config = FeatureConfig(sequence_length=self.config.ml.sequence_length)
            self._ml_pipeline = FeaturePipeline(f_config)
            self._ml_pipeline._means = model._feature_means
            self._ml_pipeline._stds = model._feature_stds

            self._ml_features_df = self._ml_pipeline.build_features(data)
            logger.info("ml_model_loaded", model_id=model.model_id)
        except Exception as e:
            logger.warning("ml_model_load_failed", error=str(e))
            self._ml_model = None

    def evaluate(self, data: pd.DataFrame, current_index: int) -> TradeSignal:
        """Evaluate all indicators and combine into a weighted signal."""
        ind = self.config.indicators
        min_bars = max(ind.macd_slow, ind.sma_slow, ind.rsi_period) + 5
        if current_index < min_bars:
            return TradeSignal(signal=Signal.HOLD, reason="warmup_period")

        # Update risk controller date
        bar_date = data.index[current_index]
        if hasattr(bar_date, "date"):
            self.risk_controller.on_new_bar(bar_date.date())
        else:
            self.risk_controller.on_new_bar(bar_date)

        # Compute individual signals
        rsi_signal = self._evaluate_rsi(current_index)
        macd_signal = self._evaluate_macd(current_index)
        ma_signal = self._evaluate_ma_crossover(current_index)
        vwap_signal = self._evaluate_vwap(data, current_index)
        ml_signal = self._evaluate_ml(data, current_index)

        # Weighted combination
        weights = self.config.weights
        composite = (
            rsi_signal * weights.rsi
            + macd_signal * weights.macd
            + ma_signal * weights.ma_crossover
            + vwap_signal * weights.vwap
            + ml_signal * weights.ml_prediction
        )

        total_weight = (
            weights.rsi + weights.macd + weights.ma_crossover
            + weights.vwap + weights.ml_prediction
        )
        if total_weight > 0:
            composite /= total_weight

        confidence = abs(composite)

        # Determine signal direction
        if composite > 0:
            direction = Signal.BUY
        elif composite < 0:
            direction = Signal.SELL
        else:
            return TradeSignal(signal=Signal.HOLD, reason="neutral_composite")

        # Apply confidence filter
        if not self.risk_controller.check_confidence(confidence):
            return TradeSignal(signal=Signal.HOLD, reason=f"low_confidence ({confidence:.3f})")

        # Check risk controls for BUY
        if direction == Signal.BUY:
            allowed, reason = self.risk_controller.can_trade(self._capital)
            if not allowed:
                return TradeSignal(signal=Signal.HOLD, reason=f"risk_blocked: {reason}")

            current_price = data.iloc[current_index]["close"]
            atr_val = self._atr.iloc[current_index] if self._atr is not None else None
            stop_loss = self.risk_controller.calculate_stop_loss(current_price, atr_val)

            return TradeSignal(
                signal=Signal.BUY,
                stop_loss=stop_loss,
                position_size_pct=1.0,
                reason=f"composite={composite:.3f} conf={confidence:.3f}",
            )

        # SELL signal
        return TradeSignal(
            signal=Signal.SELL,
            reason=f"composite={composite:.3f} conf={confidence:.3f}",
        )

    def on_trade_closed(self, pnl: float) -> None:
        """Callback from simulator when a trade closes — updates risk state."""
        self._capital += pnl
        self.risk_controller.on_trade_completed(pnl)

    def _evaluate_rsi(self, idx: int) -> float:
        """RSI signal: -1 (overbought/sell) to +1 (oversold/buy)."""
        if self._rsi is None or pd.isna(self._rsi.iloc[idx]):
            return 0.0

        rsi_val = self._rsi.iloc[idx]
        thresholds = self.config.indicators

        if rsi_val <= thresholds.rsi_oversold:
            return 1.0 * ((thresholds.rsi_oversold - rsi_val) / thresholds.rsi_oversold)
        elif rsi_val >= thresholds.rsi_overbought:
            return -1.0 * ((rsi_val - thresholds.rsi_overbought) / (100 - thresholds.rsi_overbought))
        return 0.0

    def _evaluate_macd(self, idx: int) -> float:
        """MACD signal: crossover detection with magnitude."""
        if self._macd_line is None or idx < 1:
            return 0.0

        macd_now = self._macd_line.iloc[idx]
        signal_now = self._macd_signal.iloc[idx]
        macd_prev = self._macd_line.iloc[idx - 1]
        signal_prev = self._macd_signal.iloc[idx - 1]

        if pd.isna(macd_now) or pd.isna(signal_now) or pd.isna(macd_prev) or pd.isna(signal_prev):
            return 0.0

        # Bullish crossover
        if macd_prev <= signal_prev and macd_now > signal_now:
            return min(1.0, abs(macd_now - signal_now) * 10)

        # Bearish crossover
        if macd_prev >= signal_prev and macd_now < signal_now:
            return max(-1.0, -(abs(macd_now - signal_now) * 10))

        # Momentum direction (weaker signal)
        hist = self._macd_hist.iloc[idx]
        if not pd.isna(hist):
            return np.clip(hist * 5, -0.5, 0.5)

        return 0.0

    def _evaluate_ma_crossover(self, idx: int) -> float:
        """Moving average crossover signal."""
        if self._sma_fast is None or self._sma_slow is None or idx < 1:
            return 0.0

        fast_now = self._sma_fast.iloc[idx]
        slow_now = self._sma_slow.iloc[idx]
        fast_prev = self._sma_fast.iloc[idx - 1]
        slow_prev = self._sma_slow.iloc[idx - 1]

        if pd.isna(fast_now) or pd.isna(slow_now) or pd.isna(fast_prev) or pd.isna(slow_prev):
            return 0.0

        # Golden cross
        if fast_prev <= slow_prev and fast_now > slow_now:
            return 1.0

        # Death cross
        if fast_prev >= slow_prev and fast_now < slow_now:
            return -1.0

        # Trend direction (weaker)
        if fast_now > slow_now:
            return 0.3
        elif fast_now < slow_now:
            return -0.3

        return 0.0

    def _evaluate_vwap(self, data: pd.DataFrame, idx: int) -> float:
        """VWAP signal: price relative to VWAP."""
        if self._vwap is None or not self.config.indicators.vwap_enabled:
            return 0.0

        vwap_val = self._vwap.iloc[idx]
        if pd.isna(vwap_val) or vwap_val == 0:
            return 0.0

        price = data.iloc[idx]["close"]
        deviation = (price - vwap_val) / vwap_val

        # Price below VWAP is bullish (potential mean reversion)
        # Price above VWAP is bearish
        return float(np.clip(-deviation * 10, -1.0, 1.0))

    def _evaluate_ml(self, data: pd.DataFrame, idx: int) -> float:
        """ML model signal: probability of price increase mapped to [-1, 1].

        The model only provides a probability — it does not make trade decisions.
        Returns 0.0 if ML is disabled or insufficient data.
        """
        if self._ml_model is None or self._ml_pipeline is None:
            return 0.0

        seq_len = self.config.ml.sequence_length
        if idx < seq_len:
            return 0.0

        try:
            feature_slice = self._ml_features_df.iloc[idx - seq_len:idx]
            if feature_slice.isnull().any().any() or len(feature_slice) < seq_len:
                return 0.0

            values = feature_slice.values
            normalized = (values - self._ml_pipeline._means) / self._ml_pipeline._stds

            import torch
            self._ml_model.network.eval()
            with torch.no_grad():
                x = torch.FloatTensor(normalized).unsqueeze(0)
                prob = self._ml_model.network(x).item()

            # Map [0,1] probability to [-1,1] signal
            return (prob - 0.5) * 2.0
        except Exception:
            return 0.0
