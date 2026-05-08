"""SignalAgent — generates probability-based signals from indicators and ML.

Responsibilities:
- Compute technical indicators on market data
- Optionally incorporate ML model predictions
- Output probability scores per indicator (NOT decisions)
- Write signals to shared state

This agent outputs PROBABILITIES, never trade decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.strategies.indicators import Indicators


class SignalAgent(BaseAgent):
    """Computes indicator-based signals and publishes probability scores.

    Subscribes to MARKET_DATA_READY and generates signals from indicators.
    Does NOT make buy/sell decisions — that's the StrategyAgent's job.
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        config: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "signal_agent")
        self._bus = event_bus
        self._state = state
        self._config = config or {}

        self._rsi_period = self._config.get("rsi_period", 14)
        self._macd_fast = self._config.get("macd_fast", 12)
        self._macd_slow = self._config.get("macd_slow", 26)
        self._macd_signal = self._config.get("macd_signal", 9)
        self._sma_fast = self._config.get("sma_fast", 10)
        self._sma_slow = self._config.get("sma_slow", 30)
        self._vwap_enabled = self._config.get("vwap_enabled", True)

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate signals from the current market data in shared state.

        Reads: StateKeys.MARKET_DATA
        Writes: StateKeys.SIGNALS
        """
        df = self._state.read(StateKeys.MARKET_DATA)
        if df is None or df.empty:
            raise ValueError("No market data available in state")

        self.log("computing_signals", bars=len(df))

        signals = self._compute_all_signals(df)

        await self._state.write(StateKeys.SIGNALS, signals, writer=self.agent_id)

        await self._bus.publish(Event(
            event_type=EventType.SIGNAL_GENERATED,
            payload={"n_bars": len(signals["rsi"]), "indicators": list(signals.keys())},
            source_agent=self.agent_id,
        ))

        self.log("signals_published", indicators=len(signals))
        return {"status": "success", "indicators": list(signals.keys())}

    def _compute_all_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute all indicator signals and return as a dict of Series/values."""
        close = df["close"]

        rsi = Indicators.rsi(close, self._rsi_period)
        macd_line, macd_signal, macd_hist = Indicators.macd(
            close, self._macd_fast, self._macd_slow, self._macd_signal
        )
        sma_fast = Indicators.sma(close, self._sma_fast)
        sma_slow = Indicators.sma(close, self._sma_slow)
        atr = Indicators.atr(df, period=14)

        vwap = Indicators.vwap(df) if self._vwap_enabled else pd.Series(dtype=float)

        # Normalize signals to [-1, 1] range (probabilities of direction)
        rsi_signal = self._rsi_to_signal(rsi)
        macd_signal_norm = self._macd_to_signal(macd_line, macd_signal, macd_hist)
        ma_signal = self._ma_crossover_signal(sma_fast, sma_slow)
        vwap_signal = self._vwap_to_signal(close, vwap)

        return {
            "rsi": rsi_signal,
            "macd": macd_signal_norm,
            "ma_crossover": ma_signal,
            "vwap": vwap_signal,
            "atr": atr,
            "raw_rsi": rsi,
            "raw_macd_hist": macd_hist,
            "close": close,
        }

    @staticmethod
    def _rsi_to_signal(rsi: pd.Series) -> pd.Series:
        """Map RSI to signal: oversold=bullish(+1), overbought=bearish(-1)."""
        signal = pd.Series(0.0, index=rsi.index)
        signal = signal.where(rsi.isna() | ((rsi > 30) & (rsi < 70)), other=0.0)

        oversold = rsi <= 30
        signal[oversold] = (30 - rsi[oversold]) / 30.0

        overbought = rsi >= 70
        signal[overbought] = -(rsi[overbought] - 70) / 30.0

        return signal.clip(-1, 1)

    @staticmethod
    def _macd_to_signal(macd_line: pd.Series, signal_line: pd.Series, hist: pd.Series) -> pd.Series:
        """Map MACD crossover and histogram to [-1, 1] signal."""
        signal = pd.Series(0.0, index=macd_line.index)

        prev_macd = macd_line.shift(1)
        prev_signal = signal_line.shift(1)

        bullish_cross = (prev_macd <= prev_signal) & (macd_line > signal_line)
        bearish_cross = (prev_macd >= prev_signal) & (macd_line < signal_line)

        signal[bullish_cross] = 1.0
        signal[bearish_cross] = -1.0

        # Add weaker histogram-based momentum
        neither = ~bullish_cross & ~bearish_cross
        signal[neither] = (hist[neither] * 5).clip(-0.5, 0.5)

        return signal.fillna(0.0)

    @staticmethod
    def _ma_crossover_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
        """Map MA crossover to [-1, 1] signal."""
        signal = pd.Series(0.0, index=fast.index)

        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)

        golden = (prev_fast <= prev_slow) & (fast > slow)
        death = (prev_fast >= prev_slow) & (fast < slow)

        signal[golden] = 1.0
        signal[death] = -1.0

        # Weaker trend signal
        trend_up = (~golden) & (fast > slow)
        trend_down = (~death) & (fast < slow)
        signal[trend_up] = 0.3
        signal[trend_down] = -0.3

        return signal.fillna(0.0)

    @staticmethod
    def _vwap_to_signal(close: pd.Series, vwap: pd.Series) -> pd.Series:
        """Price relative to VWAP → mean-reversion signal."""
        if vwap.empty:
            return pd.Series(0.0, index=close.index)

        deviation = (close - vwap) / vwap.replace(0, np.nan)
        return (-deviation * 10).clip(-1, 1).fillna(0.0)
