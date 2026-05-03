"""Technical indicator calculations.

All indicators are pure functions operating on pandas Series/DataFrames.
No side effects, no state — deterministic and testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Indicators:
    """Static collection of technical indicator computations."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (Wilder's smoothing).

        Returns values between 0 and 100.
        RSI = 100 when avg_loss = 0 (all gains), RSI = 0 when avg_gain = 0 (all losses).
        """
        delta = series.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Handle edge cases: all gains → RSI=100, all losses → RSI=0
        rsi = rsi.where(avg_loss > 0, other=100.0)
        rsi = rsi.where(avg_gain > 0, other=0.0)
        # Restore NaN for warmup period
        rsi.iloc[:period - 1] = np.nan

        return rsi

    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence).

        Returns:
            (macd_line, signal_line, histogram)
        """
        fast_ema = series.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price (intraday, resets daily).

        Expects DataFrame with 'high', 'low', 'close', 'volume' columns
        and a DatetimeIndex.
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        return cumulative_tp_vol / cumulative_vol.replace(0, np.nan)

    @staticmethod
    def bollinger_bands(
        series: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.

        Returns:
            (upper_band, middle_band, lower_band)
        """
        middle = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range — useful for dynamic stop-loss calculation."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
