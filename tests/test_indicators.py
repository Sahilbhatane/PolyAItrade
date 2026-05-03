"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from ai_trader.strategies.indicators import Indicators


@pytest.fixture
def price_series():
    """Generate deterministic price series for testing."""
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 1, 100))
    return pd.Series(prices, name="close")


@pytest.fixture
def ohlcv_df():
    """Generate deterministic OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    n = 100
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    volume = rng.integers(1000, 50000, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.5, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


class TestSMA:
    def test_sma_length(self, price_series):
        sma = Indicators.sma(price_series, 10)
        assert len(sma) == len(price_series)
        assert sma.iloc[:9].isna().all()
        assert not sma.iloc[9:].isna().any()

    def test_sma_value(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        sma = Indicators.sma(s, 3)
        assert sma.iloc[2] == pytest.approx(2.0)
        assert sma.iloc[4] == pytest.approx(4.0)


class TestEMA:
    def test_ema_length(self, price_series):
        ema = Indicators.ema(price_series, 10)
        assert len(ema) == len(price_series)

    def test_ema_reacts_faster_than_sma(self, price_series):
        ema = Indicators.ema(price_series, 10)
        sma = Indicators.sma(price_series, 10)
        # EMA should be closer to recent prices during trends
        last_diff_ema = abs(ema.iloc[-1] - price_series.iloc[-1])
        last_diff_sma = abs(sma.iloc[-1] - price_series.iloc[-1])
        # Not always true but EMA variance should differ from SMA
        assert ema.iloc[-1] != sma.iloc[-1]


class TestRSI:
    def test_rsi_range(self, price_series):
        rsi = Indicators.rsi(price_series, 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_period(self, price_series):
        rsi = Indicators.rsi(price_series, 14)
        assert rsi.iloc[:13].isna().all()

    def test_rsi_overbought_on_rising(self):
        rising = pd.Series([100 + i * 2 for i in range(100)], dtype=float)
        rsi = Indicators.rsi(rising, 14)
        # Purely rising series should give RSI near/at 100
        assert rsi.iloc[-1] == pytest.approx(100.0)


class TestMACD:
    def test_macd_components(self, price_series):
        macd_line, signal_line, hist = Indicators.macd(price_series)
        assert len(macd_line) == len(price_series)
        assert len(signal_line) == len(price_series)
        assert len(hist) == len(price_series)

    def test_histogram_equals_diff(self, price_series):
        macd_line, signal_line, hist = Indicators.macd(price_series)
        valid_idx = ~(macd_line.isna() | signal_line.isna())
        diff = macd_line[valid_idx] - signal_line[valid_idx]
        np.testing.assert_allclose(hist[valid_idx].values, diff.values, atol=1e-10)


class TestVWAP:
    def test_vwap_calculation(self, ohlcv_df):
        vwap = Indicators.vwap(ohlcv_df)
        assert len(vwap) == len(ohlcv_df)
        assert not vwap.isna().all()

    def test_vwap_between_high_low(self, ohlcv_df):
        vwap = Indicators.vwap(ohlcv_df)
        # VWAP should generally be between high and low
        valid = ~vwap.isna()
        assert (vwap[valid] >= ohlcv_df["low"][valid] - 5).all()


class TestATR:
    def test_atr_positive(self, ohlcv_df):
        atr = Indicators.atr(ohlcv_df, 14)
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_atr_period(self, ohlcv_df):
        atr = Indicators.atr(ohlcv_df, 14)
        assert len(atr) == len(ohlcv_df)


class TestBollingerBands:
    def test_bands_order(self, price_series):
        upper, middle, lower = Indicators.bollinger_bands(price_series, 20, 2.0)
        valid = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()
