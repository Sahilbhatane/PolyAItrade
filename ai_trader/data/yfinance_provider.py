"""YFinance-based market data provider."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from ai_trader.data.provider import BaseDataProvider, MarketData, TimeFrame
from ai_trader.logs import get_logger

_TIMEFRAME_TO_YF_INTERVAL = {
    TimeFrame.MINUTE_1: "1m",
    TimeFrame.MINUTE_5: "5m",
    TimeFrame.MINUTE_15: "15m",
    TimeFrame.HOUR_1: "1h",
    TimeFrame.DAILY: "1d",
    TimeFrame.WEEKLY: "1wk",
}

logger = get_logger(__name__)


class YFinanceProvider(BaseDataProvider):
    """Fetches OHLCV data from Yahoo Finance and normalizes to standard schema."""

    SOURCE_NAME = "yfinance"

    async def fetch_historical(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> MarketData:
        """Fetch historical data from yfinance.

        Returns normalized DataFrame with columns:
        [open, high, low, close, volume] and a UTC DatetimeIndex named 'timestamp'.
        """
        interval = _TIMEFRAME_TO_YF_INTERVAL.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for yfinance: {timeframe}")

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )

        if df.empty:
            raise ValueError(f"No data returned for {symbol} ({start} to {end})")

        df = self._normalize(df)

        market_data = MarketData(
            symbol=symbol,
            timeframe=timeframe,
            data=df,
            source=self.SOURCE_NAME,
            metadata={"rows": len(df), "interval": interval},
        )

        if not market_data.is_valid():
            raise ValueError(f"Data validation failed for {symbol}")

        logger.info("data_fetched", symbol=symbol, rows=len(df), timeframe=timeframe.value)
        return market_data

    async def fetch_realtime(self, symbol: str) -> MarketData:
        """Fetch latest 1-day intraday data as a proxy for real-time."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m", auto_adjust=True)

        if df.empty:
            raise ValueError(f"No realtime data for {symbol}")

        df = self._normalize(df)
        return MarketData(
            symbol=symbol,
            timeframe=TimeFrame.MINUTE_1,
            data=df,
            source=self.SOURCE_NAME,
        )

    async def health_check(self) -> bool:
        """Verify yfinance connectivity with a lightweight probe."""
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.fast_info
            return info is not None
        except Exception:
            return False

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance output to standard OHLCV schema."""
        df = df.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        keep_cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in keep_cols if c in df.columns]
        df = df[available]

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "timestamp"
        df = df.sort_index()
        df = df.dropna()

        df["volume"] = df["volume"].astype(int)

        return df
