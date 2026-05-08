"""MarketDataAgent — fetches, validates, and publishes market data.

Responsibilities:
- Fetch historical/realtime OHLCV data
- Validate data quality (no NaN, positive prices, chronological)
- Write validated data to shared state
- Publish MARKET_DATA_READY event on success

This agent is the ONLY writer to StateKeys.MARKET_DATA.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.data.provider import TimeFrame
from ai_trader.data.yfinance_provider import YFinanceProvider
from ai_trader.data.storage import MarketDataStore


class MarketDataAgent(BaseAgent):
    """Fetches and validates market data, then publishes it for downstream agents.

    Never makes trading decisions. Its only job is ensuring clean, timestamped data
    is available in shared state.
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        database_url: str = "sqlite:///ai_trader.db",
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "market_data_agent")
        self._bus = event_bus
        self._state = state
        self._provider = YFinanceProvider()
        self._store = MarketDataStore(database_url=database_url)

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch market data based on context parameters.

        Expected context keys:
            symbol: str — ticker symbol
            timeframe: str — e.g. "1d"
            start_date: datetime
            end_date: datetime
        """
        if not context:
            raise ValueError("MarketDataAgent requires context with symbol, timeframe, dates")

        symbol = context["symbol"]
        timeframe = TimeFrame(context.get("timeframe", "1d"))
        start = context["start_date"]
        end = context["end_date"]

        self.log("fetching_data", symbol=symbol, timeframe=timeframe.value)

        try:
            df = await self._fetch_with_cache(symbol, timeframe, start, end)
            self._validate(df)

            await self._state.write(StateKeys.MARKET_DATA, df, writer=self.agent_id)
            await self._state.write(StateKeys.MARKET_METADATA, {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "rows": len(df),
                "start": str(df.index[0]),
                "end": str(df.index[-1]),
            }, writer=self.agent_id)

            await self._bus.publish(Event(
                event_type=EventType.MARKET_DATA_READY,
                payload={"symbol": symbol, "rows": len(df)},
                source_agent=self.agent_id,
            ))

            self.log("data_published", symbol=symbol, rows=len(df))
            return {"status": "success", "rows": len(df)}

        except Exception as e:
            await self._bus.publish(Event(
                event_type=EventType.MARKET_DATA_ERROR,
                payload={"error": str(e), "symbol": symbol},
                source_agent=self.agent_id,
            ))
            raise

    async def _fetch_with_cache(
        self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Try cache first, fall back to provider."""
        cached = self._store.load(symbol, timeframe, start, end)
        if cached is not None and not cached.data.empty:
            self.log("cache_hit", symbol=symbol, rows=len(cached.data))
            return cached.data

        market_data = await self._provider.fetch_historical(symbol, timeframe, start, end)
        self._store.save(market_data)
        return market_data.data

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Refuse to proceed if data is missing, stale, or inconsistent."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if df.empty:
            raise ValueError("Empty dataset — cannot proceed")

        nulls = df[list(required)].isnull().sum()
        if nulls.any():
            raise ValueError(f"NaN values in data: {nulls[nulls > 0].to_dict()}")

        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("Non-positive prices detected")

        if not df.index.is_monotonic_increasing:
            raise ValueError("Data not sorted chronologically")
