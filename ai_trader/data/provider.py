"""Base interface for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd


class TimeFrame(str, Enum):
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    DAILY = "1d"
    WEEKLY = "1w"


@dataclass
class MarketData:
    """Container for validated market data."""

    symbol: str
    timeframe: TimeFrame
    data: pd.DataFrame
    source: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check data integrity: no gaps, sorted, required columns present."""
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(self.data.columns)):
            return False
        if self.data.isnull().any().any():
            return False
        if not self.data.index.is_monotonic_increasing:
            return False
        return True


class BaseDataProvider(ABC):
    """Abstract interface for data sources.

    All data must be validated and timestamped before downstream use.
    """

    @abstractmethod
    async def fetch_historical(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> MarketData:
        """Fetch historical OHLCV data for a symbol."""
        ...

    @abstractmethod
    async def fetch_realtime(self, symbol: str) -> MarketData:
        """Fetch latest real-time market data."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the data source is available and responsive."""
        ...
