"""Strategy interface for the backtesting engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeSignal:
    """Output of strategy evaluation at a single bar."""

    signal: Signal
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size_pct: float = 1.0  # Fraction of allowed capital to use
    reason: str = ""


class BaseStrategy(ABC):
    """Interface for backtestable strategies.

    Strategies are stateless bar-by-bar evaluators. They receive
    the full history up to the current bar and emit a signal.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    def evaluate(self, data: pd.DataFrame, current_index: int) -> TradeSignal:
        """Evaluate strategy at the given bar index.

        Args:
            data: Full OHLCV DataFrame (strategy may look back but NOT forward).
            current_index: Integer position of the current bar.

        Returns:
            TradeSignal with action and parameters.
        """
        ...

    def initialize(self, data: pd.DataFrame) -> None:
        """Optional hook called once before backtesting starts.

        Use for pre-computing indicators on the full dataset.
        """
        pass
