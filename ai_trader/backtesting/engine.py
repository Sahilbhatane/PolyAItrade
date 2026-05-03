"""Backtesting engine interface for strategy evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class BacktestResult:
    """Standardized output from a backtest run."""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BacktestEngine(ABC):
    """Abstract backtesting engine.

    Ensures deterministic, reproducible results with proper cost accounting.
    """

    def __init__(
        self,
        initial_capital: float,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.0001,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    @abstractmethod
    def run(
        self,
        data: pd.DataFrame,
        strategy: Any,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> BacktestResult:
        """Execute a backtest over historical data.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            strategy: Strategy callable or object.
            start_date: Optional start filter.
            end_date: Optional end filter.

        Returns:
            BacktestResult with full trade log and metrics.
        """
        ...

    @abstractmethod
    def calculate_metrics(self, equity_curve: list[float]) -> dict[str, float]:
        """Compute performance metrics from equity curve."""
        ...

    def apply_costs(self, price: float, quantity: int, side: str) -> float:
        """Calculate execution price after commission and slippage."""
        slippage_adjustment = self.slippage_rate * price
        commission = self.commission_rate * price * quantity

        if side == "buy":
            return (price + slippage_adjustment) * quantity + commission
        return (price - slippage_adjustment) * quantity - commission
