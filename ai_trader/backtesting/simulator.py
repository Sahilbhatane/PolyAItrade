"""Concrete backtesting engine with step-by-step trade simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from ai_trader.backtesting.engine import BacktestEngine, BacktestResult
from ai_trader.backtesting.fees import FeeModel
from ai_trader.backtesting.metrics import compute_all_metrics
from ai_trader.backtesting.strategy import BaseStrategy, Signal
from ai_trader.logs import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Tracks an open position during simulation."""

    symbol: str
    entry_price: float
    quantity: int
    entry_bar: int
    stop_loss: float | None = None
    take_profit: float | None = None
    entry_cost: float = 0.0


@dataclass
class TradeRecord:
    """Completed trade with full audit."""

    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_bar: int
    exit_bar: int
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    duration_bars: int = 0


class Simulator(BacktestEngine):
    """Deterministic bar-by-bar backtesting engine.

    Processes each bar sequentially, enforcing:
    - No look-ahead bias (strategy sees only past data)
    - Realistic cost modeling (brokerage, taxes, slippage)
    - Stop loss execution at exact trigger prices
    - Position sizing as fraction of available capital
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_model: FeeModel | None = None,
        max_position_pct: float = 0.02,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.0001,
    ):
        super().__init__(initial_capital, commission_rate, slippage_rate)
        self.fee_model = fee_model or FeeModel(
            brokerage_rate=commission_rate,
            slippage_rate=slippage_rate,
        )
        self.max_position_pct = max_position_pct

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> BacktestResult:
        """Run a full backtest simulation."""
        df = self._prepare_data(data, start_date, end_date)
        if df.empty:
            raise ValueError("No data available for the specified date range")

        strategy.initialize(df)

        cash = self.initial_capital
        position: Position | None = None
        trades: list[TradeRecord] = []
        equity_curve: list[float] = [self.initial_capital]

        for i in range(len(df)):
            bar = df.iloc[i]
            current_price = bar["close"]

            # Check stop loss / take profit on open positions
            if position is not None:
                exit_price, exit_reason = self._check_exits(position, bar)
                if exit_price is not None:
                    trade, cash = self._close_position(position, exit_price, i, bar, cash, exit_reason)
                    trades.append(trade)
                    position = None

            # Evaluate strategy signal
            signal = strategy.evaluate(df, i)

            if signal.signal == Signal.BUY and position is None:
                position, cash = self._open_position(
                    df, i, bar, cash, signal.stop_loss, signal.take_profit, signal.position_size_pct
                )
            elif signal.signal == Signal.SELL and position is not None:
                trade, cash = self._close_position(position, current_price, i, bar, cash, "signal_exit")
                trades.append(trade)
                position = None

            # Mark-to-market equity
            mtm = cash
            if position is not None:
                mtm += current_price * position.quantity
            equity_curve.append(mtm)

        # Force-close any open position at end
        if position is not None:
            final_price = df.iloc[-1]["close"]
            trade, cash = self._close_position(position, final_price, len(df) - 1, df.iloc[-1], cash, "end_of_data")
            trades.append(trade)

        trade_dicts = [self._trade_to_dict(t) for t in trades]
        metrics = compute_all_metrics(equity_curve, trade_dicts)

        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=df.index[0].to_pydatetime() if hasattr(df.index[0], "to_pydatetime") else df.index[0],
            end_date=df.index[-1].to_pydatetime() if hasattr(df.index[-1], "to_pydatetime") else df.index[-1],
            total_trades=metrics.total_trades,
            win_rate=metrics.win_rate,
            total_return=metrics.total_return,
            max_drawdown=metrics.max_drawdown,
            sharpe_ratio=metrics.sharpe_ratio,
            trades=trade_dicts,
            equity_curve=equity_curve,
            metadata={
                "initial_capital": self.initial_capital,
                "final_capital": equity_curve[-1],
                "profit_factor": metrics.profit_factor,
                "annualized_return": metrics.annualized_return,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
            },
        )

        logger.info(
            "backtest_complete",
            strategy=strategy.name,
            trades=metrics.total_trades,
            total_return=f"{metrics.total_return:.4f}",
        )
        return result

    def calculate_metrics(self, equity_curve: list[float]) -> dict[str, float]:
        """Compute metrics from equity curve alone (no trade details)."""
        from ai_trader.backtesting.metrics import (
            calculate_max_drawdown,
            calculate_sharpe_ratio,
            calculate_total_return,
        )
        return {
            "total_return": calculate_total_return(equity_curve),
            "max_drawdown": calculate_max_drawdown(equity_curve),
            "sharpe_ratio": calculate_sharpe_ratio(equity_curve),
        }

    def _prepare_data(self, data: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
        df = data.copy()
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    def _check_exits(self, position: Position, bar: pd.Series) -> tuple[float | None, str]:
        """Check if stop loss or take profit is triggered on this bar."""
        low = bar["low"]
        high = bar["high"]

        if position.stop_loss is not None and low <= position.stop_loss:
            return position.stop_loss, "stop_loss"
        if position.take_profit is not None and high >= position.take_profit:
            return position.take_profit, "take_profit"
        return None, ""

    def _open_position(
        self,
        df: pd.DataFrame,
        bar_index: int,
        bar: pd.Series,
        cash: float,
        stop_loss: float | None,
        take_profit: float | None,
        size_pct: float,
    ) -> tuple[Position, float]:
        """Open a new long position with proper sizing and cost."""
        price = bar["close"]
        risk_capital = cash * self.max_position_pct * size_pct
        quantity = int(risk_capital / price)

        if quantity <= 0:
            return None, cash  # type: ignore

        entry_cost = self.fee_model.calculate_buy_cost(price, quantity)

        if entry_cost > cash:
            quantity = int((cash * 0.99) / (price * (1 + self.fee_model.brokerage_rate + self.fee_model.slippage_rate)))
            if quantity <= 0:
                return None, cash  # type: ignore
            entry_cost = self.fee_model.calculate_buy_cost(price, quantity)

        cash -= entry_cost

        position = Position(
            symbol=df.attrs.get("symbol", "UNKNOWN"),
            entry_price=price,
            quantity=quantity,
            entry_bar=bar_index,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_cost=entry_cost,
        )
        return position, cash

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        bar_index: int,
        _bar: pd.Series,
        cash: float,
        reason: str,
    ) -> tuple[TradeRecord, float]:
        """Close position and record the trade."""
        proceeds = self.fee_model.calculate_sell_proceeds(exit_price, position.quantity)
        cash += proceeds

        pnl = proceeds - position.entry_cost
        pnl_pct = pnl / position.entry_cost if position.entry_cost > 0 else 0.0

        trade = TradeRecord(
            symbol=position.symbol,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_bar=position.entry_bar,
            exit_bar=bar_index,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            duration_bars=bar_index - position.entry_bar,
        )
        return trade, cash

    @staticmethod
    def _trade_to_dict(trade: TradeRecord) -> dict[str, Any]:
        return {
            "symbol": trade.symbol,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "entry_bar": trade.entry_bar,
            "exit_bar": trade.exit_bar,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "exit_reason": trade.exit_reason,
            "duration_bars": trade.duration_bars,
        }
