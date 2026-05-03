"""Backtesting API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ai_trader.backtesting.fees import FeeModel
from ai_trader.backtesting.simulator import Simulator
from ai_trader.backtesting.strategy import BaseStrategy, Signal, TradeSignal
from ai_trader.config import get_config
from ai_trader.data.provider import TimeFrame
from ai_trader.data.storage import MarketDataStore
from ai_trader.data.yfinance_provider import YFinanceProvider

import pandas as pd

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol")
    start_date: str = Field(..., description="Backtest start (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end (YYYY-MM-DD)")
    timeframe: str = Field(default="1d")
    initial_capital: float = Field(default=100_000.0, gt=0)
    max_position_pct: float = Field(default=0.02, gt=0, le=1.0)

    # Strategy params (simple moving average crossover for demo)
    strategy_type: str = Field(default="sma_crossover")
    strategy_params: dict[str, Any] = Field(default_factory=lambda: {"fast_period": 10, "slow_period": 30})

    # Fee overrides
    brokerage_rate: float = Field(default=0.0003, ge=0)
    slippage_rate: float = Field(default=0.0001, ge=0)
    stt_rate: float = Field(default=0.00025, ge=0)


class BacktestResponse(BaseModel):
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    total_trades: int
    win_rate: float
    trades: list[dict[str, Any]]
    equity_curve_summary: dict[str, float]


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """Run a backtest on historical data with the specified strategy."""
    try:
        timeframe = TimeFrame(request.timeframe)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {request.timeframe}")

    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD format")

    if start >= end:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    config = get_config()
    store = MarketDataStore(database_url=config.database.url)

    # Try cache first
    market_data = store.load(request.symbol, timeframe, start, end)
    if market_data is None or market_data.data.empty:
        provider = YFinanceProvider()
        try:
            market_data = await provider.fetch_historical(request.symbol, timeframe, start, end)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        store.save(market_data)

    # Build strategy
    strategy = _build_strategy(request.strategy_type, request.strategy_params)

    # Build fee model
    fee_model = FeeModel(
        brokerage_rate=request.brokerage_rate,
        slippage_rate=request.slippage_rate,
        stt_rate=request.stt_rate,
    )

    # Run simulation
    simulator = Simulator(
        initial_capital=request.initial_capital,
        fee_model=fee_model,
        max_position_pct=request.max_position_pct,
    )

    df = market_data.data.copy()
    df.attrs["symbol"] = request.symbol

    result = simulator.run(df, strategy, start, end)

    equity = result.equity_curve
    return BacktestResponse(
        strategy_name=result.strategy_name,
        symbol=request.symbol,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital,
        final_capital=equity[-1] if equity else request.initial_capital,
        total_return=result.total_return,
        max_drawdown=result.max_drawdown,
        sharpe_ratio=result.sharpe_ratio,
        profit_factor=result.metadata.get("profit_factor", 0.0),
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        trades=result.trades,
        equity_curve_summary={
            "start": equity[0] if equity else 0,
            "end": equity[-1] if equity else 0,
            "min": min(equity) if equity else 0,
            "max": max(equity) if equity else 0,
        },
    )


def _build_strategy(strategy_type: str, params: dict[str, Any]) -> BaseStrategy:
    """Factory for built-in strategies."""
    if strategy_type == "sma_crossover":
        return SMACrossoverStrategy(
            fast_period=params.get("fast_period", 10),
            slow_period=params.get("slow_period", 30),
        )
    raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy_type}")


class SMACrossoverStrategy(BaseStrategy):
    """Simple moving average crossover — long when fast > slow."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._fast_sma: pd.Series | None = None
        self._slow_sma: pd.Series | None = None

    @property
    def name(self) -> str:
        return f"SMA_{self.fast_period}_{self.slow_period}"

    def initialize(self, data: pd.DataFrame) -> None:
        self._fast_sma = data["close"].rolling(window=self.fast_period).mean()
        self._slow_sma = data["close"].rolling(window=self.slow_period).mean()

    def evaluate(self, data: pd.DataFrame, current_index: int) -> TradeSignal:
        if current_index < self.slow_period:
            return TradeSignal(signal=Signal.HOLD, reason="insufficient_data")

        fast_now = self._fast_sma.iloc[current_index]
        slow_now = self._slow_sma.iloc[current_index]
        fast_prev = self._fast_sma.iloc[current_index - 1]
        slow_prev = self._slow_sma.iloc[current_index - 1]

        current_price = data.iloc[current_index]["close"]

        if fast_prev <= slow_prev and fast_now > slow_now:
            stop_loss = current_price * 0.97  # 3% stop loss
            return TradeSignal(
                signal=Signal.BUY,
                stop_loss=stop_loss,
                reason="fast_crossed_above_slow",
            )

        if fast_prev >= slow_prev and fast_now < slow_now:
            return TradeSignal(signal=Signal.SELL, reason="fast_crossed_below_slow")

        return TradeSignal(signal=Signal.HOLD, reason="no_crossover")
