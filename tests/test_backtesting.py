"""Tests for the backtesting engine, metrics, and fee model."""

import numpy as np
import pandas as pd
import pytest

from ai_trader.backtesting.fees import FeeModel
from ai_trader.backtesting.metrics import (
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_total_return,
    compute_all_metrics,
)
from ai_trader.backtesting.simulator import Simulator
from ai_trader.backtesting.strategy import BaseStrategy, Signal, TradeSignal


class AlwaysBuyStrategy(BaseStrategy):
    """Buys on bar 5, sells on bar 15. Deterministic for testing."""

    @property
    def name(self) -> str:
        return "always_buy_test"

    def evaluate(self, data: pd.DataFrame, current_index: int) -> TradeSignal:
        if current_index == 5:
            price = data.iloc[current_index]["close"]
            return TradeSignal(signal=Signal.BUY, stop_loss=price * 0.95)
        if current_index == 15:
            return TradeSignal(signal=Signal.SELL, reason="test_exit")
        return TradeSignal(signal=Signal.HOLD)


class StopLossStrategy(BaseStrategy):
    """Buys immediately with a tight stop loss that will be hit."""

    @property
    def name(self) -> str:
        return "stop_loss_test"

    def evaluate(self, data: pd.DataFrame, current_index: int) -> TradeSignal:
        if current_index == 2:
            price = data.iloc[current_index]["close"]
            return TradeSignal(signal=Signal.BUY, stop_loss=price * 0.99)
        return TradeSignal(signal=Signal.HOLD)


def _make_ohlcv(n: int = 30, base_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = base_price + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1000, 10000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = "timestamp"
    return df


class TestMetrics:
    def test_total_return(self):
        curve = [100.0, 110.0, 105.0, 120.0]
        assert calculate_total_return(curve) == pytest.approx(0.2)

    def test_total_return_empty(self):
        assert calculate_total_return([]) == 0.0
        assert calculate_total_return([100.0]) == 0.0

    def test_max_drawdown(self):
        curve = [100.0, 120.0, 90.0, 110.0]
        # Peak 120 → trough 90 = 25% drawdown
        assert calculate_max_drawdown(curve) == pytest.approx(0.25)

    def test_max_drawdown_no_drawdown(self):
        curve = [100.0, 110.0, 120.0, 130.0]
        assert calculate_max_drawdown(curve) == 0.0

    def test_sharpe_positive(self):
        curve = [100.0 + i * 0.5 for i in range(252)]
        sharpe = calculate_sharpe_ratio(curve)
        assert sharpe > 0

    def test_profit_factor(self):
        trades = [
            {"pnl": 100},
            {"pnl": 200},
            {"pnl": -50},
            {"pnl": -100},
        ]
        assert calculate_profit_factor(trades) == pytest.approx(2.0)

    def test_profit_factor_no_losses(self):
        trades = [{"pnl": 100}, {"pnl": 50}]
        assert calculate_profit_factor(trades) == float("inf")


class TestFeeModel:
    def test_round_trip_cost_positive(self):
        model = FeeModel()
        cost = model.total_round_trip_cost(100.0, 100)
        assert cost > 0

    def test_buy_cost_exceeds_turnover(self):
        model = FeeModel()
        buy_cost = model.calculate_buy_cost(100.0, 10)
        assert buy_cost > 100.0 * 10

    def test_sell_proceeds_below_turnover(self):
        model = FeeModel()
        proceeds = model.calculate_sell_proceeds(100.0, 10)
        assert proceeds < 100.0 * 10

    def test_custom_fees(self):
        model = FeeModel(brokerage_rate=0.001, slippage_rate=0.0005)
        cost = model.total_round_trip_cost(100.0, 100)
        default_cost = FeeModel().total_round_trip_cost(100.0, 100)
        assert cost > default_cost


class TestSimulator:
    def test_basic_backtest(self):
        df = _make_ohlcv(30)
        simulator = Simulator(initial_capital=100_000.0)
        strategy = AlwaysBuyStrategy()

        result = simulator.run(df, strategy)

        assert result.strategy_name == "always_buy_test"
        assert result.total_trades == 1
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == 100_000.0

    def test_deterministic_results(self):
        """Same inputs must produce identical outputs."""
        df = _make_ohlcv(30, seed=123)
        simulator = Simulator(initial_capital=100_000.0)
        strategy = AlwaysBuyStrategy()

        result1 = simulator.run(df.copy(), strategy)
        result2 = simulator.run(df.copy(), strategy)

        assert result1.total_return == result2.total_return
        assert result1.trades == result2.trades
        assert result1.equity_curve == result2.equity_curve

    def test_stop_loss_triggers(self):
        """Generate data where price drops to trigger stop loss."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
        prices = [100.0] * 5 + [98.0, 96.0, 94.0, 92.0, 90.0] + [100.0] * 10
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1000] * 20,
        }, index=dates)
        df.index.name = "timestamp"

        simulator = Simulator(initial_capital=100_000.0)
        strategy = StopLossStrategy()
        result = simulator.run(df, strategy)

        assert result.total_trades == 1
        assert result.trades[0]["exit_reason"] == "stop_loss"

    def test_no_trade_when_insufficient_capital(self):
        df = _make_ohlcv(30, base_price=1_000_000)
        simulator = Simulator(initial_capital=100.0)
        strategy = AlwaysBuyStrategy()
        result = simulator.run(df, strategy)
        assert result.total_trades == 0

    def test_end_of_data_closes_position(self):
        """Open position should be closed at the end of data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        prices = [100.0 + i for i in range(10)]
        df = pd.DataFrame({
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000] * 10,
        }, index=dates)
        df.index.name = "timestamp"

        class BuyAndHoldStrategy(BaseStrategy):
            @property
            def name(self):
                return "buy_and_hold"

            def evaluate(self, data, current_index):
                if current_index == 1:
                    return TradeSignal(signal=Signal.BUY)
                return TradeSignal(signal=Signal.HOLD)

        simulator = Simulator(initial_capital=100_000.0)
        result = simulator.run(df, BuyAndHoldStrategy())
        assert result.total_trades == 1
        assert result.trades[0]["exit_reason"] == "end_of_data"
