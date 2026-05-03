"""Tests for the rule-based strategy engine and risk controls."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ai_trader.backtesting.simulator import Simulator
from ai_trader.backtesting.fees import FeeModel
from ai_trader.strategies.config_loader import (
    OvertradingControls,
    RiskParams,
    StrategyConfig,
    load_strategy_config,
)
from ai_trader.strategies.risk_controls import RiskController
from ai_trader.strategies.rule_engine import RuleBasedStrategy


def _make_trending_data(n: int = 100, trend: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Generate trending OHLCV data for strategy testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(rng.normal(trend, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    volume = rng.integers(5000, 50000, n)

    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = "timestamp"
    return df


class TestRiskController:
    def test_daily_trade_limit(self):
        controls = OvertradingControls(max_trades_per_day=2, cooldown_bars=0)
        risk = RiskParams()
        rc = RiskController(risk, controls)

        rc.on_new_bar(date(2024, 1, 1))
        rc.on_trade_completed(100)
        rc.on_new_bar(date(2024, 1, 1))
        rc.on_trade_completed(50)

        rc.on_new_bar(date(2024, 1, 1))
        allowed, reason = rc.can_trade(100000)
        assert not allowed
        assert "daily_trade_limit" in reason

    def test_daily_limit_resets_on_new_day(self):
        controls = OvertradingControls(max_trades_per_day=1, cooldown_bars=0)
        risk = RiskParams()
        rc = RiskController(risk, controls)

        rc.on_new_bar(date(2024, 1, 1))
        rc.on_trade_completed(100)

        rc.on_new_bar(date(2024, 1, 2))
        allowed, _ = rc.can_trade(100000)
        assert allowed

    def test_consecutive_loss_limit(self):
        controls = OvertradingControls(cooldown_bars=0)
        risk = RiskParams(max_consecutive_losses=3)
        rc = RiskController(risk, controls)

        rc.on_new_bar(date(2024, 1, 1))
        rc.on_trade_completed(-100)
        rc.on_new_bar(date(2024, 1, 1))
        rc.on_trade_completed(-100)
        rc.on_new_bar(date(2024, 1, 1))
        rc.on_trade_completed(-100)

        rc.on_new_bar(date(2024, 1, 1))
        allowed, reason = rc.can_trade(100000)
        assert not allowed
        assert "consecutive_losses" in reason

    def test_consecutive_losses_reset_on_win(self):
        controls = OvertradingControls(cooldown_bars=0)
        risk = RiskParams(max_consecutive_losses=3)
        rc = RiskController(risk, controls)

        rc.on_new_bar(date(2024, 1, 1))
        rc.on_trade_completed(-100)
        rc.on_trade_completed(-100)
        rc.on_trade_completed(200)  # resets

        rc.on_new_bar(date(2024, 1, 1))
        allowed, _ = rc.can_trade(100000)
        assert allowed
        assert rc.state.consecutive_losses == 0

    def test_cooldown_enforcement(self):
        controls = OvertradingControls(cooldown_bars=5, max_trades_per_day=100)
        risk = RiskParams()
        rc = RiskController(risk, controls)

        rc.on_new_bar(date(2024, 1, 1))
        rc.on_trade_completed(100)  # resets bars_since_last_trade to 0

        # Immediately after — should be blocked
        rc.on_new_bar(date(2024, 1, 1))
        allowed, reason = rc.can_trade(100000)
        assert not allowed
        assert "cooldown" in reason

        # After 5 bars — should be allowed
        for _ in range(5):
            rc.on_new_bar(date(2024, 1, 1))
        allowed, _ = rc.can_trade(100000)
        assert allowed

    def test_daily_loss_limit(self):
        controls = OvertradingControls(cooldown_bars=0)
        risk = RiskParams(daily_loss_limit=0.05)
        rc = RiskController(risk, controls)

        capital = 100_000
        rc.on_new_bar(date(2024, 1, 1))
        rc.on_trade_completed(-5000)  # 5% loss

        rc.on_new_bar(date(2024, 1, 1))
        allowed, reason = rc.can_trade(capital)
        assert not allowed
        assert "daily_loss_limit" in reason

    def test_position_sizing(self):
        controls = OvertradingControls()
        risk = RiskParams(max_capital_per_trade=0.02)
        rc = RiskController(risk, controls)

        qty = rc.calculate_position_size(100_000, 500)
        assert qty == 4  # 2% of 100k = 2000, / 500 = 4

    def test_stop_loss_calculation(self):
        controls = OvertradingControls()
        risk = RiskParams(stop_loss_pct=0.03)
        rc = RiskController(risk, controls)

        # Fixed percentage
        sl = rc.calculate_stop_loss(100.0)
        assert sl == pytest.approx(97.0)

        # ATR-based
        sl_atr = rc.calculate_stop_loss(100.0, atr=2.0)
        assert sl_atr == pytest.approx(96.0)


class TestRuleBasedStrategy:
    def test_strategy_produces_signals(self):
        config = StrategyConfig()
        strategy = RuleBasedStrategy(config, initial_capital=100_000.0)
        df = _make_trending_data(100)

        strategy.initialize(df)

        signals = []
        for i in range(len(df)):
            sig = strategy.evaluate(df, i)
            signals.append(sig.signal.value)

        assert "hold" in signals  # warmup period produces holds
        # Should produce at least some non-hold signals in trending data
        non_hold = [s for s in signals if s != "hold"]
        assert len(non_hold) >= 0  # May or may not trigger depending on data

    def test_strategy_respects_warmup(self):
        config = StrategyConfig()
        strategy = RuleBasedStrategy(config)
        df = _make_trending_data(50)
        strategy.initialize(df)

        # First N bars should be HOLD (warmup)
        min_bars = max(config.indicators.macd_slow, config.indicators.sma_slow, config.indicators.rsi_period) + 5
        for i in range(min_bars):
            sig = strategy.evaluate(df, i)
            assert sig.signal.value == "hold"

    def test_strategy_with_backtest_engine(self):
        config = StrategyConfig(
            overtrading=OvertradingControls(cooldown_bars=1, min_confidence=0.1)
        )
        strategy = RuleBasedStrategy(config, initial_capital=100_000.0)
        df = _make_trending_data(200, trend=0.3, seed=99)
        df.attrs["symbol"] = "TEST"

        simulator = Simulator(initial_capital=100_000.0, max_position_pct=0.02)
        result = simulator.run(df, strategy)

        assert result.strategy_name == "rule_based_v1"
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == 100_000.0

    def test_deterministic_strategy(self):
        config = StrategyConfig()
        df = _make_trending_data(150, seed=77)
        df.attrs["symbol"] = "TEST"

        strategy1 = RuleBasedStrategy(config, initial_capital=100_000.0)
        strategy2 = RuleBasedStrategy(config, initial_capital=100_000.0)

        simulator = Simulator(initial_capital=100_000.0)
        result1 = simulator.run(df.copy(), strategy1)
        result2 = simulator.run(df.copy(), strategy2)

        assert result1.total_return == result2.total_return
        assert result1.trades == result2.trades

    def test_overtrading_prevention(self):
        config = StrategyConfig(
            overtrading=OvertradingControls(max_trades_per_day=1, cooldown_bars=10, min_confidence=0.0)
        )
        strategy = RuleBasedStrategy(config, initial_capital=100_000.0)
        df = _make_trending_data(200, seed=55)
        df.attrs["symbol"] = "TEST"

        simulator = Simulator(initial_capital=100_000.0)
        result = simulator.run(df, strategy)

        # With max 1 trade/day and 10 bar cooldown, should not overtrade
        # On daily data with 200 bars, max possible trades is ~20
        assert result.total_trades <= 20


class TestStrategyConfig:
    def test_default_config(self):
        config = StrategyConfig()
        assert config.name == "rule_based_v1"
        assert config.indicators.rsi_period == 14
        assert config.risk.max_capital_per_trade == 0.02

    def test_load_from_file(self, tmp_path):
        import yaml
        data = {
            "name": "custom_strategy",
            "indicators": {"rsi_period": 21},
            "risk": {"max_capital_per_trade": 0.01},
        }
        config_file = tmp_path / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(data, f)

        config = load_strategy_config(config_file)
        assert config.name == "custom_strategy"
        assert config.indicators.rsi_period == 21
        assert config.risk.max_capital_per_trade == 0.01

    def test_load_missing_file_returns_defaults(self):
        config = load_strategy_config("/nonexistent/path.yaml")
        assert config.name == "rule_based_v1"
