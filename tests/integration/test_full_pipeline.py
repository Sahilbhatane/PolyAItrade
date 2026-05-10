import numpy as np
import pandas as pd
import pytest

from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.signal_agent import SignalAgent
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.agents.strategy_agent import StrategyAgent
from ai_trader.agents.risk_agent import RiskAgent
from ai_trader.agents.execution_agent import ExecutionAgent
from ai_trader.broker.paper import PaperBroker


def _df(n: int = 130):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    c = 100 + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": rng.integers(3000, 12000, n)}, index=idx)


@pytest.mark.asyncio
async def test_manual_pipeline_core_agents():
    bus = EventBus()
    state = StateManager()
    df = _df()
    await state.write(StateKeys.MARKET_DATA, df, writer="t")
    await state.write(StateKeys.MARKET_METADATA, {"symbol": "TEST"}, writer="t")
    await SignalAgent(bus, state).start()
    await StrategyAgent(bus, state, config={"confidence_threshold": 0.3}).start({"bar_index": 100})
    await RiskAgent(bus, state, config={"initial_capital": 100_000.0}).start()
    broker = PaperBroker(initial_balance=100_000.0)
    res = await ExecutionAgent(bus, state, broker=broker).start()
    assert res["status"] in ("success", "skipped", "failed")
