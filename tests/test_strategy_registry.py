"""Strategy registry voter smoke tests."""

import numpy as np
import pandas as pd
import pytest

from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.signal_agent import SignalAgent
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.strategies.config_loader import load_strategy_config
from ai_trader.strategies.registry import StrategyRegistry
from ai_trader.strategies.vote_types import VoteContext


@pytest.mark.asyncio
async def test_registry_vote_rule_based():
    scfg = load_strategy_config("strategy_config.yaml")
    reg = StrategyRegistry(scfg)
    rng = np.random.default_rng(4)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    c = 100 + np.cumsum(rng.normal(0, 0.2, n))
    df = pd.DataFrame(
        {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": rng.integers(5000, 15000, n)},
        index=idx,
    )
    bus = EventBus()
    st = StateManager()
    await st.write(StateKeys.MARKET_DATA, df, writer="x")
    await SignalAgent(bus, st).start()
    signals = st.read(StateKeys.SIGNALS)
    ctx = VoteContext(df=df, signals=signals, bar_index=n - 1, regime=None, strategy_config=scfg)
    r = reg.vote("rule_based_v1", ctx, {})
    assert r.action in ("BUY", "SELL", "HOLD")
