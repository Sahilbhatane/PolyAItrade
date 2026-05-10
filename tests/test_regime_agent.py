"""Tests for RegimeDetectionAgent."""

import numpy as np
import pandas as pd
import pytest

from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.regime_detection_agent import RegimeDetectionAgent
from ai_trader.agents.signal_agent import SignalAgent
from ai_trader.agents.state import StateKeys, StateManager


def _df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    c = 100 + np.cumsum(rng.normal(0.02, 0.4, n))
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": rng.integers(5000, 20000, n)}, index=idx)


@pytest.mark.asyncio
async def test_regime_writes_state():
    bus = EventBus()
    state = StateManager()
    df = _df()
    await state.write(StateKeys.MARKET_DATA, df, writer="t")
    sig = SignalAgent(bus, state)
    await sig.start()
    agent = RegimeDetectionAgent(bus, state, regime_config={"window_bars": 15})
    out = await agent.start({"bar_index": len(df) - 1})
    assert "label" in out
    assert state.read(StateKeys.REGIME)["label"] == out["label"]
    assert "features" in out
