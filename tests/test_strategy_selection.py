"""Tests for StrategySelectionAgent."""

import numpy as np
import pandas as pd
import pytest

from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.regime_detection_agent import RegimeDetectionAgent
from ai_trader.agents.signal_agent import SignalAgent
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.agents.strategy_selection_agent import StrategySelectionAgent
from ai_trader.strategies.config_loader import load_strategy_config


def _df(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    c = 100 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": rng.integers(4000, 9000, n)}, index=idx)


@pytest.mark.asyncio
async def test_strategy_selection_normalizes_weights():
    bus = EventBus()
    state = StateManager()
    df = _df()
    await state.write(StateKeys.MARKET_DATA, df, writer="t")
    await SignalAgent(bus, state).start()
    await RegimeDetectionAgent(bus, state, {}).start({"bar_index": len(df) - 1})

    scfg = load_strategy_config("strategy_config.yaml")
    sel = StrategySelectionAgent(bus, state, scfg)
    out = await sel.start()
    weights = out["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
