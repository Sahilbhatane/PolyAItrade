"""Tests for ConsensusAgent + registry."""

import numpy as np
import pandas as pd
import pytest

from ai_trader.agents.consensus_agent import ConsensusAgent
from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.regime_detection_agent import RegimeDetectionAgent
from ai_trader.agents.signal_agent import SignalAgent
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.agents.strategy_selection_agent import StrategySelectionAgent
from ai_trader.strategies.config_loader import load_strategy_config
from ai_trader.strategies.registry import VOTER_REGISTRY


@pytest.mark.asyncio
async def test_consensus_produces_audit():
    bus = EventBus()
    state = StateManager()
    rng = np.random.default_rng(3)
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    c = 100 + np.cumsum(rng.normal(0.1, 0.5, n))
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": rng.integers(8000, 40000, n)}, index=idx)
    await state.write(StateKeys.MARKET_DATA, df, writer="t")
    await SignalAgent(bus, state).start()
    await RegimeDetectionAgent(bus, state, {}).start({"bar_index": n - 1})
    scfg = load_strategy_config("strategy_config.yaml")
    await StrategySelectionAgent(bus, state, scfg).start()

    cons = ConsensusAgent(bus, state, scfg, consensus_min_weighted_confidence=0.001)
    decision = await cons.start({"bar_index": n - 1})
    assert decision["action"] in ("BUY", "SELL", "HOLD")
    audit = state.read(StateKeys.CONSENSUS_AUDIT)
    assert audit is not None
    assert "votes" in audit


def test_registry_has_all_voters():
    assert set(VOTER_REGISTRY.keys()) >= {
        "rule_based_v1",
        "momentum_breakout",
        "mean_reversion",
        "vwap_reversion",
        "ma_crossover",
        "fibonacci_confluence",
    }
