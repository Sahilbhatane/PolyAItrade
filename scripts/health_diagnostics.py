#!/usr/bin/env python3
"""Quick synthetic pipeline timing diagnostics."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd

from ai_trader.agents.consensus_agent import ConsensusAgent
from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.regime_detection_agent import RegimeDetectionAgent
from ai_trader.agents.signal_agent import SignalAgent
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.agents.strategy_selection_agent import StrategySelectionAgent
from ai_trader.strategies.config_loader import load_strategy_config


def _df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.integers(1000, 10000, n),
        },
        index=idx,
    )


async def main() -> None:
    bus = EventBus()
    state = StateManager()
    df = _df()
    await state.write(StateKeys.MARKET_DATA, df, writer="diag")
    scfg = load_strategy_config()

    t0 = time.perf_counter()
    sig = SignalAgent(event_bus=bus, state=state)
    await sig.start()
    print(f"signal_agent_ms={(time.perf_counter()-t0)*1000:.1f}")

    t1 = time.perf_counter()
    reg = RegimeDetectionAgent(bus, state, regime_config={})
    await reg.start({"bar_index": len(df) - 1})
    print(f"regime_agent_ms={(time.perf_counter()-t1)*1000:.1f}")

    t2 = time.perf_counter()
    sel = StrategySelectionAgent(bus, state, scfg)
    await sel.start()
    print(f"strategy_selection_ms={(time.perf_counter()-t2)*1000:.1f}")

    t3 = time.perf_counter()
    cons = ConsensusAgent(bus, state, scfg, consensus_min_weighted_confidence=0.01)
    await cons.start({"bar_index": len(df) - 1})
    print(f"consensus_agent_ms={(time.perf_counter()-t3)*1000:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
