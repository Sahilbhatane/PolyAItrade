"""Vote datatypes shared by registry and voters (no circular imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ai_trader.strategies.config_loader import StrategyConfig


@dataclass
class VoteContext:
    """Inputs for a single-strategy voter at one bar (no future leakage)."""

    df: pd.DataFrame
    signals: dict[str, Any]
    bar_index: int
    regime: dict[str, Any] | None = None
    strategy_config: StrategyConfig | None = None


@dataclass
class VoteResult:
    """One strategy vote."""

    name: str
    action: str  # BUY / SELL / HOLD
    confidence: float
    reasoning: str
    signals_used: dict[str, float] = field(default_factory=dict)
