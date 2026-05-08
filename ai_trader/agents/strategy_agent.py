"""StrategyAgent — combines signals into trade decisions with confidence scores.

Responsibilities:
- Read probability signals from state
- Apply weighted voting to produce composite BUY/SELL/HOLD
- Compute confidence score
- Write trade decision to state (does NOT execute)

Every decision is explainable: input signals, reasoning, confidence are logged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager


class StrategyAgent(BaseAgent):
    """Combines indicator signals into a trade decision.

    Reads signals from shared state and produces a decision dict containing:
    - action: BUY / SELL / HOLD
    - confidence: 0.0 to 1.0
    - reasoning: human-readable explanation
    - bar_index: which bar the decision applies to
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        config: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "strategy_agent")
        self._bus = event_bus
        self._state = state
        self._config = config or {}

        # Signal weights (configurable, no hardcoded values)
        self._weights = {
            "rsi": self._config.get("weight_rsi", 0.25),
            "macd": self._config.get("weight_macd", 0.25),
            "ma_crossover": self._config.get("weight_ma", 0.25),
            "vwap": self._config.get("weight_vwap", 0.25),
        }
        self._confidence_threshold = self._config.get("confidence_threshold", 0.5)

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate signals and produce a trade decision.

        Expected context:
            bar_index: int — which bar to evaluate (default: last bar)

        Reads: StateKeys.SIGNALS
        Writes: StateKeys.TRADE_DECISION
        """
        signals = self._state.read(StateKeys.SIGNALS)
        if signals is None:
            raise ValueError("No signals available in state")

        bar_index = (context or {}).get("bar_index")
        if bar_index is None:
            bar_index = len(signals["rsi"]) - 1

        decision = self._make_decision(signals, bar_index)

        await self._state.write(StateKeys.TRADE_DECISION, decision, writer=self.agent_id)

        event_type = EventType.TRADE_DECISION if decision["action"] != "HOLD" else EventType.TRADE_REJECTED
        await self._bus.publish(Event(
            event_type=event_type,
            payload=decision,
            source_agent=self.agent_id,
        ))

        self.log(
            "decision_made",
            action=decision["action"],
            confidence=f"{decision['confidence']:.3f}",
        )
        return decision

    def _make_decision(self, signals: dict[str, Any], bar_index: int) -> dict[str, Any]:
        """Weighted composite of indicator signals → action + confidence."""
        component_values = {}
        for indicator, weight in self._weights.items():
            series = signals.get(indicator)
            if series is None or bar_index >= len(series):
                component_values[indicator] = 0.0
                continue
            val = series.iloc[bar_index]
            component_values[indicator] = float(val) if not pd.isna(val) else 0.0

        # Weighted sum
        total_weight = sum(self._weights.values())
        if total_weight == 0:
            composite = 0.0
        else:
            composite = sum(
                component_values[k] * self._weights[k] for k in self._weights
            ) / total_weight

        confidence = abs(composite)

        # Determine action
        if confidence < self._confidence_threshold:
            action = "HOLD"
            reasoning = f"Low confidence ({confidence:.3f} < {self._confidence_threshold})"
        elif composite > 0:
            action = "BUY"
            reasoning = f"Bullish composite={composite:.3f}"
        else:
            action = "SELL"
            reasoning = f"Bearish composite={composite:.3f}"

        # Get price context
        close_series = signals.get("close")
        current_price = float(close_series.iloc[bar_index]) if close_series is not None else 0.0

        # Get ATR for stop-loss suggestion
        atr_series = signals.get("atr")
        atr_val = float(atr_series.iloc[bar_index]) if (atr_series is not None and not pd.isna(atr_series.iloc[bar_index])) else None

        return {
            "action": action,
            "confidence": confidence,
            "composite_score": composite,
            "reasoning": reasoning,
            "bar_index": bar_index,
            "current_price": current_price,
            "atr": atr_val,
            "signals": component_values,
        }
