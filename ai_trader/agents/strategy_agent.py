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

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.strategies.composite_signal import weighted_signal_decision


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

        decision = weighted_signal_decision(signals, bar_index, self._weights, self._confidence_threshold)

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
