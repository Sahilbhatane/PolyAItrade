"""StrategySelectionAgent — regime-aware strategy weights (+ optional RL proposal blend)."""

from __future__ import annotations

from typing import Any

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.strategies.config_loader import StrategyConfig


class StrategySelectionAgent(BaseAgent):
    """Writes normalized strategy weights into state for ConsensusAgent."""

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        strategy_config: StrategyConfig,
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "strategy_selection_agent")
        self._bus = event_bus
        self._state = state
        self._strategy_config = strategy_config

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        regime = self._state.read(StateKeys.REGIME) or {}
        label = str(regime.get("label", "sideways"))

        specs = self._strategy_config.strategies or {}
        regime_row = (self._strategy_config.regime_weights or {}).get(label, {})

        weights: dict[str, float] = {}
        for name, spec in specs.items():
            if isinstance(spec, dict) and not spec.get("enabled", True):
                continue
            base_w = float(spec.get("weight", 1.0)) if isinstance(spec, dict) else 1.0
            mult = float(regime_row.get(name, 1.0))
            weights[name] = base_w * mult

        rl = self._state.read(StateKeys.RL_WEIGHT_PROPOSAL) or {}
        rl_weights = rl.get("strategy_weights") if isinstance(rl, dict) else None
        if isinstance(rl_weights, dict):
            for k, delta in rl_weights.items():
                if k in weights and isinstance(delta, (int, float)):
                    weights[k] = float(weights[k]) * (1.0 + float(delta))

        # Normalize
        total = sum(max(v, 0.0) for v in weights.values()) or 1.0
        weights = {k: max(v, 0.0) / total for k, v in weights.items()}

        max_trades = None
        if isinstance(rl, dict) and rl.get("max_trades_per_day") is not None:
            max_trades = rl.get("max_trades_per_day")

        payload = {
            "regime": label,
            "weights": weights,
            "rl_adjusted": bool(rl_weights),
            "max_trades_per_day_hint": max_trades,
        }
        await self._state.write(StateKeys.STRATEGY_WEIGHTS, payload, writer=self.agent_id)

        await self._bus.publish(Event(
            event_type=EventType.SIGNAL_GENERATED,
            payload={"strategy_weights": weights},
            source_agent=self.agent_id,
        ))
        self.log("strategy_weights_selected", regime=label, n_strategies=len(weights))
        return payload
