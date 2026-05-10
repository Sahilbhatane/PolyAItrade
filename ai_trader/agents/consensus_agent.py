"""ConsensusAgent — weighted multi-strategy voting with audit trail."""

from __future__ import annotations

from typing import Any

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.strategies.config_loader import StrategyConfig
from ai_trader.strategies.registry import StrategyRegistry
from ai_trader.strategies.vote_types import VoteContext


class ConsensusAgent(BaseAgent):
    """Aggregates registered strategies; never executes trades."""

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        strategy_config: StrategyConfig,
        consensus_min_weighted_confidence: float = 0.35,
        min_voter_confidence: float = 0.12,
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "consensus_agent")
        self._bus = event_bus
        self._state = state
        self._strategy_config = strategy_config
        self._min_weighted = float(
            consensus_min_weighted_confidence or strategy_config.consensus_min_weighted_confidence
        )
        self._min_voter_confidence = float(min_voter_confidence)
        self._registry = StrategyRegistry(strategy_config)

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        signals = self._state.read(StateKeys.SIGNALS)
        df = self._state.read(StateKeys.MARKET_DATA)
        weights_state = self._state.read(StateKeys.STRATEGY_WEIGHTS) or {}
        regime = self._state.read(StateKeys.REGIME)

        if signals is None or df is None:
            raise ValueError("ConsensusAgent requires SIGNALS and MARKET_DATA")

        ctx_dict = context or {}
        bar_index = ctx_dict.get("bar_index")
        if bar_index is None:
            bar_index = len(signals["rsi"]) - 1

        strategy_weights: dict[str, float] = weights_state.get("weights", {}) if isinstance(weights_state, dict) else {}
        if not strategy_weights:
            specs = self._strategy_config.strategies or {}
            strategy_weights = {
                n: float(v.get("weight", 1.0))
                for n, v in specs.items()
                if isinstance(v, dict) and v.get("enabled", True)
            }
            total = sum(strategy_weights.values()) or 1.0
            strategy_weights = {k: v / total for k, v in strategy_weights.items()}

        vote_ctx = VoteContext(df=df, signals=signals, bar_index=bar_index, regime=regime)

        votes_raw: list[dict[str, Any]] = []
        for name, w in strategy_weights.items():
            if w <= 0:
                continue
            spec = (self._strategy_config.strategies or {}).get(name, {})
            params = spec.get("params", {}) if isinstance(spec, dict) else {}
            vr = self._registry.vote(name, vote_ctx, params)
            votes_raw.append(
                {
                    "name": vr.name,
                    "weight": float(w),
                    "action": vr.action,
                    "confidence": float(vr.confidence),
                    "reasoning": vr.reasoning,
                    "signals_used": vr.signals_used,
                }
            )

        decision, audit = self._combine(votes_raw, regime, bar_index)

        await self._state.write(StateKeys.TRADE_DECISION, decision, writer=self.agent_id)
        await self._state.write(StateKeys.CONSENSUS_AUDIT, audit, writer=self.agent_id)

        event_type = EventType.TRADE_DECISION if decision["action"] != "HOLD" else EventType.TRADE_REJECTED
        await self._bus.publish(Event(event_type=event_type, payload=decision, source_agent=self.agent_id))
        self.log("consensus_decision", action=decision["action"], confidence=f"{decision['confidence']:.3f}")
        return decision

    def _combine(
        self,
        votes: list[dict[str, Any]],
        regime: dict[str, Any] | None,
        bar_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Weighted voting + conflict checks."""
        strong = [
            v
            for v in votes
            if v["action"] != "HOLD" and v["confidence"] >= self._min_voter_confidence
        ]

        buy_mass = sum(v["weight"] * v["confidence"] for v in strong if v["action"] == "BUY")
        sell_mass = sum(v["weight"] * v["confidence"] for v in strong if v["action"] == "SELL")
        conflict = False
        if buy_mass > 1e-9 and sell_mass > 1e-9:
            ratio = min(buy_mass, sell_mass) / max(buy_mass, sell_mass)
            conflict = ratio >= 0.35

        buyers = [v for v in strong if v["action"] == "BUY"]
        sellers = [v for v in strong if v["action"] == "SELL"]

        signed_sum = sum(
            v["weight"] * v["confidence"] * (1.0 if v["action"] == "BUY" else -1.0)
            for v in strong
        )
        w_sum = sum(v["weight"] for v in strong) or 1.0
        composite = signed_sum / w_sum

        reason_rules: list[str] = []

        if conflict:
            reason_rules.append("conflict_buy_sell_mass")
            decision = self._hold_decision(votes, composite, "conflicting_mass", bar_index)
            audit = self._audit(votes, regime, composite, reason_rules, buyers, sellers)
            return decision, audit

        # No single-strategy force: need >=2 agreeing strategies on winning side
        if composite > 0 and len(buyers) < 2:
            reason_rules.append("insufficient_buy_consensus")
            decision = self._hold_decision(votes, composite, "<2_buy_strategies", bar_index)
            audit = self._audit(votes, regime, composite, reason_rules, buyers, sellers)
            return decision, audit
        if composite < 0 and len(sellers) < 2:
            reason_rules.append("insufficient_sell_consensus")
            decision = self._hold_decision(votes, composite, "<2_sell_strategies", bar_index)
            audit = self._audit(votes, regime, composite, reason_rules, buyers, sellers)
            return decision, audit

        conf = abs(composite)
        if conf < self._min_weighted:
            reason_rules.append("weak_weighted_confidence")
            decision = self._hold_decision(
                votes, composite, f"below_threshold_{self._min_weighted}", bar_index
            )
            audit = self._audit(votes, regime, composite, reason_rules, buyers, sellers)
            return decision, audit

        action = "BUY" if composite > 0 else "SELL"

        signals = self._state.read(StateKeys.SIGNALS) or {}
        close_series = signals.get("close")
        current_price = float(close_series.iloc[bar_index]) if close_series is not None else 0.0
        atr_series = signals.get("atr")
        atr_val = None
        if atr_series is not None and bar_index < len(atr_series):
            atr_val = float(atr_series.iloc[bar_index])

        decision = {
            "action": action,
            "confidence": float(conf),
            "composite_score": float(composite),
            "reasoning": f"consensus_{action.lower()} composite={composite:.3f}",
            "bar_index": bar_index,
            "current_price": current_price,
            "atr": atr_val,
            "signals": {v["name"]: v["confidence"] * (1 if v["action"] == action else -1) for v in votes},
        }
        audit = self._audit(votes, regime, composite, reason_rules + ["accepted"], buyers, sellers)
        return decision, audit

    def _hold_decision(
        self,
        votes: list[dict[str, Any]],
        composite: float,
        why: str,
        bar_index: int,
    ) -> dict[str, Any]:
        signals = self._state.read(StateKeys.SIGNALS) or {}
        close_series = signals.get("close")
        current_price = float(close_series.iloc[bar_index]) if close_series is not None else 0.0
        atr_series = signals.get("atr")
        atr_val = None
        if atr_series is not None and bar_index < len(atr_series):
            atr_val = float(atr_series.iloc[bar_index])
        return {
            "action": "HOLD",
            "confidence": abs(composite),
            "composite_score": composite,
            "reasoning": f"consensus_rejected:{why}",
            "bar_index": bar_index,
            "current_price": current_price,
            "atr": atr_val,
            "signals": {v["name"]: 0.0 for v in votes},
        }

    def _audit(
        self,
        votes: list[dict[str, Any]],
        regime: dict[str, Any] | None,
        composite: float,
        rules: list[str],
        buyers: list[dict[str, Any]],
        sellers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "regime": (regime or {}).get("label"),
            "composite": composite,
            "votes": votes,
            "rules_fired": rules,
            "buy_strategies": [b["name"] for b in buyers],
            "sell_strategies": [s["name"] for s in sellers],
            "min_weighted_confidence": self._min_weighted,
        }
