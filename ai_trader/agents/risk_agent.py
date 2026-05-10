"""RiskAgent — highest authority agent that approves or rejects trades.

Responsibilities:
- Enforce capital limits (max 1-2% per trade)
- Enforce daily loss limits
- Enforce consecutive loss limits
- Calculate position size
- Set mandatory stop loss
- Can override ANY trade decision

If this agent rejects, the trade DOES NOT happen. Period.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.logs import get_logger

logger = get_logger(__name__)


class RiskAgent(BaseAgent):
    """Highest-authority agent — can veto any trade.

    Reads trade decisions from state, applies comprehensive risk checks,
    and writes an approved/rejected verdict to state.
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        config: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "risk_agent")
        self._bus = event_bus
        self._state = state
        self._config = config or {}

        # Risk parameters (all configurable)
        self._max_capital_pct = self._config.get("max_capital_per_trade", 0.02)
        self._stop_loss_pct = self._config.get("stop_loss_pct", 0.03)
        self._trailing_stop_pct = self._config.get("trailing_stop_pct", 0.02)
        self._daily_loss_limit = self._config.get("daily_loss_limit", 0.05)
        self._max_consecutive_losses = self._config.get("max_consecutive_losses", 3)
        self._max_trades_per_day = self._config.get("max_trades_per_day", 5)
        self._atr_stop_multiplier = self._config.get("atr_stop_multiplier", 2.0)
        self._volatile_regime_mult = float(self._config.get("volatile_regime_multiplier", 0.5))
        self._low_liquidity_mult = float(self._config.get("low_liquidity_regime_multiplier", 0.55))

        # Internal risk state
        self._consecutive_losses = 0
        self._trades_today = 0
        self._daily_pnl = 0.0
        self._current_date: date | None = None
        self._capital = self._config.get("initial_capital", 100_000.0)

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate a trade decision against risk constraints.

        Reads: StateKeys.TRADE_DECISION
        Writes: StateKeys.RISK_VERDICT
        """
        decision = self._state.read(StateKeys.TRADE_DECISION)
        if decision is None:
            raise ValueError("No trade decision in state")

        # HOLD decisions pass through untouched
        if decision["action"] == "HOLD":
            verdict = self._build_verdict(decision, approved=False, reason="HOLD — no action needed")
            await self._state.write(StateKeys.RISK_VERDICT, verdict, writer=self.agent_id)
            return verdict

        verdict = self._evaluate_risk(decision)

        await self._state.write(StateKeys.RISK_VERDICT, verdict, writer=self.agent_id)

        if verdict["approved"]:
            await self._bus.publish(Event(
                event_type=EventType.RISK_APPROVED,
                payload=verdict,
                source_agent=self.agent_id,
            ))
            self.log("trade_approved", action=decision["action"], size=verdict.get("position_size"))
        else:
            await self._bus.publish(Event(
                event_type=EventType.RISK_REJECTED,
                payload=verdict,
                source_agent=self.agent_id,
            ))
            self.log("trade_rejected", reason=verdict["reason"])

        return verdict

    def _evaluate_risk(self, decision: dict[str, Any]) -> dict[str, Any]:
        """Run all risk checks. Returns verdict with approval status."""
        action = decision["action"]
        price = decision.get("current_price", 0.0)
        atr = decision.get("atr")

        # Check consecutive losses
        if self._consecutive_losses >= self._max_consecutive_losses:
            return self._build_verdict(
                decision, approved=False,
                reason=f"Consecutive losses limit reached ({self._consecutive_losses})"
            )

        # Check daily trade count
        if self._trades_today >= self._max_trades_per_day:
            return self._build_verdict(
                decision, approved=False,
                reason=f"Daily trade limit reached ({self._trades_today})"
            )

        # Check daily loss limit
        if self._capital > 0 and self._daily_pnl < 0:
            daily_loss_pct = abs(self._daily_pnl) / self._capital
            if daily_loss_pct >= self._daily_loss_limit:
                return self._build_verdict(
                    decision, approved=False,
                    reason=f"Daily loss limit hit ({daily_loss_pct:.2%})"
                )

        if action == "BUY" and price > 0:
            # Calculate position size
            position_size = self._calculate_position_size(price)
            if position_size <= 0:
                return self._build_verdict(
                    decision, approved=False,
                    reason="Insufficient capital for minimum position"
                )

            # Calculate stop loss (mandatory)
            stop_loss = self._calculate_stop_loss(price, atr)
            trailing_stop = price * (1.0 - self._trailing_stop_pct)

            return self._build_verdict(
                decision, approved=True,
                reason="All risk checks passed",
                position_size=position_size,
                stop_loss=stop_loss,
                trailing_stop=trailing_stop,
                risk_reward_ratio=self._estimate_risk_reward(price, stop_loss),
            )

        if action == "SELL":
            return self._build_verdict(decision, approved=True, reason="Exit approved")

        return self._build_verdict(decision, approved=False, reason="Unknown action")

    def _calculate_position_size(self, price: float) -> int:
        """Size position to risk at most max_capital_pct of total capital (regime-adjusted)."""
        base = self._capital * self._max_capital_pct / price
        regime = self._state.read(StateKeys.REGIME) or {}
        label = str(regime.get("label", ""))
        mult = 1.0
        if label == "volatile":
            mult = self._volatile_regime_mult
        elif label == "low_liquidity":
            mult = self._low_liquidity_mult
        return max(int(base * mult), 0)

    def _calculate_stop_loss(self, entry_price: float, atr: float | None) -> float:
        """ATR-based stop loss if available, otherwise fixed percentage."""
        if atr is not None and atr > 0:
            return entry_price - (atr * self._atr_stop_multiplier)
        return entry_price * (1.0 - self._stop_loss_pct)

    @staticmethod
    def _estimate_risk_reward(entry: float, stop_loss: float) -> float:
        """Estimate risk/reward ratio assuming 2x reward target."""
        risk = entry - stop_loss
        if risk <= 0:
            return 0.0
        reward = risk * 2.0
        return reward / risk

    def on_trade_result(self, pnl: float, trade_date: date | None = None) -> None:
        """Update internal risk state after a trade completes."""
        if trade_date and trade_date != self._current_date:
            self._current_date = trade_date
            self._trades_today = 0
            self._daily_pnl = 0.0

        self._trades_today += 1
        self._daily_pnl += pnl
        self._capital += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def reset(self) -> None:
        """Reset all risk state (for new backtest runs)."""
        self._consecutive_losses = 0
        self._trades_today = 0
        self._daily_pnl = 0.0
        self._current_date = None
        self._capital = self._config.get("initial_capital", 100_000.0)

    @staticmethod
    def _build_verdict(
        decision: dict[str, Any],
        approved: bool,
        reason: str,
        **extras: Any,
    ) -> dict[str, Any]:
        return {
            "approved": approved,
            "reason": reason,
            "original_decision": decision,
            **extras,
        }
