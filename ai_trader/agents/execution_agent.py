"""ExecutionAgent — executes ONLY risk-approved trades.

Responsibilities:
- Read risk-approved verdicts from state
- Place orders via broker interface
- Verify order success
- Report execution results

This agent NEVER decides what to trade. It only executes what the pipeline approves.
"""

from __future__ import annotations

from typing import Any

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.broker.base import BaseBroker, Order, OrderSide, OrderType, OrderStatus
from ai_trader.logs import get_logger

logger = get_logger(__name__)


class ExecutionAgent(BaseAgent):
    """Executes trades that have been approved by the RiskAgent.

    Only operates on RISK_APPROVED verdicts. Never makes independent decisions.
    Verifies order success and publishes execution results.
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        broker: BaseBroker,
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "execution_agent")
        self._bus = event_bus
        self._state = state
        self._broker = broker

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a trade based on risk-approved verdict.

        Reads: StateKeys.RISK_VERDICT
        Writes: StateKeys.ORDER_RESULT
        """
        verdict = self._state.read(StateKeys.RISK_VERDICT)
        if verdict is None:
            raise ValueError("No risk verdict in state")

        if not verdict.get("approved"):
            self.log("skipping_execution", reason="Not approved by RiskAgent")
            result = {"status": "skipped", "reason": verdict.get("reason", "rejected")}
            await self._state.write(StateKeys.ORDER_RESULT, result, writer=self.agent_id)
            return result

        decision = verdict.get("original_decision", {})
        action = decision.get("action", "")
        symbol = self._get_symbol()
        price = decision.get("current_price", 0.0)

        try:
            order = self._build_order(verdict, symbol, action, price)
            executed_order = await self._broker.place_order(order)

            if executed_order.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED):
                result = {
                    "status": "success",
                    "order_id": executed_order.order_id,
                    "side": executed_order.side.value,
                    "quantity": executed_order.quantity,
                    "price": executed_order.filled_price or price,
                    "stop_loss": order.stop_loss,
                }
                await self._bus.publish(Event(
                    event_type=EventType.ORDER_FILLED,
                    payload=result,
                    source_agent=self.agent_id,
                ))
                self.log("order_executed", order_id=executed_order.order_id, side=action)
            else:
                result = {
                    "status": "failed",
                    "order_id": executed_order.order_id,
                    "order_status": executed_order.status.value,
                }
                await self._bus.publish(Event(
                    event_type=EventType.ORDER_FAILED,
                    payload=result,
                    source_agent=self.agent_id,
                ))
                self.log("order_failed", order_status=executed_order.status.value)

        except Exception as e:
            result = {"status": "error", "error": str(e)}
            await self._bus.publish(Event(
                event_type=EventType.ORDER_FAILED,
                payload=result,
                source_agent=self.agent_id,
            ))
            self.log("execution_error", level="error", error=str(e))

        await self._state.write(StateKeys.ORDER_RESULT, result, writer=self.agent_id)
        return result

    def _get_symbol(self) -> str:
        """Get current symbol from market metadata."""
        meta = self._state.read(StateKeys.MARKET_METADATA)
        if meta and "symbol" in meta:
            return meta["symbol"]
        return "UNKNOWN"

    @staticmethod
    def _build_order(verdict: dict, symbol: str, action: str, price: float) -> Order:
        """Construct an Order object from the approved verdict."""
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        quantity = verdict.get("position_size", 1)
        stop_loss = verdict.get("stop_loss")

        return Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
        )
