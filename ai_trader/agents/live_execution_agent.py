"""LiveExecutionAgent — wraps execution with approval gate, kill switch, and broker fallback.

This agent sits between RiskAgent and the actual broker, enforcing:
1. Kill switch check — halts immediately if engaged
2. Human approval — waits for explicit consent before placing orders
3. Broker execution with retry — places orders via AngelOneBroker (or fallback)
4. Order verification — confirms order was filled correctly
5. Partial execution handling — reports and logs partial fills

NO trade is placed without passing through ALL gates.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.broker.base import BaseBroker, Order, OrderSide, OrderType, OrderStatus
from ai_trader.broker.approval import ApprovalGate, ApprovalStatus
from ai_trader.broker.kill_switch import KillSwitch
from ai_trader.logs import get_logger

logger = get_logger(__name__)


class LiveExecutionAgent(BaseAgent):
    """Execution agent for live trading with full safety layers.

    Pipeline: KillSwitch → Approval → Slippage → Broker → Verify → Report
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        broker: BaseBroker,
        approval_gate: ApprovalGate,
        kill_switch: KillSwitch,
        fallback_broker: BaseBroker | None = None,
        config: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(agent_id=agent_id or "live_execution_agent")
        self._bus = event_bus
        self._state = state
        self._broker = broker
        self._fallback_broker = fallback_broker
        self._approval_gate = approval_gate
        self._kill_switch = kill_switch
        self._config = config or {}

        self._slippage_bps = self._config.get("slippage_bps", 5)
        self._seed = self._config.get("seed", None)
        self._rng = np.random.default_rng(self._seed)

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Full execution pipeline with all safety gates.

        Steps:
        1. Check kill switch
        2. Read risk verdict
        3. Request human approval
        4. Apply slippage estimation
        5. Execute via broker (with fallback)
        6. Verify order
        7. Publish result
        """
        # Gate 1: Kill switch
        if self._kill_switch.is_active:
            result = self._blocked_result("kill_switch_active", self._kill_switch.status.get("reason", ""))
            await self._state.write(StateKeys.ORDER_RESULT, result, writer=self.agent_id)
            return result

        # Read risk verdict
        verdict = self._state.read(StateKeys.RISK_VERDICT)
        if verdict is None:
            raise ValueError("No risk verdict in state")

        if not verdict.get("approved"):
            result = {"status": "skipped", "reason": verdict.get("reason", "rejected")}
            await self._state.write(StateKeys.ORDER_RESULT, result, writer=self.agent_id)
            return result

        decision = verdict.get("original_decision", {})
        action = decision.get("action", "")
        symbol = self._get_symbol()
        price = decision.get("current_price", 0.0)
        position_size = verdict.get("position_size", 1)
        stop_loss = verdict.get("stop_loss")

        # Gate 2: Human approval.
        # Callers (e.g. the TradingService) may supply the request_id so the
        # same id is surfaced to the operator UI and correlated with the
        # ApprovalGate; otherwise we generate one.
        request_id = (context or {}).get("request_id") or str(uuid.uuid4())
        trade_details = {
            "symbol": symbol,
            "side": action,
            "quantity": position_size,
            "price": price,
            "stop_loss": stop_loss,
            "confidence": decision.get("confidence", 0.0),
            "risk_reward": verdict.get("risk_reward"),
        }

        approval_status = await self._approval_gate.request_approval(request_id, trade_details)

        if approval_status != ApprovalStatus.APPROVED:
            result = self._blocked_result(
                f"approval_{approval_status.value}",
                f"Trade not approved: {approval_status.value}",
            )
            await self._state.write(StateKeys.ORDER_RESULT, result, writer=self.agent_id)
            await self._bus.publish(Event(
                event_type=EventType.ORDER_FAILED,
                payload=result,
                source_agent=self.agent_id,
            ))
            return result

        # Re-check kill switch after approval wait
        if self._kill_switch.is_active:
            result = self._blocked_result("kill_switch_engaged_during_approval", "")
            await self._state.write(StateKeys.ORDER_RESULT, result, writer=self.agent_id)
            return result

        # Apply slippage estimation
        slippage_pct = self._slippage_bps / 10_000.0
        slippage_jitter = self._rng.uniform(0.5, 1.5)
        actual_slippage = slippage_pct * slippage_jitter

        if action == "BUY":
            estimated_fill_price = price * (1.0 + actual_slippage)
        else:
            estimated_fill_price = price * (1.0 - actual_slippage)

        # Build order
        order = self._build_order(
            symbol=symbol,
            action=action,
            price=price,
            quantity=position_size,
            stop_loss=stop_loss,
        )

        # Execute with fallback
        executed_order = await self._execute_with_fallback(order)

        if executed_order.status == OrderStatus.FILLED:
            actual_fill = executed_order.filled_price or estimated_fill_price
            slippage_cost = abs(actual_fill - price) * position_size

            result = {
                "status": "success",
                "order_id": executed_order.order_id,
                "broker_order_id": executed_order.metadata.get("broker_order_id", ""),
                "side": executed_order.side.value,
                "quantity": executed_order.quantity,
                "requested_price": price,
                "fill_price": actual_fill,
                "slippage_cost": slippage_cost,
                "stop_loss": stop_loss,
                "approval_id": request_id,
            }

            await self._bus.publish(Event(
                event_type=EventType.ORDER_FILLED,
                payload=result,
                source_agent=self.agent_id,
            ))
            self._kill_switch.reset_api_failures()
            self.log("order_executed_live", order_id=executed_order.order_id, symbol=symbol)

        elif executed_order.status == OrderStatus.PARTIALLY_FILLED:
            result = {
                "status": "partial",
                "order_id": executed_order.order_id,
                "broker_order_id": executed_order.metadata.get("broker_order_id", ""),
                "filled_qty": executed_order.metadata.get("filled_qty", 0),
                "requested_qty": position_size,
                "fill_price": executed_order.filled_price,
                "approval_id": request_id,
            }
            await self._bus.publish(Event(
                event_type=EventType.ORDER_FILLED,
                payload=result,
                source_agent=self.agent_id,
            ))
            self.log("order_partial_fill", order_id=executed_order.order_id)

        else:
            self._kill_switch.report_api_failure()
            result = {
                "status": "failed",
                "order_id": executed_order.order_id,
                "order_status": executed_order.status.value,
                "reject_reason": executed_order.metadata.get("reject_reason", "unknown"),
                "slippage_cost": 0.0,
                "approval_id": request_id,
            }
            await self._bus.publish(Event(
                event_type=EventType.ORDER_FAILED,
                payload=result,
                source_agent=self.agent_id,
            ))
            self.log("order_failed_live", order_status=executed_order.status.value)

        await self._state.write(StateKeys.ORDER_RESULT, result, writer=self.agent_id)
        return result

    async def _execute_with_fallback(self, order: Order) -> Order:
        """Try primary broker, fall back to secondary on failure."""
        try:
            executed = await self._broker.place_order(order)
            if executed.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED):
                return executed
        except Exception as e:
            self.log("primary_broker_failed", level="error", error=str(e))

        if self._fallback_broker:
            self.log("falling_back_to_secondary_broker")
            try:
                return await self._fallback_broker.place_order(order)
            except Exception as e:
                self.log("fallback_broker_failed", level="error", error=str(e))
                order.status = OrderStatus.REJECTED
                order.metadata["reject_reason"] = f"Both brokers failed: {e}"
                return order

        if order.status not in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "Primary broker failed, no fallback available"
        return order

    def _get_symbol(self) -> str:
        meta = self._state.read(StateKeys.MARKET_METADATA)
        if meta and "symbol" in meta:
            return meta["symbol"]
        return "UNKNOWN"

    @staticmethod
    def _build_order(
        symbol: str, action: str, price: float, quantity: int, stop_loss: float | None
    ) -> Order:
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        return Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
        )

    @staticmethod
    def _blocked_result(reason: str, detail: str) -> dict[str, Any]:
        return {
            "status": "blocked",
            "reason": reason,
            "detail": detail,
            "slippage_cost": 0.0,
        }
