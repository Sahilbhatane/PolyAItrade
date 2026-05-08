"""Paper broker — simulates order execution without real money.

Used for testing and backtesting the agent pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ai_trader.broker.base import BaseBroker, Order, OrderStatus
from ai_trader.logs import get_logger

logger = get_logger(__name__)


class PaperBroker(BaseBroker):
    """Simulated broker that instantly fills all orders at requested price.

    Tracks positions and account balance for pipeline testing.
    """

    def __init__(self, initial_balance: float = 100_000.0):
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._positions: list[dict[str, Any]] = []
        self._orders: dict[str, Order] = {}

    @property
    def balance(self) -> float:
        return self._balance

    async def place_order(self, order: Order) -> Order:
        """Simulate instant fill at the order price."""
        fill_price = order.price or 0.0
        cost = fill_price * order.quantity

        if order.side.value == "buy":
            if cost > self._balance:
                order.status = OrderStatus.REJECTED
                order.metadata["reject_reason"] = "Insufficient funds"
                logger.warning("order_rejected", reason="insufficient_funds", cost=cost)
                return order

            self._balance -= cost
            self._positions.append({
                "symbol": order.symbol,
                "quantity": order.quantity,
                "entry_price": fill_price,
                "order_id": order.order_id,
            })
        else:
            self._balance += cost
            self._positions = [
                p for p in self._positions if p.get("symbol") != order.symbol
            ]

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_at = datetime.now(timezone.utc)
        self._orders[order.order_id] = order

        logger.info(
            "paper_order_filled",
            side=order.side.value,
            symbol=order.symbol,
            qty=order.quantity,
            price=fill_price,
        )
        return order

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        order = self._orders.get(order_id)
        return order.status if order else OrderStatus.REJECTED

    async def get_positions(self) -> list[dict[str, Any]]:
        return self._positions.copy()

    async def get_account_balance(self) -> dict[str, float]:
        return {
            "cash": self._balance,
            "initial": self._initial_balance,
            "pnl": self._balance - self._initial_balance,
        }

    async def health_check(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset broker state."""
        self._balance = self._initial_balance
        self._positions.clear()
        self._orders.clear()
