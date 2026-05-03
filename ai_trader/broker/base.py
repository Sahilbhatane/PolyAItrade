"""Base interface for broker integrations."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trade order with full audit trail."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float | None = None
    filled_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseBroker(ABC):
    """Abstract broker interface.

    Handles order placement, position tracking, and account queries.
    ExecutionAgent is the only consumer of this interface.
    """

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Submit an order to the broker. Returns updated order with status."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Attempt to cancel a pending order."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Query current status of an order."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions."""
        ...

    @abstractmethod
    async def get_account_balance(self) -> dict[str, float]:
        """Get current account balance and margin info."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify broker connectivity."""
        ...
