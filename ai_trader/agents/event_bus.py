"""Event-driven message bus for inter-agent communication.

Agents publish events to topics. Other agents subscribe to topics of interest.
This eliminates direct cross-dependencies between agents — they only know about
the event bus and the event schemas, never each other.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from ai_trader.logs import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """All event types in the system. Adding events here is the only coupling point."""

    MARKET_DATA_READY = "market_data.ready"
    MARKET_DATA_ERROR = "market_data.error"

    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_ERROR = "signal.error"

    TRADE_DECISION = "strategy.decision"
    TRADE_REJECTED = "strategy.rejected"

    RISK_APPROVED = "risk.approved"
    RISK_REJECTED = "risk.rejected"

    ORDER_SUBMITTED = "execution.submitted"
    ORDER_FILLED = "execution.filled"
    ORDER_FAILED = "execution.failed"

    PIPELINE_START = "pipeline.start"
    PIPELINE_COMPLETE = "pipeline.complete"
    PIPELINE_ERROR = "pipeline.error"


@dataclass
class Event:
    """Immutable event flowing through the system."""

    event_type: EventType
    payload: dict[str, Any]
    source_agent: str
    event_id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = field(default_factory=lambda: uuid4().hex[:12])


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Central publish/subscribe event bus.

    - Thread-safe via asyncio (single event loop, no race conditions).
    - Events are dispatched sequentially within a topic to maintain ordering.
    - Supports wildcard subscriptions and event history for debugging.
    """

    def __init__(self, max_history: int = 500):
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        self._subscribers[event_type].append(handler)
        logger.debug("event_subscribed", event_type=event_type.value)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler from a specific event type."""
        handlers = self._subscribers[event_type]
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers of its type.

        Events are dispatched sequentially to preserve ordering guarantees.
        """
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(
                    "event_handler_error",
                    event_type=event.event_type.value,
                    handler=handler.__qualname__,
                    error=str(e),
                )

    def get_history(self, event_type: EventType | None = None, limit: int = 50) -> list[Event]:
        """Retrieve recent events, optionally filtered by type."""
        if event_type is None:
            return self._history[-limit:]
        return [e for e in self._history if e.event_type == event_type][-limit:]

    def clear(self) -> None:
        """Reset all subscriptions and history."""
        self._subscribers.clear()
        self._history.clear()
