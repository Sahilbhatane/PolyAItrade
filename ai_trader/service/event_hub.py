"""Fan-out bridge from the in-process ``EventBus`` to many async subscribers.

The agent :class:`~ai_trader.agents.event_bus.EventBus` dispatches events to
handlers sequentially inside a single event loop. The TUI, however, needs to
stream those events to one or more HTTP (SSE) clients without any single slow
client stalling the pipeline.

``EventHub`` subscribes to every :class:`EventType` on the bus and pushes a
JSON-safe copy of each event onto a bounded per-subscriber queue. If a
subscriber cannot keep up, the *oldest* item in its queue is dropped (and
counted) rather than blocking the producer — this directly guards the
"frozen UI blocks the execution engine" and "dropped events" failure modes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any

from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.logs import get_logger

logger = get_logger(__name__)


def json_safe(value: Any, _depth: int = 0) -> Any:
    """Best-effort conversion of arbitrary values into JSON-serializable data.

    Event payloads can contain numpy scalars, datetimes, enums, or nested
    dataclasses. We never want serialization of one event to crash the stream,
    so unknown types fall back to their ``str`` representation.
    """
    if _depth > 6:
        return str(value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v, _depth + 1) for v in value]

    # numpy scalars / arrays expose .item()/.tolist() without importing numpy here
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item(), _depth + 1)
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return json_safe(tolist(), _depth + 1)
        except Exception:
            pass
    return str(value)


def serialize_event(event: Event, seq: int) -> dict[str, Any]:
    """Convert an :class:`Event` into a JSON-safe dict for transport."""
    return {
        "seq": seq,
        "event_id": event.event_id,
        "type": event.event_type.value,
        "source": event.source_agent,
        "timestamp": event.timestamp.isoformat(),
        "correlation_id": event.correlation_id,
        "payload": json_safe(event.payload),
    }


class EventHub:
    """Broadcast events from an :class:`EventBus` to many async consumers."""

    def __init__(self, bus: EventBus, max_queue: int = 1000):
        self._bus = bus
        self._max_queue = max_queue
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._seq = 0
        self._dropped = 0
        self._attach()

    def _attach(self) -> None:
        for event_type in EventType:
            self._bus.subscribe(event_type, self._on_event)

    async def _on_event(self, event: Event) -> None:
        self._seq += 1
        item = serialize_event(event, self._seq)
        # Copy the set so a client unsubscribing mid-dispatch cannot mutate it.
        for queue in list(self._subscribers):
            self._offer(queue, item)

    def _offer(self, queue: asyncio.Queue[dict[str, Any]], item: dict[str, Any]) -> None:
        """Enqueue with drop-oldest backpressure so producers never block."""
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                self._dropped += 1
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                self._dropped += 1

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a new consumer and return its bounded queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.add(queue)
        logger.debug("event_hub_subscribed", subscribers=len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)
        logger.debug("event_hub_unsubscribed", subscribers=len(self._subscribers))

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def dropped_count(self) -> int:
        return self._dropped

    @property
    def sequence(self) -> int:
        return self._seq

    def stats(self) -> dict[str, int]:
        return {
            "subscribers": self.subscriber_count,
            "dropped_events": self._dropped,
            "sequence": self._seq,
            "max_queue": self._max_queue,
        }

    def detach(self) -> None:
        """Unsubscribe from the bus and drop all consumers (for shutdown/tests)."""
        for event_type in EventType:
            self._bus.unsubscribe(event_type, self._on_event)
        self._subscribers.clear()
