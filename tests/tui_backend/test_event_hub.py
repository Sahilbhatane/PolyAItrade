"""Tests for the EventBus -> EventHub SSE fan-out bridge."""

import asyncio
from datetime import datetime, timezone
from enum import Enum

import pytest

from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.service.event_hub import EventHub, json_safe, serialize_event


class _Color(Enum):
    RED = "red"


def test_json_safe_primitives_and_containers():
    assert json_safe({"a": 1, "b": [1, 2.5, "x"]}) == {"a": 1, "b": [1, 2.5, "x"]}
    assert json_safe(_Color.RED) == "red"
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert json_safe(ts) == ts.isoformat()


def test_json_safe_unknown_type_falls_back_to_str():
    class Weird:
        def __repr__(self):
            return "weird"

    assert json_safe(Weird()) == "weird"


def test_serialize_event_shape():
    ev = Event(event_type=EventType.RISK_APPROVED, payload={"approved": True}, source_agent="risk")
    out = serialize_event(ev, seq=7)
    assert out["seq"] == 7
    assert out["type"] == "risk.approved"
    assert out["source"] == "risk"
    assert out["payload"] == {"approved": True}
    assert "timestamp" in out and "event_id" in out


@pytest.mark.asyncio
async def test_hub_delivers_events_to_subscriber():
    bus = EventBus()
    hub = EventHub(bus)
    queue = hub.subscribe()

    await bus.publish(Event(event_type=EventType.PIPELINE_START, payload={"x": 1}, source_agent="t"))

    item = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert item["type"] == "pipeline.start"
    assert item["payload"] == {"x": 1}
    assert hub.subscriber_count == 1


@pytest.mark.asyncio
async def test_hub_drop_oldest_backpressure():
    bus = EventBus()
    hub = EventHub(bus, max_queue=2)
    queue = hub.subscribe()

    for i in range(5):
        await bus.publish(
            Event(event_type=EventType.ORDER_FILLED, payload={"i": i}, source_agent="t")
        )

    # Queue holds only the newest 2 events; older ones were dropped.
    assert queue.qsize() == 2
    first = queue.get_nowait()
    second = queue.get_nowait()
    assert first["payload"]["i"] == 3
    assert second["payload"]["i"] == 4
    assert hub.dropped_count >= 3


@pytest.mark.asyncio
async def test_hub_unsubscribe_and_detach():
    bus = EventBus()
    hub = EventHub(bus)
    queue = hub.subscribe()
    assert hub.subscriber_count == 1

    hub.unsubscribe(queue)
    assert hub.subscriber_count == 0

    hub.detach()
    await bus.publish(Event(event_type=EventType.PIPELINE_START, payload={}, source_agent="t"))
    # No subscribers, sequence unchanged after detach.
    assert hub.subscriber_count == 0
