"""Tests for the server-side TradingService safety routing."""

import asyncio

import pytest

from ai_trader.broker.approval import ApprovalGate
from ai_trader.broker.kill_switch import KillSwitch
from ai_trader.broker.paper import PaperBroker
from ai_trader.service.trading_service import (
    ServiceBusyError,
    TradeIntentError,
    TradingService,
)

RISK_CFG = {"risk": {"max_capital_per_trade": 0.02, "initial_capital": 100_000.0}}


def _service(auto_approve=True):
    broker = PaperBroker(initial_balance=100_000.0)
    gate = ApprovalGate(timeout_s=5.0, auto_approve=auto_approve)
    ks = KillSwitch()
    return TradingService(broker, gate, ks, config=RISK_CFG), broker, gate, ks


async def _drain(service):
    for _ in range(200):
        if not service.is_pending:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("pending trade did not resolve")


@pytest.mark.asyncio
async def test_buy_intent_approved_and_filled_paper():
    service, broker, _, _ = _service(auto_approve=True)
    result = await service.submit_trade(
        {"symbol": "RELIANCE-EQ", "side": "buy", "price": 100.0, "confidence": 0.7}
    )
    assert result["accepted"] is True
    assert result["status"] == "pending_approval"
    assert result["position_size"] == 20  # 100000 * 0.02 / 100
    assert result["stop_loss"] is not None

    await _drain(service)
    snap = await service.snapshot()
    assert snap["last_trade"] is not None
    assert snap["last_trade"]["symbol"] == "RELIANCE-EQ"
    assert snap["positions"]["count"] == 1


@pytest.mark.asyncio
async def test_risk_rejects_after_consecutive_losses():
    service, _, _, _ = _service(auto_approve=True)
    for _ in range(3):
        service._risk_agent.on_trade_result(pnl=-100.0)

    result = await service.submit_trade(
        {"symbol": "X-EQ", "side": "buy", "price": 100.0}
    )
    assert result["accepted"] is False
    assert result["status"] == "risk_rejected"
    assert "onsecutive" in result["reason"]


@pytest.mark.asyncio
async def test_kill_switch_blocks_intent():
    service, _, _, ks = _service()
    ks.engage(reason="test", triggered_by="test")
    with pytest.raises(ServiceBusyError):
        await service.submit_trade({"symbol": "X-EQ", "side": "buy", "price": 100.0})


@pytest.mark.asyncio
async def test_only_one_pending_trade_at_a_time():
    # Human approval mode (no auto-approve) keeps the first trade pending.
    service, _, gate, _ = _service(auto_approve=False)
    first = await service.submit_trade({"symbol": "A-EQ", "side": "buy", "price": 100.0})
    assert first["status"] == "pending_approval"
    assert service.is_pending is True

    with pytest.raises(ServiceBusyError):
        await service.submit_trade({"symbol": "B-EQ", "side": "buy", "price": 100.0})

    # Wait for the background task to register the request with the gate
    # (this is what the operator UI polls before approving).
    for _ in range(200):
        if any(r.request_id == first["request_id"] for r in gate.pending_requests):
            break
        await asyncio.sleep(0.01)

    # Resolve the first approval so the pending guard clears.
    assert gate.approve(first["request_id"], by="test") is True
    await _drain(service)


@pytest.mark.asyncio
async def test_invalid_intent_rejected():
    service, _, _, _ = _service()
    with pytest.raises(TradeIntentError):
        await service.submit_trade({"symbol": "X", "side": "hold", "price": 100.0})
    with pytest.raises(TradeIntentError):
        await service.submit_trade({"symbol": "X", "side": "buy", "price": 0.0})


@pytest.mark.asyncio
async def test_events_stream_through_hub_on_submit():
    service, _, _, _ = _service(auto_approve=True)
    queue = service.event_hub.subscribe()
    await service.submit_trade({"symbol": "X-EQ", "side": "buy", "price": 100.0})
    await _drain(service)

    seen = []
    while not queue.empty():
        seen.append(queue.get_nowait()["type"])
    assert "risk.approved" in seen
    assert any(t.startswith("execution.") for t in seen)
