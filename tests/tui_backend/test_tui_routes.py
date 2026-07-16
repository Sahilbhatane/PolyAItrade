"""Tests for the /tui/* read-model and intent routes."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from ai_trader.app import create_app
from ai_trader.broker.approval import ApprovalGate
from ai_trader.broker.kill_switch import KillSwitch
from ai_trader.broker.paper import PaperBroker
from ai_trader.routes import tui
from ai_trader.service.trading_service import TradingService

RISK_CFG = {"risk": {"max_capital_per_trade": 0.02, "initial_capital": 100_000.0}}


@pytest.fixture
def service():
    broker = PaperBroker(initial_balance=100_000.0)
    gate = ApprovalGate(timeout_s=5.0, auto_approve=True)
    ks = KillSwitch()
    svc = TradingService(broker, gate, ks, config=RISK_CFG)
    tui.set_dependencies(svc)
    return svc


@pytest.fixture
async def client(service):
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_snapshot(client):
    resp = await client.get("/tui/snapshot")
    assert resp.status_code == 200
    body = resp.json()
    assert "kill_switch" in body
    assert "risk" in body
    assert "event_hub" in body
    assert body["positions"]["count"] == 0


@pytest.mark.asyncio
async def test_submit_trade_accepted(client, service):
    resp = await client.post(
        "/tui/trade/submit",
        json={"symbol": "RELIANCE-EQ", "side": "buy", "price": 100.0, "confidence": 0.7},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["status"] == "pending_approval"

    for _ in range(200):
        if not service.is_pending:
            break
        await asyncio.sleep(0.01)
    assert service.is_pending is False


@pytest.mark.asyncio
async def test_submit_trade_invalid_price(client):
    resp = await client.post(
        "/tui/trade/submit",
        json={"symbol": "X", "side": "buy", "price": -5.0},
    )
    # pydantic validation (gt=0) rejects before reaching the service.
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_submit_trade_busy_returns_409(client, service):
    # Switch to human-approval mode so the first trade stays pending.
    service._approval_gate._auto_approve = False
    first = await client.post(
        "/tui/trade/submit", json={"symbol": "A-EQ", "side": "buy", "price": 100.0}
    )
    assert first.status_code == 200

    second = await client.post(
        "/tui/trade/submit", json={"symbol": "B-EQ", "side": "buy", "price": 100.0}
    )
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_logs_endpoint(client, tmp_path, monkeypatch):
    import ai_trader.routes.tui as tui_module

    log_dir = tmp_path
    (log_dir / "app.log").write_text(
        '{"level":"INFO","event":"a","logger":"x"}\n'
        '{"level":"ERROR","event":"b","logger":"y"}\n',
        encoding="utf-8",
    )

    class _Cfg:
        class logging:  # noqa: N801
            output_dir = str(log_dir)

    monkeypatch.setattr(tui_module, "get_config", lambda: _Cfg())

    resp = await client.get("/tui/logs?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["records"]) == 2

    resp2 = await client.get("/tui/logs?limit=10&level=ERROR")
    assert resp2.status_code == 200
    assert len(resp2.json()["records"]) == 1


@pytest.mark.asyncio
async def test_diagnostics_endpoint(client):
    resp = await client.get("/tui/diagnostics")
    assert resp.status_code == 200
    body = resp.json()
    assert "broker" in body
    assert "event_hub" in body
    assert "async_tasks" in body
    assert "memory" in body


@pytest.mark.asyncio
async def test_strategies_endpoint(client):
    resp = await client.get("/tui/strategies")
    assert resp.status_code == 200
    body = resp.json()
    assert "strategies" in body
    assert isinstance(body["strategies"], list)


@pytest.mark.asyncio
async def test_agents_endpoint(client):
    resp = await client.get("/tui/agents")
    assert resp.status_code == 200
    body = resp.json()
    assert "agents" in body
    assert "recent_events" in body


@pytest.mark.asyncio
async def test_rl_endpoint(client):
    resp = await client.get("/tui/rl")
    assert resp.status_code == 200
    body = resp.json()
    assert "checkpoint_exists" in body


@pytest.mark.asyncio
async def test_integrations_endpoint_no_secrets(client):
    resp = await client.get("/tui/config/integrations")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["broker"]["api_key"], bool)
    assert isinstance(body["integrations"]["openai"], bool)
