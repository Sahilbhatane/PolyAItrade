"""Tests for FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from ai_trader.app import create_app


@pytest.fixture
def fastapi_app():
    return create_app()


@pytest.fixture
async def client(fastapi_app):
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["health"] == "/health"
    assert body["docs"] == "/docs"


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ready(client):
    resp = await client.get("/ready")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_fetch_data_invalid_timeframe(client):
    resp = await client.post("/data/fetch", json={
        "symbol": "AAPL",
        "timeframe": "invalid",
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_fetch_data_invalid_dates(client):
    resp = await client.post("/data/fetch", json={
        "symbol": "AAPL",
        "timeframe": "1d",
        "start_date": "2024-02-01",
        "end_date": "2024-01-01",
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_backtest_invalid_timeframe(client):
    resp = await client.post("/backtest/run", json={
        "symbol": "AAPL",
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "timeframe": "invalid",
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_backtest_bad_date_order(client):
    resp = await client.post("/backtest/run", json={
        "symbol": "AAPL",
        "start_date": "2024-06-01",
        "end_date": "2024-01-01",
        "timeframe": "1d",
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_backtest_unknown_strategy(client):
    resp = await client.post("/backtest/run", json={
        "symbol": "AAPL",
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "timeframe": "1d",
        "strategy_type": "nonexistent",
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_rl_status(client):
    resp = await client.get("/rl/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "checkpoint_exists" in body
