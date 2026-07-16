"""Phase 6 hardening tests — virtualization, reconnect, memory, accessibility."""

from __future__ import annotations

import gc
import json
import tracemalloc

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from ai_trader.tui.app import PolyVITradeApp
from ai_trader.tui.screens.logs import _MAX_VISIBLE_ROWS, LogsPane
from ai_trader.tui.store import ConnectionState
from ai_trader.tui.transport import BackendTransport, TransportError
from ai_trader.tui.widgets.status_pill import StatusPill


def _app(fake_transport):
    return PolyVITradeApp(transport=fake_transport, enable_workers=False)


@pytest.mark.asyncio
async def test_logs_virtualization_caps_rows(fake_transport):
    """Feeding many log lines must keep DataTable row count bounded."""
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.switch_pane("logs")
        await pilot.pause()
        pane = app.query_one(LogsPane)
        table = app.query_one("#logs-table")

        # Simulate large tail (backend paging tested separately at 100k).
        big_batch = {
            "records": [
                {
                    "level": "INFO",
                    "event": f"event_{i}",
                    "timestamp": "2024-01-01T09:15:00Z",
                    "logger": "test",
                }
                for i in range(5_000)
            ],
            "next_cursor": None,
            "bof": True,
        }
        pane.apply_logs(big_batch, reset=True)
        await pilot.pause()
        assert table.row_count <= _MAX_VISIBLE_ROWS
        assert table.row_count == _MAX_VISIBLE_ROWS


@pytest.mark.asyncio
async def test_reconnect_degraded_to_connected(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.store.connection = ConnectionState.DEGRADED
        app.store.last_error = "timeout"
        app._broadcast()
        await pilot.pause()

        app.store.connection = ConnectionState.CONNECTED
        app.store.last_error = None
        app._broadcast()
        await pilot.pause()
        assert app.store.connection == ConnectionState.CONNECTED


@pytest.mark.asyncio
async def test_transport_retries_on_server_error():
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503, json={"detail": "busy"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    transport = BackendTransport(client=client)
    result = await transport._get("/tui/snapshot")
    assert result == {"ok": True}
    assert calls["n"] == 2
    await transport.close()


@pytest.mark.asyncio
async def test_transport_no_retry_on_4xx():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "missing"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    transport = BackendTransport(client=client)
    with pytest.raises(TransportError):
        await transport._get("/nope")
    await transport.close()


@pytest.mark.asyncio
async def test_memory_stable_after_pane_switching(fake_transport):
    """Mount/unmount cycles should not grow unbounded (leak guard)."""
    tracemalloc.start()
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        for _ in range(20):
            for sid in ("dashboard", "trade", "logs", "strategies", "agents", "help"):
                app.switch_pane(sid)
                await pilot.pause()
        gc.collect()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Loose bound — catches runaway growth, not exact RSS.
        assert peak < 80 * 1024 * 1024


@pytest.mark.asyncio
async def test_status_pill_uses_glyph_not_color_only():
    """Accessibility: state conveyed by symbol + text, not color alone."""
    pill = StatusPill("TEST", "ok", state="crit")
    rendered = pill.render()
    assert "✖" in rendered
    assert "TEST" in rendered
    assert "ok" in rendered


@pytest.mark.asyncio
async def test_tiny_terminal_all_core_screens(fake_transport):
    app = _app(fake_transport)
    async with app.run_test(size=(40, 15)) as pilot:
        for sid in ("dashboard", "trade", "approvals", "logs", "help"):
            app.switch_pane(sid)
            await pilot.pause()
        assert True  # no crash


@pytest.mark.asyncio
async def test_backend_integrations_never_returns_secrets():
    from ai_trader.app import create_app

    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/tui/config/integrations")
        assert resp.status_code == 200
        text = json.dumps(resp.json())
        # Must be booleans only for credential fields, never raw keys.
        assert "sk-" not in text
        assert "api_key" in text  # field name ok
        body = resp.json()
        assert isinstance(body["broker"]["api_key"], bool)
