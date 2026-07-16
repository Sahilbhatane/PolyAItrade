"""Headless Pilot tests for the operator TUI (keyboard, nav, intents, resize)."""

from __future__ import annotations

import pytest
from textual.widgets import ContentSwitcher, Input

from ai_trader.tui.app import PolyVITradeApp
from ai_trader.tui.modals import ConfirmModal
from ai_trader.tui.screens.trade import TradePane
from ai_trader.tui.store import ConnectionState
from ai_trader.tui.widgets.metric_card import MetricCard


def _app(fake_transport):
    return PolyVITradeApp(transport=fake_transport, enable_workers=False)


@pytest.mark.asyncio
async def test_app_boots_with_all_panes(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        switcher = app.query_one("#workspace", ContentSwitcher)
        assert switcher.current == "dashboard"
        assert len(list(switcher.children)) == 12
        await pilot.pause()


@pytest.mark.asyncio
async def test_nav_by_number_key(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        # Number nav works when focus is not inside a text input.
        await pilot.press("3")  # -> positions
        assert app.query_one("#workspace", ContentSwitcher).current == "positions"
        await pilot.press("5")  # -> logs
        assert app.query_one("#workspace", ContentSwitcher).current == "logs"
        await pilot.press("1")  # -> dashboard
        assert app.query_one("#workspace", ContentSwitcher).current == "dashboard"


@pytest.mark.asyncio
async def test_dashboard_renders_snapshot(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.store.snapshot = fake_transport.snapshot_data
        app._broadcast()
        await pilot.pause()
        pnl_card = app.query_one("#card-pnl", MetricCard)
        assert "250" in pnl_card.value


@pytest.mark.asyncio
async def test_trade_submit_paper_posts_intent(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.switch_pane("trade")
        await pilot.pause()
        app.query_one("#in-symbol", Input).value = "RELIANCE-EQ"
        app.query_one("#in-price", Input).value = "200"
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()
        assert fake_transport.called("submit_trade")
        _, args, _ = fake_transport.last("submit_trade")
        assert args[0]["symbol"] == "RELIANCE-EQ"
        assert args[0]["side"] == "BUY"


@pytest.mark.asyncio
async def test_trade_submit_blocks_without_price(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.switch_pane("trade")
        await pilot.pause()
        app.query_one("#in-symbol", Input).value = "X-EQ"
        # no price
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert not fake_transport.called("submit_trade")


@pytest.mark.asyncio
async def test_live_trade_requires_confirmation(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.switch_pane("trade")
        await pilot.pause()
        pane = app.query_one(TradePane)
        pane.action_set_mode("live")
        app.query_one("#in-symbol", Input).value = "X-EQ"
        app.query_one("#in-price", Input).value = "100"
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmModal)
        # Cancelling must NOT submit.
        await pilot.press("n")
        await pilot.pause()
        assert not fake_transport.called("submit_trade")


@pytest.mark.asyncio
async def test_approval_approve_posts_intent(fake_transport):
    fake_transport.approvals_data = [
        {
            "request_id": "abc-123",
            "trade_details": {"side": "BUY", "symbol": "X-EQ", "quantity": 5, "price": 100},
            "created_at": "2024-01-01T09:15:00Z",
        }
    ]
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.store.approvals = fake_transport.approvals_data
        app.switch_pane("approvals")
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        await pilot.pause()
        assert fake_transport.called("respond_approval")
        _, args, _ = fake_transport.last("respond_approval")
        assert args[0] == "abc-123"
        assert args[1] == "approve"


@pytest.mark.asyncio
async def test_kill_switch_confirm_flow(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmModal)
        await pilot.press("y")
        await pilot.pause()
        await pilot.pause()
        assert fake_transport.called("set_kill_switch")
        _, args, _ = fake_transport.last("set_kill_switch")
        assert args[0] == "engage"


@pytest.mark.asyncio
async def test_logs_load_on_activate(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.switch_pane("logs")
        await pilot.pause()
        await pilot.pause()
        assert fake_transport.called("logs")


@pytest.mark.asyncio
async def test_help_modal_opens(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        await pilot.press("f12")
        await pilot.pause()
        from ai_trader.tui.modals import HelpModal

        assert isinstance(app.screen, HelpModal)


@pytest.mark.asyncio
async def test_tiny_terminal_boots(fake_transport):
    app = _app(fake_transport)
    async with app.run_test(size=(40, 15)) as pilot:
        await pilot.pause()
        assert app.query_one("#workspace", ContentSwitcher).current == "dashboard"


@pytest.mark.asyncio
async def test_degraded_connection_shows_in_status(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.store.connection = ConnectionState.DEGRADED
        app.store.last_error = "boom"
        app._broadcast()
        await pilot.pause()
        # No crash; status bar reflects degraded state.
        assert app.store.connection == ConnectionState.DEGRADED
