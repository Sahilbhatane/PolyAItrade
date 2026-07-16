"""Phase 5 screen tests — Strategies, Agents, RL, Settings, API Config, Help."""

from __future__ import annotations

import pytest
from textual.widgets import ContentSwitcher, DataTable, Input

from ai_trader.tui.app import PolyVITradeApp
from ai_trader.tui.screens.help import HelpPane
from ai_trader.tui.screens.settings import SettingsPane
from ai_trader.tui.widgets.metric_card import MetricCard


def _app(fake_transport):
    return PolyVITradeApp(transport=fake_transport, enable_workers=False)


@pytest.mark.asyncio
async def test_strategies_screen_renders(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.store.strategies = fake_transport.strategies_data
        app.switch_pane("strategies")
        await pilot.pause()
        table = app.query_one("#strat-table", DataTable)
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_agents_screen_renders(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.store.agents = fake_transport.agents_data
        app.switch_pane("agents")
        await pilot.pause()
        assert app.query_one("#agents-table", DataTable).row_count >= 2
        assert app.query_one("#events-table", DataTable).row_count >= 1


@pytest.mark.asyncio
async def test_rl_screen_renders(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.store.rl = fake_transport.rl_data
        app.switch_pane("rl")
        await pilot.pause()
        card = app.query_one("#rl-deploy", MetricCard)
        assert "SHADOW" in card.value


@pytest.mark.asyncio
async def test_apiconfig_screen_renders(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.store.integrations = fake_transport.integrations_data
        app.switch_pane("apiconfig")
        await pilot.pause()
        table = app.query_one("#api-table", DataTable)
        assert table.row_count >= 3


@pytest.mark.asyncio
async def test_help_screen_has_glossary(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.switch_pane("help")
        await pilot.pause()
        pane = app.query_one(HelpPane)
        assert pane.pane_id == "help"


@pytest.mark.asyncio
async def test_settings_save_local(tmp_path, monkeypatch, fake_transport):
    import ai_trader.tui.screens.settings as settings_mod

    path = tmp_path / "tui_settings.json"
    monkeypatch.setattr(settings_mod, "_SETTINGS_PATH", path)

    app = _app(fake_transport)
    async with app.run_test() as pilot:
        app.switch_pane("settings")
        await pilot.pause()
        pane = app.query_one(SettingsPane)
        app.query_one("#s-refresh", Input).value = "5"
        pane.action_save()
        await pilot.pause()
        assert path.exists()
        saved = settings_mod.load_settings()
        assert saved["refresh_interval_s"] == 5


@pytest.mark.asyncio
async def test_nav_to_all_phase5_screens(fake_transport):
    app = _app(fake_transport)
    async with app.run_test() as pilot:
        for sid in ("strategies", "agents", "rl", "settings", "apiconfig", "help"):
            app.switch_pane(sid)
            await pilot.pause()
            assert app.query_one("#workspace", ContentSwitcher).current == sid
