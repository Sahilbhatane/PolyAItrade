"""PolyVITradeApp — the Textual application shell.

Owns the transport, the AppStore, and the background workers that keep the
store in sync (SSE event stream + adaptive polling for read-model slices).
Widgets never do I/O; they render the store and post intents that the App
executes against the backend.
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import ContentSwitcher

from ai_trader.tui.commands import PolyVICommands
from ai_trader.tui.messages import (
    ApprovalDecision,
    KillSwitchIntent,
    LoadLogsIntent,
    SubmitTradeIntent,
)
from ai_trader.tui.modals import ConfirmModal, HelpModal
from ai_trader.tui.screens.agents import AgentsPane
from ai_trader.tui.screens.apiconfig import ApiConfigPane
from ai_trader.tui.screens.approvals import ApprovalsPane
from ai_trader.tui.screens.base import Pane
from ai_trader.tui.screens.dashboard import DashboardPane
from ai_trader.tui.screens.diagnostics import DiagnosticsPane
from ai_trader.tui.screens.help import HelpPane
from ai_trader.tui.screens.logs import LogsPane
from ai_trader.tui.screens.positions import PositionsPane
from ai_trader.tui.screens.rl import RLPane
from ai_trader.tui.screens.settings import SettingsPane
from ai_trader.tui.screens.strategies import StrategiesPane
from ai_trader.tui.screens.trade import TradePane
from ai_trader.tui.store import HELP_TEXT, AppStore, ConnectionState
from ai_trader.tui.transport import BackendTransport, TransportError
from ai_trader.tui.widgets.header import HeaderBar
from ai_trader.tui.widgets.metric_card import MetricCard
from ai_trader.tui.widgets.nav import NAV_ITEMS, NavRail
from ai_trader.tui.widgets.statusbar import StatusBar

# Number hotkeys: 1..9 -> first nine screens, 0 -> tenth screen.
_NAV_KEYS: dict[str, str] = {str(i + 1): sid for i, (sid, _) in enumerate(NAV_ITEMS[:9])}
if len(NAV_ITEMS) >= 10:
    _NAV_KEYS["0"] = NAV_ITEMS[9][0]


class PolyVITradeApp(App):
    """The operator interface application."""

    CSS_PATH = "app.tcss"
    COMMANDS = App.COMMANDS | {PolyVICommands}

    BINDINGS = [
        ("ctrl+p", "command_palette", "Palette"),
        ("ctrl+k", "toggle_kill_switch", "Kill switch"),
        ("f12", "contextual_help", "Help"),
        ("question_mark", "contextual_help", "Help"),
        ("q", "request_quit", "Quit"),
        ("ctrl+q", "request_quit", "Quit"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_previous", "Prev"),
    ] + [(k, f"nav('{sid}')", "") for k, sid in _NAV_KEYS.items()]

    def __init__(
        self,
        transport: BackendTransport | None = None,
        base_url: str = "http://localhost:8000",
        enable_workers: bool = True,
    ):
        super().__init__()
        self.store = AppStore()
        self._transport = transport or BackendTransport(base_url=base_url)
        self._enable_workers = enable_workers
        self._mode = "PAPER"
        self._current = "dashboard"

    # --- Composition ---------------------------------------------------

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with Horizontal(id="body"):
            yield NavRail(id="nav")
            with ContentSwitcher(initial="dashboard", id="workspace"):
                yield DashboardPane(id="dashboard")
                yield TradePane(id="trade")
                yield PositionsPane(id="positions")
                yield ApprovalsPane(id="approvals")
                yield LogsPane(id="logs")
                yield StrategiesPane(id="strategies")
                yield AgentsPane(id="agents")
                yield RLPane(id="rl")
                yield DiagnosticsPane(id="diagnostics")
                yield SettingsPane(id="settings")
                yield ApiConfigPane(id="apiconfig")
                yield HelpPane(id="help")
        yield StatusBar(id="statusbar")

    def on_mount(self) -> None:
        self.query_one(NavRail).highlight_screen(self._current)
        self._broadcast()
        # Clock/status tick — repaints only the changed pills.
        self.set_interval(1.0, self._tick)
        if self._enable_workers:
            self.start_workers()

    # --- Navigation ----------------------------------------------------

    def action_nav(self, screen_id: str) -> None:
        self.switch_pane(screen_id)

    def switch_pane(self, screen_id: str) -> None:
        try:
            switcher = self.query_one("#workspace", ContentSwitcher)
        except Exception:
            return
        if screen_id not in {c.id for c in switcher.children}:
            return
        switcher.current = screen_id
        self._current = screen_id
        self.query_one(NavRail).highlight_screen(screen_id)
        pane = self._active_pane()
        if pane is not None:
            pane.on_activate()
            pane.refresh_from_store(self.store)
            self.query_one(StatusBar).set_hints(pane.key_hints)
            self._broadcast()

    def on_nav_rail_selected(self, message: NavRail.Selected) -> None:
        self.switch_pane(message.screen_id)

    def _active_pane(self) -> Pane | None:
        try:
            switcher = self.query_one("#workspace", ContentSwitcher)
            pane = switcher.get_child_by_id(switcher.current) if switcher.current else None
            return pane if isinstance(pane, Pane) else None
        except Exception:
            return None

    # --- Store broadcast ----------------------------------------------

    def _broadcast(self) -> None:
        try:
            self.query_one(HeaderBar).refresh_from_store(self.store)
            self.query_one(StatusBar).refresh_from_store(self.store, self._mode)
        except Exception:
            return
        pane = self._active_pane()
        if pane is not None:
            pane.refresh_from_store(self.store)

    def _tick(self) -> None:
        self._broadcast()

    # --- Workers -------------------------------------------------------

    def start_workers(self) -> None:
        self.run_worker(self._poll_loop("snapshot", self._transport.snapshot, 2.0), exclusive=False)
        self.run_worker(self._poll_loop("positions", self._transport.positions, 3.0), exclusive=False)
        self.run_worker(self._poll_loop("approvals", self._transport.pending_approvals, 2.0), exclusive=False)
        self.run_worker(self._poll_loop("diagnostics", self._transport.diagnostics, 5.0), exclusive=False)
        self.run_worker(self._poll_loop("strategies", self._transport.strategies, 10.0), exclusive=False)
        self.run_worker(self._poll_loop("agents", self._transport.agents, 5.0), exclusive=False)
        self.run_worker(self._poll_loop("rl", self._transport.rl, 15.0), exclusive=False)
        self.run_worker(self._poll_loop("integrations", self._transport.integrations, 30.0), exclusive=False)
        self.run_worker(self._event_stream(), exclusive=False)

    async def _poll_loop(self, key: str, fetch, interval: float) -> None:
        while True:
            try:
                data = await fetch()
                self._apply_data(key, data)
                if self.store.connection == ConnectionState.DISCONNECTED:
                    self.store.connection = ConnectionState.CONNECTED
                self.store.last_error = None
            except TransportError as e:
                self.store.connection = ConnectionState.DEGRADED
                self.store.last_error = str(e)
            except Exception as e:  # never let a worker die silently
                self.store.last_error = f"{key}: {e}"
            self._broadcast()
            await asyncio.sleep(interval)

    def _apply_data(self, key: str, data) -> None:
        if key == "snapshot":
            self.store.snapshot = data
        elif key == "positions":
            self.store.positions = data
        elif key == "approvals":
            self.store.approvals = data
        elif key == "diagnostics":
            self.store.diagnostics = data
        elif key == "strategies":
            self.store.strategies = data
        elif key == "agents":
            self.store.agents = data
        elif key == "rl":
            self.store.rl = data
        elif key == "integrations":
            self.store.integrations = data

    async def _event_stream(self) -> None:
        backoff = 1.0
        while True:
            try:
                self.store.connection = ConnectionState.CONNECTING
                self._broadcast()
                async for event in self._transport.stream_events():
                    self.store.connection = ConnectionState.CONNECTED
                    self.store.add_event(event)
                    self._broadcast()
                    backoff = 1.0
            except TransportError as e:
                self.store.connection = ConnectionState.DISCONNECTED
                self.store.last_error = str(e)
                self._broadcast()
            except Exception as e:
                self.store.last_error = f"events: {e}"
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    # --- Intents -------------------------------------------------------

    async def on_submit_trade_intent(self, message: SubmitTradeIntent) -> None:
        intent = message.intent
        if intent.get("mode") == "live":
            def _after(confirmed: bool | None) -> None:
                if confirmed:
                    self.run_worker(self._do_submit(intent), exclusive=False)

            self.push_screen(
                ConfirmModal(
                    f"Submit LIVE {intent.get('side')} {intent.get('symbol')} "
                    f"@ {intent.get('price')}?  This can place a REAL order."
                ),
                _after,
            )
            return
        self.run_worker(self._do_submit(intent), exclusive=False)

    async def _do_submit(self, intent: dict) -> None:
        try:
            result = await self._transport.submit_trade(intent)
        except TransportError as e:
            result = {"accepted": False, "status": "error", "reason": str(e)}
        pane = self._active_pane()
        if isinstance(pane, TradePane):
            pane.show_result(result)
        self.notify(
            f"Trade {result.get('status', '?')}",
            severity="information" if result.get("accepted") else "warning",
        )

    async def on_approval_decision(self, message: ApprovalDecision) -> None:
        self.run_worker(self._do_approval(message), exclusive=False)

    async def _do_approval(self, message: ApprovalDecision) -> None:
        try:
            await self._transport.respond_approval(message.request_id, message.action, message.reason)
            self.notify(f"Approval {message.action}ed", severity="information")
        except TransportError as e:
            self.notify(f"Approval failed: {e}", severity="error")

    async def on_kill_switch_intent(self, message: KillSwitchIntent) -> None:
        self.run_worker(self._do_kill_switch(message.action, message.reason), exclusive=False)

    async def _do_kill_switch(self, action: str, reason: str) -> None:
        try:
            await self._transport.set_kill_switch(action, reason)
            self.notify(f"Kill switch {action}d", severity="warning")
        except TransportError as e:
            self.notify(f"Kill switch failed: {e}", severity="error")

    async def on_load_logs_intent(self, message: LoadLogsIntent) -> None:
        self.run_worker(self._do_load_logs(message), exclusive=False)

    async def _do_load_logs(self, message: LoadLogsIntent) -> None:
        pane = self._active_pane()
        if not isinstance(pane, LogsPane):
            return
        params = pane.query_params()
        try:
            payload = await self._transport.logs(limit=200, cursor=message.cursor, **params)
        except TransportError as e:
            self.store.last_error = str(e)
            self._broadcast()
            return
        pane.apply_logs(payload, reset=message.reset)

    # --- Global actions ------------------------------------------------

    def action_toggle_kill_switch(self) -> None:
        active = self.store.kill_switch_active
        action = "disengage" if active else "engage"
        prompt = (
            "DISENGAGE kill switch and allow trading to resume?"
            if active
            else "ENGAGE kill switch? This HALTS ALL TRADING immediately."
        )

        def _after(confirmed: bool | None) -> None:
            if confirmed:
                self.post_message(KillSwitchIntent(action, reason="operator via TUI"))

        self.push_screen(ConfirmModal(prompt), _after)

    def action_contextual_help(self) -> None:
        focused = self.focused
        if isinstance(focused, MetricCard) and focused.help_key in HELP_TEXT:
            self.push_screen(HelpModal(focused.border_title or "Help", HELP_TEXT[focused.help_key]))
            return
        body = "\n".join(f"• {v}" for v in list(HELP_TEXT.values())[:6])
        self.push_screen(HelpModal("PolyVITrade — quick help", body))

    def action_request_quit(self) -> None:
        def _after(confirmed: bool | None) -> None:
            if confirmed:
                self.exit()

        self.push_screen(ConfirmModal("Quit PolyVITrade operator interface?"), _after)

    # --- Palette action bridge ----------------------------------------

    def run_palette_action(self, action: str) -> None:
        mapping = {
            "engage_kill_switch": lambda: self.post_message(KillSwitchIntent("engage", "palette")),
            "disengage_kill_switch": lambda: self.post_message(KillSwitchIntent("disengage", "palette")),
            "help": self.action_contextual_help,
            "reconnect": self.start_workers,
            "quit": self.action_request_quit,
        }
        fn = mapping.get(action)
        if fn:
            fn()

    async def on_unmount(self) -> None:
        await self._transport.close()


def run(base_url: str = "http://localhost:8000") -> None:  # pragma: no cover
    PolyVITradeApp(base_url=base_url).run()
