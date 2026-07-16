"""Persistent top header — global system status at a glance."""

from __future__ import annotations

from datetime import datetime

from textual.containers import Horizontal
from textual.widgets import Static

from ai_trader.tui.store import AppStore, ConnectionState
from ai_trader.tui.widgets.status_pill import StatusPill


class HeaderBar(Horizontal):
    """Row of status pills. Updates only the pills whose value/state changed."""

    DEFAULT_CSS = """
    HeaderBar {
        height: 1;
        dock: top;
        background: $panel;
        width: 100%;
    }
    HeaderBar Static.title { padding: 0 1; text-style: bold; color: $accent; }
    """

    def compose(self):
        yield Static("PolyVITrade", classes="title")
        yield StatusPill("MKT", id="pill-market")
        yield StatusPill("TIME", id="pill-time")
        yield StatusPill("BRK", id="pill-broker")
        yield StatusPill("WRK", id="pill-worker")
        yield StatusPill("WS", id="pill-ws")
        yield StatusPill("API", id="pill-api")
        yield StatusPill("RISK", id="pill-risk")
        yield StatusPill("APPR", id="pill-approvals")
        yield StatusPill("NOTIF", id="pill-notif")

    def refresh_from_store(self, store: AppStore) -> None:
        snap = store.snapshot
        conn = store.connection

        self._pill("pill-time").update_pill(datetime.now().strftime("%H:%M:%S"), "info")

        market = "OPEN" if store.market_open else "CLOSED"
        self._pill("pill-market").update_pill(market, "ok" if store.market_open else "off")

        broker_lat = snap.get("broker", {}).get("latency_ms")
        broker_val = f"{broker_lat:.0f}ms" if isinstance(broker_lat, (int, float)) else "—"
        broker_state = "ok" if isinstance(broker_lat, (int, float)) else "off"
        self._pill("pill-broker").update_pill(broker_val, broker_state)

        connected = conn == ConnectionState.CONNECTED
        degraded = conn == ConnectionState.DEGRADED
        worker_state = "ok" if connected else ("warn" if degraded else "crit")
        self._pill("pill-worker").update_pill(conn.value, worker_state)

        ws_state = "ok" if connected else ("warn" if degraded else "off")
        self._pill("pill-ws").update_pill("live" if connected else "down", ws_state)

        self._pill("pill-api").update_pill(
            "ok" if conn != ConnectionState.DISCONNECTED else "down",
            "ok" if conn != ConnectionState.DISCONNECTED else "crit",
        )

        if store.kill_switch_active:
            self._pill("pill-risk").update_pill("HALTED", "crit")
        else:
            cl = store.risk.get("consecutive_losses", 0)
            maxcl = store.risk.get("max_consecutive_losses", 3)
            state = "warn" if cl and maxcl and cl >= maxcl - 1 else "ok"
            self._pill("pill-risk").update_pill("ok", state)

        pending = store.pending_approval_count
        self._pill("pill-approvals").update_pill(str(pending), "warn" if pending else "ok")

        dropped = store.dropped_local + int(snap.get("event_hub", {}).get("dropped_events", 0))
        self._pill("pill-notif").update_pill(str(store.event_count), "warn" if dropped else "off")

    def _pill(self, pill_id: str) -> StatusPill:
        return self.query_one(f"#{pill_id}", StatusPill)
