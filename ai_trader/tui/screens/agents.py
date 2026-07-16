"""Agents — live agent status + recent event-bus activity."""

from __future__ import annotations

from textual.widgets import DataTable, Static

from ai_trader.tui.screens.base import Pane
from ai_trader.tui.store import AppStore

_STATUS_GLYPH = {
    "idle": "●",
    "running": "◆",
    "error": "✖",
    "stopped": "○",
    "pipeline": "·",
}


class AgentsPane(Pane):
    pane_id = "agents"

    DEFAULT_CSS = """
    AgentsPane { padding: 1; }
    AgentsPane #agents-summary { height: 2; }
    AgentsPane #agents-table { height: 40%; }
    AgentsPane #events-table { height: 1fr; }
    """

    def compose(self):
        yield Static("", id="agents-summary")
        yield DataTable(id="agents-table", zebra_stripes=True, cursor_type="row")
        yield Static("[dim]Recent events[/dim]")
        yield DataTable(id="events-table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        self.query_one("#agents-table", DataTable).add_columns(
            "", "Agent", "Status", "Live"
        )
        self.query_one("#events-table", DataTable).add_columns(
            "Time", "Type", "Source", "Correlation"
        )

    @property
    def key_hints(self) -> str:
        return "j/k move  ^P palette  F12 help"

    def on_activate(self) -> None:
        self.query_one("#agents-table", DataTable).focus()

    def refresh_from_store(self, store: AppStore) -> None:
        data = store.agents
        if not data:
            return

        summary = self.query_one("#agents-summary", Static)
        ks = "HALTED" if data.get("kill_switch_active") else "ok"
        pending = data.get("pending_request_id") or "none"
        summary.update(f"Kill switch: {ks}  │  pending trade: {pending[:12]}")

        agents_table = self.query_one("#agents-table", DataTable)
        agent_rows = []
        for a in data.get("agents", []):
            status = str(a.get("status", "—"))
            glyph = _STATUS_GLYPH.get(status, "?")
            agent_rows.append((
                glyph,
                a.get("name", "—"),
                status,
                "yes" if a.get("live") else "pipeline",
            ))

        sig = tuple(r[1] + r[2] for r in agent_rows)
        if getattr(self, "_agent_sig", None) != sig:
            self._agent_sig = sig
            agents_table.clear()
            for row in agent_rows:
                agents_table.add_row(*row)

        events_table = self.query_one("#events-table", DataTable)
        events = data.get("recent_events", [])[:20]
        evt_sig = tuple(e.get("correlation_id", "") for e in events)
        if getattr(self, "_evt_sig", None) != evt_sig:
            self._evt_sig = evt_sig
            events_table.clear()
            for e in events:
                ts = e.get("timestamp", "")
                if "T" in ts:
                    ts = ts.split("T", 1)[1][:8]
                events_table.add_row(
                    ts,
                    e.get("type", "—"),
                    e.get("source", "—"),
                    (e.get("correlation_id") or "—")[:8],
                )
