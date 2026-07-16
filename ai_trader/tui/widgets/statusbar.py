"""Persistent bottom status bar: shortcuts, mode, connection, memory, queue."""

from __future__ import annotations

from textual.widgets import Static

from ai_trader.tui.store import AppStore, ConnectionState

_CONN_GLYPH = {
    ConnectionState.CONNECTED: "●",
    ConnectionState.DEGRADED: "▲",
    ConnectionState.CONNECTING: "◌",
    ConnectionState.DISCONNECTED: "✖",
}


class StatusBar(Static):
    """Single-line footer, updated only when its computed text changes."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._hints = "^P palette  ^K kill  F12 help  q quit"
        self._last = ""

    def set_hints(self, hints: str) -> None:
        self._hints = hints

    def refresh_from_store(self, store: AppStore, mode: str = "PAPER") -> None:
        conn = store.connection
        glyph = _CONN_GLYPH.get(conn, "?")
        mem = store.diagnostics.get("memory", {}).get("rss_mb")
        mem_txt = f"{mem:.0f}MB" if isinstance(mem, (int, float)) else "—"
        queue = store.snapshot.get("event_hub", {}).get("subscribers", 0)
        tasks = store.diagnostics.get("async_tasks", "—")

        text = (
            f"{self._hints}  │  mode {mode}  │  {glyph} {conn.value}  "
            f"│  mem {mem_txt}  │  subs {queue}  │  tasks {tasks}  │  evt {store.event_count}"
        )
        if store.last_error:
            text += f"  │  ! {store.last_error[:40]}"
        if text != self._last:
            self._last = text
            self.update(text)
