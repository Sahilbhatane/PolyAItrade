"""Logs — lazy, filterable, seekable. Never loads the whole file."""

from __future__ import annotations

from textual.containers import Horizontal
from textual.widgets import DataTable, Input

from ai_trader.tui.messages import LoadLogsIntent
from ai_trader.tui.screens.base import Pane

_LEVEL_STATE = {"ERROR": "✖", "CRITICAL": "✖", "WARNING": "▲", "INFO": "●", "DEBUG": "○"}
# Cap in-memory table rows so 100k+ log pages never blow up widget count.
_MAX_VISIBLE_ROWS = 500


class LogsPane(Pane):
    pane_id = "logs"

    BINDINGS = [
        ("slash", "focus_search", "Search"),
        ("f", "focus_level", "Filter level"),
        ("c", "clear_filters", "Clear filters"),
        ("g", "reload", "Reload"),
    ]

    DEFAULT_CSS = """
    LogsPane { padding: 1; }
    LogsPane #log-filters { height: 3; }
    LogsPane #log-filters Input { width: 1fr; margin-right: 1; }
    LogsPane #logs-table { height: 1fr; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._next_cursor: int | None = None
        self._bof = False
        self.level = ""
        self.component = ""
        self.search = ""

    def compose(self):
        with Horizontal(id="log-filters"):
            yield Input(placeholder="level (INFO/WARNING/ERROR)", id="f-level")
            yield Input(placeholder="component", id="f-component")
            yield Input(placeholder="search", id="f-search")
        yield DataTable(id="logs-table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#logs-table", DataTable)
        table.add_columns("", "Time", "Level", "Component", "Event")

    @property
    def key_hints(self) -> str:
        return "/ search  f level  c clear  g reload  PgUp older  ^P palette"

    def on_activate(self) -> None:
        self.post_message(LoadLogsIntent(reset=True))

    # --- Filters -------------------------------------------------------

    def action_focus_search(self) -> None:
        self.query_one("#f-search", Input).focus()

    def action_focus_level(self) -> None:
        self.query_one("#f-level", Input).focus()

    def action_clear_filters(self) -> None:
        for wid in ("f-level", "f-component", "f-search"):
            self.query_one(f"#{wid}", Input).value = ""
        self.level = self.component = self.search = ""
        self.post_message(LoadLogsIntent(reset=True))

    def action_reload(self) -> None:
        self.post_message(LoadLogsIntent(reset=True))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.level = self.query_one("#f-level", Input).value.strip()
        self.component = self.query_one("#f-component", Input).value.strip()
        self.search = self.query_one("#f-search", Input).value.strip()
        self.post_message(LoadLogsIntent(reset=True))

    def on_key(self, event) -> None:
        if event.key == "pageup" and not self._bof and self._next_cursor is not None:
            self.post_message(LoadLogsIntent(cursor=self._next_cursor))

    # --- Rendering (called by App after fetch) -------------------------

    def apply_logs(self, payload: dict, reset: bool) -> None:
        table = self.query_one("#logs-table", DataTable)
        if reset:
            table.clear()
            self._row_count = 0
        records = payload.get("records", [])
        for rec in records:
            level = rec.get("level", "")
            glyph = _LEVEL_STATE.get(level, "·")
            table.add_row(
                glyph,
                _short_ts(rec.get("timestamp", "")),
                level,
                rec.get("logger", ""),
                rec.get("event", "") or rec.get("raw", "")[:80],
            )
        self._row_count = table.row_count
        _trim_table(table)
        self._next_cursor = payload.get("next_cursor")
        self._bof = bool(payload.get("bof"))

    def query_params(self) -> dict:
        return {
            "level": self.level or None,
            "component": self.component or None,
            "search": self.search or None,
        }


def _trim_table(table: DataTable) -> None:
    """Evict oldest rows so the widget never holds more than _MAX_VISIBLE_ROWS."""
    while table.row_count > _MAX_VISIBLE_ROWS:
        try:
            key = table.get_row_key_at(0)
            table.remove_row(key)
        except (AttributeError, IndexError, KeyError):
            rows = getattr(table, "rows", None)
            if rows:
                table.remove_row(next(iter(rows)))
            else:
                break


def _short_ts(ts: str) -> str:
    # ISO timestamp -> HH:MM:SS if possible.
    if "T" in ts:
        return ts.split("T", 1)[1][:8]
    if " " in ts:
        return ts.split(" ", 1)[1][:8]
    return ts[:8]
