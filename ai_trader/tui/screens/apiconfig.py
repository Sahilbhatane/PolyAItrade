"""API Configuration — integration status only, never secrets."""

from __future__ import annotations

from textual.widgets import DataTable, Static

from ai_trader.tui.screens.base import Pane
from ai_trader.tui.store import AppStore

_INTEGRATIONS = [
    ("Angel One API Key", "broker.api_key"),
    ("Angel One Client ID", "broker.client_id"),
    ("Angel One Password", "broker.password"),
    ("Angel One TOTP", "broker.totp_secret"),
    ("PostgreSQL / SQLite", "database.url_configured"),
    ("Alpha Vantage", "integrations.alpha_vantage"),
    ("Polygon", "integrations.polygon"),
    ("Finnhub", "integrations.finnhub"),
    ("OpenAI", "integrations.openai"),
    ("Anthropic", "integrations.anthropic"),
    ("Telegram", "integrations.telegram"),
    ("Discord", "integrations.discord"),
]


class ApiConfigPane(Pane):
    pane_id = "apiconfig"

    DEFAULT_CSS = """
    ApiConfigPane { padding: 1; }
    ApiConfigPane #api-note { height: 2; color: $text-muted; }
    ApiConfigPane DataTable { height: 1fr; }
    """

    def compose(self):
        yield Static(
            "Status only — secrets are never shown. Configure via .env or config.yaml.",
            id="api-note",
        )
        yield DataTable(id="api-table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        self.query_one("#api-table", DataTable).add_columns("", "Integration", "Status")

    @property
    def key_hints(self) -> str:
        return "j/k move  ^P palette  F12 help"

    def on_activate(self) -> None:
        self.query_one("#api-table", DataTable).focus()

    def refresh_from_store(self, store: AppStore) -> None:
        data = store.integrations
        if not data:
            return

        table = self.query_one("#api-table", DataTable)
        rows = []
        for label, path in _INTEGRATIONS:
            ok = _resolve(data, path)
            rows.append(("●" if ok else "○", label, "configured" if ok else "missing"))

        sig = tuple(r[1] + r[2] for r in rows)
        if getattr(self, "_sig", None) == sig:
            return
        self._sig = sig
        table.clear()
        for row in rows:
            table.add_row(*row)


def _resolve(data: dict, path: str) -> bool:
    parts = path.split(".")
    cur: object = data
    for p in parts:
        if not isinstance(cur, dict):
            return False
        cur = cur.get(p)
    return bool(cur)
