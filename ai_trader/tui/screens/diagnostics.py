"""Diagnostics — latencies, queues, tasks, threads, memory."""

from __future__ import annotations

from textual.widgets import DataTable

from ai_trader.tui.screens.base import Pane
from ai_trader.tui.store import AppStore


class DiagnosticsPane(Pane):
    pane_id = "diagnostics"

    DEFAULT_CSS = """
    DiagnosticsPane { padding: 1; }
    DiagnosticsPane DataTable { height: 1fr; }
    """

    def compose(self):
        yield DataTable(id="diag-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#diag-table", DataTable)
        table.add_columns("Metric", "Value")

    def refresh_from_store(self, store: AppStore) -> None:
        diag = store.diagnostics
        if not diag:
            return
        table = self.query_one("#diag-table", DataTable)
        rows = self._flatten(diag)
        sig = tuple(f"{k}={v}" for k, v in rows)
        if getattr(self, "_sig", None) == sig:
            return
        self._sig = sig
        table.clear()
        for key, val in rows:
            table.add_row(key, str(val))

    @staticmethod
    def _flatten(diag: dict) -> list[tuple[str, object]]:
        rows: list[tuple[str, object]] = []
        for key, val in diag.items():
            if isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    rows.append((f"{key}.{sub_k}", sub_v))
            else:
                rows.append((key, val))
        return rows
