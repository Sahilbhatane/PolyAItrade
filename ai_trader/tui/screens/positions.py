"""Positions — table only, no graphs."""

from __future__ import annotations

from textual.widgets import DataTable

from ai_trader.tui.screens.base import Pane
from ai_trader.tui.store import AppStore

COLUMNS = [
    "Symbol", "Qty", "Entry", "Current", "PnL",
    "Exposure", "Stop", "Target", "Confidence", "Risk", "Age",
]


class PositionsPane(Pane):
    pane_id = "positions"

    DEFAULT_CSS = """
    PositionsPane { padding: 1; }
    PositionsPane DataTable { height: 1fr; }
    """

    def compose(self):
        table = DataTable(id="positions-table", zebra_stripes=True, cursor_type="row")
        yield table

    def on_mount(self) -> None:
        table = self.query_one("#positions-table", DataTable)
        table.add_columns(*COLUMNS)

    @property
    def key_hints(self) -> str:
        return "j/k move  Enter detail  ^P palette  F12 help"

    def on_activate(self) -> None:
        self.query_one("#positions-table", DataTable).focus()

    def refresh_from_store(self, store: AppStore) -> None:
        table = self.query_one("#positions-table", DataTable)
        # Rebuild only when the row set changed; DataTable clears cheaply and we
        # keep row counts small (a portfolio, not a log).
        rows = []
        for p in store.positions:
            qty = p.get("quantity", 0)
            entry = p.get("entry_price", 0.0)
            exposure = _num(entry) * _num(qty)
            rows.append((
                str(p.get("symbol", "—")),
                str(qty),
                _fmt(entry),
                _fmt(p.get("current_price")),
                _fmt(p.get("pnl")),
                _fmt(exposure),
                _fmt(p.get("stop_loss")),
                _fmt(p.get("target")),
                _fmt(p.get("confidence")),
                str(p.get("risk", "—")),
                str(p.get("age", "—")),
            ))

        signature = tuple(r[0] + r[1] for r in rows)
        if getattr(self, "_last_sig", None) == signature:
            return
        self._last_sig = signature

        table.clear()
        for row in rows:
            table.add_row(*row)


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)
