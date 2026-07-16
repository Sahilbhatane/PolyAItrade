"""Strategies — active strategies, weights, regime, ML status."""

from __future__ import annotations

from textual.widgets import DataTable, Static

from ai_trader.tui.screens.base import Pane
from ai_trader.tui.store import AppStore


class StrategiesPane(Pane):
    pane_id = "strategies"

    DEFAULT_CSS = """
    StrategiesPane { padding: 1; }
    StrategiesPane #strat-summary { height: 3; color: $text-muted; }
    StrategiesPane DataTable { height: 1fr; }
    """

    def compose(self):
        yield Static("", id="strat-summary")
        yield DataTable(id="strat-table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#strat-table", DataTable)
        table.add_columns("Strategy", "Enabled", "Weight", "Regime Mult", "ML")

    @property
    def key_hints(self) -> str:
        return "j/k move  ^P palette  F12 help"

    def on_activate(self) -> None:
        self.query_one("#strat-table", DataTable).focus()

    def refresh_from_store(self, store: AppStore) -> None:
        data = store.strategies
        if not data:
            return

        summary = self.query_one("#strat-summary", Static)
        regime = data.get("current_regime") or "—"
        ml = "on" if data.get("ml_enabled") else "off"
        summary.update(
            f"[b]{data.get('name', '—')}[/b] v{data.get('version', '?')}  "
            f"regime: {regime}  ML: {ml}  "
            f"consensus min: {data.get('consensus_min_weighted_confidence', '—')}"
        )

        table = self.query_one("#strat-table", DataTable)
        rows = []
        live_weights = data.get("live_weights") or {}
        for s in data.get("strategies", []):
            name = s.get("name", "—")
            rows.append((
                name,
                "yes" if s.get("enabled") else "no",
                _fmt(s.get("weight")),
                _fmt(live_weights.get(name, "—")),
                "—",
            ))

        sig = tuple(r[0] + r[1] for r in rows)
        if getattr(self, "_sig", None) == sig:
            return
        self._sig = sig
        table.clear()
        for row in rows:
            table.add_row(*row)


def _fmt(v) -> str:
    if v is None or v == "—":
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)
