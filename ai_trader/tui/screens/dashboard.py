"""Dashboard — cards only, no charts. The at-a-glance operational view."""

from __future__ import annotations

from textual.containers import Grid

from ai_trader.tui.screens.base import Pane
from ai_trader.tui.store import AppStore
from ai_trader.tui.widgets.metric_card import MetricCard


def _fmt_money(value) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


class DashboardPane(Pane):
    pane_id = "dashboard"

    DEFAULT_CSS = """
    DashboardPane { padding: 1; }
    DashboardPane Grid {
        grid-size: 4;
        grid-gutter: 1;
        grid-rows: 5;
        height: auto;
    }
    """

    # (card_id, title, help_key)
    CARDS = [
        ("pnl", "Today's P&L", "pnl"),
        ("positions", "Open Positions", "exposure"),
        ("pending", "Pending Approval", "approval"),
        ("regime", "Current Regime", "regime"),
        ("risk", "Risk State", "kill_switch"),
        ("drawdown", "Daily Loss", "drawdown_daily"),
        ("capital", "Capital", "risk_budget"),
        ("risk_budget", "Risk Budget Left", "risk_budget"),
        ("trades", "Trades Today", "risk_budget"),
        ("consecutive", "Consecutive Losses", "drawdown"),
        ("market", "Market", "regime"),
        ("broker", "Broker Latency", "kill_switch"),
        ("exposure", "Exposure", "exposure"),
        ("confidence", "Confidence Thr.", "confidence"),
        ("last_trade", "Last Trade", "pnl"),
        ("queue", "Queue Health", "kill_switch"),
    ]

    def compose(self):
        with Grid():
            for card_id, title, help_key in self.CARDS:
                yield MetricCard(title, help_key=help_key, id=f"card-{card_id}")

    def on_activate(self) -> None:
        cards = list(self.query(MetricCard))
        if cards:
            cards[0].focus()

    def _card(self, card_id: str) -> MetricCard:
        return self.query_one(f"#card-{card_id}", MetricCard)

    def refresh_from_store(self, store: AppStore) -> None:
        snap = store.snapshot
        if not snap:
            return

        risk = store.risk
        balance = snap.get("balance", {})

        pnl = balance.get("pnl", risk.get("daily_pnl", 0.0))
        pnl_state = "ok" if (pnl or 0) >= 0 else "crit"
        self._card("pnl").set_metric(_fmt_money(pnl), pnl_state)

        pos = snap.get("positions", {})
        self._card("positions").set_metric(str(pos.get("count", 0)), "ok")
        self._card("exposure").set_metric(_fmt_money(pos.get("exposure", 0.0)), "ok")

        pending = store.pending_approval_count
        self._card("pending").set_metric(str(pending), "warn" if pending else "ok")

        regime = (snap.get("regime") or {}).get("label", "—")
        self._card("regime").set_metric(str(regime).upper() if regime else "—", "info")

        if store.kill_switch_active:
            self._card("risk").set_metric("HALTED", "crit")
        else:
            self._card("risk").set_metric("ACTIVE", "ok")

        dl = risk.get("daily_loss_pct", 0.0)
        dl_limit = risk.get("daily_loss_limit", 0.05) or 0.05
        dl_state = "crit" if dl >= dl_limit else ("warn" if dl >= dl_limit * 0.6 else "ok")
        self._card("drawdown").set_metric(_fmt_pct(dl), dl_state)

        self._card("capital").set_metric(_fmt_money(risk.get("capital")), "ok")
        self._card("risk_budget").set_metric(_fmt_money(snap.get("remaining_risk_budget")), "ok")

        trades = risk.get("trades_today", 0)
        max_trades = risk.get("max_trades_per_day", 5)
        t_state = "warn" if max_trades and trades >= max_trades - 1 else "ok"
        self._card("trades").set_metric(f"{trades}/{max_trades}", t_state)

        cl = risk.get("consecutive_losses", 0)
        maxcl = risk.get("max_consecutive_losses", 3)
        cl_state = "crit" if maxcl and cl >= maxcl else ("warn" if cl else "ok")
        self._card("consecutive").set_metric(f"{cl}/{maxcl}", cl_state)

        self._card("market").set_metric(
            "OPEN" if store.market_open else "CLOSED", "ok" if store.market_open else "off"
        )

        lat = snap.get("broker", {}).get("latency_ms")
        lat_txt = f"{lat:.0f}ms" if isinstance(lat, (int, float)) else "—"
        self._card("broker").set_metric(lat_txt, "ok" if isinstance(lat, (int, float)) else "off")

        self._card("confidence").set_metric("—", "info")

        last = snap.get("last_trade")
        if last:
            self._card("last_trade").set_metric(
                f"{last.get('side', '')} {last.get('symbol', '')}", "info"
            )
        else:
            self._card("last_trade").set_metric("none", "off")

        hub = snap.get("event_hub", {})
        dropped = int(hub.get("dropped_events", 0)) + store.dropped_local
        self._card("queue").set_metric(
            f"{hub.get('subscribers', 0)} subs", "warn" if dropped else "ok"
        )
