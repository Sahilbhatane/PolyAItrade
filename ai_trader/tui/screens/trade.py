"""Trade — the primary, keyboard-optimized order ticket.

Hotkeys: F1 Buy · F2 Sell · F3 Paper · F4 Live · Enter Submit · Esc Clear.
The ticket only *proposes* a trade; RiskAgent (server-side) sizes the position
and sets the mandatory stop loss, and the ApprovalGate must clear it. The
result panel shows the server's verdict.
"""

from __future__ import annotations

from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Label, Select, Static

from ai_trader.tui.messages import SubmitTradeIntent
from ai_trader.tui.screens.base import Pane


class TradePane(Pane):
    pane_id = "trade"

    BINDINGS = [
        ("f1", "set_side('BUY')", "Buy"),
        ("f2", "set_side('SELL')", "Sell"),
        ("f3", "set_mode('paper')", "Paper"),
        ("f4", "set_mode('live')", "Live"),
        ("ctrl+s", "submit", "Submit"),
        ("escape", "clear", "Clear"),
    ]

    DEFAULT_CSS = """
    TradePane { padding: 1; }
    TradePane #ticket { width: 2fr; }
    TradePane #ticket-info { width: 1fr; border-left: solid $panel; padding-left: 1; }
    TradePane Input { margin-bottom: 1; }
    TradePane Label { color: $text-muted; }
    TradePane #side-buy { color: $success; text-style: bold; }
    TradePane #side-sell { color: $error; text-style: bold; }
    TradePane #mode-live { color: $error; text-style: bold; }
    TradePane #warnings { color: $warning; }
    TradePane #result { color: $accent; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._side = "BUY"
        self._mode = "paper"

    def compose(self):
        with Horizontal():
            with Vertical(id="ticket"):
                yield Label("Symbol")
                yield Input(placeholder="RELIANCE-EQ", id="in-symbol")
                yield Label("Quantity (requested — RiskAgent may resize)")
                yield Input(placeholder="0", id="in-qty", type="integer")
                yield Label("Price")
                yield Input(placeholder="0.00", id="in-price", type="number")
                yield Label("Order Type")
                yield Select(
                    [("MARKET", "MARKET"), ("LIMIT", "LIMIT")],
                    value="MARKET",
                    id="in-ordertype",
                    allow_blank=False,
                )
                yield Label("Stop Loss")
                yield Input(placeholder="optional", id="in-stop", type="number")
                yield Label("Target")
                yield Input(placeholder="optional", id="in-target", type="number")
                yield Label("Trailing Stop")
                yield Input(placeholder="optional", id="in-trail", type="number")
                yield Label("Confidence (0-1)")
                yield Input(placeholder="0.0", id="in-conf", type="number")
            with Vertical(id="ticket-info"):
                yield Static("BUY", id="side-buy")
                yield Static("PAPER", id="mode-paper")
                yield Static("", id="est")
                yield Static("", id="warnings")
                yield Static("", id="result")

    @property
    def key_hints(self) -> str:
        return "F1 Buy  F2 Sell  F3 Paper  F4 Live  ^S submit  Esc clear"

    def on_activate(self) -> None:
        self._render_state()
        try:
            self.query_one("#in-symbol", Input).focus()
        except Exception:
            pass

    # --- Actions -------------------------------------------------------

    def action_set_side(self, side: str) -> None:
        self._side = side
        self._render_state()

    def action_set_mode(self, mode: str) -> None:
        self._mode = mode
        self._render_state()

    def action_clear(self) -> None:
        for wid in ("in-symbol", "in-qty", "in-price", "in-stop", "in-target", "in-trail", "in-conf"):
            self.query_one(f"#{wid}", Input).value = ""
        self.query_one("#result", Static).update("")
        self.query_one("#warnings", Static).update("")

    def action_submit(self) -> None:
        intent, warnings = self._build_intent()
        if warnings:
            self.query_one("#warnings", Static).update("⚠ " + "  ".join(warnings))
            return
        self.query_one("#warnings", Static).update("")
        self.post_message(SubmitTradeIntent(intent))

    def show_result(self, result: dict) -> None:
        if result.get("accepted"):
            self.query_one("#result", Static).update(
                f"✔ {result.get('status')}  size={result.get('position_size')}  "
                f"stop={result.get('stop_loss')}  rr={result.get('risk_reward_ratio')}"
            )
        else:
            self.query_one("#result", Static).update(
                f"✖ {result.get('status')}: {result.get('reason', '')}"
            )

    # --- Internals -----------------------------------------------------

    def _render_state(self) -> None:
        buy = self.query_one("#side-buy", Static)
        buy.update(f"SIDE: {self._side}")
        buy.set_class(self._side == "BUY", "-active")
        mode = self.query_one("#mode-paper", Static)
        mode.update(f"MODE: {self._mode.upper()}")
        mode.styles.color = "red" if self._mode == "live" else None

    def _build_intent(self) -> tuple[dict, list[str]]:
        def _val(wid: str) -> str:
            return self.query_one(f"#{wid}", Input).value.strip()

        blockers: list[str] = []
        symbol = _val("in-symbol")
        if not symbol:
            blockers.append("symbol required")

        price = _to_float(_val("in-price"))
        if price is None or price <= 0:
            blockers.append("price must be > 0")

        try:
            order_type = self.query_one("#in-ordertype", Select).value
        except Exception:
            order_type = "MARKET"

        intent = {
            "symbol": symbol,
            "side": self._side,
            "quantity": _to_int(_val("in-qty")) or 0,
            "price": price or 0.0,
            "order_type": order_type,
            "stop_loss": _to_float(_val("in-stop")),
            "target": _to_float(_val("in-target")),
            "trailing_stop": _to_float(_val("in-trail")),
            "confidence": _to_float(_val("in-conf")) or 0.0,
            "strategy": "manual",
            "mode": self._mode,
        }
        return intent, blockers


def _to_float(v: str):
    try:
        return float(v) if v != "" else None
    except ValueError:
        return None


def _to_int(v: str):
    try:
        return int(v) if v != "" else None
    except ValueError:
        return None
