"""Approvals — human-in-the-loop gate. One key to approve, one to reject."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static

from ai_trader.tui.messages import ApprovalDecision
from ai_trader.tui.screens.base import Pane
from ai_trader.tui.store import AppStore


class ApprovalsPane(Pane):
    pane_id = "approvals"

    can_focus = True

    BINDINGS = [
        ("a", "approve", "Approve"),
        ("r", "reject", "Reject"),
    ]

    DEFAULT_CSS = """
    ApprovalsPane { padding: 1; }
    ApprovalsPane #approvals-list { height: 1fr; }
    ApprovalsPane .approval-card {
        border: round $warning;
        padding: 1;
        margin-bottom: 1;
    }
    ApprovalsPane .approval-card.-selected { border: round $accent; }
    ApprovalsPane #approvals-empty { color: $text-muted; padding: 1; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected = 0
        self._ids: list[str] = []

    def compose(self):
        yield Static("No pending approvals.", id="approvals-empty")
        yield VerticalScroll(id="approvals-list")

    @property
    def key_hints(self) -> str:
        return "a approve  r reject  j/k move  ^P palette  F12 help"

    def on_activate(self) -> None:
        self.focus()

    def refresh_from_store(self, store: AppStore) -> None:
        approvals = store.approvals
        ids = [a.get("request_id", "") for a in approvals]
        empty = self.query_one("#approvals-empty", Static)
        listing = self.query_one("#approvals-list", VerticalScroll)

        if ids == self._ids:
            self._apply_selection()
            return
        self._ids = ids
        self._selected = min(self._selected, max(len(ids) - 1, 0))

        listing.remove_children()
        empty.display = not approvals
        listing.display = bool(approvals)

        for a in approvals:
            td = a.get("trade_details", {})
            text = (
                f"[b]{td.get('side', '?')} {td.get('symbol', '?')}[/b]  "
                f"qty {td.get('quantity', '?')}  @ {td.get('price', '?')}\n"
                f"  stop_loss: {td.get('stop_loss', '—')}   "
                f"confidence: {td.get('confidence', '—')}   "
                f"risk/reward: {td.get('risk_reward', '—')}\n"
                f"  reasoning: {td.get('reasoning', '—')}\n"
                f"  requested: {a.get('created_at', '—')}   id: {a.get('request_id', '')[:8]}"
            )
            card = Static(text, classes="approval-card")
            listing.mount(card)
        self._apply_selection()

    def _apply_selection(self) -> None:
        cards = list(self.query(".approval-card"))
        for i, card in enumerate(cards):
            card.set_class(i == self._selected, "-selected")

    def action_approve(self) -> None:
        rid = self._current_id()
        if rid:
            self.post_message(ApprovalDecision(rid, "approve"))

    def action_reject(self) -> None:
        rid = self._current_id()
        if rid:
            self.post_message(ApprovalDecision(rid, "reject", reason="Rejected by operator"))

    def _current_id(self) -> str | None:
        if 0 <= self._selected < len(self._ids):
            return self._ids[self._selected]
        return None

    def on_key(self, event) -> None:
        if event.key in ("j", "down"):
            self._selected = min(self._selected + 1, max(len(self._ids) - 1, 0))
            self._apply_selection()
            event.stop()
        elif event.key in ("k", "up"):
            self._selected = max(self._selected - 1, 0)
            self._apply_selection()
            event.stop()
