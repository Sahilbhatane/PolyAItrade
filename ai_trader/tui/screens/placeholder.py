"""Lightweight placeholder panes for screens implemented in a later phase.

Kept intentionally simple so navigation, keymap, and layout can be exercised
end-to-end now; each will be replaced by a full implementation in Phase 5.
"""

from __future__ import annotations

from textual.widgets import Static

from ai_trader.tui.screens.base import Pane


class PlaceholderPane(Pane):
    DEFAULT_CSS = """
    PlaceholderPane { padding: 2; align: center middle; }
    PlaceholderPane Static { color: $text-muted; }
    """

    def __init__(self, pane_id: str, title: str, **kwargs):
        super().__init__(**kwargs)
        self.pane_id = pane_id
        self._title = title

    def compose(self):
        yield Static(f"{self._title}\n\n(coming in a later phase)")
