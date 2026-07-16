"""Persistent left navigation rail (keyboard-first)."""

from __future__ import annotations

from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

# (screen_id, label, number hotkey shown)
NAV_ITEMS: list[tuple[str, str]] = [
    ("dashboard", "Dashboard"),
    ("trade", "Trade"),
    ("positions", "Positions"),
    ("approvals", "Approvals"),
    ("logs", "Logs"),
    ("strategies", "Strategies"),
    ("agents", "Agents"),
    ("rl", "RL"),
    ("diagnostics", "Diagnostics"),
    ("settings", "Settings"),
    ("apiconfig", "API Config"),
    ("help", "Help"),
]


class NavRail(OptionList):
    """List of screens. Emits :class:`NavRail.Selected` when a screen is chosen."""

    DEFAULT_CSS = """
    NavRail {
        dock: left;
        width: 18;
        height: 100%;
        border-right: solid $panel;
        padding: 0;
    }
    NavRail:focus { border-right: solid $accent; }
    """

    class Selected(Message):
        def __init__(self, screen_id: str) -> None:
            self.screen_id = screen_id
            super().__init__()

    def __init__(self, **kwargs):
        options = [
            Option(f"{self._hotkey(i)}  {label}", id=screen_id)
            for i, (screen_id, label) in enumerate(NAV_ITEMS)
        ]
        super().__init__(*options, **kwargs)

    @staticmethod
    def _hotkey(index: int) -> str:
        # 1..9 then 0 for the tenth; blank for the rest (palette-only).
        if index < 9:
            return str(index + 1)
        if index == 9:
            return "0"
        return " "

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.post_message(self.Selected(event.option.id))

    def highlight_screen(self, screen_id: str) -> None:
        for i, (sid, _) in enumerate(NAV_ITEMS):
            if sid == screen_id:
                self.highlighted = i
                return
