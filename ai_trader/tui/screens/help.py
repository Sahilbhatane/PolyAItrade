"""Help — full keyboard map and plain-English glossary."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static

from ai_trader.tui.screens.base import Pane
from ai_trader.tui.store import HELP_TEXT

_KEYBOARD = """
[b]Global[/b]
  Ctrl+P   Command palette (search every action)
  1-9, 0   Jump to screen by number
  Tab      Move focus forward
  Ctrl+K   Kill switch (confirm)
  F12 / ?  Contextual help
  q        Quit (confirm)

[b]Trade[/b]
  F1/F2    Buy / Sell
  F3/F4    Paper / Live mode (Live confirms)
  Ctrl+S   Submit ticket
  Esc      Clear ticket

[b]Approvals[/b]
  a        Approve selected
  r        Reject selected
  j/k      Move selection

[b]Logs[/b]
  /        Focus search
  f        Focus level filter
  c        Clear filters
  g        Reload
  PgUp     Load older page
"""

_GLOSSARY = "\n".join(f"[b]{k}[/b]: {v}" for k, v in HELP_TEXT.items())


class HelpPane(Pane):
    pane_id = "help"

    DEFAULT_CSS = """
    HelpPane { padding: 1; }
    HelpPane VerticalScroll { height: 1fr; }
    HelpPane Static { padding-bottom: 1; }
    """

    def compose(self):
        with VerticalScroll():
            yield Static(_KEYBOARD, id="help-keys")
            yield Static("\n[b]Glossary[/b]\n" + _GLOSSARY, id="help-glossary")

    @property
    def key_hints(self) -> str:
        return "j/k scroll  ^P palette  F12 contextual help on other screens"
