"""Compact status indicator that conveys state by symbol AND colour.

Accessibility: colour is never the only signal — each state has a distinct
leading glyph so the pill is readable on monochrome terminals and by
colour-blind operators.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

# state -> (glyph, css class)
_STATE_STYLE: dict[str, tuple[str, str]] = {
    "ok": ("●", "pill-ok"),
    "warn": ("▲", "pill-warn"),
    "crit": ("✖", "pill-crit"),
    "off": ("○", "pill-off"),
    "info": ("◆", "pill-info"),
}


class StatusPill(Static):
    """A single ``label: glyph value`` indicator with a semantic state."""

    DEFAULT_CSS = """
    StatusPill { width: auto; padding: 0 1; }
    StatusPill.pill-ok { color: $success; }
    StatusPill.pill-warn { color: $warning; }
    StatusPill.pill-crit { color: $error; text-style: bold; }
    StatusPill.pill-off { color: $text-muted; }
    StatusPill.pill-info { color: $accent; }
    """

    label: reactive[str] = reactive("")
    value: reactive[str] = reactive("")
    state: reactive[str] = reactive("off")

    def __init__(self, label: str, value: str = "", state: str = "off", **kwargs):
        super().__init__(**kwargs)
        self.set_reactive(StatusPill.label, label)
        self.set_reactive(StatusPill.value, value)
        self.set_reactive(StatusPill.state, state)

    def _apply_class(self) -> None:
        for _, css in _STATE_STYLE.values():
            self.remove_class(css)
        _, css = _STATE_STYLE.get(self.state, _STATE_STYLE["off"])
        self.add_class(css)

    def render(self) -> str:
        glyph, _ = _STATE_STYLE.get(self.state, _STATE_STYLE["off"])
        text = f"{glyph} {self.label}"
        if self.value:
            text = f"{text} {self.value}"
        return text

    def watch_state(self) -> None:
        if self.is_mounted:
            self._apply_class()

    def on_mount(self) -> None:
        self._apply_class()

    def update_pill(self, value: str, state: str) -> None:
        """Update value+state together; only repaints if something changed."""
        if value != self.value:
            self.value = value
        if state != self.state:
            self.state = state
