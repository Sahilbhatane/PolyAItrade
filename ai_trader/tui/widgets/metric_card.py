"""Dashboard metric card — title, big value, and plain-English help.

Cards are the only Dashboard visual (no charts, per spec). Each card carries a
``help_key`` into ``store.HELP_TEXT`` so pressing F12 explains it in plain
English for non-traders.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class MetricCard(Static):
    """A focusable card showing one metric and its semantic state."""

    DEFAULT_CSS = """
    MetricCard {
        width: 1fr;
        height: 5;
        min-width: 18;
        border: round $panel;
        padding: 0 1;
        content-align: left top;
    }
    MetricCard:focus { border: round $accent; }
    MetricCard.state-ok { border-title-color: $success; }
    MetricCard.state-warn { border-title-color: $warning; }
    MetricCard.state-crit { border-title-color: $error; }
    MetricCard .metric-value { text-style: bold; }
    """

    can_focus = True

    value: reactive[str] = reactive("—")
    state: reactive[str] = reactive("ok")

    def __init__(self, title: str, help_key: str = "", value: str = "—", **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self.help_key = help_key
        self.border_title = title
        self.set_reactive(MetricCard.value, value)

    def render(self) -> str:
        return f"[b]{self.value}[/b]"

    def watch_state(self, old: str, new: str) -> None:
        for cls in ("state-ok", "state-warn", "state-crit"):
            self.remove_class(cls)
        self.add_class(f"state-{new}")

    def on_mount(self) -> None:
        self.add_class(f"state-{self.state}")

    def set_metric(self, value: str, state: str = "ok") -> None:
        if value != self.value:
            self.value = value
        if state != self.state:
            self.state = state
