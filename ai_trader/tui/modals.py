"""Modal overlays: destructive-action confirmation and contextual help."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ConfirmModal(ModalScreen[bool]):
    """Yes/No confirmation for destructive actions (Live trade, kill switch)."""

    BINDINGS = [
        ("y", "confirm", "Yes"),
        ("n", "cancel", "No"),
        ("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    ConfirmModal { align: center middle; }
    ConfirmModal Vertical {
        width: 60; height: auto; padding: 1 2;
        border: thick $error; background: $surface;
    }
    ConfirmModal Label { text-style: bold; margin-bottom: 1; }
    ConfirmModal .buttons { height: 3; align: center middle; }
    ConfirmModal Button { margin: 0 1; }
    """

    def __init__(self, prompt: str):
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._prompt)
            yield Static("[dim]y = confirm    n / Esc = cancel[/dim]")
            with Vertical(classes="buttons"):
                yield Button("Confirm", variant="error", id="confirm")
                yield Button("Cancel", variant="primary", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class HelpModal(ModalScreen[None]):
    """Plain-English explanation for the focused metric (F12)."""

    BINDINGS = [("escape", "dismiss", "Close"), ("f12", "dismiss", "Close")]

    DEFAULT_CSS = """
    HelpModal { align: center middle; }
    HelpModal Vertical {
        width: 70; height: auto; padding: 1 2;
        border: round $accent; background: $surface;
    }
    HelpModal Label { text-style: bold; color: $accent; margin-bottom: 1; }
    """

    def __init__(self, title: str, body: str):
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title)
            yield Static(self._body)
            yield Static("\n[dim]Esc to close[/dim]")

    def action_dismiss(self) -> None:
        self.dismiss(None)
