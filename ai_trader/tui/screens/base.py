"""Base class for workspace panes.

Panes live inside a ``ContentSwitcher`` and are mounted exactly once — switching
between them toggles visibility rather than rebuilding, so navigation never
triggers a full-screen redraw.
"""

from __future__ import annotations

from textual.containers import Vertical

from ai_trader.tui.store import AppStore


class Pane(Vertical):
    """A workspace pane. Subclasses set ``pane_id`` and implement hooks."""

    pane_id: str = ""

    def refresh_from_store(self, store: AppStore) -> None:
        """Update reactive widgets from the store. Default: no-op."""

    def on_activate(self) -> None:
        """Called when this pane becomes visible (trigger lazy data loads)."""

    @property
    def key_hints(self) -> str:
        """Screen-specific key hints for the status bar."""
        return "^P palette  ^K kill  F12 help  q quit"
