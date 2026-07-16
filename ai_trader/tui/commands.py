"""Command palette provider (Ctrl+P) — every action is searchable."""

from __future__ import annotations

from functools import partial

from textual.command import DiscoveryHit, Hit, Hits, Provider

from ai_trader.tui.widgets.nav import NAV_ITEMS

# (title, action-callback-name on the app)
_ACTIONS: list[tuple[str, str]] = [
    ("Kill switch: ENGAGE (halt all trading)", "engage_kill_switch"),
    ("Kill switch: DISENGAGE (resume trading)", "disengage_kill_switch"),
    ("Show contextual help", "help"),
    ("Reconnect backend", "reconnect"),
    ("Quit", "quit"),
]


class PolyVICommands(Provider):
    """Offers navigation and global actions to the command palette."""

    def _commands(self) -> list[tuple[str, object]]:
        app = self.app
        items: list[tuple[str, object]] = []
        for screen_id, label in NAV_ITEMS:
            items.append((f"Go to: {label}", partial(app.switch_pane, screen_id)))
        for title, action in _ACTIONS:
            items.append((title, partial(app.run_palette_action, action)))
        return items

    async def discover(self) -> Hits:
        for title, callback in self._commands():
            yield DiscoveryHit(title, callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for title, callback in self._commands():
            score = matcher.match(title)
            if score > 0:
                yield Hit(score, matcher.highlight(title), callback)
