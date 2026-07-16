"""Settings — local TUI preferences (persisted client-side only)."""

from __future__ import annotations

import json
from pathlib import Path

from textual.widgets import Input, Label, Select, Static

from ai_trader.tui.screens.base import Pane

_SETTINGS_PATH = Path.home() / ".polyvitrade" / "tui_settings.json"

DEFAULTS = {
    "theme": "dark",
    "refresh_interval_s": 2,
    "notification_level": "info",
    "auto_scroll_logs": True,
    "logging_level": "INFO",
    "timezone": "Asia/Kolkata",
}


def load_settings() -> dict:
    if not _SETTINGS_PATH.exists():
        return dict(DEFAULTS)
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(DEFAULTS)
        if isinstance(data, dict):
            merged.update(data)
        return merged
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULTS)


def save_settings(settings: dict) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


class SettingsPane(Pane):
    pane_id = "settings"

    BINDINGS = [("ctrl+s", "save", "Save settings")]

    DEFAULT_CSS = """
    SettingsPane { padding: 1; }
    SettingsPane Label { color: $text-muted; margin-top: 1; }
    SettingsPane Input, SettingsPane Select { margin-bottom: 1; }
    SettingsPane #settings-status { color: $success; height: 2; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._settings = load_settings()

    def compose(self):
        yield Label("Refresh interval (seconds)")
        yield Input(str(self._settings["refresh_interval_s"]), id="s-refresh", type="integer")
        yield Label("Notification level")
        yield Select(
            [("info", "info"), ("warning", "warning"), ("error", "error"), ("none", "none")],
            value=self._settings["notification_level"],
            id="s-notify",
            allow_blank=False,
        )
        yield Label("Logging level")
        yield Select(
            [("DEBUG", "DEBUG"), ("INFO", "INFO"), ("WARNING", "WARNING"), ("ERROR", "ERROR")],
            value=self._settings["logging_level"],
            id="s-loglevel",
            allow_blank=False,
        )
        yield Label("Timezone")
        yield Input(self._settings["timezone"], id="s-tz")
        yield Static("", id="settings-status")

    @property
    def key_hints(self) -> str:
        return "^S save  ^P palette  F12 help"

    def action_save(self) -> None:
        try:
            refresh = int(self.query_one("#s-refresh", Input).value or DEFAULTS["refresh_interval_s"])
        except ValueError:
            refresh = DEFAULTS["refresh_interval_s"]
        self._settings["refresh_interval_s"] = max(1, min(refresh, 60))
        self._settings["notification_level"] = self.query_one("#s-notify", Select).value or "info"
        self._settings["logging_level"] = self.query_one("#s-loglevel", Select).value or "INFO"
        self._settings["timezone"] = self.query_one("#s-tz", Input).value.strip() or DEFAULTS["timezone"]
        save_settings(self._settings)
        self.query_one("#settings-status", Static).update("✔ Settings saved locally.")
