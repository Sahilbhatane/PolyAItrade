"""System settings editor and local TUI preferences."""

from __future__ import annotations

import json
from pathlib import Path

from textual.containers import Horizontal
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from ai_trader.tui.messages import LoadConfigIntent, ResetConfigIntent, SaveConfigIntent
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
    SettingsPane { padding: 1; overflow-y: auto; }
    SettingsPane Label { color: $text-muted; margin-top: 1; }
    SettingsPane Input, SettingsPane Select { margin-bottom: 1; }
    SettingsPane #config-editor { height: 1fr; min-height: 12; }
    SettingsPane #config-actions { height: 3; }
    SettingsPane #settings-status { color: $success; height: 2; }
    SettingsPane #config-warning { color: $warning; height: auto; }
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
        yield Label("System configuration file")
        yield Select(
            [(name, name) for name in ("config.yaml", "ml_config.yaml", "strategy_config.yaml", ".env")],
            value="config.yaml",
            id="config-file",
            allow_blank=False,
        )
        yield Static(
            "Edit YAML or .env directly. Changes affect system settings only, not trade submission. "
            "Live mode, broker, strategy, and credential changes require confirmation.",
            id="config-warning",
        )
        yield TextArea("", id="config-editor")
        with Horizontal(id="config-actions"):
            yield Button("Load", id="config-load")
            yield Button("Save", variant="success", id="config-save")
            yield Button("Reset to safe defaults", variant="error", id="config-reset")
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

    def on_activate(self) -> None:
        self.post_message(LoadConfigIntent())

    def _selected_file(self) -> str:
        return str(self.query_one("#config-file", Select).value or "config.yaml")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "config-file":
            self.post_message(LoadConfigIntent(self._selected_file()))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "config-load":
            self.post_message(LoadConfigIntent(self._selected_file()))
        elif event.button.id == "config-save":
            name = self._selected_file()
            content = self.query_one("#config-editor", TextArea).text
            self.post_message(SaveConfigIntent(name, content, _is_sensitive_change(name, content)))
        elif event.button.id == "config-reset":
            self.post_message(ResetConfigIntent(self._selected_file()))

    def set_config_files(self, names: list[str]) -> None:
        select = self.query_one("#config-file", Select)
        current = self._selected_file()
        select.set_options([(name, name) for name in names])
        if names:
            select.value = current if current in names else names[0]

    def set_config_content(self, content: str) -> None:
        self.query_one("#config-editor", TextArea).text = content

    def show_config_status(self, message: str, error: bool = False) -> None:
        status = self.query_one("#settings-status", Static)
        status.update(message)
        status.styles.color = "red" if error else "green"


def _is_sensitive_change(name: str, content: str) -> bool:
    lowered = content.lower()
    return (
        name in {".env", "strategy_config.yaml"}
        or "environment:" in lowered
        or "broker:" in lowered and "name:" in lowered
        or "mode:" in lowered
        or "deployment_mode:" in lowered
        or any(token in lowered for token in ("api_key:", "api_secret:", "totp_secret:", "traderkey"))
    )
