"""Intent messages posted by panes and handled by the App.

Panes never touch the transport directly — they post an intent that bubbles to
the App, which performs the network call in a worker. This keeps all I/O in one
place and guarantees the safety chain is always the backend's responsibility.
"""

from __future__ import annotations

from typing import Any

from textual.message import Message


class SubmitTradeIntent(Message):
    def __init__(self, intent: dict[str, Any]) -> None:
        self.intent = intent
        super().__init__()


class ApprovalDecision(Message):
    def __init__(self, request_id: str, action: str, reason: str = "") -> None:
        self.request_id = request_id
        self.action = action
        self.reason = reason
        super().__init__()


class KillSwitchIntent(Message):
    def __init__(self, action: str, reason: str = "") -> None:
        self.action = action
        self.reason = reason
        super().__init__()


class LoadLogsIntent(Message):
    def __init__(self, cursor: int | None = None, reset: bool = False) -> None:
        self.cursor = cursor
        self.reset = reset
        super().__init__()


class LoadSectionIntent(Message):
    """Request a lazy fetch of a read-model section (strategies/agents/rl/integrations)."""

    def __init__(self, section: str) -> None:
        self.section = section
        super().__init__()
