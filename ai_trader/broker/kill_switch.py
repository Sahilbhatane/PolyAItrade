"""Kill switch — emergency halt for all trading activity.

Provides a global, thread-safe mechanism to immediately stop all trading.
When engaged:
- No new orders can be placed
- Pending approvals are auto-rejected
- All agents check this before acting

The kill switch can be triggered by:
- Manual API call
- Automatic rules (e.g., daily loss limit, connectivity loss)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_trader.logs import get_logger

logger = get_logger(__name__)


@dataclass
class KillSwitchEvent:
    """Audit record of a kill switch state change."""

    timestamp: datetime
    activated: bool
    reason: str
    triggered_by: str


class KillSwitch:
    """Global trading halt mechanism.

    Thread-safe. All components must check is_active() before proceeding.
    """

    def __init__(self, auto_triggers: dict[str, Any] | None = None):
        self._lock = threading.Lock()
        self._active = False
        self._reason = ""
        self._triggered_by = ""
        self._triggered_at: datetime | None = None
        self._history: list[KillSwitchEvent] = []

        # Auto-trigger thresholds (optional)
        self._auto_triggers = auto_triggers or {}
        self._daily_loss_limit = self._auto_triggers.get("daily_loss_limit", None)
        self._max_api_failures = self._auto_triggers.get("max_api_failures", 5)
        self._api_failure_count = 0

    @property
    def is_active(self) -> bool:
        """Check if the kill switch is currently engaged."""
        with self._lock:
            return self._active

    @property
    def status(self) -> dict[str, Any]:
        """Get full kill switch state."""
        with self._lock:
            return {
                "active": self._active,
                "reason": self._reason,
                "triggered_by": self._triggered_by,
                "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
            }

    @property
    def history(self) -> list[KillSwitchEvent]:
        return self._history.copy()

    def engage(self, reason: str, triggered_by: str = "manual") -> None:
        """Activate the kill switch — halts ALL trading immediately."""
        with self._lock:
            if self._active:
                logger.warning("kill_switch_already_active", reason=self._reason)
                return

            self._active = True
            self._reason = reason
            self._triggered_by = triggered_by
            self._triggered_at = datetime.now(timezone.utc)

            event = KillSwitchEvent(
                timestamp=self._triggered_at,
                activated=True,
                reason=reason,
                triggered_by=triggered_by,
            )
            self._history.append(event)

        logger.critical(
            "KILL_SWITCH_ENGAGED",
            reason=reason,
            triggered_by=triggered_by,
        )

    def disengage(self, by: str = "manual") -> None:
        """Deactivate the kill switch — resume trading capability.

        This should require elevated privileges in production.
        """
        with self._lock:
            if not self._active:
                return

            self._active = False
            now = datetime.now(timezone.utc)

            event = KillSwitchEvent(
                timestamp=now,
                activated=False,
                reason=f"Disengaged by {by}",
                triggered_by=by,
            )
            self._history.append(event)
            self._reason = ""
            self._triggered_by = ""
            self._triggered_at = None

        logger.warning("kill_switch_disengaged", by=by)

    def check_daily_loss(self, current_loss_pct: float) -> None:
        """Auto-engage if daily loss exceeds threshold."""
        if self._daily_loss_limit is None:
            return

        if current_loss_pct >= self._daily_loss_limit:
            self.engage(
                reason=f"Daily loss {current_loss_pct:.2%} exceeded limit {self._daily_loss_limit:.2%}",
                triggered_by="auto_daily_loss",
            )

    def report_api_failure(self) -> None:
        """Track API failures — auto-engage after threshold."""
        with self._lock:
            self._api_failure_count += 1
            count = self._api_failure_count

        if count >= self._max_api_failures:
            self.engage(
                reason=f"API failure count ({count}) exceeded threshold ({self._max_api_failures})",
                triggered_by="auto_api_failure",
            )

    def reset_api_failures(self) -> None:
        """Reset the API failure counter (called on successful API call)."""
        with self._lock:
            self._api_failure_count = 0
