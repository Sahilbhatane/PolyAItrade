"""Client-side UI state store.

``AppStore`` is the single source of truth the widgets render from. It holds
*mirrored* backend state only — no business logic, no broker access. The App's
transport workers write into it; widgets read slices and update their own
reactive attributes so only changed widgets repaint.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConnectionState(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


@dataclass
class AppStore:
    """Mirror of backend read-model + live event feed."""

    connection: ConnectionState = ConnectionState.CONNECTING
    last_error: str | None = None

    snapshot: dict[str, Any] = field(default_factory=dict)
    positions: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    strategies: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    rl: dict[str, Any] = field(default_factory=dict)
    integrations: dict[str, Any] = field(default_factory=dict)

    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))
    event_count: int = 0
    dropped_local: int = 0

    def add_event(self, event: dict[str, Any]) -> None:
        if len(self.events) == self.events.maxlen:
            self.dropped_local += 1
        self.events.append(event)
        self.event_count += 1

    # Convenience accessors used by widgets (never raise on missing data) ---

    @property
    def market_open(self) -> bool:
        return bool(self.snapshot.get("market_open", False))

    @property
    def kill_switch_active(self) -> bool:
        return bool(self.snapshot.get("kill_switch", {}).get("active", False))

    @property
    def pending_approval_count(self) -> int:
        return int(self.snapshot.get("approvals", {}).get("pending", len(self.approvals)))

    @property
    def risk(self) -> dict[str, Any]:
        return self.snapshot.get("risk", {})


HELP_TEXT: dict[str, str] = {
    "pnl": (
        "Profit & Loss: how much money you have made or lost today. Green is "
        "profit, red is loss."
    ),
    "confidence": (
        "Confidence: how strongly the strategy believes in a trade, from 0 to "
        "100%. Trades below the threshold are not taken."
    ),
    "exposure": (
        "Exposure: the total value of the market you are currently invested in. "
        "Higher exposure means more risk."
    ),
    "drawdown": (
        "Drawdown: how far your account has fallen from its highest point. A "
        "small drawdown is healthier."
    ),
    "regime": (
        "Regime: the current market mood — trending, sideways, volatile, or "
        "low-liquidity. Position sizes shrink in risky regimes."
    ),
    "approval": (
        "Approval: no real trade is placed until a human approves it here. "
        "Unapproved trades are automatically rejected after a timeout."
    ),
    "risk_budget": (
        "Remaining risk budget: how much capital you can still put at risk "
        "today before hitting your limits."
    ),
    "kill_switch": (
        "Kill switch: an emergency stop that halts ALL trading immediately. "
        "Use it if something looks wrong."
    ),
    "drawdown_daily": (
        "Daily loss: how much you have lost today as a percent of capital. "
        "Trading stops automatically past the daily limit."
    ),
}
