"""Central shared state manager for the agent pipeline.

Agents read/write to named slots via this manager rather than
holding direct references to each other. This eliminates coupling
and prevents race conditions (single-writer per slot, many readers).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_trader.logs import get_logger

logger = get_logger(__name__)


@dataclass
class StateEntry:
    """A single state slot with metadata."""

    value: Any
    writer: str
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1


class StateManager:
    """Thread-safe central state for the agent pipeline.

    Each slot has a single designated writer (enforced by convention).
    Multiple agents can read any slot without coordination.
    """

    def __init__(self):
        self._store: dict[str, StateEntry] = {}
        self._lock = asyncio.Lock()

    async def write(self, key: str, value: Any, writer: str) -> None:
        """Write a value to a named state slot."""
        async with self._lock:
            existing = self._store.get(key)
            version = existing.version + 1 if existing else 1
            self._store[key] = StateEntry(
                value=value,
                writer=writer,
                updated_at=datetime.now(timezone.utc),
                version=version,
            )

    def write_sync(self, key: str, value: Any, writer: str = "sync") -> None:
        """Synchronous write — for use in tests or non-async contexts."""
        existing = self._store.get(key)
        version = existing.version + 1 if existing else 1
        self._store[key] = StateEntry(
            value=value,
            writer=writer,
            updated_at=datetime.now(timezone.utc),
            version=version,
        )

    def read(self, key: str) -> Any | None:
        """Read the current value of a state slot. Returns None if not set."""
        entry = self._store.get(key)
        return entry.value if entry else None

    def read_entry(self, key: str) -> StateEntry | None:
        """Read the full state entry including metadata."""
        return self._store.get(key)

    def has(self, key: str) -> bool:
        """Check if a key exists in state."""
        return key in self._store

    def keys(self) -> list[str]:
        """List all current state keys."""
        return list(self._store.keys())

    async def clear(self) -> None:
        """Reset all state."""
        async with self._lock:
            self._store.clear()

    def snapshot(self) -> dict[str, Any]:
        """Get a read-only snapshot of all current state values."""
        return {k: v.value for k, v in self._store.items()}


# Well-known state keys used across agents
class StateKeys:
    """Constants for state slot names. Prevents typo-based bugs."""

    MARKET_DATA = "market_data"
    MARKET_METADATA = "market_metadata"
    SIGNALS = "signals"
    TRADE_DECISION = "trade_decision"
    RISK_VERDICT = "risk_verdict"
    ORDER_RESULT = "order_result"
    PORTFOLIO = "portfolio"
    PIPELINE_STATUS = "pipeline_status"
    TRADE_LOG = "trade_log"
    REFLECTION_REPORT = "reflection_report"
    WEIGHT_ADJUSTMENTS = "weight_adjustments"
    REGIME = "regime"
    STRATEGY_WEIGHTS = "strategy_weights"
    CONSENSUS_AUDIT = "consensus_audit"
    RL_WEIGHT_PROPOSAL = "rl_weight_proposal"
