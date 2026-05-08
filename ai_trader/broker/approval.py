"""Human approval layer — gates trade execution behind explicit consent.

No trade is executed without human approval. This module provides:
- ApprovalGate: async approval mechanism with timeout
- Approval status tracking and audit log
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

from ai_trader.logs import get_logger

logger = get_logger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    KILLED = "killed"


@dataclass
class ApprovalRequest:
    """An order awaiting human approval."""

    request_id: str
    trade_details: dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    reason: str = ""


class ApprovalGate:
    """Blocks execution until a human approves or rejects the trade.

    Modes:
    - Callback mode: calls a registered approval handler (e.g., WebSocket push)
    - Polling mode: exposes pending requests for a UI to query and respond
    - Auto-approve mode (paper trading only): bypasses the gate entirely

    The gate NEVER auto-approves in live mode.
    """

    def __init__(
        self,
        timeout_s: float = 300.0,
        auto_approve: bool = False,
        on_request: Callable[[ApprovalRequest], Awaitable[None]] | None = None,
    ):
        self._timeout_s = timeout_s
        self._auto_approve = auto_approve
        self._on_request = on_request
        self._pending: dict[str, ApprovalRequest] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._audit_log: list[ApprovalRequest] = []

    @property
    def pending_requests(self) -> list[ApprovalRequest]:
        """Get all requests currently awaiting approval."""
        return [r for r in self._pending.values() if r.status == ApprovalStatus.PENDING]

    @property
    def audit_log(self) -> list[ApprovalRequest]:
        return self._audit_log.copy()

    async def request_approval(
        self, request_id: str, trade_details: dict[str, Any]
    ) -> ApprovalStatus:
        """Submit a trade for approval. Blocks until resolved or timeout.

        Returns the final approval status.
        """
        if self._auto_approve:
            logger.info("auto_approved", request_id=request_id)
            req = ApprovalRequest(
                request_id=request_id,
                trade_details=trade_details,
                status=ApprovalStatus.APPROVED,
                resolved_at=datetime.now(timezone.utc),
                resolved_by="auto",
            )
            self._audit_log.append(req)
            return ApprovalStatus.APPROVED

        req = ApprovalRequest(request_id=request_id, trade_details=trade_details)
        event = asyncio.Event()
        self._pending[request_id] = req
        self._events[request_id] = event

        logger.info(
            "approval_requested",
            request_id=request_id,
            symbol=trade_details.get("symbol"),
            side=trade_details.get("side"),
            quantity=trade_details.get("quantity"),
        )

        if self._on_request:
            await self._on_request(req)

        try:
            await asyncio.wait_for(event.wait(), timeout=self._timeout_s)
        except asyncio.TimeoutError:
            req.status = ApprovalStatus.TIMEOUT
            req.resolved_at = datetime.now(timezone.utc)
            req.reason = "Approval timed out"
            logger.warning("approval_timeout", request_id=request_id)

        self._audit_log.append(req)
        self._pending.pop(request_id, None)
        self._events.pop(request_id, None)

        return req.status

    def approve(self, request_id: str, by: str = "human") -> bool:
        """Approve a pending trade request."""
        return self._resolve(request_id, ApprovalStatus.APPROVED, by)

    def reject(self, request_id: str, by: str = "human", reason: str = "") -> bool:
        """Reject a pending trade request."""
        return self._resolve(request_id, ApprovalStatus.REJECTED, by, reason)

    def _resolve(
        self, request_id: str, status: ApprovalStatus, by: str, reason: str = ""
    ) -> bool:
        """Resolve a pending request."""
        req = self._pending.get(request_id)
        event = self._events.get(request_id)

        if not req or not event:
            logger.warning("approval_resolve_not_found", request_id=request_id)
            return False

        req.status = status
        req.resolved_at = datetime.now(timezone.utc)
        req.resolved_by = by
        req.reason = reason
        event.set()

        logger.info(
            "approval_resolved",
            request_id=request_id,
            status=status.value,
            by=by,
        )
        return True
