"""Base agent interface for the multi-agent trading system."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ai_trader.logs import get_logger


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentMessage:
    """Message passed between agents for inter-agent communication."""

    sender: str
    recipient: str
    payload: dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None


class BaseAgent(ABC):
    """Abstract base class for all agents in the trading system.

    Provides structured logging, inter-agent communication,
    and lifecycle management. Subclasses implement domain logic in run().
    """

    def __init__(self, agent_id: str | None = None, **kwargs: Any):
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self._status = AgentStatus.IDLE
        self._logger = get_logger(self.__class__.__name__, agent_id=self.agent_id)
        self._message_inbox: list[AgentMessage] = []
        self._message_outbox: list[AgentMessage] = []

    @property
    def status(self) -> AgentStatus:
        return self._status

    @abstractmethod
    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the agent's primary task.

        Args:
            context: Runtime context/data passed to the agent.

        Returns:
            Result dictionary with agent output.
        """
        ...

    def log(self, message: str, level: str = "info", **extra: Any) -> None:
        """Emit a structured log entry with agent context."""
        log_fn = getattr(self._logger, level, self._logger.info)
        log_fn(message, agent_id=self.agent_id, status=self._status.value, **extra)

    def communicate(self, recipient: str, payload: dict[str, Any], correlation_id: str | None = None) -> AgentMessage:
        """Send a message to another agent.

        Args:
            recipient: Target agent ID.
            payload: Data to send.
            correlation_id: Optional ID linking related messages.

        Returns:
            The constructed message (also stored in outbox).
        """
        msg = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            payload=payload,
            correlation_id=correlation_id,
        )
        self._message_outbox.append(msg)
        self.log("message_sent", recipient=recipient, message_id=msg.message_id)
        return msg

    def receive(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        self._message_inbox.append(message)
        self.log("message_received", sender=message.sender, message_id=message.message_id)

    async def start(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Lifecycle wrapper around run() with status tracking and error handling."""
        self._status = AgentStatus.RUNNING
        self.log("agent_started")
        try:
            result = await self.run(context)
            self._status = AgentStatus.IDLE
            self.log("agent_completed")
            return result
        except Exception as e:
            self._status = AgentStatus.ERROR
            self.log("agent_error", level="error", error=str(e), error_type=type(e).__name__)
            raise

    def stop(self) -> None:
        """Gracefully stop the agent."""
        self._status = AgentStatus.STOPPED
        self.log("agent_stopped")
