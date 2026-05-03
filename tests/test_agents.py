"""Tests for the base agent system."""

from typing import Any

import pytest

from ai_trader.agents.base import AgentStatus, BaseAgent, AgentMessage


class MockAgent(BaseAgent):
    """Concrete agent for testing."""

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"result": "success", "context": context}


class FailingAgent(BaseAgent):
    """Agent that raises during run()."""

    async def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        raise RuntimeError("intentional failure")


@pytest.mark.asyncio
async def test_agent_lifecycle():
    agent = MockAgent(agent_id="test_agent_1")
    assert agent.status == AgentStatus.IDLE

    result = await agent.start({"key": "value"})
    assert result == {"result": "success", "context": {"key": "value"}}
    assert agent.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_agent_error_handling():
    agent = FailingAgent(agent_id="fail_agent")
    with pytest.raises(RuntimeError, match="intentional failure"):
        await agent.start()
    assert agent.status == AgentStatus.ERROR


def test_agent_communication():
    sender = MockAgent(agent_id="sender_1")
    receiver = MockAgent(agent_id="receiver_1")

    msg = sender.communicate("receiver_1", {"signal": 0.8})
    assert msg.sender == "sender_1"
    assert msg.recipient == "receiver_1"
    assert msg.payload == {"signal": 0.8}
    assert len(sender._message_outbox) == 1

    receiver.receive(msg)
    assert len(receiver._message_inbox) == 1
    assert receiver._message_inbox[0].payload == {"signal": 0.8}


def test_agent_stop():
    agent = MockAgent(agent_id="stop_test")
    agent.stop()
    assert agent.status == AgentStatus.STOPPED
