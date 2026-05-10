import pytest

from ai_trader.broker.approval import ApprovalGate, ApprovalStatus


@pytest.mark.asyncio
async def test_approval_gate_timeout_rejects():
    gate = ApprovalGate(timeout_s=0.01, auto_approve=False)
    status = await gate.request_approval(
        "rid-test",
        {"symbol": "X", "side": "BUY", "quantity": 1, "price": 1.0},
    )
    assert status == ApprovalStatus.TIMEOUT
