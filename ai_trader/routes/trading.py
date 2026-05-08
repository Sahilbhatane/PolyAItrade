"""Trading control routes — approval management, kill switch, positions, balances.

Provides endpoints for human-in-the-loop trading control.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/trading", tags=["trading"])

# These will be injected at startup via app.state
_approval_gate = None
_kill_switch = None
_broker = None


def set_dependencies(approval_gate, kill_switch, broker) -> None:
    """Called at app startup to inject live instances."""
    global _approval_gate, _kill_switch, _broker
    _approval_gate = approval_gate
    _kill_switch = kill_switch
    _broker = broker


# --- Request/Response Models ---


class ApprovalResponse(BaseModel):
    request_id: str
    action: str  # "approve" or "reject"
    reason: str = ""


class KillSwitchAction(BaseModel):
    action: str = Field(..., pattern="^(engage|disengage)$")
    reason: str = ""


# --- Approval Endpoints ---


@router.get("/approvals/pending")
async def get_pending_approvals() -> list[dict[str, Any]]:
    """List all trades waiting for human approval."""
    if _approval_gate is None:
        raise HTTPException(status_code=503, detail="Approval gate not initialized")

    return [
        {
            "request_id": r.request_id,
            "trade_details": r.trade_details,
            "created_at": r.created_at.isoformat(),
        }
        for r in _approval_gate.pending_requests
    ]


@router.post("/approvals/respond")
async def respond_to_approval(response: ApprovalResponse) -> dict[str, Any]:
    """Approve or reject a pending trade."""
    if _approval_gate is None:
        raise HTTPException(status_code=503, detail="Approval gate not initialized")

    if response.action == "approve":
        success = _approval_gate.approve(response.request_id, by="api_user")
    elif response.action == "reject":
        success = _approval_gate.reject(
            response.request_id, by="api_user", reason=response.reason
        )
    else:
        raise HTTPException(status_code=400, detail="Action must be 'approve' or 'reject'")

    if not success:
        raise HTTPException(status_code=404, detail="Request not found or already resolved")

    return {"status": "ok", "request_id": response.request_id, "action": response.action}


@router.get("/approvals/history")
async def get_approval_history() -> list[dict[str, Any]]:
    """Get audit log of all approval decisions."""
    if _approval_gate is None:
        raise HTTPException(status_code=503, detail="Approval gate not initialized")

    return [
        {
            "request_id": r.request_id,
            "status": r.status.value,
            "trade_details": r.trade_details,
            "created_at": r.created_at.isoformat(),
            "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
            "resolved_by": r.resolved_by,
            "reason": r.reason,
        }
        for r in _approval_gate.audit_log
    ]


# --- Kill Switch Endpoints ---


@router.get("/kill-switch/status")
async def get_kill_switch_status() -> dict[str, Any]:
    """Get current kill switch state."""
    if _kill_switch is None:
        raise HTTPException(status_code=503, detail="Kill switch not initialized")
    return _kill_switch.status


@router.post("/kill-switch")
async def toggle_kill_switch(action: KillSwitchAction) -> dict[str, Any]:
    """Engage or disengage the kill switch."""
    if _kill_switch is None:
        raise HTTPException(status_code=503, detail="Kill switch not initialized")

    if action.action == "engage":
        _kill_switch.engage(reason=action.reason or "Manual API trigger", triggered_by="api_user")
    else:
        _kill_switch.disengage(by="api_user")

    return _kill_switch.status


@router.get("/kill-switch/history")
async def get_kill_switch_history() -> list[dict[str, Any]]:
    """Get kill switch event history."""
    if _kill_switch is None:
        raise HTTPException(status_code=503, detail="Kill switch not initialized")

    return [
        {
            "timestamp": e.timestamp.isoformat(),
            "activated": e.activated,
            "reason": e.reason,
            "triggered_by": e.triggered_by,
        }
        for e in _kill_switch.history
    ]


# --- Position & Balance Endpoints ---


@router.get("/positions")
async def get_positions() -> list[dict[str, Any]]:
    """Get current open positions from the broker."""
    if _broker is None:
        raise HTTPException(status_code=503, detail="Broker not initialized")

    try:
        return await _broker.get_positions()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Broker error: {e}")


@router.get("/balance")
async def get_balance() -> dict[str, float]:
    """Get current account balance."""
    if _broker is None:
        raise HTTPException(status_code=503, detail="Broker not initialized")

    try:
        return await _broker.get_account_balance()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Broker error: {e}")


@router.get("/health")
async def broker_health() -> dict[str, Any]:
    """Check broker connectivity."""
    if _broker is None:
        return {"healthy": False, "reason": "Broker not initialized"}

    try:
        healthy = await _broker.health_check()
        return {"healthy": healthy}
    except Exception as e:
        return {"healthy": False, "reason": str(e)}
