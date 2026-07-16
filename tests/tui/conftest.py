"""Shared fixtures for TUI tests: an in-memory fake transport."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest


class FakeTransport:
    """Records intent calls and returns canned read-model data (no network)."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        self.snapshot_data: dict[str, Any] = {
            "market_open": True,
            "kill_switch": {"active": False},
            "broker": {"latency_ms": 12.0},
            "balance": {"cash": 100000.0, "pnl": 250.0},
            "positions": {"count": 1, "exposure": 2000.0},
            "approvals": {"pending": 0},
            "risk": {
                "capital": 100000.0,
                "daily_pnl": 250.0,
                "daily_loss_pct": 0.0,
                "consecutive_losses": 0,
                "max_consecutive_losses": 3,
                "trades_today": 1,
                "max_trades_per_day": 5,
                "daily_loss_limit": 0.05,
                "max_capital_per_trade": 0.02,
            },
            "remaining_risk_budget": 0.0,
            "last_trade": {"side": "BUY", "symbol": "RELIANCE-EQ"},
            "event_hub": {"subscribers": 1, "dropped_events": 0},
        }
        self.positions_data: list[dict[str, Any]] = [
            {"symbol": "RELIANCE-EQ", "quantity": 10, "entry_price": 200.0}
        ]
        self.approvals_data: list[dict[str, Any]] = []
        self.diagnostics_data: dict[str, Any] = {
            "broker": {"healthy": True, "latency_ms": 12.0},
            "event_hub": {"subscribers": 1, "dropped_events": 0},
            "async_tasks": 5,
            "threads": 3,
            "memory": {"rss_mb": 90.0},
        }
        self.logs_data: dict[str, Any] = {
            "records": [
                {"level": "INFO", "event": "hello", "timestamp": "2024-01-01T09:15:00Z", "logger": "x"}
            ],
            "next_cursor": None,
            "bof": True,
        }
        self.submit_result: dict[str, Any] = {
            "accepted": True,
            "status": "pending_approval",
            "position_size": 20,
            "stop_loss": 194.0,
            "risk_reward_ratio": 2.0,
        }
        self.strategies_data: dict[str, Any] = {
            "name": "rule_based_v1",
            "version": "1.0.0",
            "ml_enabled": False,
            "consensus_min_weighted_confidence": 0.35,
            "current_regime": "sideways",
            "live_weights": {},
            "strategies": [
                {"name": "rule_based_v1", "enabled": True, "weight": 1.0},
                {"name": "momentum_breakout", "enabled": True, "weight": 0.8},
            ],
        }
        self.agents_data: dict[str, Any] = {
            "agents": [
                {"name": "risk_agent", "status": "idle", "live": True},
                {"name": "market_data_agent", "status": "pipeline", "live": False},
            ],
            "recent_events": [
                {
                    "type": "risk.approved",
                    "source": "risk_agent",
                    "timestamp": "2024-01-01T09:15:00Z",
                    "correlation_id": "abc",
                },
            ],
            "kill_switch_active": False,
            "pending_request_id": None,
        }
        self.rl_data: dict[str, Any] = {
            "policy_version": "v1",
            "checkpoint_exists": False,
            "checkpoint_dir": "models/rl",
            "deployment_mode": "shadow",
            "seed": 42,
            "config": {},
        }
        self.integrations_data: dict[str, Any] = {
            "broker": {"name": "paper", "api_key": False, "client_id": False},
            "database": {"url_configured": True},
            "integrations": {
                "openai": False,
                "polygon": False,
            },
        }
        self._fail_next: set[str] = set()

    async def snapshot(self):
        return self.snapshot_data

    async def positions(self):
        return self.positions_data

    async def pending_approvals(self):
        return self.approvals_data

    async def diagnostics(self):
        return self.diagnostics_data

    async def strategies(self):
        self.calls.append(("strategies", (), {}))
        return self.strategies_data

    async def agents(self):
        self.calls.append(("agents", (), {}))
        return self.agents_data

    async def rl(self):
        self.calls.append(("rl", (), {}))
        return self.rl_data

    async def integrations(self):
        self.calls.append(("integrations", (), {}))
        return self.integrations_data

    async def logs(self, **kwargs):
        self.calls.append(("logs", (), kwargs))
        return self.logs_data

    async def submit_trade(self, intent):
        self.calls.append(("submit_trade", (intent,), {}))
        return self.submit_result

    async def respond_approval(self, request_id, action, reason=""):
        self.calls.append(("respond_approval", (request_id, action, reason), {}))
        return {"status": "ok"}

    async def set_kill_switch(self, action, reason=""):
        self.calls.append(("set_kill_switch", (action, reason), {}))
        return {"active": action == "engage"}

    async def stream_events(self) -> AsyncIterator[dict]:
        if False:  # pragma: no cover - never yields in tests
            yield {}

    async def close(self):
        self.calls.append(("close", (), {}))

    def called(self, name: str) -> bool:
        return any(c[0] == name for c in self.calls)

    def last(self, name: str):
        for c in reversed(self.calls):
            if c[0] == name:
                return c
        return None


@pytest.fixture
def fake_transport():
    return FakeTransport()
