"""Tests for broker integration: AngelOne, ApprovalGate, KillSwitch, LiveExecutionAgent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_trader.broker.approval import ApprovalGate, ApprovalStatus
from ai_trader.broker.kill_switch import KillSwitch
from ai_trader.broker.paper import PaperBroker
from ai_trader.broker.base import Order, OrderSide, OrderType, OrderStatus
from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.state import StateManager, StateKeys
from ai_trader.agents.live_execution_agent import LiveExecutionAgent


# ============================================================
# ApprovalGate Tests
# ============================================================


class TestApprovalGate:
    """Test human approval layer."""

    @pytest.mark.asyncio
    async def test_auto_approve_mode(self):
        gate = ApprovalGate(auto_approve=True)
        status = await gate.request_approval("req_1", {"symbol": "RELIANCE", "side": "BUY"})
        assert status == ApprovalStatus.APPROVED
        assert len(gate.audit_log) == 1
        assert gate.audit_log[0].resolved_by == "auto"

    @pytest.mark.asyncio
    async def test_manual_approve(self):
        gate = ApprovalGate(timeout_s=5.0)

        async def approve_after_delay():
            await asyncio.sleep(0.1)
            gate.approve("req_2", by="tester")

        task = asyncio.create_task(approve_after_delay())
        status = await gate.request_approval("req_2", {"symbol": "TCS"})
        await task

        assert status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_manual_reject(self):
        gate = ApprovalGate(timeout_s=5.0)

        async def reject_after_delay():
            await asyncio.sleep(0.1)
            gate.reject("req_3", by="risk_manager", reason="Too risky")

        task = asyncio.create_task(reject_after_delay())
        status = await gate.request_approval("req_3", {"symbol": "INFY"})
        await task

        assert status == ApprovalStatus.REJECTED
        assert gate.audit_log[0].reason == "Too risky"

    @pytest.mark.asyncio
    async def test_timeout(self):
        gate = ApprovalGate(timeout_s=0.2)
        status = await gate.request_approval("req_4", {"symbol": "HDFC"})
        assert status == ApprovalStatus.TIMEOUT
        assert len(gate.pending_requests) == 0

    @pytest.mark.asyncio
    async def test_pending_requests_visible(self):
        gate = ApprovalGate(timeout_s=5.0)

        async def check_pending():
            await asyncio.sleep(0.05)
            assert len(gate.pending_requests) == 1
            assert gate.pending_requests[0].request_id == "req_5"
            gate.approve("req_5")

        task = asyncio.create_task(check_pending())
        await gate.request_approval("req_5", {"symbol": "SBIN"})
        await task

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_returns_false(self):
        gate = ApprovalGate(auto_approve=False)
        assert gate.approve("nonexistent") is False
        assert gate.reject("nonexistent") is False

    @pytest.mark.asyncio
    async def test_callback_on_request(self):
        callback_called = []

        async def on_request(req):
            callback_called.append(req.request_id)

        gate = ApprovalGate(timeout_s=5.0, on_request=on_request)

        async def approve_quickly():
            await asyncio.sleep(0.05)
            gate.approve("req_cb")

        task = asyncio.create_task(approve_quickly())
        await gate.request_approval("req_cb", {"symbol": "ITC"})
        await task

        assert "req_cb" in callback_called


# ============================================================
# KillSwitch Tests
# ============================================================


class TestKillSwitch:
    """Test emergency halt mechanism."""

    def test_initial_state_inactive(self):
        ks = KillSwitch()
        assert ks.is_active is False
        assert ks.status["active"] is False

    def test_engage_disengage(self):
        ks = KillSwitch()
        ks.engage(reason="test halt", triggered_by="unit_test")
        assert ks.is_active is True
        assert ks.status["reason"] == "test halt"

        ks.disengage(by="unit_test")
        assert ks.is_active is False

    def test_engage_idempotent(self):
        ks = KillSwitch()
        ks.engage(reason="first", triggered_by="a")
        ks.engage(reason="second", triggered_by="b")
        assert ks.status["reason"] == "first"  # first one wins

    def test_daily_loss_auto_trigger(self):
        ks = KillSwitch(auto_triggers={"daily_loss_limit": 0.03})
        ks.check_daily_loss(0.02)
        assert ks.is_active is False

        ks.check_daily_loss(0.04)
        assert ks.is_active is True
        assert "exceeded" in ks.status["reason"]

    def test_api_failure_auto_trigger(self):
        ks = KillSwitch(auto_triggers={"max_api_failures": 3})
        ks.report_api_failure()
        ks.report_api_failure()
        assert ks.is_active is False

        ks.report_api_failure()
        assert ks.is_active is True

    def test_reset_api_failures(self):
        ks = KillSwitch(auto_triggers={"max_api_failures": 3})
        ks.report_api_failure()
        ks.report_api_failure()
        ks.reset_api_failures()
        ks.report_api_failure()
        assert ks.is_active is False  # counter was reset

    def test_history_tracking(self):
        ks = KillSwitch()
        ks.engage(reason="r1", triggered_by="t1")
        ks.disengage(by="t2")
        assert len(ks.history) == 2
        assert ks.history[0].activated is True
        assert ks.history[1].activated is False


# ============================================================
# LiveExecutionAgent Tests
# ============================================================


class TestLiveExecutionAgent:
    """Test the full execution pipeline with safety gates."""

    def _setup_state(self, state: StateManager, approved: bool = True):
        """Populate state with a typical risk verdict."""
        state.write_sync(StateKeys.RISK_VERDICT, {
            "approved": approved,
            "reason": "all checks passed" if approved else "rejected",
            "original_decision": {
                "action": "BUY",
                "current_price": 1500.0,
                "confidence": 0.75,
            },
            "position_size": 10,
            "stop_loss": 1470.0,
        })
        state.write_sync(StateKeys.MARKET_METADATA, {"symbol": "RELIANCE"})

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_execution(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker()
        gate = ApprovalGate(auto_approve=True)
        ks = KillSwitch()
        ks.engage(reason="test", triggered_by="test")

        self._setup_state(state)

        agent = LiveExecutionAgent(bus, state, broker, gate, ks)
        result = await agent.run()

        assert result["status"] == "blocked"
        assert "kill_switch" in result["reason"]

    @pytest.mark.asyncio
    async def test_approval_rejected_blocks_execution(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker()
        gate = ApprovalGate(timeout_s=5.0)
        ks = KillSwitch()

        self._setup_state(state)

        async def reject_quickly():
            await asyncio.sleep(0.05)
            pending = gate.pending_requests
            if pending:
                gate.reject(pending[0].request_id, reason="not now")

        task = asyncio.create_task(reject_quickly())
        agent = LiveExecutionAgent(bus, state, broker, gate, ks)
        result = await agent.run()
        await task

        assert result["status"] == "blocked"
        assert "rejected" in result["reason"]

    @pytest.mark.asyncio
    async def test_approval_timeout_blocks_execution(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker()
        gate = ApprovalGate(timeout_s=0.1)
        ks = KillSwitch()

        self._setup_state(state)

        agent = LiveExecutionAgent(bus, state, broker, gate, ks)
        result = await agent.run()

        assert result["status"] == "blocked"
        assert "timeout" in result["reason"]

    @pytest.mark.asyncio
    async def test_successful_execution_with_auto_approve(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker(initial_balance=500_000.0)
        gate = ApprovalGate(auto_approve=True)
        ks = KillSwitch()

        self._setup_state(state)

        agent = LiveExecutionAgent(bus, state, broker, gate, ks, config={"seed": 42})
        result = await agent.run()

        assert result["status"] == "success"
        assert result["side"] == "buy"
        assert result["quantity"] == 10
        assert result["fill_price"] > 0
        assert "slippage_cost" in result

    @pytest.mark.asyncio
    async def test_not_approved_verdict_skips(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker()
        gate = ApprovalGate(auto_approve=True)
        ks = KillSwitch()

        self._setup_state(state, approved=False)

        agent = LiveExecutionAgent(bus, state, broker, gate, ks)
        result = await agent.run()

        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_broker_failure_triggers_kill_switch_counter(self):
        bus = EventBus()
        state = StateManager()
        gate = ApprovalGate(auto_approve=True)
        ks = KillSwitch(auto_triggers={"max_api_failures": 2})

        self._setup_state(state)

        # Broker that always rejects
        broker = PaperBroker(initial_balance=0.0)

        agent = LiveExecutionAgent(bus, state, broker, gate, ks)
        result1 = await agent.run()
        assert result1["status"] == "failed"

        # Re-populate state for second attempt
        self._setup_state(state)
        result2 = await agent.run()
        assert ks.is_active is True  # triggered after 2 failures

    @pytest.mark.asyncio
    async def test_fallback_broker_used_on_primary_failure(self):
        bus = EventBus()
        state = StateManager()
        gate = ApprovalGate(auto_approve=True)
        ks = KillSwitch()

        self._setup_state(state)

        # Primary broker always raises
        primary = MagicMock(spec=PaperBroker)
        primary.place_order = AsyncMock(side_effect=ConnectionError("API down"))

        # Fallback broker succeeds
        fallback = PaperBroker(initial_balance=500_000.0)

        agent = LiveExecutionAgent(
            bus, state, primary, gate, ks, fallback_broker=fallback, config={"seed": 42}
        )
        result = await agent.run()

        assert result["status"] == "success"
        assert result["fill_price"] > 0

    @pytest.mark.asyncio
    async def test_kill_switch_engaged_during_approval_blocks(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker()
        gate = ApprovalGate(timeout_s=5.0)
        ks = KillSwitch()

        self._setup_state(state)

        async def approve_then_kill():
            await asyncio.sleep(0.05)
            ks.engage(reason="emergency", triggered_by="test")
            pending = gate.pending_requests
            if pending:
                gate.approve(pending[0].request_id)

        task = asyncio.create_task(approve_then_kill())
        agent = LiveExecutionAgent(bus, state, broker, gate, ks)
        result = await agent.run()
        await task

        assert result["status"] == "blocked"
        assert "during_approval" in result["reason"]


# ============================================================
# AngelOneBroker Tests (mocked)
# ============================================================


class TestAngelOneBrokerMocked:
    """Test AngelOne broker with mocked SmartAPI calls."""

    @pytest.mark.asyncio
    async def test_place_order_success(self):
        with patch.dict("sys.modules", {"SmartApi": MagicMock(), "pyotp": MagicMock()}):
            from ai_trader.broker.angelone import AngelOneBroker

            broker = AngelOneBroker(
                api_key="test_key",
                client_id="test_id",
                password="test_pass",
                totp_secret="JBSWY3DPEHPK3PXP",
                max_retries=1,
            )

            mock_session = MagicMock()
            mock_session.placeOrder.return_value = "order_123"
            mock_session.orderBook.return_value = {
                "data": [{"orderid": "order_123", "orderstatus": "complete", "averageprice": "1505.5"}]
            }
            broker._session = mock_session
            broker._last_login = 9999999999.0  # prevent re-login

            order = Order(
                symbol="RELIANCE-EQ",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=5,
                price=1500.0,
                metadata={"token": "2885"},
            )

            result = await broker.place_order(order)
            assert result.status == OrderStatus.FILLED
            assert result.metadata["broker_order_id"] == "order_123"
            assert result.filled_price == 1505.5

    @pytest.mark.asyncio
    async def test_place_order_api_failure_retries(self):
        with patch.dict("sys.modules", {"SmartApi": MagicMock(), "pyotp": MagicMock()}):
            from ai_trader.broker.angelone import AngelOneBroker

            broker = AngelOneBroker(
                api_key="k",
                client_id="c",
                password="p",
                totp_secret="s",
                max_retries=2,
                retry_delay_s=0.01,
            )

            mock_session = MagicMock()
            mock_session.placeOrder.side_effect = ConnectionError("timeout")
            broker._session = mock_session
            broker._last_login = 9999999999.0

            order = Order(
                symbol="TCS-EQ",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1,
                price=3500.0,
                metadata={"token": "11536"},
            )

            result = await broker.place_order(order)
            assert result.status == OrderStatus.REJECTED
            assert "failed after retries" in result.metadata.get("reject_reason", "")

    @pytest.mark.asyncio
    async def test_get_positions(self):
        with patch.dict("sys.modules", {"SmartApi": MagicMock(), "pyotp": MagicMock()}):
            from ai_trader.broker.angelone import AngelOneBroker

            broker = AngelOneBroker(
                api_key="k", client_id="c", password="p", totp_secret="s"
            )

            mock_session = MagicMock()
            mock_session.position.return_value = {
                "data": [
                    {
                        "tradingsymbol": "RELIANCE-EQ",
                        "exchange": "NSE",
                        "netqty": "10",
                        "averageprice": "1500.0",
                        "pnl": "250.0",
                        "producttype": "INTRADAY",
                    }
                ]
            }
            broker._session = mock_session
            broker._last_login = 9999999999.0

            positions = await broker.get_positions()
            assert len(positions) == 1
            assert positions[0]["symbol"] == "RELIANCE-EQ"
            assert positions[0]["quantity"] == 10

    @pytest.mark.asyncio
    async def test_get_balance(self):
        with patch.dict("sys.modules", {"SmartApi": MagicMock(), "pyotp": MagicMock()}):
            from ai_trader.broker.angelone import AngelOneBroker

            broker = AngelOneBroker(
                api_key="k", client_id="c", password="p", totp_secret="s"
            )

            mock_session = MagicMock()
            mock_session.rmsLimit.return_value = {
                "data": {
                    "net": "500000",
                    "utiliseddebits": "15000",
                    "availablecash": "485000",
                    "collateral": "0",
                }
            }
            broker._session = mock_session
            broker._last_login = 9999999999.0

            balance = await broker.get_account_balance()
            assert balance["cash"] == 500000.0
            assert balance["margin_available"] == 485000.0

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        with patch.dict("sys.modules", {"SmartApi": MagicMock(), "pyotp": MagicMock()}):
            from ai_trader.broker.angelone import AngelOneBroker

            broker = AngelOneBroker(
                api_key="k", client_id="c", password="p", totp_secret="s"
            )

            mock_session = MagicMock()
            mock_session.getProfile.return_value = {"status": True, "data": {"name": "Test"}}
            broker._session = mock_session
            broker._auth_token = "jwt_token"
            broker._last_login = 9999999999.0

            assert await broker.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        with patch.dict("sys.modules", {"SmartApi": MagicMock(), "pyotp": MagicMock()}):
            from ai_trader.broker.angelone import AngelOneBroker

            broker = AngelOneBroker(
                api_key="k", client_id="c", password="p", totp_secret="s",
                max_retries=1, retry_delay_s=0.01,
            )

            mock_session = MagicMock()
            mock_session.getProfile.side_effect = Exception("connection refused")
            broker._session = mock_session
            broker._auth_token = "jwt"
            broker._last_login = 9999999999.0

            assert await broker.health_check() is False


# ============================================================
# Trading Routes Tests
# ============================================================


class TestTradingRoutes:
    """Test the FastAPI trading control endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from ai_trader.app import create_app

        app = create_app()
        with TestClient(app) as c:
            yield c

    def test_kill_switch_status(self, client):
        resp = client.get("/trading/kill-switch/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is False

    def test_kill_switch_engage_disengage(self, client):
        resp = client.post("/trading/kill-switch", json={"action": "engage", "reason": "test"})
        assert resp.status_code == 200
        assert resp.json()["active"] is True

        resp = client.post("/trading/kill-switch", json={"action": "disengage"})
        assert resp.status_code == 200
        assert resp.json()["active"] is False

    def test_kill_switch_history(self, client):
        client.post("/trading/kill-switch", json={"action": "engage", "reason": "test"})
        client.post("/trading/kill-switch", json={"action": "disengage"})

        resp = client.get("/trading/kill-switch/history")
        assert resp.status_code == 200
        assert len(resp.json()) >= 2

    def test_positions_endpoint(self, client):
        resp = client.get("/trading/positions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_balance_endpoint(self, client):
        resp = client.get("/trading/balance")
        assert resp.status_code == 200
        data = resp.json()
        assert "cash" in data

    def test_broker_health(self, client):
        resp = client.get("/trading/health")
        assert resp.status_code == 200
        assert resp.json()["healthy"] is True

    def test_approvals_pending_empty(self, client):
        resp = client.get("/trading/approvals/pending")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_approval_respond_not_found(self, client):
        resp = client.post("/trading/approvals/respond", json={
            "request_id": "nonexistent",
            "action": "approve",
        })
        assert resp.status_code == 404
