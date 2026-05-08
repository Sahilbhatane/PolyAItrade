"""Tests for the multi-agent pipeline: event bus, state, individual agents, and orchestrator."""

import asyncio

import numpy as np
import pandas as pd
import pytest

from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.agents.market_data_agent import MarketDataAgent
from ai_trader.agents.signal_agent import SignalAgent
from ai_trader.agents.strategy_agent import StrategyAgent
from ai_trader.agents.risk_agent import RiskAgent
from ai_trader.agents.execution_agent import ExecutionAgent
from ai_trader.agents.orchestrator import Orchestrator
from ai_trader.broker.paper import PaperBroker


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    volume = rng.integers(5000, 50000, n).astype(int)
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = "timestamp"
    return df


# ── Event Bus ─────────────────────────────────────────────────────────────────


class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.MARKET_DATA_READY, handler)
        event = Event(
            event_type=EventType.MARKET_DATA_READY,
            payload={"symbol": "TEST"},
            source_agent="test",
        )
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].payload["symbol"] == "TEST"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        bus = EventBus()
        counts = {"a": 0, "b": 0}

        async def handler_a(event: Event):
            counts["a"] += 1

        async def handler_b(event: Event):
            counts["b"] += 1

        bus.subscribe(EventType.SIGNAL_GENERATED, handler_a)
        bus.subscribe(EventType.SIGNAL_GENERATED, handler_b)

        await bus.publish(Event(
            event_type=EventType.SIGNAL_GENERATED,
            payload={},
            source_agent="test",
        ))

        assert counts["a"] == 1
        assert counts["b"] == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.RISK_APPROVED, handler)
        bus.unsubscribe(EventType.RISK_APPROVED, handler)

        await bus.publish(Event(
            event_type=EventType.RISK_APPROVED, payload={}, source_agent="test"
        ))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_history_tracking(self):
        bus = EventBus(max_history=10)
        for i in range(15):
            await bus.publish(Event(
                event_type=EventType.PIPELINE_START, payload={"i": i}, source_agent="test"
            ))

        history = bus.get_history()
        assert len(history) == 10

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self):
        bus = EventBus()

        async def bad_handler(event: Event):
            raise RuntimeError("oops")

        async def good_handler(event: Event):
            pass  # should still be called

        bus.subscribe(EventType.PIPELINE_START, bad_handler)
        bus.subscribe(EventType.PIPELINE_START, good_handler)

        # Should not raise
        await bus.publish(Event(
            event_type=EventType.PIPELINE_START, payload={}, source_agent="test"
        ))


# ── State Manager ─────────────────────────────────────────────────────────────


class TestStateManager:
    @pytest.mark.asyncio
    async def test_write_and_read(self):
        state = StateManager()
        await state.write("key1", {"data": 42}, writer="agent_a")

        assert state.read("key1") == {"data": 42}
        assert state.has("key1")
        assert not state.has("key2")

    @pytest.mark.asyncio
    async def test_version_increments(self):
        state = StateManager()
        await state.write("k", "v1", writer="a")
        await state.write("k", "v2", writer="a")

        entry = state.read_entry("k")
        assert entry.version == 2
        assert entry.value == "v2"

    @pytest.mark.asyncio
    async def test_snapshot(self):
        state = StateManager()
        await state.write("a", 1, writer="x")
        await state.write("b", 2, writer="y")

        snap = state.snapshot()
        assert snap == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_clear(self):
        state = StateManager()
        await state.write("a", 1, writer="x")
        await state.clear()
        assert state.read("a") is None


# ── Signal Agent ──────────────────────────────────────────────────────────────


class TestSignalAgent:
    @pytest.mark.asyncio
    async def test_generates_signals(self):
        bus = EventBus()
        state = StateManager()
        df = _make_ohlcv(100)
        await state.write(StateKeys.MARKET_DATA, df, writer="test")

        agent = SignalAgent(event_bus=bus, state=state)
        result = await agent.start()

        assert result["status"] == "success"
        signals = state.read(StateKeys.SIGNALS)
        assert "rsi" in signals
        assert "macd" in signals
        assert "ma_crossover" in signals

    @pytest.mark.asyncio
    async def test_signals_in_range(self):
        bus = EventBus()
        state = StateManager()
        df = _make_ohlcv(200)
        await state.write(StateKeys.MARKET_DATA, df, writer="test")

        agent = SignalAgent(event_bus=bus, state=state)
        await agent.start()

        signals = state.read(StateKeys.SIGNALS)
        for key in ["rsi", "macd", "ma_crossover", "vwap"]:
            series = signals[key]
            assert series.min() >= -1.0
            assert series.max() <= 1.0

    @pytest.mark.asyncio
    async def test_raises_without_data(self):
        bus = EventBus()
        state = StateManager()
        agent = SignalAgent(event_bus=bus, state=state)

        with pytest.raises(ValueError, match="No market data"):
            await agent.start()


# ── Strategy Agent ────────────────────────────────────────────────────────────


class TestStrategyAgent:
    @pytest.mark.asyncio
    async def test_produces_decision(self):
        bus = EventBus()
        state = StateManager()
        df = _make_ohlcv(100)
        await state.write(StateKeys.MARKET_DATA, df, writer="test")

        signal_agent = SignalAgent(event_bus=bus, state=state)
        await signal_agent.start()

        strategy_agent = StrategyAgent(event_bus=bus, state=state)
        result = await strategy_agent.start({"bar_index": 80})

        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0
        assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_hold_on_low_confidence(self):
        bus = EventBus()
        state = StateManager()
        df = _make_ohlcv(100)
        await state.write(StateKeys.MARKET_DATA, df, writer="test")

        signal_agent = SignalAgent(event_bus=bus, state=state)
        await signal_agent.start()

        # Very high confidence threshold → should produce HOLD
        strategy_agent = StrategyAgent(
            event_bus=bus, state=state,
            config={"confidence_threshold": 0.99}
        )
        result = await strategy_agent.start({"bar_index": 50})

        assert result["action"] == "HOLD"


# ── Risk Agent ────────────────────────────────────────────────────────────────


class TestRiskAgent:
    @pytest.mark.asyncio
    async def test_approves_valid_trade(self):
        bus = EventBus()
        state = StateManager()

        decision = {
            "action": "BUY",
            "confidence": 0.8,
            "current_price": 100.0,
            "atr": 2.5,
            "signals": {},
        }
        await state.write(StateKeys.TRADE_DECISION, decision, writer="test")

        agent = RiskAgent(event_bus=bus, state=state, config={"initial_capital": 100_000.0})
        result = await agent.start()

        assert result["approved"] is True
        assert result["position_size"] > 0
        assert result["stop_loss"] > 0

    @pytest.mark.asyncio
    async def test_rejects_after_consecutive_losses(self):
        bus = EventBus()
        state = StateManager()

        agent = RiskAgent(
            event_bus=bus, state=state,
            config={"initial_capital": 100_000.0, "max_consecutive_losses": 3}
        )

        # Simulate 3 losses
        for _ in range(3):
            agent.on_trade_result(-500.0)

        decision = {"action": "BUY", "confidence": 0.9, "current_price": 100.0, "atr": 2.0, "signals": {}}
        await state.write(StateKeys.TRADE_DECISION, decision, writer="test")

        result = await agent.start()
        assert result["approved"] is False
        assert "consecutive" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_rejects_daily_trade_limit(self):
        bus = EventBus()
        state = StateManager()

        agent = RiskAgent(
            event_bus=bus, state=state,
            config={"initial_capital": 100_000.0, "max_trades_per_day": 2}
        )

        from datetime import date
        today = date.today()
        agent.on_trade_result(100.0, today)
        agent.on_trade_result(100.0, today)

        decision = {"action": "BUY", "confidence": 0.9, "current_price": 100.0, "atr": 2.0, "signals": {}}
        await state.write(StateKeys.TRADE_DECISION, decision, writer="test")

        result = await agent.start()
        assert result["approved"] is False
        assert "daily trade limit" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_hold_passes_through(self):
        bus = EventBus()
        state = StateManager()

        decision = {"action": "HOLD", "confidence": 0.2, "current_price": 100.0, "atr": None, "signals": {}}
        await state.write(StateKeys.TRADE_DECISION, decision, writer="test")

        agent = RiskAgent(event_bus=bus, state=state)
        result = await agent.start()
        assert result["approved"] is False
        assert "HOLD" in result["reason"]

    @pytest.mark.asyncio
    async def test_mandatory_stop_loss(self):
        bus = EventBus()
        state = StateManager()

        decision = {"action": "BUY", "confidence": 0.8, "current_price": 200.0, "atr": 5.0, "signals": {}}
        await state.write(StateKeys.TRADE_DECISION, decision, writer="test")

        agent = RiskAgent(event_bus=bus, state=state, config={"initial_capital": 100_000.0})
        result = await agent.start()

        assert result["approved"]
        assert result["stop_loss"] < 200.0
        assert result["stop_loss"] == 200.0 - (5.0 * 2.0)


# ── Execution Agent ───────────────────────────────────────────────────────────


class TestExecutionAgent:
    @pytest.mark.asyncio
    async def test_executes_approved_trade(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker(initial_balance=100_000.0)

        await state.write(StateKeys.MARKET_METADATA, {"symbol": "RELIANCE.NS"}, writer="test")
        verdict = {
            "approved": True,
            "reason": "All checks passed",
            "original_decision": {"action": "BUY", "current_price": 2500.0},
            "position_size": 10,
            "stop_loss": 2425.0,
        }
        await state.write(StateKeys.RISK_VERDICT, verdict, writer="test")

        agent = ExecutionAgent(event_bus=bus, state=state, broker=broker)
        result = await agent.start()

        assert result["status"] == "success"
        assert result["quantity"] == 10
        assert broker.balance < 100_000.0

    @pytest.mark.asyncio
    async def test_skips_rejected_trade(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker()

        verdict = {"approved": False, "reason": "Risk limit hit", "original_decision": {"action": "BUY"}}
        await state.write(StateKeys.RISK_VERDICT, verdict, writer="test")

        agent = ExecutionAgent(event_bus=bus, state=state, broker=broker)
        result = await agent.start()

        assert result["status"] == "skipped"
        assert broker.balance == 100_000.0

    @pytest.mark.asyncio
    async def test_handles_insufficient_funds(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker(initial_balance=100.0)

        await state.write(StateKeys.MARKET_METADATA, {"symbol": "TEST"}, writer="test")
        verdict = {
            "approved": True,
            "reason": "ok",
            "original_decision": {"action": "BUY", "current_price": 5000.0},
            "position_size": 100,
            "stop_loss": 4850.0,
        }
        await state.write(StateKeys.RISK_VERDICT, verdict, writer="test")

        agent = ExecutionAgent(event_bus=bus, state=state, broker=broker)
        result = await agent.start()

        assert result["status"] == "failed"


# ── Paper Broker ──────────────────────────────────────────────────────────────


class TestPaperBroker:
    @pytest.mark.asyncio
    async def test_buy_reduces_balance(self):
        broker = PaperBroker(initial_balance=50_000.0)
        from ai_trader.broker.base import Order, OrderSide, OrderType
        order = Order(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10, price=100.0)
        result = await broker.place_order(order)

        assert result.status.value == "filled"
        assert broker.balance == 49_000.0

    @pytest.mark.asyncio
    async def test_sell_increases_balance(self):
        broker = PaperBroker(initial_balance=50_000.0)
        from ai_trader.broker.base import Order, OrderSide, OrderType
        order = Order(symbol="TEST", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=10, price=150.0)
        result = await broker.place_order(order)

        assert result.status.value == "filled"
        assert broker.balance == 51_500.0

    @pytest.mark.asyncio
    async def test_health_check(self):
        broker = PaperBroker()
        assert await broker.health_check() is True


# ── Orchestrator (Integration) ────────────────────────────────────────────────


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_synthetic_data(self):
        """Test the complete pipeline using pre-loaded state (skipping yfinance)."""
        broker = PaperBroker(initial_balance=100_000.0)
        orch = Orchestrator(broker=broker, config={
            "strategy": {"confidence_threshold": 0.3},
            "risk": {"initial_capital": 100_000.0},
        })

        # Pre-load market data into state to avoid network calls
        df = _make_ohlcv(150)
        await orch.state.write(StateKeys.MARKET_DATA, df, writer="test")
        await orch.state.write(StateKeys.MARKET_METADATA, {
            "symbol": "TEST", "timeframe": "1d", "rows": len(df),
        }, writer="test")

        # Run signal → strategy → risk → execution manually
        signal_agent = orch._signal_agent
        await signal_agent.start()

        strategy_agent = orch._strategy_agent
        await strategy_agent.start({"bar_index": 100})

        risk_agent = orch._risk_agent
        await risk_agent.start()

        exec_agent = orch._execution_agent
        result = await exec_agent.start()

        # The pipeline should complete without error regardless of trade decision
        assert result["status"] in ("success", "skipped", "failed")

    @pytest.mark.asyncio
    async def test_event_flow_is_sequential(self):
        """Verify events fire in correct order."""
        broker = PaperBroker()
        orch = Orchestrator(broker=broker, config={
            "risk": {"initial_capital": 100_000.0},
        })

        events_received = []

        async def track(event: Event):
            events_received.append(event.event_type)

        orch.event_bus.subscribe(EventType.SIGNAL_GENERATED, track)
        orch.event_bus.subscribe(EventType.TRADE_DECISION, track)
        orch.event_bus.subscribe(EventType.TRADE_REJECTED, track)
        orch.event_bus.subscribe(EventType.RISK_APPROVED, track)
        orch.event_bus.subscribe(EventType.RISK_REJECTED, track)

        df = _make_ohlcv(100)
        await orch.state.write(StateKeys.MARKET_DATA, df, writer="test")
        await orch.state.write(StateKeys.MARKET_METADATA, {"symbol": "TEST"}, writer="test")

        await orch._signal_agent.start()
        await orch._strategy_agent.start({"bar_index": 80})
        await orch._risk_agent.start()

        # Signal should always fire first, then a decision event, then risk
        assert events_received[0] == EventType.SIGNAL_GENERATED
        assert events_received[1] in (EventType.TRADE_DECISION, EventType.TRADE_REJECTED)

    @pytest.mark.asyncio
    async def test_agents_are_loosely_coupled(self):
        """Verify agents don't hold direct references to each other."""
        broker = PaperBroker()
        orch = Orchestrator(broker=broker)

        # Each agent only knows about bus and state
        agents = [
            orch._signal_agent,
            orch._strategy_agent,
            orch._risk_agent,
            orch._execution_agent,
        ]

        for agent in agents:
            assert hasattr(agent, "_bus")
            assert hasattr(agent, "_state")
            # Should NOT have references to other agents
            for other in agents:
                if other is agent:
                    continue
                for attr_val in agent.__dict__.values():
                    assert attr_val is not other, f"{agent.agent_id} holds ref to {other.agent_id}"

    @pytest.mark.asyncio
    async def test_pipeline_error_propagation(self):
        """If SignalAgent fails (no data), pipeline should not proceed."""
        broker = PaperBroker()
        orch = Orchestrator(broker=broker)

        # Don't load any market data
        with pytest.raises(ValueError, match="No market data"):
            await orch._signal_agent.start()

        # State should NOT have signals
        assert orch.state.read(StateKeys.SIGNALS) is None

    @pytest.mark.asyncio
    async def test_risk_agent_resets(self):
        """RiskAgent reset clears all accumulated state."""
        broker = PaperBroker()
        orch = Orchestrator(broker=broker, config={"risk": {"initial_capital": 50_000.0}})

        orch.risk_agent.on_trade_result(-1000.0)
        orch.risk_agent.on_trade_result(-1000.0)
        assert orch.risk_agent._consecutive_losses == 2

        orch.risk_agent.reset()
        assert orch.risk_agent._consecutive_losses == 0
        assert orch.risk_agent._capital == 50_000.0
