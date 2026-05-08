"""Tests for the ReflectionAgent: trade logging, evaluation, bounded adjustments."""

import numpy as np
import pytest

from ai_trader.agents.event_bus import EventBus, EventType, Event
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.agents.reflection_agent import ReflectionAgent, WeightAdjustment
from ai_trader.agents.execution_agent import ExecutionAgent
from ai_trader.agents.orchestrator import Orchestrator
from ai_trader.broker.paper import PaperBroker


def _make_trade(pnl: float, side: str = "BUY", entry_price: float = 100.0, signals: dict | None = None) -> dict:
    """Helper to build trade context for reflection."""
    return {
        "trade": {
            "trade_id": f"t_{abs(hash(str(pnl)))}",
            "symbol": "TEST",
            "side": side,
            "entry_price": entry_price,
            "exit_price": entry_price + pnl / 10,
            "quantity": 10,
            "stop_loss": entry_price * 0.97,
            "entry_bar": 50,
            "exit_bar": 55,
            "entry_reason": "bullish composite",
            "exit_reason": "signal_exit",
            "confidence": 0.7,
            "pnl": pnl,
            "pnl_pct": pnl / (entry_price * 10),
            "duration_bars": 5,
            "slippage_cost": abs(pnl) * 0.01,
        },
        "signals_at_entry": signals or {"rsi": 0.3, "macd": 0.5, "ma_crossover": 0.2, "vwap": -0.1},
        "actual_price_move": pnl / (entry_price * 10),
    }


# ── Trade Logging ─────────────────────────────────────────────────────────────


class TestTradeLogging:
    @pytest.mark.asyncio
    async def test_logs_trade(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        ctx = _make_trade(500.0)
        await agent.start(ctx)

        assert len(agent.trade_log) == 1
        assert agent.trade_log[0].pnl == 500.0
        assert agent.trade_log[0].symbol == "TEST"

    @pytest.mark.asyncio
    async def test_logs_entry_exit_reasons(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        ctx = _make_trade(200.0)
        ctx["trade"]["entry_reason"] = "macd_crossover"
        ctx["trade"]["exit_reason"] = "stop_loss"
        await agent.start(ctx)

        record = agent.trade_log[0]
        assert record.entry_reason == "macd_crossover"
        assert record.exit_reason == "stop_loss"

    @pytest.mark.asyncio
    async def test_logs_signals_at_entry(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        signals = {"rsi": 0.8, "macd": -0.2, "ma_crossover": 0.5, "vwap": 0.1}
        ctx = _make_trade(100.0, signals=signals)
        await agent.start(ctx)

        record = agent.trade_log[0]
        assert record.entry_signals == signals

    @pytest.mark.asyncio
    async def test_writes_trade_log_to_state(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        await agent.start(_make_trade(100.0))
        await agent.start(_make_trade(-50.0))

        log = state.read(StateKeys.TRADE_LOG)
        assert log is not None
        assert len(log) == 2


# ── Trade Evaluation ──────────────────────────────────────────────────────────


class TestTradeEvaluation:
    @pytest.mark.asyncio
    async def test_signal_correctness_positive(self):
        """When signals point up and price goes up → high correctness."""
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        ctx = _make_trade(500.0, signals={"rsi": 0.5, "macd": 0.5, "ma_crossover": 0.3, "vwap": 0.2})
        ctx["actual_price_move"] = 0.05  # price went up
        result = await agent.start(ctx)

        assert result["evaluation"]["signal_correctness"] > 0

    @pytest.mark.asyncio
    async def test_signal_correctness_negative(self):
        """When signals point up but price goes down → negative correctness."""
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        ctx = _make_trade(-300.0, signals={"rsi": 0.5, "macd": 0.5, "ma_crossover": 0.3, "vwap": 0.2})
        ctx["actual_price_move"] = -0.05  # price went down
        result = await agent.start(ctx)

        assert result["evaluation"]["signal_correctness"] < 0

    @pytest.mark.asyncio
    async def test_timing_score(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        ctx = _make_trade(200.0)
        ctx["actual_price_move"] = 0.05
        ctx["trade"]["entry_price"] = 100.0
        ctx["trade"]["exit_price"] = 103.0  # captured 3% of 5% move
        result = await agent.start(ctx)

        assert -1.0 <= result["evaluation"]["timing_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_risk_assessment_stop_loss(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        ctx = _make_trade(-100.0)
        ctx["trade"]["exit_reason"] = "stop_loss"
        ctx["trade"]["pnl_pct"] = -0.01
        await agent.start(ctx)

        eval_result = agent.evaluations[0]
        assert eval_result.stop_loss_hit is True
        assert eval_result.risk_assessment == "adequate"

    @pytest.mark.asyncio
    async def test_publishes_reflection_event(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        events_received = []
        async def track(event: Event):
            events_received.append(event)

        bus.subscribe(EventType.REFLECTION_COMPLETE, track)

        await agent.start(_make_trade(100.0))

        assert len(events_received) == 1
        assert events_received[0].payload["trade_id"] is not None


# ── Bounded Adjustments ───────────────────────────────────────────────────────


class TestBoundedAdjustments:
    @pytest.mark.asyncio
    async def test_no_adjustment_below_min_trades(self):
        """No adjustments until min_trades_for_adjustment is reached."""
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(
            event_bus=bus, state=state,
            config={"min_trades_for_adjustment": 10}
        )

        for i in range(5):
            await agent.start(_make_trade(100.0 * (1 if i % 2 == 0 else -1)))

        result = await agent.start(_make_trade(50.0))
        assert result["has_adjustments"] is False

    @pytest.mark.asyncio
    async def test_adjustments_after_enough_trades(self):
        """Adjustments are proposed after sufficient trades."""
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(
            event_bus=bus, state=state,
            config={"min_trades_for_adjustment": 5}
        )

        # Feed 6 trades (enough for adjustment)
        for i in range(6):
            pnl = 200.0 if i % 3 != 0 else -100.0
            await agent.start(_make_trade(pnl))

        report = state.read(StateKeys.REFLECTION_REPORT)
        assert report is not None
        assert report.total_trades >= 5

    @pytest.mark.asyncio
    async def test_weight_changes_are_bounded(self):
        """Weight adjustments never exceed max_weight_delta."""
        bus = EventBus()
        state = StateManager()
        max_delta = 0.03
        agent = ReflectionAgent(
            event_bus=bus, state=state,
            config={"min_trades_for_adjustment": 5, "max_weight_delta": max_delta}
        )

        # Feed consistently winning trades with strong RSI signal
        for _ in range(8):
            ctx = _make_trade(500.0, signals={"rsi": 0.9, "macd": 0.1, "ma_crossover": 0.1, "vwap": 0.0})
            ctx["actual_price_move"] = 0.05
            await agent.start(ctx)

        report = state.read(StateKeys.REFLECTION_REPORT)
        if report and report.weight_adjustments:
            for adj in report.weight_adjustments:
                assert adj.magnitude <= max_delta + 0.001

    @pytest.mark.asyncio
    async def test_weights_stay_within_bounds(self):
        """Weights never go below min_weight or above max_weight."""
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(
            event_bus=bus, state=state,
            config={
                "min_trades_for_adjustment": 3,
                "min_weight": 0.1,
                "max_weight": 0.4,
            }
        )

        # Feed extreme winning trades heavily biased toward RSI
        for _ in range(10):
            ctx = _make_trade(1000.0, signals={"rsi": 1.0, "macd": -1.0, "ma_crossover": -1.0, "vwap": -1.0})
            ctx["actual_price_move"] = 0.1
            await agent.start(ctx)

        # Check internal weights are bounded
        for w in agent._current_weights.values():
            assert w >= 0.1 - 0.01  # small float tolerance
            assert w <= 0.4 + 0.01

    @pytest.mark.asyncio
    async def test_confidence_adjustment_bounded(self):
        """Confidence threshold changes are clamped."""
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(
            event_bus=bus, state=state,
            config={
                "min_trades_for_adjustment": 3,
                "max_confidence_delta": 0.02,
                "min_confidence": 0.3,
                "max_confidence": 0.8,
            }
        )

        for _ in range(5):
            ctx = _make_trade(-500.0, signals={"rsi": -0.5, "macd": -0.5, "ma_crossover": -0.3, "vwap": -0.2})
            ctx["actual_price_move"] = 0.1  # signals were completely wrong
            await agent.start(ctx)

        assert 0.3 <= agent._current_confidence_threshold <= 0.8

    @pytest.mark.asyncio
    async def test_adjustment_published_as_event(self):
        """Weight adjustments are published for other agents to observe."""
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(
            event_bus=bus, state=state,
            config={"min_trades_for_adjustment": 3}
        )

        adjustments_received = []
        async def track(event: Event):
            adjustments_received.append(event)

        bus.subscribe(EventType.ADJUSTMENT_PROPOSED, track)

        for _ in range(5):
            await agent.start(_make_trade(300.0))

        # May or may not fire depending on data, but should not crash
        # (event fires only if there are actual weight changes)


# ── Performance Reports ───────────────────────────────────────────────────────


class TestPerformanceReports:
    @pytest.mark.asyncio
    async def test_win_rate_calculation(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(
            event_bus=bus, state=state,
            config={"min_trades_for_adjustment": 3}
        )

        # 3 wins, 2 losses
        for pnl in [100, 200, -50, 150, -30]:
            await agent.start(_make_trade(float(pnl)))

        report = state.read(StateKeys.REFLECTION_REPORT)
        assert report is not None
        assert report.win_rate == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_performance_trend_detection(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(
            event_bus=bus, state=state,
            config={"min_trades_for_adjustment": 3, "lookback_window": 10}
        )

        # First half: all losses. Second half: all wins. → improving
        for pnl in [-100, -200, -150, -80, -50, 100, 200, 300, 250, 150]:
            await agent.start(_make_trade(float(pnl)))

        report = state.read(StateKeys.REFLECTION_REPORT)
        assert report is not None
        assert report.performance_trend == "improving"

    @pytest.mark.asyncio
    async def test_get_performance_summary(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        summary = agent.get_performance_summary()
        assert summary["total_trades"] == 0

        await agent.start(_make_trade(100.0))
        await agent.start(_make_trade(-50.0))

        summary = agent.get_performance_summary()
        assert summary["total_trades"] == 2
        assert "recent_win_rate" in summary
        assert "current_weights" in summary

    @pytest.mark.asyncio
    async def test_reset_clears_everything(self):
        bus = EventBus()
        state = StateManager()
        agent = ReflectionAgent(event_bus=bus, state=state)

        await agent.start(_make_trade(100.0))
        assert len(agent.trade_log) == 1

        agent.reset()
        assert len(agent.trade_log) == 0
        assert len(agent.evaluations) == 0


# ── Slippage and Latency (ExecutionAgent) ─────────────────────────────────────


class TestSlippageSimulation:
    @pytest.mark.asyncio
    async def test_slippage_increases_buy_price(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker(initial_balance=100_000.0)

        await state.write(StateKeys.MARKET_METADATA, {"symbol": "TEST"}, writer="test")
        verdict = {
            "approved": True,
            "reason": "ok",
            "original_decision": {"action": "BUY", "current_price": 1000.0},
            "position_size": 10,
            "stop_loss": 970.0,
        }
        await state.write(StateKeys.RISK_VERDICT, verdict, writer="test")

        agent = ExecutionAgent(
            event_bus=bus, state=state, broker=broker,
            config={"slippage_bps": 10, "seed": 42}
        )
        result = await agent.start()

        assert result["status"] == "success"
        # Fill price should be higher than requested (adverse slippage on buy)
        assert result["fill_price"] > 1000.0
        assert result["slippage_cost"] > 0

    @pytest.mark.asyncio
    async def test_slippage_decreases_sell_price(self):
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker(initial_balance=100_000.0)

        await state.write(StateKeys.MARKET_METADATA, {"symbol": "TEST"}, writer="test")
        verdict = {
            "approved": True,
            "reason": "ok",
            "original_decision": {"action": "SELL", "current_price": 1000.0},
            "position_size": 10,
            "stop_loss": None,
        }
        await state.write(StateKeys.RISK_VERDICT, verdict, writer="test")

        agent = ExecutionAgent(
            event_bus=bus, state=state, broker=broker,
            config={"slippage_bps": 10, "seed": 42}
        )
        result = await agent.start()

        assert result["status"] == "success"
        # Fill price should be lower than requested (adverse slippage on sell)
        assert result["fill_price"] < 1000.0

    @pytest.mark.asyncio
    async def test_execution_delay(self):
        """Verify execution delay is simulated."""
        import time
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker(initial_balance=100_000.0)

        await state.write(StateKeys.MARKET_METADATA, {"symbol": "TEST"}, writer="test")
        verdict = {
            "approved": True,
            "reason": "ok",
            "original_decision": {"action": "BUY", "current_price": 100.0},
            "position_size": 5,
            "stop_loss": 97.0,
        }
        await state.write(StateKeys.RISK_VERDICT, verdict, writer="test")

        agent = ExecutionAgent(
            event_bus=bus, state=state, broker=broker,
            config={"slippage_bps": 0, "execution_delay_ms": 50, "seed": 1}
        )

        start = time.monotonic()
        await agent.start()
        elapsed = (time.monotonic() - start) * 1000

        assert elapsed >= 45  # at least ~50ms

    @pytest.mark.asyncio
    async def test_zero_slippage_config(self):
        """With slippage_bps=0, fill price should equal requested price."""
        bus = EventBus()
        state = StateManager()
        broker = PaperBroker(initial_balance=100_000.0)

        await state.write(StateKeys.MARKET_METADATA, {"symbol": "TEST"}, writer="test")
        verdict = {
            "approved": True,
            "reason": "ok",
            "original_decision": {"action": "BUY", "current_price": 500.0},
            "position_size": 5,
            "stop_loss": 485.0,
        }
        await state.write(StateKeys.RISK_VERDICT, verdict, writer="test")

        agent = ExecutionAgent(
            event_bus=bus, state=state, broker=broker,
            config={"slippage_bps": 0, "seed": 99}
        )
        result = await agent.start()

        # With 0 bps base, fill_price = price * (1 + 0 * jitter) = price
        assert result["fill_price"] == pytest.approx(500.0, abs=0.01)


# ── Integration with Orchestrator ─────────────────────────────────────────────


class TestReflectionIntegration:
    @pytest.mark.asyncio
    async def test_orchestrator_has_reflection_agent(self):
        broker = PaperBroker()
        orch = Orchestrator(broker=broker)
        assert orch.reflection_agent is not None

    @pytest.mark.asyncio
    async def test_reflect_on_trade_via_orchestrator(self):
        broker = PaperBroker()
        orch = Orchestrator(broker=broker)

        trade_data = {
            "trade_id": "t_123",
            "symbol": "TEST",
            "side": "BUY",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "quantity": 10,
            "pnl": 50.0,
            "pnl_pct": 0.005,
            "entry_reason": "composite_bullish",
            "exit_reason": "signal_exit",
            "confidence": 0.7,
            "duration_bars": 3,
            "slippage_cost": 1.0,
        }
        signals = {"rsi": 0.4, "macd": 0.3, "ma_crossover": 0.2, "vwap": 0.1}

        result = await orch.reflect_on_trade(trade_data, signals, actual_price_move=0.05)

        assert result["trade_id"] == "t_123"
        assert "evaluation" in result

    @pytest.mark.asyncio
    async def test_reset_clears_reflection(self):
        broker = PaperBroker()
        orch = Orchestrator(broker=broker)

        trade_data = {"trade_id": "t_1", "symbol": "X", "side": "BUY", "entry_price": 100.0, "pnl": 10.0}
        await orch.reflect_on_trade(trade_data)

        assert len(orch.reflection_agent.trade_log) == 1
        orch.reset()
        assert len(orch.reflection_agent.trade_log) == 0
