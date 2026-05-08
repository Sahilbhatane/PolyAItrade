"""Orchestrator — coordinates the agent pipeline sequentially.

The pipeline flows strictly:
  MarketDataAgent → SignalAgent → StrategyAgent → RiskAgent → ExecutionAgent

No agent can bypass this chain. The orchestrator ensures:
1. Correct ordering (data before signals, signals before decisions, etc.)
2. Error propagation (failure at any stage stops the pipeline)
3. Event publishing for pipeline lifecycle
4. Logging of every step for auditability
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ai_trader.agents.base import BaseAgent
from ai_trader.agents.event_bus import Event, EventBus, EventType
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.agents.market_data_agent import MarketDataAgent
from ai_trader.agents.signal_agent import SignalAgent
from ai_trader.agents.strategy_agent import StrategyAgent
from ai_trader.agents.risk_agent import RiskAgent
from ai_trader.agents.execution_agent import ExecutionAgent
from ai_trader.agents.reflection_agent import ReflectionAgent
from ai_trader.broker.base import BaseBroker
from ai_trader.logs import get_logger

logger = get_logger(__name__)


class PipelineResult:
    """Result of a single pipeline execution."""

    def __init__(self):
        self.steps: list[dict[str, Any]] = []
        self.success: bool = False
        self.error: str | None = None
        self.final_state: dict[str, Any] = {}

    def add_step(self, agent_name: str, result: dict[str, Any]) -> None:
        self.steps.append({"agent": agent_name, "result": result})

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "steps": self.steps,
            "final_state": self.final_state,
        }


class Orchestrator:
    """Schedules and runs the agent pipeline.

    Agents are instantiated once and reused across pipeline runs.
    Each run produces a PipelineResult with full audit trail.
    """

    def __init__(
        self,
        broker: BaseBroker,
        config: dict[str, Any] | None = None,
        database_url: str = "sqlite:///ai_trader.db",
    ):
        self._config = config or {}
        self._bus = EventBus()
        self._state = StateManager()

        # Instantiate agents — loosely coupled via bus + state only
        self._market_data_agent = MarketDataAgent(
            event_bus=self._bus,
            state=self._state,
            database_url=database_url,
        )
        self._signal_agent = SignalAgent(
            event_bus=self._bus,
            state=self._state,
            config=self._config.get("signal", {}),
        )
        self._strategy_agent = StrategyAgent(
            event_bus=self._bus,
            state=self._state,
            config=self._config.get("strategy", {}),
        )
        self._risk_agent = RiskAgent(
            event_bus=self._bus,
            state=self._state,
            config=self._config.get("risk", {}),
        )
        self._execution_agent = ExecutionAgent(
            event_bus=self._bus,
            state=self._state,
            broker=broker,
            config=self._config.get("execution", {}),
        )
        self._reflection_agent = ReflectionAgent(
            event_bus=self._bus,
            state=self._state,
            config=self._config.get("reflection", {}),
        )

        self._pipeline_agents: list[tuple[str, BaseAgent]] = [
            ("MarketData", self._market_data_agent),
            ("Signal", self._signal_agent),
            ("Strategy", self._strategy_agent),
            ("Risk", self._risk_agent),
            ("Execution", self._execution_agent),
        ]

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    @property
    def state(self) -> StateManager:
        return self._state

    @property
    def risk_agent(self) -> RiskAgent:
        return self._risk_agent

    @property
    def reflection_agent(self) -> ReflectionAgent:
        return self._reflection_agent

    async def run_pipeline(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        bar_index: int | None = None,
    ) -> PipelineResult:
        """Execute the full agent pipeline once.

        Args:
            symbol: Ticker symbol to process.
            timeframe: Data timeframe.
            start_date: Historical data start.
            end_date: Historical data end.
            bar_index: Specific bar to evaluate (None = last).

        Returns:
            PipelineResult with step-by-step audit.
        """
        result = PipelineResult()

        await self._bus.publish(Event(
            event_type=EventType.PIPELINE_START,
            payload={"symbol": symbol, "timeframe": timeframe},
            source_agent="orchestrator",
        ))

        contexts = self._build_contexts(symbol, timeframe, start_date, end_date, bar_index)

        for agent_name, agent in self._pipeline_agents:
            ctx = contexts.get(agent_name, {})
            try:
                step_result = await agent.start(ctx)
                result.add_step(agent_name, step_result)
            except Exception as e:
                result.error = f"{agent_name} failed: {str(e)}"
                result.add_step(agent_name, {"status": "error", "error": str(e)})

                await self._bus.publish(Event(
                    event_type=EventType.PIPELINE_ERROR,
                    payload={"stage": agent_name, "error": str(e)},
                    source_agent="orchestrator",
                ))
                logger.error("pipeline_failed", stage=agent_name, error=str(e))
                return result

        # Run reflection if a trade was executed
        order_result = self._state.read(StateKeys.ORDER_RESULT)
        if order_result and order_result.get("status") == "success":
            try:
                reflection_ctx = self._build_reflection_context()
                ref_result = await self._reflection_agent.start(reflection_ctx)
                result.add_step("Reflection", ref_result)
            except Exception as e:
                logger.warning("reflection_failed", error=str(e))

        result.success = True
        result.final_state = self._state.snapshot()

        await self._bus.publish(Event(
            event_type=EventType.PIPELINE_COMPLETE,
            payload={"steps": len(result.steps)},
            source_agent="orchestrator",
        ))

        logger.info("pipeline_complete", symbol=symbol, steps=len(result.steps))
        return result

    async def run_bar_by_bar(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[PipelineResult]:
        """Run pipeline for each bar sequentially (backtest-style).

        Fetches data once, then runs signal → strategy → risk → execution
        for each bar. The MarketDataAgent runs only once at the start.
        """
        # Step 1: Fetch and validate data
        data_ctx = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
        }
        await self._market_data_agent.start(data_ctx)

        # Step 2: Compute signals on full dataset
        await self._signal_agent.start()

        signals = self._state.read(StateKeys.SIGNALS)
        if signals is None:
            return []

        n_bars = len(signals.get("rsi", []))
        results = []

        # Step 3: For each bar, run strategy → risk → execution
        for bar_idx in range(n_bars):
            bar_result = PipelineResult()

            try:
                strat_result = await self._strategy_agent.start({"bar_index": bar_idx})
                bar_result.add_step("Strategy", strat_result)

                risk_result = await self._risk_agent.start()
                bar_result.add_step("Risk", risk_result)

                exec_result = await self._execution_agent.start()
                bar_result.add_step("Execution", exec_result)

                bar_result.success = True
            except Exception as e:
                bar_result.error = str(e)

            results.append(bar_result)

        return results

    async def reflect_on_trade(self, trade_data: dict[str, Any], signals_at_entry: dict[str, float] = None, actual_price_move: float = 0.0) -> dict[str, Any]:
        """Manually trigger reflection on a closed trade."""
        ctx = {
            "trade": trade_data,
            "signals_at_entry": signals_at_entry or {},
            "actual_price_move": actual_price_move,
        }
        return await self._reflection_agent.start(ctx)

    def _build_reflection_context(self) -> dict[str, Any]:
        """Build context for reflection from current pipeline state."""
        decision = self._state.read(StateKeys.TRADE_DECISION) or {}
        order_result = self._state.read(StateKeys.ORDER_RESULT) or {}
        verdict = self._state.read(StateKeys.RISK_VERDICT) or {}

        trade_data = {
            "trade_id": order_result.get("order_id", ""),
            "symbol": self._state.read(StateKeys.MARKET_METADATA) or {},
            "side": order_result.get("side", "BUY"),
            "entry_price": order_result.get("fill_price", order_result.get("price", 0.0)),
            "quantity": order_result.get("quantity", 0),
            "stop_loss": verdict.get("stop_loss"),
            "entry_reason": decision.get("reasoning", ""),
            "confidence": decision.get("confidence", 0.0),
            "slippage_cost": order_result.get("slippage_cost", 0.0),
        }

        return {
            "trade": trade_data,
            "signals_at_entry": decision.get("signals", {}),
            "actual_price_move": 0.0,
        }

    def reset(self) -> None:
        """Reset all state for a fresh run."""
        self._risk_agent.reset()
        self._reflection_agent.reset()
        self._bus.clear()

    @staticmethod
    def _build_contexts(
        symbol: str,
        timeframe: str,
        start_date: datetime | None,
        end_date: datetime | None,
        bar_index: int | None,
    ) -> dict[str, dict[str, Any]]:
        """Build per-agent context dicts."""
        return {
            "MarketData": {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
            },
            "Signal": {},
            "Strategy": {"bar_index": bar_index} if bar_index is not None else {},
            "Risk": {},
            "Execution": {},
        }
