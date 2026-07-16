"""Server-side trading service — the single owner of the live pipeline.

Responsibilities:
- Own a shared :class:`EventBus` + :class:`StateManager` for the running server.
- Bridge every bus event to the :class:`EventHub` (for SSE streaming).
- Accept human-initiated trade intents from the TUI and route them strictly
  through ``RiskAgent -> ApprovalGate -> LiveExecutionAgent (KillSwitch)``.
- Expose a read-model snapshot for the dashboard.

Safety invariants enforced here:
- The TUI never receives a broker handle; it can only submit *intents*.
- No intent can reach the broker without an approved RiskAgent verdict AND an
  explicit ApprovalGate resolution AND a disengaged KillSwitch.
- Only one trade may be pending approval at a time (guards duplicate
  approvals / duplicate orders and state-slot races).
- RiskAgent internal counters (consecutive losses, trades/day, daily P&L) are a
  persistent singleton, so daily limits survive across intents.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import date
from typing import Any

from ai_trader.agents.event_bus import EventBus
from ai_trader.agents.risk_agent import RiskAgent
from ai_trader.agents.state import StateKeys, StateManager
from ai_trader.broker.approval import ApprovalGate
from ai_trader.broker.base import BaseBroker
from ai_trader.broker.kill_switch import KillSwitch
from ai_trader.logs import get_logger
from ai_trader.service.event_hub import EventHub
from ai_trader.utils.time import is_market_open, now_ist

logger = get_logger(__name__)


class TradeIntentError(Exception):
    """Raised when a trade intent is malformed or cannot be accepted."""


class ServiceBusyError(Exception):
    """Raised when a trade is already pending approval."""


class TradingService:
    """Owns the live event bus, risk/execution agents, and the read-model."""

    def __init__(
        self,
        broker: BaseBroker,
        approval_gate: ApprovalGate,
        kill_switch: KillSwitch,
        config: dict[str, Any] | None = None,
        max_event_queue: int = 1000,
    ):
        self._config = config or {}
        self._broker = broker
        self._approval_gate = approval_gate
        self._kill_switch = kill_switch

        self._bus = EventBus()
        self._state = StateManager()
        self._hub = EventHub(self._bus, max_queue=max_event_queue)

        self._risk_agent = RiskAgent(
            event_bus=self._bus,
            state=self._state,
            config=self._config.get("risk", {}),
        )

        # LiveExecutionAgent is imported lazily to avoid importing numpy at
        # module import time for callers that only need the read-model.
        from ai_trader.agents.live_execution_agent import LiveExecutionAgent

        self._execution_agent = LiveExecutionAgent(
            event_bus=self._bus,
            state=self._state,
            broker=broker,
            approval_gate=approval_gate,
            kill_switch=kill_switch,
            config=self._config.get("execution", {}),
        )

        self._pending_request_id: str | None = None
        self._pending_task: asyncio.Task[Any] | None = None
        self._last_broker_latency_ms: float | None = None
        self._last_trade: dict[str, Any] | None = None

    # --- Accessors -----------------------------------------------------

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    @property
    def event_hub(self) -> EventHub:
        return self._hub

    @property
    def state(self) -> StateManager:
        return self._state

    @property
    def is_pending(self) -> bool:
        """True while a trade is awaiting approval/execution."""
        return self._has_pending()

    # --- Trade intent --------------------------------------------------

    async def submit_trade(self, intent: dict[str, Any]) -> dict[str, Any]:
        """Accept a human trade intent and route it through the safety chain.

        Returns immediately after RiskAgent evaluation:
        - rejected verdict  -> ``{"accepted": False, ...}``
        - approved verdict  -> ``{"accepted": True, "status": "pending_approval", ...}``
          and execution proceeds in the background (blocking on the human
          ApprovalGate + KillSwitch).
        """
        if self._kill_switch.is_active:
            raise ServiceBusyError("Kill switch is engaged — no intents accepted")

        if self._has_pending():
            raise ServiceBusyError(
                f"A trade is already pending approval ({self._pending_request_id})"
            )

        decision = self._build_decision(intent)
        symbol = str(intent["symbol"]).strip()

        await self._state.write(StateKeys.MARKET_METADATA, {"symbol": symbol}, writer="tui")
        await self._state.write(StateKeys.TRADE_DECISION, decision, writer="tui")

        verdict = await self._risk_agent.start()

        if not verdict.get("approved"):
            return {
                "accepted": False,
                "status": "risk_rejected",
                "reason": verdict.get("reason", "rejected"),
                "verdict": verdict,
            }

        request_id = str(uuid.uuid4())
        self._pending_request_id = request_id
        self._pending_task = asyncio.create_task(self._execute(request_id, symbol, decision))

        return {
            "accepted": True,
            "status": "pending_approval",
            "request_id": request_id,
            "symbol": symbol,
            "side": decision["action"],
            "position_size": verdict.get("position_size"),
            "stop_loss": verdict.get("stop_loss"),
            "trailing_stop": verdict.get("trailing_stop"),
            "risk_reward_ratio": verdict.get("risk_reward_ratio"),
            "reason": verdict.get("reason"),
        }

    async def _execute(self, request_id: str, symbol: str, decision: dict[str, Any]) -> None:
        """Background execution: blocks on approval, then broker, via the agent."""
        try:
            result = await self._execution_agent.start({"request_id": request_id})
            if result.get("status") in ("success", "partial"):
                self._last_trade = {
                    "symbol": symbol,
                    "side": decision.get("action"),
                    "status": result.get("status"),
                    "fill_price": result.get("fill_price"),
                    "quantity": result.get("quantity") or result.get("filled_qty"),
                    "at": now_ist().isoformat(),
                }
                # Update persistent risk counters. Realized P&L is only known on
                # position close; entries record a trade without P&L impact.
                self._risk_agent.on_trade_result(pnl=0.0, trade_date=date.today())
        except Exception as e:  # never let a background task die silently
            logger.error("trade_execution_failed", request_id=request_id, error=str(e))
        finally:
            self._pending_request_id = None
            self._pending_task = None

    def _has_pending(self) -> bool:
        return self._pending_task is not None and not self._pending_task.done()

    def _build_decision(self, intent: dict[str, Any]) -> dict[str, Any]:
        side = str(intent.get("side", "")).upper()
        if side not in ("BUY", "SELL"):
            raise TradeIntentError("side must be BUY or SELL")

        try:
            price = float(intent.get("price", 0.0))
        except (TypeError, ValueError):
            raise TradeIntentError("price must be numeric")
        if price <= 0:
            raise TradeIntentError("price must be > 0")

        confidence = float(intent.get("confidence", 0.0) or 0.0)
        atr = intent.get("atr")
        return {
            "action": side,
            "current_price": price,
            "atr": float(atr) if atr is not None else None,
            "confidence": confidence,
            "requested_quantity": int(intent.get("quantity", 0) or 0),
            "reasoning": str(intent.get("reasoning", "manual_intent")),
            "strategy": str(intent.get("strategy", "manual")),
            "signals": intent.get("signals", {}) or {},
            "source": "tui_manual",
        }

    # --- Read model ----------------------------------------------------

    async def snapshot(self) -> dict[str, Any]:
        """Aggregate dashboard metrics from live components (no polling loops)."""
        balance = await self._safe_balance()
        positions = await self._safe_positions()
        risk = self._risk_agent.state_snapshot()

        exposure = 0.0
        for pos in positions:
            try:
                exposure += float(pos.get("entry_price", 0.0)) * float(pos.get("quantity", 0.0))
            except (TypeError, ValueError):
                continue

        pending = self._approval_gate.pending_requests
        capital = risk.get("capital", 0.0) or 0.0
        remaining_risk_budget = max(
            capital * float(risk.get("max_capital_per_trade", 0.0)) - exposure, 0.0
        )

        return {
            "timestamp": now_ist().isoformat(),
            "market_open": is_market_open(),
            "kill_switch": self._kill_switch.status,
            "broker": {"latency_ms": self._last_broker_latency_ms},
            "balance": balance,
            "positions": {"count": len(positions), "exposure": exposure},
            "approvals": {"pending": len(pending)},
            "risk": risk,
            "remaining_risk_budget": remaining_risk_budget,
            "pending_request_id": self._pending_request_id,
            "last_trade": self._last_trade,
            "event_hub": self._hub.stats(),
        }

    def agents_snapshot(self) -> dict[str, Any]:
        """Live status of the server-owned agents + recent bus events.

        Only the agents this service actually instantiates are reported with a
        live status. Other pipeline agents run inside per-request orchestrators,
        so we surface them as informational names rather than faking a status.
        """
        owned = [
            {"name": self._risk_agent.agent_id, "status": self._risk_agent.status.value, "live": True},
            {
                "name": self._execution_agent.agent_id,
                "status": self._execution_agent.status.value,
                "live": True,
            },
        ]
        pipeline_only = [
            "market_data_agent",
            "signal_agent",
            "regime_detection_agent",
            "strategy_selection_agent",
            "consensus_agent",
            "reflection_agent",
        ]
        for name in pipeline_only:
            owned.append({"name": name, "status": "pipeline", "live": False})

        recent = [
            {
                "type": e.event_type.value,
                "source": e.source_agent,
                "timestamp": e.timestamp.isoformat(),
                "correlation_id": e.correlation_id,
            }
            for e in self._bus.get_history(limit=25)
        ]
        return {
            "agents": owned,
            "recent_events": list(reversed(recent)),
            "kill_switch_active": self._kill_switch.is_active,
            "pending_request_id": self._pending_request_id,
        }

    async def broker_health(self) -> bool:
        start = time.perf_counter()
        try:
            healthy = await self._broker.health_check()
            return bool(healthy)
        finally:
            self._last_broker_latency_ms = (time.perf_counter() - start) * 1000.0

    async def _safe_balance(self) -> dict[str, float]:
        try:
            return await self._broker.get_account_balance()
        except Exception as e:
            logger.warning("snapshot_balance_failed", error=str(e))
            return {}

    async def _safe_positions(self) -> list[dict[str, Any]]:
        try:
            return await self._broker.get_positions()
        except Exception as e:
            logger.warning("snapshot_positions_failed", error=str(e))
            return []

    async def shutdown(self) -> None:
        if self._pending_task is not None and not self._pending_task.done():
            self._pending_task.cancel()
        self._hub.detach()
