"""Long-running server-side trading service and event bridge.

This package wires the in-process agent ``EventBus`` into the FastAPI process
so that clients (the Textual TUI) can consume a live, event-driven stream and a
small read-model over HTTP.

Nothing in this package owns business logic beyond orchestrating the existing,
already-tested safety components (``RiskAgent`` -> ``ApprovalGate`` ->
``LiveExecutionAgent``/``KillSwitch``). The TUI never talks to a broker directly.
"""

from ai_trader.service.event_hub import EventHub
from ai_trader.service.trading_service import TradingService

__all__ = ["EventHub", "TradingService"]
