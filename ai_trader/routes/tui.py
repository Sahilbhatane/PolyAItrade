"""Read-model + event-stream API consumed by the Textual TUI.

Design constraints:
- The TUI is event-driven: ``GET /tui/events`` is a Server-Sent Events stream
  bridged from the in-process ``EventBus`` via the :class:`EventHub`.
- The TUI only submits *intents*; ``POST /tui/trade/submit`` routes strictly
  through ``RiskAgent -> ApprovalGate -> LiveExecutionAgent (KillSwitch)``.
- Read endpoints never load unbounded data (logs are paged/seekable).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai_trader.config import get_config
from ai_trader.logs import get_logger
from ai_trader.service.log_reader import read_logs
from ai_trader.service.trading_service import (
    ServiceBusyError,
    TradeIntentError,
    TradingService,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/tui", tags=["tui"])

_service: TradingService | None = None

_HEARTBEAT_S = 15.0


def set_dependencies(service: TradingService) -> None:
    """Inject the live TradingService instance at app startup."""
    global _service
    _service = service


def _require_service() -> TradingService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Trading service not initialized")
    return _service


class TradeIntentRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    side: str = Field(..., pattern="^(?i)(buy|sell)$")
    quantity: int = Field(default=0, ge=0)
    price: float = Field(..., gt=0)
    order_type: str = Field(default="MARKET")
    stop_loss: float | None = Field(default=None, ge=0)
    target: float | None = Field(default=None, ge=0)
    trailing_stop: float | None = Field(default=None, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: str = Field(default="manual", max_length=64)
    atr: float | None = Field(default=None, ge=0)
    mode: str = Field(default="paper", pattern="^(?i)(paper|live)$")


@router.get("/events")
async def stream_events(request: Request) -> StreamingResponse:
    """Server-Sent Events stream of live pipeline events."""
    service = _require_service()
    queue = service.event_hub.subscribe()

    async def event_generator():
        try:
            # Prime the connection so clients know the stream is live.
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_S)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield f"event: {item['type']}\ndata: {json.dumps(item)}\n\n"
        finally:
            service.event_hub.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/snapshot")
async def snapshot() -> dict[str, Any]:
    """Aggregate dashboard read-model."""
    service = _require_service()
    return await service.snapshot()


@router.post("/trade/submit")
async def submit_trade(intent: TradeIntentRequest) -> dict[str, Any]:
    """Submit a human trade intent through the full safety chain."""
    service = _require_service()
    try:
        return await service.submit_trade(intent.model_dump())
    except ServiceBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except TradeIntentError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/logs")
async def logs(
    limit: int = 200,
    cursor: int | None = None,
    level: str | None = None,
    component: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """Paged, seekable log reader (newest-first pages, never loads full file)."""
    config = get_config()
    log_path = os.path.join(config.logging.output_dir, "app.log")
    # Offload blocking file I/O to a thread so the event loop is never blocked.
    return await asyncio.to_thread(
        read_logs,
        log_path,
        limit=limit,
        cursor=cursor,
        level=level,
        component=component,
        search=search,
    )


@router.get("/strategies")
async def strategies() -> dict[str, Any]:
    """Read-model of configured strategies, weights, and regime weighting."""
    from ai_trader.strategies.config_loader import load_strategy_config

    service = _require_service()
    cfg = load_strategy_config()

    specs = []
    for name, spec in (cfg.strategies or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        specs.append({
            "name": name,
            "enabled": bool(spec.get("enabled", True)),
            "weight": spec.get("weight"),
            "params": spec.get("params", {}),
        })

    live_weights = service.state.read("strategy_weights") or {}
    regime = service.state.read("regime") or {}

    return {
        "name": cfg.name,
        "version": cfg.version,
        "signal_weights": cfg.weights.model_dump(),
        "consensus_min_weighted_confidence": cfg.consensus_min_weighted_confidence,
        "ml_enabled": cfg.ml.enabled,
        "strategies": specs,
        "current_regime": regime.get("label"),
        "live_weights": live_weights,
    }


@router.get("/agents")
async def agents() -> dict[str, Any]:
    """Live status of server-owned agents + recent event-bus activity."""
    service = _require_service()
    return service.agents_snapshot()


@router.get("/rl")
async def rl() -> dict[str, Any]:
    """RL policy/checkpoint read-model (mirrors /rl/status with more context)."""
    import yaml

    cfg_path = "ml_config.yaml"
    rl_cfg: dict[str, Any] = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            rl_cfg = data.get("rl", {}) if isinstance(data, dict) else {}
        except (OSError, yaml.YAMLError):
            rl_cfg = {}

    checkpoint_dir = rl_cfg.get("checkpoint_dir", "models/rl")
    checkpoint = os.path.join(checkpoint_dir, "ppo_polyvitrade.zip")
    return {
        "policy_version": rl_cfg.get("policy_version", "n/a"),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_exists": os.path.exists(checkpoint),
        "seed": rl_cfg.get("seed"),
        "deployment_mode": rl_cfg.get("deployment_mode", "shadow"),
        "config": {k: v for k, v in rl_cfg.items() if not isinstance(v, dict)},
    }


@router.get("/config/integrations")
async def config_integrations() -> dict[str, Any]:
    """Configuration STATUS only — booleans, never secret values.

    Security: this endpoint must never return API keys or tokens. It reports
    solely whether each integration/credential is configured.
    """
    config = get_config()
    integrations = config.integrations

    def _configured(value: str | None) -> bool:
        return bool(value and str(value).strip())

    broker = config.broker
    return {
        "broker": {
            "name": broker.name,
            "api_key": _configured(broker.api_key),
            "client_id": _configured(broker.client_id),
            "password": _configured(broker.password),
            "totp_secret": _configured(broker.totp_secret),
        },
        "database": {"url_configured": _configured(config.database.url)},
        "integrations": {
            "alpha_vantage": _configured(integrations.alpha_vantage_api_key),
            "polygon": _configured(integrations.polygon_api_key),
            "finnhub": _configured(integrations.finnhub_api_key),
            "openai": _configured(integrations.openai_api_key),
            "anthropic": _configured(integrations.anthropic_api_key),
            "telegram": _configured(integrations.telegram_bot_token),
            "discord": _configured(integrations.discord_webhook_url),
        },
    }


@router.get("/diagnostics")
async def diagnostics() -> dict[str, Any]:
    """Runtime diagnostics: latencies, queues, tasks, threads, memory."""
    service = _require_service()

    broker_healthy = await service.broker_health()
    snap = await service.snapshot()

    diag: dict[str, Any] = {
        "broker": {
            "healthy": broker_healthy,
            "latency_ms": snap["broker"]["latency_ms"],
        },
        "event_hub": snap["event_hub"],
        "kill_switch": snap["kill_switch"],
        "async_tasks": len(asyncio.all_tasks()),
        "threads": threading.active_count(),
        "pid": os.getpid(),
        "memory": _memory_stats(),
    }
    return diag


def _memory_stats() -> dict[str, Any]:
    """Best-effort process memory (psutil if present, else stdlib fallback)."""
    try:
        import psutil  # optional dependency

        proc = psutil.Process()
        mem = proc.memory_info()
        return {
            "rss_mb": round(mem.rss / (1024 * 1024), 2),
            "cpu_percent": proc.cpu_percent(interval=None),
            "source": "psutil",
        }
    except Exception:
        try:
            import resource  # not available on Windows

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return {"rss_mb": round(usage.ru_maxrss / 1024, 2), "source": "resource"}
        except Exception:
            return {"rss_mb": None, "source": "unavailable"}
