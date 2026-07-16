"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from ai_trader.config import get_config
from ai_trader.logs import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    config = get_config()
    setup_logging(level=config.logging.level, output_dir=config.logging.output_dir)

    # Initialize broker infrastructure
    from ai_trader.broker.approval import ApprovalGate
    from ai_trader.broker.kill_switch import KillSwitch
    from ai_trader.routes import trading

    is_paper = config.broker.name == "paper"

    approval_gate = ApprovalGate(
        timeout_s=config.approval.timeout_s,
        auto_approve=is_paper and config.approval.auto_approve_paper,
    )

    kill_switch = KillSwitch(auto_triggers={
        "daily_loss_limit": config.kill_switch.daily_loss_limit,
        "max_api_failures": config.kill_switch.max_api_failures,
        "volatility_spike_zscore": config.kill_switch.volatility_spike_zscore,
        "runaway_loss_pct": config.kill_switch.runaway_loss_pct,
    })

    broker = _create_broker(config)
    trading.set_dependencies(approval_gate, kill_switch, broker)

    # Long-running service that owns the shared EventBus and bridges events to
    # the TUI. Constructed with the SAME approval gate / kill switch / broker so
    # intents submitted from the TUI pass through the identical safety gates.
    from ai_trader.routes import tui
    from ai_trader.service import TradingService

    trading_service = TradingService(
        broker=broker,
        approval_gate=approval_gate,
        kill_switch=kill_switch,
        config={
            "risk": {
                "max_capital_per_trade": config.trading.max_risk_per_trade,
                "daily_loss_limit": config.trading.max_daily_loss,
                "max_consecutive_losses": config.trading.max_consecutive_losses,
            },
        },
    )
    tui.set_dependencies(trading_service)

    app.state.approval_gate = approval_gate
    app.state.kill_switch = kill_switch
    app.state.broker = broker
    app.state.trading_service = trading_service

    try:
        yield
    finally:
        await trading_service.shutdown()


def _create_broker(config):
    """Factory to create the appropriate broker instance."""
    if config.broker.name == "angelone":
        from ai_trader.broker.angelone import AngelOneBroker
        return AngelOneBroker(
            api_key=config.broker.api_key,
            client_id=config.broker.client_id,
            password=config.broker.password,
            totp_secret=config.broker.totp_secret,
            max_retries=config.broker.max_retries,
            retry_delay_s=config.broker.retry_delay_s,
            product_type=config.broker.product_type,
            exchange=config.broker.exchange,
        )
    else:
        from ai_trader.broker.paper import PaperBroker
        return PaperBroker()


def create_app() -> FastAPI:
    """Application factory with dependency injection."""
    config = get_config()

    app = FastAPI(
        title=config.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    from ai_trader.routes import backtest, data, health, ml, rl, trading, tui

    app.include_router(health.router)
    app.include_router(data.router)
    app.include_router(backtest.router)
    app.include_router(ml.router)
    app.include_router(rl.router)
    app.include_router(trading.router)
    app.include_router(tui.router)

    return app
