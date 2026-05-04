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
    yield


def create_app() -> FastAPI:
    """Application factory with dependency injection."""
    config = get_config()

    app = FastAPI(
        title=config.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    from ai_trader.routes import backtest, data, health, ml

    app.include_router(health.router)
    app.include_router(data.router)
    app.include_router(backtest.router)
    app.include_router(ml.router)

    return app
