"""Centralized environment variables via pydantic-settings (.env + process env).

Flat keys (APP_ENV, ANGELONE_API_KEY, ...) merge into YAML-loaded config before
AI_TRADER_* nested overrides are applied (see ConfigLoader).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Application secrets and operational overrides.

    Extra keys in .env are ignored to avoid accidental coupling.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # App
    APP_ENV: str | None = Field(default=None, description="Synonym for config environment")
    APP_HOST: str | None = Field(default=None)
    APP_PORT: int | None = Field(default=None, ge=1, le=65535)

    # Database
    DATABASE_URL: str | None = None

    # Logging
    LOG_LEVEL: str | None = None

    # Broker (Angel One flat aliases → mapped to broker.*)
    ANGELONE_API_KEY: str | None = None
    ANGELONE_CLIENT_ID: str | None = None
    ANGELONE_PASSWORD: str | None = None
    ANGELONE_TOTP_SECRET: str | None = None

    # Market data APIs (stored under integrations.* for optional future use)
    ALPHA_VANTAGE_API_KEY: str | None = None
    POLYGON_API_KEY: str | None = None
    FINNHUB_API_KEY: str | None = None

    # AI / LLM
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    # Notifications
    TELEGRAM_BOT_TOKEN: str | None = None
    DISCORD_WEBHOOK_URL: str | None = None


def env_settings_to_nested_raw(settings: EnvSettings) -> dict[str, Any]:
    """Map flat env settings to nested dict keys matching config.yaml / AppConfig."""
    raw: dict[str, Any] = {}

    if settings.APP_ENV:
        raw["environment"] = settings.APP_ENV

    if settings.APP_HOST is not None or settings.APP_PORT is not None:
        srv = raw.setdefault("server", {})
        if settings.APP_HOST is not None:
            srv["host"] = settings.APP_HOST
        if settings.APP_PORT is not None:
            srv["port"] = settings.APP_PORT

    if settings.DATABASE_URL:
        raw.setdefault("database", {})["url"] = settings.DATABASE_URL

    if settings.LOG_LEVEL:
        raw.setdefault("logging", {})["level"] = settings.LOG_LEVEL.upper()

    broker_keys = (
        settings.ANGELONE_API_KEY,
        settings.ANGELONE_CLIENT_ID,
        settings.ANGELONE_PASSWORD,
        settings.ANGELONE_TOTP_SECRET,
    )
    if any(v is not None for v in broker_keys):
        b = raw.setdefault("broker", {})
        if settings.ANGELONE_API_KEY is not None:
            b["api_key"] = settings.ANGELONE_API_KEY
        if settings.ANGELONE_CLIENT_ID is not None:
            b["client_id"] = settings.ANGELONE_CLIENT_ID
        if settings.ANGELONE_PASSWORD is not None:
            b["password"] = settings.ANGELONE_PASSWORD
        if settings.ANGELONE_TOTP_SECRET is not None:
            b["totp_secret"] = settings.ANGELONE_TOTP_SECRET

    integ = {}
    if settings.ALPHA_VANTAGE_API_KEY:
        integ["alpha_vantage_api_key"] = settings.ALPHA_VANTAGE_API_KEY
    if settings.POLYGON_API_KEY:
        integ["polygon_api_key"] = settings.POLYGON_API_KEY
    if settings.FINNHUB_API_KEY:
        integ["finnhub_api_key"] = settings.FINNHUB_API_KEY
    if settings.OPENAI_API_KEY:
        integ["openai_api_key"] = settings.OPENAI_API_KEY
    if settings.ANTHROPIC_API_KEY:
        integ["anthropic_api_key"] = settings.ANTHROPIC_API_KEY
    if settings.TELEGRAM_BOT_TOKEN:
        integ["telegram_bot_token"] = settings.TELEGRAM_BOT_TOKEN
    if settings.DISCORD_WEBHOOK_URL:
        integ["discord_webhook_url"] = settings.DISCORD_WEBHOOK_URL
    if integ:
        raw["integrations"] = integ

    return raw


def validate_production_secrets(config: Any) -> None:
    """Raise if production live broker is enabled without required credentials."""
    env_name = str(getattr(config, "environment", "") or "").lower()
    broker_name = str(getattr(config.broker, "name", "") or "").lower()

    if env_name != "production" or broker_name != "angelone":
        return

    b = config.broker
    missing = []
    if not getattr(b, "api_key", None):
        missing.append("broker.api_key / ANGELONE_API_KEY")
    if not getattr(b, "client_id", None):
        missing.append("broker.client_id / ANGELONE_CLIENT_ID")
    if not getattr(b, "totp_secret", None):
        missing.append("broker.totp_secret / ANGELONE_TOTP_SECRET")

    if missing:
        raise ValueError(
            "Production Angel One trading requires: " + ", ".join(missing)
        )


def load_env_settings() -> EnvSettings:
    """Parse process env + optional .env file."""
    return EnvSettings()
