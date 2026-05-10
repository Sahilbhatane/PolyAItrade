"""Configuration loader with YAML support and environment variable overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ai_trader.config.env import env_settings_to_nested_raw, load_env_settings, validate_production_secrets


class TradingConfig(BaseModel):
    max_risk_per_trade: float = Field(default=0.02, ge=0.0, le=1.0)
    max_daily_loss: float = Field(default=0.05, ge=0.0, le=1.0)
    max_consecutive_losses: int = Field(default=3, ge=1)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    market_open: str = Field(default="09:15")
    market_close: str = Field(default="15:30")


class DatabaseConfig(BaseModel):
    url: str = Field(default="sqlite:///ai_trader.db")
    echo: bool = Field(default=False)
    pool_size: int = Field(default=5, ge=1)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    output_dir: str = Field(default="ai_trader/logs")


class BacktestConfig(BaseModel):
    default_capital: float = Field(default=100_000.0, gt=0)
    brokerage_rate: float = Field(default=0.0003, ge=0)
    stt_rate: float = Field(default=0.00025, ge=0)
    gst_rate: float = Field(default=0.18, ge=0)
    slippage_rate: float = Field(default=0.0001, ge=0)
    max_position_pct: float = Field(default=0.02, gt=0, le=1.0)


class BrokerConfig(BaseModel):
    name: str = Field(default="paper")
    api_key: str = Field(default="")
    api_secret: str = Field(default="")
    client_id: str = Field(default="")
    password: str = Field(default="")
    totp_secret: str = Field(default="")
    base_url: str = Field(default="")
    max_retries: int = Field(default=3, ge=1)
    retry_delay_s: float = Field(default=1.0, ge=0.1)
    product_type: str = Field(default="INTRADAY")
    exchange: str = Field(default="NSE")


class ApprovalConfig(BaseModel):
    enabled: bool = Field(default=True)
    timeout_s: float = Field(default=300.0, ge=10.0)
    auto_approve_paper: bool = Field(default=True)


class KillSwitchConfig(BaseModel):
    daily_loss_limit: float | None = Field(default=0.05, ge=0.0, le=1.0)
    max_api_failures: int = Field(default=5, ge=1)
    volatility_spike_zscore: float | None = Field(default=None)
    runaway_loss_pct: float | None = Field(default=None)


class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)


class IntegrationsConfig(BaseModel):
    alpha_vantage_api_key: str = Field(default="")
    polygon_api_key: str = Field(default="")
    finnhub_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    telegram_bot_token: str = Field(default="")
    discord_webhook_url: str = Field(default="")


class RegimeConfig(BaseModel):
    window_bars: int = Field(default=20, ge=5)
    sma_slope_period: int = Field(default=10, ge=2)
    trend_slope_threshold: float = Field(default=0.0005, ge=0.0)
    atr_percentile_volatile: float = Field(default=0.75, ge=0.0, le=1.0)
    volume_z_low_liquidity: float = Field(default=-1.25)
    sideways_atr_pct_max: float = Field(default=0.55, ge=0.0, le=1.0)


class AgentsPipelineConfig(BaseModel):
    consensus_enabled: bool = Field(default=True)
    consensus_min_weighted_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class RiskSizingConfig(BaseModel):
    """Regime-aware multipliers for position sizing (applied in RiskAgent)."""

    volatile_multiplier: float = Field(default=0.5, ge=0.1, le=1.0)
    low_liquidity_multiplier: float = Field(default=0.5, ge=0.1, le=1.0)


class AppConfig(BaseModel):
    app_name: str = Field(default="PolyVITrade")
    environment: str = Field(default="development")
    trading: TradingConfig = Field(default_factory=TradingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    agents: AgentsPipelineConfig = Field(default_factory=AgentsPipelineConfig)
    risk_sizing: RiskSizingConfig = Field(default_factory=RiskSizingConfig)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base (mutates base)."""
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], val)
        elif isinstance(val, dict):
            base[key] = val.copy()
        else:
            base[key] = val
    return base


class ConfigLoader:
    """Loads configuration from YAML files with environment variable overrides."""

    def __init__(self, config_path: str | Path | None = None):
        self._config_path = self._resolve_path(config_path)
        self._raw: dict[str, Any] = {}
        self._config: AppConfig | None = None

    @staticmethod
    def _resolve_path(config_path: str | Path | None) -> Path:
        if config_path:
            return Path(config_path)
        env_path = os.getenv("AI_TRADER_CONFIG")
        if env_path:
            return Path(env_path)
        return Path("config.yaml")

    def load(self) -> AppConfig:
        """Load YAML, merge .env flat keys, apply AI_TRADER_* overrides (highest precedence)."""
        self._raw = self._load_yaml()

        try:
            env_raw = env_settings_to_nested_raw(load_env_settings())
            deep_merge(self._raw, env_raw)
        except Exception:
            pass

        self._apply_env_overrides()
        self._config = AppConfig(**self._raw)
        validate_production_secrets(self._config)
        return self._config

    def _load_yaml(self) -> dict[str, Any]:
        if not self._config_path.exists():
            return {}
        with open(self._config_path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def _apply_env_overrides(self) -> None:
        """Override config values with AI_TRADER_ prefixed env vars.

        e.g. AI_TRADER_DATABASE__URL overrides database.url
        """
        prefix = "AI_TRADER_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix) :].lower().split("__")
            self._set_nested(self._raw, parts, value)

    @staticmethod
    def _set_nested(data: dict, keys: list[str], value: str) -> None:
        for key in keys[:-1]:
            data = data.setdefault(key, {})
        data[keys[-1]] = value

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            return self.load()
        return self._config


def clear_config_cache() -> None:
    """Clear singleton config cache (for tests / reload)."""
    get_config.cache_clear()


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> AppConfig:
    """Singleton accessor for application config."""
    loader = ConfigLoader(config_path)
    return loader.load()
