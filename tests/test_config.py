"""Tests for the configuration system."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from ai_trader.config.loader import AppConfig, ConfigLoader


@pytest.fixture
def temp_config_file():
    config_data = {
        "app_name": "TestApp",
        "environment": "testing",
        "trading": {
            "max_risk_per_trade": 0.01,
            "confidence_threshold": 0.7,
        },
        "database": {
            "url": "sqlite:///test.db",
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        yield f.name
    os.unlink(f.name)


def test_load_from_yaml(temp_config_file):
    loader = ConfigLoader(temp_config_file)
    config = loader.load()
    assert config.app_name == "TestApp"
    assert config.environment == "testing"
    assert config.trading.max_risk_per_trade == 0.01
    assert config.trading.confidence_threshold == 0.7


def test_default_config_when_no_file():
    loader = ConfigLoader("/nonexistent/path.yaml")
    config = loader.load()
    assert config.app_name == "PolyVITrade"
    assert config.environment == "development"


def test_env_override(temp_config_file, monkeypatch):
    monkeypatch.setenv("AI_TRADER_DATABASE__URL", "sqlite:///override.db")
    loader = ConfigLoader(temp_config_file)
    config = loader.load()
    assert config.database.url == "sqlite:///override.db"


def test_config_validation():
    config = AppConfig(trading={"max_risk_per_trade": 0.5})
    assert config.trading.max_risk_per_trade == 0.5


def test_config_validation_rejects_invalid():
    with pytest.raises(Exception):
        AppConfig(trading={"max_risk_per_trade": 5.0})


def test_clear_config_cache_runs():
    from ai_trader.config.loader import clear_config_cache

    clear_config_cache()


def test_production_angelone_requires_secrets(monkeypatch, temp_config_file):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_TRADER_BROKER__NAME", "angelone")
    monkeypatch.setenv("AI_TRADER_BROKER__API_KEY", "")
    monkeypatch.setenv("AI_TRADER_BROKER__CLIENT_ID", "")
    monkeypatch.setenv("AI_TRADER_BROKER__TOTP_SECRET", "")
    from ai_trader.config.loader import clear_config_cache

    clear_config_cache()
    loader = ConfigLoader(temp_config_file)
    with pytest.raises(ValueError, match="Production Angel"):
        loader.load()
