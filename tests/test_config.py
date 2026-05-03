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
