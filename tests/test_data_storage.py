"""Tests for the market data storage layer."""

import tempfile
from datetime import datetime, timezone

import pandas as pd
import pytest

from ai_trader.data.provider import MarketData, TimeFrame
from ai_trader.data.storage import MarketDataStore


@pytest.fixture
def temp_store():
    """Create a store backed by a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_url = f"sqlite:///{f.name}"
    return MarketDataStore(database_url=db_url)


@pytest.fixture
def sample_market_data():
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": range(100, 110),
        "high": range(101, 111),
        "low": range(99, 109),
        "close": range(100, 110),
        "volume": [1000] * 10,
    }, index=dates)
    df.index.name = "timestamp"
    return MarketData(
        symbol="TEST",
        timeframe=TimeFrame.DAILY,
        data=df,
        source="test",
    )


def test_save_and_load(temp_store, sample_market_data):
    rows = temp_store.save(sample_market_data)
    assert rows == 10

    loaded = temp_store.load(
        "TEST",
        TimeFrame.DAILY,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    assert loaded is not None
    assert len(loaded.data) == 10
    assert loaded.source == "cache"


def test_no_duplicate_inserts(temp_store, sample_market_data):
    temp_store.save(sample_market_data)
    rows = temp_store.save(sample_market_data)
    assert rows == 0


def test_load_empty(temp_store):
    result = temp_store.load(
        "NONEXIST",
        TimeFrame.DAILY,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    assert result is None


def test_has_data(temp_store, sample_market_data):
    assert not temp_store.has_data(
        "TEST", TimeFrame.DAILY,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    temp_store.save(sample_market_data)
    assert temp_store.has_data(
        "TEST", TimeFrame.DAILY,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 15, tzinfo=timezone.utc),
    )


def test_partial_date_range(temp_store, sample_market_data):
    temp_store.save(sample_market_data)
    loaded = temp_store.load(
        "TEST",
        TimeFrame.DAILY,
        datetime(2024, 1, 3, tzinfo=timezone.utc),
        datetime(2024, 1, 7, tzinfo=timezone.utc),
    )
    assert loaded is not None
    assert len(loaded.data) == 5
