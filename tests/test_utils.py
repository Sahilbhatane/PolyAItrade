"""Tests for utility modules."""

from datetime import datetime, time, timezone, timedelta

import pytest

from ai_trader.utils.di import Container
from ai_trader.utils.time import is_market_open, IST


class TestContainer:
    def test_register_and_resolve_singleton(self):
        container = Container()
        container.register_singleton("db", {"connection": "mock"})
        assert container.resolve("db") == {"connection": "mock"}

    def test_register_and_resolve_factory(self):
        container = Container()
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"instance": call_count}

        container.register_factory("service", factory)
        result = container.resolve("service")
        assert result == {"instance": 1}
        # Second resolve returns same singleton
        assert container.resolve("service") == {"instance": 1}
        assert call_count == 1

    def test_resolve_missing_key_raises(self):
        container = Container()
        with pytest.raises(KeyError):
            container.resolve("missing")

    def test_reset(self):
        container = Container()
        container.register_singleton("x", 1)
        container.reset()
        assert not container.has("x")


class TestMarketHours:
    def test_market_open_weekday(self):
        # Wednesday at 10:00 IST
        dt = datetime(2025, 1, 8, 10, 0, tzinfo=IST)
        assert is_market_open(dt) is True

    def test_market_closed_weekend(self):
        # Saturday at 10:00 IST
        dt = datetime(2025, 1, 11, 10, 0, tzinfo=IST)
        assert is_market_open(dt) is False

    def test_market_closed_before_open(self):
        # Wednesday at 8:00 IST
        dt = datetime(2025, 1, 8, 8, 0, tzinfo=IST)
        assert is_market_open(dt) is False

    def test_market_closed_after_close(self):
        # Wednesday at 16:00 IST
        dt = datetime(2025, 1, 8, 16, 0, tzinfo=IST)
        assert is_market_open(dt) is False
