"""Database-backed market data storage with caching."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, Index
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ai_trader.data.provider import MarketData, TimeFrame
from ai_trader.logs import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class OHLCVRecord(Base):
    __tablename__ = "ohlcv"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    source = Column(String(50), nullable=False, default="yfinance")

    __table_args__ = (
        Index("idx_symbol_timeframe_ts", "symbol", "timeframe", "timestamp", unique=True),
    )


class MarketDataStore:
    """Persists and caches OHLCV data in SQLite/PostgreSQL.

    Provides a write-through cache: saves fetched data, serves from DB on repeat requests.
    """

    def __init__(self, database_url: str = "sqlite:///ai_trader.db", echo: bool = False):
        self._engine = create_engine(database_url, echo=echo)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)

    def save(self, market_data: MarketData) -> int:
        """Persist market data to the database. Returns number of rows inserted."""
        df = market_data.data.copy()
        df["symbol"] = market_data.symbol
        df["timeframe"] = market_data.timeframe.value
        df["source"] = market_data.source
        df = df.reset_index()
        df = df.rename(columns={"index": "timestamp"} if "timestamp" not in df.columns else {})

        rows_inserted = 0
        with self._session_factory() as session:
            for _, row in df.iterrows():
                existing = (
                    session.query(OHLCVRecord)
                    .filter_by(
                        symbol=row["symbol"],
                        timeframe=row["timeframe"],
                        timestamp=row["timestamp"],
                    )
                    .first()
                )
                if existing:
                    continue

                record = OHLCVRecord(
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=int(row["volume"]),
                    source=row["source"],
                )
                session.add(record)
                rows_inserted += 1
            session.commit()

        logger.info("data_saved", symbol=market_data.symbol, rows=rows_inserted)
        return rows_inserted

    def load(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> MarketData | None:
        """Load cached data from database. Returns None if no data found."""
        with self._session_factory() as session:
            records = (
                session.query(OHLCVRecord)
                .filter(
                    OHLCVRecord.symbol == symbol,
                    OHLCVRecord.timeframe == timeframe.value,
                    OHLCVRecord.timestamp >= start,
                    OHLCVRecord.timestamp <= end,
                )
                .order_by(OHLCVRecord.timestamp)
                .all()
            )

        if not records:
            return None

        rows = [
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in records
        ]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        logger.info("data_loaded_from_cache", symbol=symbol, rows=len(df))
        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            data=df,
            source="cache",
            fetched_at=datetime.now(timezone.utc),
        )

    def has_data(self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime) -> bool:
        """Check if data exists in cache for the given range."""
        with self._session_factory() as session:
            count = (
                session.query(OHLCVRecord)
                .filter(
                    OHLCVRecord.symbol == symbol,
                    OHLCVRecord.timeframe == timeframe.value,
                    OHLCVRecord.timestamp >= start,
                    OHLCVRecord.timestamp <= end,
                )
                .count()
            )
        return count > 0
