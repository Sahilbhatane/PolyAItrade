"""Data fetching API endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ai_trader.config import get_config
from ai_trader.data.provider import TimeFrame
from ai_trader.data.storage import MarketDataStore
from ai_trader.data.yfinance_provider import YFinanceProvider

router = APIRouter(prefix="/data", tags=["data"])


class FetchRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol (e.g., RELIANCE.NS, AAPL)")
    timeframe: str = Field(default="1d", description="Candle interval: 1m, 5m, 15m, 1h, 1d, 1w")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    use_cache: bool = Field(default=True, description="Return cached data if available")


class FetchResponse(BaseModel):
    symbol: str
    timeframe: str
    rows: int
    source: str
    data: list[dict]


@router.post("/fetch", response_model=FetchResponse)
async def fetch_data(request: FetchRequest) -> FetchResponse:
    """Fetch historical OHLCV data for a symbol.

    Returns normalized data from cache or yfinance.
    """
    try:
        timeframe = TimeFrame(request.timeframe)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {request.timeframe}")

    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD format")

    if start >= end:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    config = get_config()
    store = MarketDataStore(database_url=config.database.url)

    if request.use_cache:
        cached = store.load(request.symbol, timeframe, start, end)
        if cached is not None and not cached.data.empty:
            records = _df_to_records(cached.data)
            return FetchResponse(
                symbol=request.symbol,
                timeframe=request.timeframe,
                rows=len(records),
                source="cache",
                data=records,
            )

    provider = YFinanceProvider()
    try:
        market_data = await provider.fetch_historical(request.symbol, timeframe, start, end)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    store.save(market_data)

    records = _df_to_records(market_data.data)
    return FetchResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        rows=len(records),
        source="yfinance",
        data=records,
    )


def _df_to_records(df) -> list[dict]:
    """Convert DataFrame to list of dicts with ISO timestamp strings."""
    df = df.copy()
    df.index = df.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.reset_index()
    df = df.rename(columns={"index": "timestamp"} if "index" in df.columns else {})
    return df.to_dict(orient="records")
