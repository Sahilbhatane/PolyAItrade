"""ML model training and prediction API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ai_trader.config import get_config
from ai_trader.data.provider import TimeFrame
from ai_trader.data.storage import MarketDataStore
from ai_trader.data.yfinance_provider import YFinanceProvider
from ai_trader.logs import get_logger

router = APIRouter(prefix="/ml", tags=["ml"])
logger = get_logger(__name__)


class TrainRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol for training data")
    start_date: str = Field(..., description="Training data start (YYYY-MM-DD)")
    end_date: str = Field(..., description="Training data end (YYYY-MM-DD)")
    timeframe: str = Field(default="1d")
    config_path: str = Field(default="ml_config.yaml")
    save_path: str = Field(default="models/lstm_latest")


class TrainResponse(BaseModel):
    status: str
    folds_completed: int
    aggregate_metrics: dict[str, float]
    model_path: str | None


class PredictRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol")
    model_path: str = Field(default="models/lstm_latest")
    lookback_days: int = Field(default=60, ge=30)


class PredictResponse(BaseModel):
    symbol: str
    probability_up: float
    signal: float
    confidence: float
    model_id: str


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest) -> TrainResponse:
    """Train the LSTM model on historical data using walk-forward validation."""
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD format")

    if start >= end:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    try:
        timeframe = TimeFrame(request.timeframe)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {request.timeframe}")

    config = get_config()
    store = MarketDataStore(database_url=config.database.url)

    # Fetch data
    market_data = store.load(request.symbol, timeframe, start, end)
    if market_data is None or market_data.data.empty:
        provider = YFinanceProvider()
        try:
            market_data = await provider.fetch_historical(request.symbol, timeframe, start, end)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        store.save(market_data)

    df = market_data.data

    if len(df) < 100:
        raise HTTPException(status_code=400, detail="Insufficient data for training (need >= 100 rows)")

    # Load ML config and run pipeline
    from ai_trader.models.config import load_ml_config
    from ai_trader.models.training import TrainingPipeline

    ml_config = load_ml_config(request.config_path)
    pipeline = TrainingPipeline(ml_config.training, ml_config.features)

    try:
        result = pipeline.run(df, save_path=request.save_path)
    except Exception as e:
        logger.error("training_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    return TrainResponse(
        status="success",
        folds_completed=len(result.fold_results),
        aggregate_metrics=result.aggregate_metrics,
        model_path=result.final_model_path,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Get prediction from a trained model for the latest market data."""
    from ai_trader.models.lstm import LSTMPredictor
    from ai_trader.models.features import FeaturePipeline, FeatureConfig

    # Load model
    try:
        model = LSTMPredictor()
        model.load(request.model_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found at: {request.model_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Fetch recent market data
    config = get_config()
    provider = YFinanceProvider()
    end = datetime.utcnow()
    start = datetime(end.year, end.month, end.day)

    from datetime import timedelta
    start = end - timedelta(days=request.lookback_days)

    try:
        market_data = await provider.fetch_historical(
            request.symbol, TimeFrame.DAY_1, start, end
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Build features and predict
    f_config = FeatureConfig(sequence_length=30)
    pipeline = FeaturePipeline(f_config)
    pipeline._means = model._feature_means
    pipeline._stds = model._feature_stds

    sequence = pipeline.transform_single(market_data.data)
    if sequence is None:
        raise HTTPException(status_code=400, detail="Insufficient data for prediction")

    predictions = model.predict(sequence)
    pred = predictions[0]
    pred.symbol = request.symbol

    return PredictResponse(
        symbol=request.symbol,
        probability_up=pred.metadata["raw_probability"],
        signal=pred.signal,
        confidence=pred.confidence,
        model_id=model.model_id,
    )
