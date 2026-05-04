"""ML-specific configuration loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ai_trader.models.features import FeatureConfig
from ai_trader.models.training import TrainingConfig


class ModelArchConfig(BaseModel):
    """LSTM architecture parameters."""

    type: str = Field(default="lstm")
    input_size: int = Field(default=15)
    hidden_size: int = Field(default=64, ge=8)
    num_layers: int = Field(default=2, ge=1, le=5)
    dropout: float = Field(default=0.2, ge=0.0, le=0.5)
    learning_rate: float = Field(default=1e-3, gt=0)
    seed: int = Field(default=42)


class RetrainingConfig(BaseModel):
    """Periodic retraining schedule to combat model drift."""

    retrain_every_n_bars: int = Field(default=60, ge=10)
    min_new_data_points: int = Field(default=30, ge=10)


class MLPipelineConfig(BaseModel):
    """Complete ML pipeline configuration."""

    model: ModelArchConfig = Field(default_factory=ModelArchConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)


def load_ml_config(path: str | Path | None = None) -> MLPipelineConfig:
    """Load ML config from YAML. Falls back to defaults if missing."""
    if path is None:
        path = Path("ml_config.yaml")
    else:
        path = Path(path)

    if not path.exists():
        return MLPipelineConfig()

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return MLPipelineConfig()

    # Map nested YAML to pydantic models
    model_data = data.get("model", {})
    features_data = data.get("features", {})
    training_data = data.get("training", {})
    retraining_data = data.get("retraining", {})

    return MLPipelineConfig(
        model=ModelArchConfig(**model_data),
        features=FeatureConfig(**features_data),
        training=TrainingConfig(**{
            **training_data,
            "hidden_size": model_data.get("hidden_size", 64),
            "num_layers": model_data.get("num_layers", 2),
            "dropout": model_data.get("dropout", 0.2),
            "learning_rate": model_data.get("learning_rate", 1e-3),
            "seed": model_data.get("seed", 42),
        }),
        retraining=RetrainingConfig(**retraining_data),
    )
