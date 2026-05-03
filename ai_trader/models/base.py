"""Base interface for ML models used in signal generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class ModelPrediction:
    """Standardized prediction output from any ML model."""

    symbol: str
    signal: float  # -1.0 (strong sell) to 1.0 (strong buy)
    confidence: float  # 0.0 to 1.0
    features_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseModel(ABC):
    """Abstract base class for ML models.

    Models produce probability-based signals, never direct trade decisions.
    """

    def __init__(self, model_id: str, version: str = "0.1.0"):
        self.model_id = model_id
        self.version = version
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @abstractmethod
    def train(self, features: np.ndarray, targets: np.ndarray, **kwargs: Any) -> dict[str, float]:
        """Train the model on historical data.

        Returns:
            Dictionary of training metrics.
        """
        ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> list[ModelPrediction]:
        """Generate predictions from input features.

        Must return probabilities/signals, NOT binary decisions.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights/state to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights/state from disk."""
        ...

    def validate_input(self, features: np.ndarray) -> bool:
        """Sanity-check input shape and values before prediction."""
        if features is None or features.size == 0:
            return False
        if np.isnan(features).any() or np.isinf(features).any():
            return False
        return True
