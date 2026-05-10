"""Rolling z-score anomaly hooks for volatility / PnL spikes."""

from __future__ import annotations

from collections import deque
from typing import Any


class RollingZDetector:
    """Maintains a rolling window and flags z-score anomalies."""

    def __init__(self, window: int = 60):
        self._window = max(window, 5)
        self._vals: deque[float] = deque(maxlen=self._window)

    def update(self, x: float) -> dict[str, Any]:
        self._vals.append(float(x))
        if len(self._vals) < 5:
            return {"zscore": 0.0, "anomaly": False}
        import numpy as np

        arr = np.array(self._vals, dtype=float)
        mu = float(arr.mean())
        sigma = float(arr.std()) or 1e-9
        z = (float(x) - mu) / sigma
        return {"zscore": float(z), "anomaly": abs(z) >= 4.0}
