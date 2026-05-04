"""Feature engineering pipeline for ML models.

Converts raw OHLCV data + technical indicators into normalized
sequential feature arrays suitable for LSTM input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_trader.strategies.indicators import Indicators


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    sequence_length: int = 30
    forecast_horizon: int = 1
    price_change_threshold: float = 0.0

    # Which features to include
    use_returns: bool = True
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    use_atr: bool = True
    use_volume_ratio: bool = True
    use_ma_ratios: bool = True

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    atr_period: int = 14
    sma_fast: int = 10
    sma_slow: int = 30


@dataclass
class FeaturePipelineResult:
    """Output of the feature engineering pipeline."""

    features: np.ndarray       # (n_samples, seq_len, n_features)
    targets: np.ndarray        # (n_samples,) binary
    feature_names: list[str]
    timestamps: list           # timestamp for each sample's prediction point
    means: np.ndarray          # per-feature means (from training data only)
    stds: np.ndarray           # per-feature stds (from training data only)


class FeaturePipeline:
    """Transforms OHLCV DataFrame into LSTM-ready sequences.

    Enforces strict temporal ordering to prevent data leakage:
    - Normalization stats computed only from training portion
    - No future data in any feature computation
    - Returns and indicators use only past data relative to each sample
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all feature columns from raw OHLCV.

        All features are computed from past data only (no leakage).
        """
        features = pd.DataFrame(index=df.index)
        cfg = self.config

        if cfg.use_returns:
            features["return_1"] = df["close"].pct_change(1)
            features["return_5"] = df["close"].pct_change(5)
            features["return_10"] = df["close"].pct_change(10)
            features["log_return"] = np.log(df["close"] / df["close"].shift(1))

        if cfg.use_rsi:
            features["rsi"] = Indicators.rsi(df["close"], cfg.rsi_period) / 100.0

        if cfg.use_macd:
            macd_line, signal_line, hist = Indicators.macd(
                df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
            )
            features["macd_line"] = macd_line
            features["macd_signal"] = signal_line
            features["macd_hist"] = hist

        if cfg.use_bollinger:
            upper, middle, lower = Indicators.bollinger_bands(
                df["close"], cfg.bollinger_period
            )
            features["bb_width"] = (upper - lower) / middle
            features["bb_position"] = (df["close"] - lower) / (upper - lower)

        if cfg.use_atr:
            features["atr_ratio"] = Indicators.atr(df, cfg.atr_period) / df["close"]

        if cfg.use_volume_ratio:
            vol_ma = df["volume"].rolling(20).mean()
            features["volume_ratio"] = df["volume"] / vol_ma

        if cfg.use_ma_ratios:
            sma_fast = Indicators.sma(df["close"], cfg.sma_fast)
            sma_slow = Indicators.sma(df["close"], cfg.sma_slow)
            features["ma_ratio"] = sma_fast / sma_slow - 1.0
            features["price_to_sma"] = df["close"] / sma_slow - 1.0

        # High-Low range normalized by close
        features["hl_range"] = (df["high"] - df["low"]) / df["close"]

        return features

    def build_targets(self, df: pd.DataFrame) -> pd.Series:
        """Build binary target: 1 if price goes up in forecast_horizon, 0 otherwise.

        Uses future close relative to current close (the thing we're predicting).
        """
        future_return = df["close"].shift(-self.config.forecast_horizon) / df["close"] - 1.0
        return (future_return > self.config.price_change_threshold).astype(float)

    def create_sequences(
        self,
        feature_df: pd.DataFrame,
        targets: pd.Series,
        fit_normalization: bool = True,
    ) -> FeaturePipelineResult:
        """Convert feature DataFrame into overlapping sequences for LSTM.

        Args:
            feature_df: DataFrame of computed features.
            targets: Series of binary targets.
            fit_normalization: If True, compute mean/std from this data.
                Set False for test data to reuse training stats.

        Returns:
            FeaturePipelineResult with arrays ready for model.
        """
        # Drop rows where features or targets are NaN
        combined = feature_df.copy()
        combined["_target"] = targets
        combined = combined.dropna()

        feature_cols = [c for c in combined.columns if c != "_target"]
        feature_values = combined[feature_cols].values
        target_values = combined["_target"].values
        timestamps = combined.index.tolist()

        if fit_normalization:
            self._means = np.mean(feature_values, axis=0)
            self._stds = np.std(feature_values, axis=0)
            self._stds[self._stds == 0] = 1.0  # prevent division by zero

        if self._means is None:
            raise ValueError("Normalization params not set. Call with fit_normalization=True first.")

        normalized = (feature_values - self._means) / self._stds

        seq_len = self.config.sequence_length
        X, y, ts = [], [], []

        for i in range(seq_len, len(normalized)):
            X.append(normalized[i - seq_len:i])
            y.append(target_values[i])
            ts.append(timestamps[i])

        return FeaturePipelineResult(
            features=np.array(X, dtype=np.float32),
            targets=np.array(y, dtype=np.float32),
            feature_names=feature_cols,
            timestamps=ts,
            means=self._means,
            stds=self._stds,
        )

    def transform_single(self, df: pd.DataFrame) -> np.ndarray | None:
        """Transform the latest N bars into a single sequence for live prediction.

        Returns None if insufficient data.
        """
        feature_df = self.build_features(df)
        feature_df = feature_df.dropna()

        if len(feature_df) < self.config.sequence_length:
            return None

        if self._means is None:
            return None

        feature_cols = list(feature_df.columns)
        values = feature_df[feature_cols].values[-self.config.sequence_length:]
        normalized = (values - self._means) / self._stds

        return np.array([normalized], dtype=np.float32)
