"""LSTM-based price prediction model using PyTorch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ai_trader.models.base import BaseModel, ModelPrediction
from ai_trader.logs import get_logger

logger = get_logger(__name__)


class LSTMNetwork(nn.Module):
    """Multi-layer LSTM with dropout for time-series prediction.

    Outputs a single probability (price-up likelihood) per sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take last timestep
        return self.fc(last_hidden).squeeze(-1)


class LSTMPredictor(BaseModel):
    """LSTM model that predicts probability of price increase.

    Implements the BaseModel interface. Outputs probability scores only —
    never makes trade decisions directly.
    """

    def __init__(
        self,
        model_id: str = "lstm_v1",
        version: str = "0.1.0",
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        super().__init__(model_id, version)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.seed = seed

        torch.manual_seed(seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._network = LSTMNetwork(input_size, hidden_size, num_layers, dropout).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=learning_rate)
        self._criterion = nn.BCELoss()

        self._feature_names: list[str] = []
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    @property
    def network(self) -> LSTMNetwork:
        return self._network

    def train(self, features: np.ndarray, targets: np.ndarray, **kwargs: Any) -> dict[str, float]:
        """Train the LSTM on sequential feature data.

        Args:
            features: Shape (n_samples, seq_len, n_features).
            targets: Shape (n_samples,) — binary labels (1=price up, 0=price down).
            **kwargs: epochs, batch_size, validation_split.

        Returns:
            Training metrics dict.
        """
        if not self.validate_input(features):
            raise ValueError("Invalid training features: contains NaN or Inf")

        epochs = kwargs.get("epochs", 50)
        batch_size = kwargs.get("batch_size", 32)
        val_split = kwargs.get("validation_split", 0.0)

        torch.manual_seed(self.seed)

        # Split validation (chronological, no shuffle — prevents data leakage)
        if val_split > 0:
            split_idx = int(len(features) * (1 - val_split))
            X_train, X_val = features[:split_idx], features[split_idx:]
            y_train, y_val = targets[:split_idx], targets[split_idx:]
        else:
            X_train, y_train = features, targets
            X_val, y_val = None, None

        X_tensor = torch.FloatTensor(X_train).to(self._device)
        y_tensor = torch.FloatTensor(y_train).to(self._device)

        self._network.train()
        train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            indices = list(range(len(X_tensor)))
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]

                self._optimizer.zero_grad()
                predictions = self._network(X_batch)
                loss = self._criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info("training_epoch", epoch=epoch + 1, loss=f"{avg_loss:.4f}")

        metrics = {
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "epochs_trained": float(epochs),
        }

        if X_val is not None:
            val_metrics = self._evaluate_set(X_val, y_val)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        self._is_trained = True
        logger.info("training_complete", **{k: f"{v:.4f}" for k, v in metrics.items()})
        return metrics

    def predict(self, features: np.ndarray) -> list[ModelPrediction]:
        """Generate probability predictions.

        Args:
            features: Shape (n_samples, seq_len, n_features).

        Returns:
            List of ModelPrediction with probability-based signal and confidence.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        if not self.validate_input(features):
            raise ValueError("Invalid prediction features: contains NaN or Inf")

        self._network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(features).to(self._device)
            probabilities = self._network(X_tensor).cpu().numpy()

        predictions = []
        for prob in probabilities:
            # Convert probability to signal: 0.5 is neutral, >0.5 bullish, <0.5 bearish
            signal = (float(prob) - 0.5) * 2.0  # maps [0,1] → [-1,1]
            confidence = abs(signal)

            predictions.append(ModelPrediction(
                symbol="",
                signal=signal,
                confidence=confidence,
                features_used=self._feature_names,
                metadata={"raw_probability": float(prob)},
            ))

        return predictions

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return raw probability array (convenience method for evaluation)."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self._network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(features).to(self._device)
            return self._network(X_tensor).cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights and metadata to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._network.state_dict(), save_dir / "weights.pt")

        meta = {
            "model_id": self.model_id,
            "version": self.version,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "feature_names": self._feature_names,
            "is_trained": self._is_trained,
        }
        if self._feature_means is not None:
            meta["feature_means"] = self._feature_means.tolist()
            meta["feature_stds"] = self._feature_stds.tolist()

        with open(save_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("model_saved", path=str(save_dir))

    def load(self, path: str) -> None:
        """Load model weights and metadata from disk."""
        load_dir = Path(path)

        with open(load_dir / "meta.json", "r") as f:
            meta = json.load(f)

        self.input_size = meta["input_size"]
        self.hidden_size = meta["hidden_size"]
        self.num_layers = meta["num_layers"]
        self.dropout = meta["dropout"]
        self._feature_names = meta.get("feature_names", [])
        self._is_trained = meta.get("is_trained", False)

        if "feature_means" in meta:
            self._feature_means = np.array(meta["feature_means"])
            self._feature_stds = np.array(meta["feature_stds"])

        self._network = LSTMNetwork(
            self.input_size, self.hidden_size, self.num_layers, self.dropout
        ).to(self._device)
        self._network.load_state_dict(
            torch.load(load_dir / "weights.pt", map_location=self._device, weights_only=True)
        )
        logger.info("model_loaded", path=str(load_dir))

    def set_normalization_params(self, means: np.ndarray, stds: np.ndarray, feature_names: list[str]) -> None:
        """Store normalization parameters computed from training data."""
        self._feature_means = means
        self._feature_stds = stds
        self._feature_names = feature_names

    def _evaluate_set(self, features: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        """Evaluate model on a dataset. Returns loss and accuracy."""
        self._network.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self._device)
            y = torch.FloatTensor(targets).to(self._device)
            preds = self._network(X)
            loss = self._criterion(preds, y).item()

            binary_preds = (preds.cpu().numpy() >= 0.5).astype(int)
            accuracy = float((binary_preds == targets).mean())

        return {"loss": loss, "accuracy": accuracy}
