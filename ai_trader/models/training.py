"""Training pipeline with walk-forward validation for LSTM models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai_trader.models.features import FeatureConfig, FeaturePipeline
from ai_trader.models.lstm import LSTMPredictor
from ai_trader.models.evaluation import evaluate_predictions, EvaluationResult
from ai_trader.logs import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    seed: int = 42

    # Walk-forward params
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    # test_ratio is implicit: 1 - train - val

    # Walk-forward folds
    n_folds: int = 3
    min_train_size: int = 200


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""

    fold: int
    train_size: int
    test_size: int
    train_metrics: dict[str, float]
    eval_result: EvaluationResult


@dataclass
class TrainingResult:
    """Complete result from the training pipeline."""

    fold_results: list[FoldResult]
    final_model_path: str | None = None
    aggregate_metrics: dict[str, float] = field(default_factory=dict)


class TrainingPipeline:
    """Reproducible training pipeline with walk-forward validation.

    Walk-forward validation prevents data leakage by always training
    on past data and testing on future data, simulating real deployment.

    Timeline: [====TRAIN====][==VAL==][==TEST==]
    Each fold slides forward, growing the training window.
    """

    def __init__(
        self,
        training_config: TrainingConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        self.t_config = training_config or TrainingConfig()
        self.f_config = feature_config or FeatureConfig()

    def run(self, df: pd.DataFrame, save_path: str | None = None) -> TrainingResult:
        """Execute the full training pipeline.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            save_path: Directory to save the final model.

        Returns:
            TrainingResult with fold metrics and model path.
        """
        logger.info("pipeline_start", rows=len(df), folds=self.t_config.n_folds)

        pipeline = FeaturePipeline(self.f_config)
        feature_df = pipeline.build_features(df)
        targets = pipeline.build_targets(df)

        fold_results = self._walk_forward(df, feature_df, targets, pipeline)
        aggregate = self._aggregate_metrics(fold_results)

        # Train final model on all available data for deployment
        final_model = self._train_final_model(feature_df, targets, pipeline)

        final_path = None
        if save_path and final_model is not None:
            final_model.save(save_path)
            final_path = save_path

        result = TrainingResult(
            fold_results=fold_results,
            final_model_path=final_path,
            aggregate_metrics=aggregate,
        )

        logger.info("pipeline_complete", folds=len(fold_results), **{k: f"{v:.4f}" for k, v in aggregate.items()})
        return result

    def _walk_forward(
        self,
        _df: pd.DataFrame,
        feature_df: pd.DataFrame,
        targets: pd.Series,
        _pipeline: FeaturePipeline,
    ) -> list[FoldResult]:
        """Walk-forward cross-validation.

        Each fold trains on a growing window and tests on
        the next unseen segment. Strictly chronological.
        """
        n = len(feature_df)
        n_folds = self.t_config.n_folds
        test_size = int(n * (1 - self.t_config.train_ratio - self.t_config.validation_ratio) / n_folds)
        test_size = max(test_size, 50)

        results = []

        for fold in range(n_folds):
            test_end = n - (n_folds - fold - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start

            if train_end < self.t_config.min_train_size:
                logger.warning("fold_skipped", fold=fold, reason="insufficient training data")
                continue

            # Split — strictly chronological, no overlap
            train_features = feature_df.iloc[:train_end]
            train_targets = targets.iloc[:train_end]
            test_features = feature_df.iloc[test_start:test_end]
            test_targets = targets.iloc[test_start:test_end]

            # Create sequences (fit normalization on train only)
            train_pipeline = FeaturePipeline(self.f_config)
            train_result = train_pipeline.create_sequences(train_features, train_targets, fit_normalization=True)

            if len(train_result.features) < self.t_config.batch_size:
                logger.warning("fold_skipped", fold=fold, reason="too few training sequences")
                continue

            # Test sequences use training normalization params (no leakage)
            test_result = train_pipeline.create_sequences(test_features, test_targets, fit_normalization=False)

            if len(test_result.features) == 0:
                logger.warning("fold_skipped", fold=fold, reason="no test sequences")
                continue

            # Train model for this fold
            model = self._create_model(train_result.features.shape[2])
            train_metrics = model.train(
                train_result.features,
                train_result.targets,
                epochs=self.t_config.epochs,
                batch_size=self.t_config.batch_size,
            )

            # Evaluate on test set
            probas = model.predict_proba(test_result.features)
            eval_result = evaluate_predictions(
                y_true=test_result.targets,
                y_proba=probas,
            )

            fold_results = FoldResult(
                fold=fold,
                train_size=len(train_result.features),
                test_size=len(test_result.features),
                train_metrics=train_metrics,
                eval_result=eval_result,
            )
            results.append(fold_results)

            logger.info(
                "fold_complete",
                fold=fold,
                accuracy=f"{eval_result.accuracy:.4f}",
                precision=f"{eval_result.precision:.4f}",
                recall=f"{eval_result.recall:.4f}",
            )

        return results

    def _train_final_model(
        self,
        feature_df: pd.DataFrame,
        targets: pd.Series,
        pipeline: FeaturePipeline,
    ) -> LSTMPredictor | None:
        """Train final model on all data (minus forecast horizon gap)."""
        result = pipeline.create_sequences(feature_df, targets, fit_normalization=True)

        if len(result.features) < self.t_config.batch_size:
            logger.warning("final_model_skipped", reason="insufficient data")
            return None

        model = self._create_model(result.features.shape[2])
        model.train(
            result.features,
            result.targets,
            epochs=self.t_config.epochs,
            batch_size=self.t_config.batch_size,
        )
        model.set_normalization_params(result.means, result.stds, result.feature_names)
        return model

    def _create_model(self, input_size: int) -> LSTMPredictor:
        return LSTMPredictor(
            input_size=input_size,
            hidden_size=self.t_config.hidden_size,
            num_layers=self.t_config.num_layers,
            dropout=self.t_config.dropout,
            learning_rate=self.t_config.learning_rate,
            seed=self.t_config.seed,
        )

    @staticmethod
    def _aggregate_metrics(fold_results: list[FoldResult]) -> dict[str, float]:
        """Average metrics across all folds."""
        if not fold_results:
            return {}

        keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        aggregate = {}
        for key in keys:
            values = [getattr(fr.eval_result, key) for fr in fold_results]
            aggregate[f"mean_{key}"] = float(np.mean(values))
            aggregate[f"std_{key}"] = float(np.std(values))

        return aggregate
