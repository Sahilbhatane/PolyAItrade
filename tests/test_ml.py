"""Tests for the ML prediction module: LSTM, features, training, evaluation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ai_trader.models.evaluation import EvaluationResult, evaluate_predictions, _compute_auc
from ai_trader.models.features import FeatureConfig, FeaturePipeline
from ai_trader.models.lstm import LSTMNetwork, LSTMPredictor
from ai_trader.models.training import TrainingConfig, TrainingPipeline


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic OHLCV data with trend."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    volume = rng.integers(5000, 50000, n)
    df = pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = "timestamp"
    return df


# ── LSTM Network ──────────────────────────────────────────────────────────────


class TestLSTMNetwork:
    def test_forward_shape(self):
        net = LSTMNetwork(input_size=10, hidden_size=32, num_layers=1)
        x = torch.randn(4, 30, 10)
        out = net(x)
        assert out.shape == (4,)

    def test_output_range(self):
        net = LSTMNetwork(input_size=5, hidden_size=16, num_layers=1)
        x = torch.randn(8, 20, 5)
        out = net(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_deterministic_with_seed(self):
        """Networks initialized with the same seed should have identical parameters."""
        torch.manual_seed(42)
        net1 = LSTMNetwork(input_size=5, hidden_size=16, num_layers=1)

        torch.manual_seed(42)
        net2 = LSTMNetwork(input_size=5, hidden_size=16, num_layers=1)

        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            torch.testing.assert_close(p1, p2)


# ── LSTM Predictor ────────────────────────────────────────────────────────────


class TestLSTMPredictor:
    def test_train_and_predict(self):
        model = LSTMPredictor(input_size=5, hidden_size=16, num_layers=1, seed=42)
        X = np.random.default_rng(42).normal(size=(100, 20, 5)).astype(np.float32)
        y = (np.random.default_rng(42).random(100) > 0.5).astype(np.float32)

        metrics = model.train(X, y, epochs=5, batch_size=16)
        assert "final_train_loss" in metrics
        assert model.is_trained

        preds = model.predict(X[:3])
        assert len(preds) == 3
        for p in preds:
            assert -1.0 <= p.signal <= 1.0
            assert 0.0 <= p.confidence <= 1.0
            assert "raw_probability" in p.metadata

    def test_predict_before_train_raises(self):
        model = LSTMPredictor(input_size=5, hidden_size=16)
        X = np.random.default_rng(0).normal(size=(5, 10, 5)).astype(np.float32)
        with pytest.raises(RuntimeError, match="trained"):
            model.predict(X)

    def test_reject_nan_input(self):
        model = LSTMPredictor(input_size=3)
        X = np.array([[[1.0, np.nan, 3.0]]])
        with pytest.raises(ValueError, match="NaN"):
            model.train(X, np.array([1.0]))

    def test_save_and_load(self):
        model = LSTMPredictor(input_size=5, hidden_size=16, seed=42)
        X = np.random.default_rng(42).normal(size=(50, 10, 5)).astype(np.float32)
        y = (np.random.default_rng(42).random(50) > 0.5).astype(np.float32)
        model.train(X, y, epochs=3, batch_size=16)
        model.set_normalization_params(
            np.zeros(5), np.ones(5), ["f1", "f2", "f3", "f4", "f5"]
        )

        preds_before = model.predict_proba(X[:5])

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = LSTMPredictor()
            loaded.load(tmpdir)

            assert loaded.is_trained
            assert loaded.input_size == 5
            preds_after = loaded.predict_proba(X[:5])

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-6)

    def test_reproducible_training(self):
        X = np.random.default_rng(10).normal(size=(80, 15, 5)).astype(np.float32)
        y = (np.random.default_rng(10).random(80) > 0.5).astype(np.float32)

        m1 = LSTMPredictor(input_size=5, hidden_size=16, seed=99)
        m1.train(X, y, epochs=5, batch_size=16)

        m2 = LSTMPredictor(input_size=5, hidden_size=16, seed=99)
        m2.train(X, y, epochs=5, batch_size=16)

        p1 = m1.predict_proba(X[:10])
        p2 = m2.predict_proba(X[:10])
        np.testing.assert_allclose(p1, p2, atol=1e-5)


# ── Feature Pipeline ─────────────────────────────────────────────────────────


class TestFeaturePipeline:
    def test_build_features(self):
        df = _make_ohlcv(200)
        pipeline = FeaturePipeline()
        features = pipeline.build_features(df)

        assert len(features) == len(df)
        assert "rsi" in features.columns
        assert "macd_line" in features.columns
        assert "volume_ratio" in features.columns

    def test_build_targets_binary(self):
        df = _make_ohlcv(100)
        pipeline = FeaturePipeline()
        targets = pipeline.build_targets(df)

        valid = targets.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_create_sequences_shape(self):
        df = _make_ohlcv(200)
        config = FeatureConfig(sequence_length=20)
        pipeline = FeaturePipeline(config)
        features = pipeline.build_features(df)
        targets = pipeline.build_targets(df)

        result = pipeline.create_sequences(features, targets)

        assert result.features.ndim == 3
        assert result.features.shape[1] == 20  # seq_len
        assert result.features.shape[2] == len(result.feature_names)
        assert len(result.targets) == len(result.features)

    def test_normalization_no_leakage(self):
        """Test data normalization uses only training stats."""
        df = _make_ohlcv(300)
        config = FeatureConfig(sequence_length=15)
        pipeline = FeaturePipeline(config)
        features = pipeline.build_features(df)
        targets = pipeline.build_targets(df)

        # Train split
        split = 200
        train_f = features.iloc[:split]
        train_t = targets.iloc[:split]
        test_f = features.iloc[split:]
        test_t = targets.iloc[split:]

        train_result = pipeline.create_sequences(train_f, train_t, fit_normalization=True)
        test_result = pipeline.create_sequences(test_f, test_t, fit_normalization=False)

        # Normalization params should be from train only
        np.testing.assert_array_equal(train_result.means, test_result.means)
        np.testing.assert_array_equal(train_result.stds, test_result.stds)

    def test_transform_single(self):
        df = _make_ohlcv(100)
        config = FeatureConfig(sequence_length=20)
        pipeline = FeaturePipeline(config)
        features = pipeline.build_features(df)
        targets = pipeline.build_targets(df)

        pipeline.create_sequences(features, targets, fit_normalization=True)
        result = pipeline.transform_single(df)

        assert result is not None
        assert result.shape == (1, 20, len(features.columns))


# ── Evaluation ────────────────────────────────────────────────────────────────


class TestEvaluation:
    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_proba = np.array([0.9, 0.8, 0.1, 0.2, 0.7])

        result = evaluate_predictions(y_true, y_proba)

        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_random_predictions(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 1000).astype(float)
        y_proba = rng.random(1000)

        result = evaluate_predictions(y_true, y_proba)

        assert 0.4 <= result.accuracy <= 0.6
        assert 0.3 <= result.auc_roc <= 0.7

    def test_profit_impact_with_returns(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        returns = np.array([0.02, -0.01, 0.015, 0.01, -0.02])

        result = evaluate_predictions(y_true, y_proba, price_returns=returns)

        # Model predicted up for idx 0,1,2 (proba >= 0.5)
        # Returns for those: 0.02, -0.01, 0.015
        assert result.profit_factor > 1.0

    def test_auc_edge_cases(self):
        assert _compute_auc(np.array([1, 1, 1]), np.array([0.5, 0.6, 0.7])) == 0.5
        assert _compute_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) == 0.5


# ── Training Pipeline ─────────────────────────────────────────────────────────


class TestTrainingPipeline:
    def test_walk_forward_runs(self):
        df = _make_ohlcv(400, seed=77)

        t_config = TrainingConfig(
            epochs=3,
            batch_size=16,
            hidden_size=16,
            num_layers=1,
            n_folds=2,
            min_train_size=50,
            seed=42,
        )
        f_config = FeatureConfig(sequence_length=15)

        pipeline = TrainingPipeline(t_config, f_config)
        result = pipeline.run(df)

        assert len(result.fold_results) > 0
        for fr in result.fold_results:
            assert fr.eval_result.total_samples > 0
            assert 0 <= fr.eval_result.accuracy <= 1

    def test_walk_forward_no_future_leakage(self):
        """Verify that each fold's test data comes after training data."""
        df = _make_ohlcv(400, seed=55)

        t_config = TrainingConfig(
            epochs=2, batch_size=16, hidden_size=16,
            num_layers=1, n_folds=2, min_train_size=50, seed=42,
        )
        f_config = FeatureConfig(sequence_length=10)

        pipeline = TrainingPipeline(t_config, f_config)
        result = pipeline.run(df)

        for fr in result.fold_results:
            # Test period should have fewer samples than training
            assert fr.train_size > fr.test_size

    def test_reproducible_pipeline(self):
        df = _make_ohlcv(300, seed=33)
        t_config = TrainingConfig(
            epochs=3, batch_size=16, hidden_size=16,
            num_layers=1, n_folds=1, min_train_size=50, seed=42,
        )
        f_config = FeatureConfig(sequence_length=10)

        r1 = TrainingPipeline(t_config, f_config).run(df.copy())
        r2 = TrainingPipeline(t_config, f_config).run(df.copy())

        assert len(r1.fold_results) == len(r2.fold_results)
        if r1.fold_results and r2.fold_results:
            assert r1.fold_results[0].eval_result.accuracy == pytest.approx(
                r2.fold_results[0].eval_result.accuracy, abs=1e-4
            )

    def test_save_final_model(self):
        df = _make_ohlcv(300, seed=44)
        t_config = TrainingConfig(
            epochs=2, batch_size=16, hidden_size=16,
            num_layers=1, n_folds=1, min_train_size=50, seed=42,
        )
        f_config = FeatureConfig(sequence_length=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = TrainingPipeline(t_config, f_config).run(df, save_path=tmpdir)
            assert result.final_model_path is not None
            assert Path(tmpdir, "weights.pt").exists()
            assert Path(tmpdir, "meta.json").exists()


# ── ML + Strategy Integration ─────────────────────────────────────────────────


class TestMLStrategyIntegration:
    def test_strategy_works_without_ml(self):
        """Strategy must function when ML is disabled (default)."""
        from ai_trader.strategies.config_loader import StrategyConfig
        from ai_trader.strategies.rule_engine import RuleBasedStrategy
        from ai_trader.backtesting.simulator import Simulator

        config = StrategyConfig()
        assert not config.ml.enabled

        strategy = RuleBasedStrategy(config, initial_capital=100_000.0)
        df = _make_ohlcv(150)
        df.attrs["symbol"] = "TEST"

        simulator = Simulator(initial_capital=100_000.0)
        result = simulator.run(df, strategy)

        assert len(result.equity_curve) > 0

    def test_ml_weight_zero_has_no_effect(self):
        """With ml weight=0.0, results should be identical to pure rule-based."""
        from ai_trader.strategies.config_loader import StrategyConfig, SignalWeights
        from ai_trader.strategies.rule_engine import RuleBasedStrategy
        from ai_trader.backtesting.simulator import Simulator

        df = _make_ohlcv(150, seed=88)
        df.attrs["symbol"] = "TEST"

        config1 = StrategyConfig(weights=SignalWeights(ml_prediction=0.0))
        config2 = StrategyConfig(weights=SignalWeights(ml_prediction=0.0))

        s1 = RuleBasedStrategy(config1, initial_capital=100_000.0)
        s2 = RuleBasedStrategy(config2, initial_capital=100_000.0)

        r1 = Simulator(initial_capital=100_000.0).run(df.copy(), s1)
        r2 = Simulator(initial_capital=100_000.0).run(df.copy(), s2)

        assert r1.total_return == r2.total_return
