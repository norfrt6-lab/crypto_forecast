"""Tests for src.evaluation."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.evaluation import (
    Metrics,
    compute_direction_accuracy_price,
    compute_direction_accuracy_returns,
    compute_metrics,
    compute_per_step_direction_accuracy,
    inverse_transform_predictions,
)


class TestDirectionAccuracyReturns:
    def test_perfect_prediction(self):
        y_true = np.array([[0.01, -0.02], [0.03, 0.01]])
        acc = compute_direction_accuracy_returns(y_true, y_true)
        assert acc == 1.0

    def test_opposite_prediction(self):
        y_true = np.array([[0.01, -0.02]])
        y_pred = np.array([[-0.01, 0.02]])
        acc = compute_direction_accuracy_returns(y_true, y_pred)
        assert acc == 0.0

    def test_partial_accuracy(self):
        y_true = np.array([[0.01, -0.02]])
        y_pred = np.array([[0.05, 0.03]])  # 1 correct, 1 wrong
        acc = compute_direction_accuracy_returns(y_true, y_pred)
        assert acc == 0.5


class TestDirectionAccuracyPrice:
    def test_same_directions(self):
        y_true = np.array([[100, 101, 102]])
        y_pred = np.array([[100, 101.5, 103]])
        acc = compute_direction_accuracy_price(y_true, y_pred)
        assert acc == 1.0

    def test_empty_horizon(self):
        y_true = np.array([[100]])
        y_pred = np.array([[101]])
        acc = compute_direction_accuracy_price(y_true, y_pred)
        assert acc == 0.0


class TestComputeMetrics:
    def test_returns_metrics_dataclass(self):
        np.random.seed(42)
        y_true = np.random.randn(50, 5)
        y_pred = y_true + 0.1 * np.random.randn(50, 5)
        metrics = compute_metrics(y_true, y_pred, target_is_return=True)
        assert isinstance(metrics, Metrics)
        assert metrics.rmse >= 0
        assert metrics.mae >= 0
        assert 0 <= metrics.direction_accuracy <= 1

    def test_per_step_shapes(self):
        y_true = np.random.randn(30, 5)
        y_pred = np.random.randn(30, 5)
        metrics = compute_metrics(y_true, y_pred)
        assert len(metrics.per_step_mse) == 5
        assert len(metrics.per_step_mae) == 5
        assert len(metrics.per_step_direction_acc) == 5

    def test_perfect_prediction_zero_error(self):
        y_true = np.random.randn(20, 3)
        metrics = compute_metrics(y_true, y_true, target_is_return=True)
        assert metrics.mse < 1e-10
        assert metrics.mae < 1e-10


class TestPerStepDirectionAccuracy:
    def test_return_target(self):
        y_true = np.array([[0.1, -0.2, 0.3], [0.1, 0.2, -0.3]])
        accs = compute_per_step_direction_accuracy(y_true, y_true, target_is_return=True)
        assert len(accs) == 3
        np.testing.assert_array_equal(accs, [1.0, 1.0, 1.0])

    def test_price_target(self):
        y_true = np.array([[100, 101, 102]])
        y_pred = np.array([[100, 101.5, 103]])
        accs = compute_per_step_direction_accuracy(y_true, y_pred, target_is_return=False)
        assert len(accs) == 3


class TestInverseTransform:
    def test_roundtrip(self):
        scaler = StandardScaler()
        original = np.random.randn(50, 1) * 10 + 100
        scaler.fit(original)
        scaled = scaler.transform(original)
        # Tile to simulate horizon
        y_scaled = np.tile(scaled, (1, 3))
        result = inverse_transform_predictions(y_scaled, scaler)
        np.testing.assert_array_almost_equal(result[:, 0], original.flatten(), decimal=5)

    def test_output_shape(self):
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 1))
        y_scaled = np.random.randn(20, 5)
        result = inverse_transform_predictions(y_scaled, scaler)
        assert result.shape == (20, 5)
