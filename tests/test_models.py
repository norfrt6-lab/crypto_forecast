"""Tests for src.models (baseline + LSTM)."""

import numpy as np
import pytest
import torch

from src.models.baseline import (
    GBTBaselineModel,
    LinearBaselineModel,
    NaiveRepeatModel,
    ZeroBaselineModel,
)
from src.models.lstm import LSTMForecaster


class TestZeroBaseline:
    def test_predict_shape(self):
        model = ZeroBaselineModel(horizon=5)
        X = np.random.randn(32, 60, 10)
        pred = model.predict(X)
        assert pred.shape == (32, 5)

    def test_predict_all_zeros(self):
        model = ZeroBaselineModel(horizon=3)
        X = np.random.randn(10, 30, 5)
        pred = model.predict(X)
        assert np.all(pred == 0)


class TestNaiveRepeat:
    def test_predict_shape(self):
        model = NaiveRepeatModel(target_col_idx=0, horizon=5)
        X = np.random.randn(20, 60, 10)
        pred = model.predict(X)
        assert pred.shape == (20, 5)

    def test_repeats_last_value(self):
        model = NaiveRepeatModel(target_col_idx=2, horizon=3)
        X = np.zeros((5, 10, 4))
        X[:, -1, 2] = [1.0, 2.0, 3.0, 4.0, 5.0]
        pred = model.predict(X)
        for i in range(5):
            np.testing.assert_array_equal(pred[i], [i + 1.0] * 3)


class TestLinearBaseline:
    def test_fit_predict(self):
        model = LinearBaselineModel(alpha=1.0)
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(100, 3)
        model.fit(X, y)
        pred = model.predict(X[:10])
        assert pred.shape == (10, 3)

    def test_predict_before_fit_raises(self):
        model = LinearBaselineModel()
        X = np.random.randn(10, 5, 3)
        with pytest.raises(RuntimeError, match="must be fit"):
            model.predict(X)


class TestGBTBaseline:
    def test_fit_predict(self):
        model = GBTBaselineModel(n_estimators=10, max_depth=2)
        X = np.random.randn(50, 10, 5)
        y = np.random.randn(50, 3)
        model.fit(X, y)
        pred = model.predict(X[:5])
        assert pred.shape == (5, 3)

    def test_predict_before_fit_raises(self):
        model = GBTBaselineModel()
        X = np.random.randn(10, 5, 3)
        with pytest.raises(RuntimeError, match="must be fit"):
            model.predict(X)


class TestLSTMForecaster:
    def test_forward_shape(self):
        model = LSTMForecaster(input_size=10, hidden_size=32, num_layers=2, horizon=5)
        x = torch.randn(8, 60, 10)
        out = model(x)
        assert out.shape == (8, 5)

    def test_bidirectional(self):
        model = LSTMForecaster(
            input_size=5, hidden_size=16, num_layers=1, horizon=3, bidirectional=True
        )
        x = torch.randn(4, 20, 5)
        out = model(x)
        assert out.shape == (4, 3)

    def test_gradient_flow(self):
        model = LSTMForecaster(input_size=5, hidden_size=16, num_layers=1, horizon=3)
        x = torch.randn(4, 10, 5)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.lstm.weight_ih_l0.grad is not None
