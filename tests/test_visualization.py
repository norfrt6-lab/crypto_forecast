"""Tests for src.visualization."""

import numpy as np
import pandas as pd

from src.evaluation import Metrics
from src.visualization import (
    _ensure_dir,
    plot_all_horizon_steps,
    plot_model_comparison,
    plot_per_step_metrics,
    plot_predictions_vs_actual,
    plot_training_loss,
)


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        target = str(tmp_path / "sub" / "dir")
        result = _ensure_dir(target)
        assert result.exists()


class TestPlotPredictionsVsActual:
    def test_generates_plot(self, tmp_path):
        y_true = np.random.randn(50, 3)
        y_pred = np.random.randn(50, 3)
        dates = pd.date_range("2020-01-01", periods=50)
        plot_predictions_vs_actual(y_true, y_pred, dates, horizon=3, output_dir=str(tmp_path))
        assert any(tmp_path.glob("*.png"))

    def test_no_dates(self, tmp_path):
        y_true = np.random.randn(30, 2)
        y_pred = np.random.randn(30, 2)
        plot_predictions_vs_actual(y_true, y_pred, None, horizon=2, output_dir=str(tmp_path))
        assert any(tmp_path.glob("*.png"))


class TestPlotAllHorizonSteps:
    def test_generates_plot(self, tmp_path):
        y_true = np.random.randn(40, 5)
        y_pred = np.random.randn(40, 5)
        plot_all_horizon_steps(y_true, y_pred, None, output_dir=str(tmp_path))
        assert any(tmp_path.glob("*.png"))

    def test_single_horizon(self, tmp_path):
        y_true = np.random.randn(20, 1)
        y_pred = np.random.randn(20, 1)
        plot_all_horizon_steps(y_true, y_pred, None, output_dir=str(tmp_path))
        assert any(tmp_path.glob("*.png"))


class TestPlotTrainingLoss:
    def test_generates_plot(self, tmp_path):
        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.5],
            "val_loss": [1.1, 0.9, 0.7, 0.65],
        }
        plot_training_loss(history, output_dir=str(tmp_path))
        assert any(tmp_path.glob("*.png"))


def _make_metrics(rmse=0.1, mae=0.08, horizon=3):
    return Metrics(
        mse=rmse**2,
        mae=mae,
        rmse=rmse,
        direction_accuracy=0.55,
        per_step_mse=np.array([rmse**2] * horizon),
        per_step_mae=np.array([mae] * horizon),
        per_step_direction_acc=np.array([0.55] * horizon),
    )


class TestPlotPerStepMetrics:
    def test_generates_plot(self, tmp_path):
        metrics_dict = {
            "LSTM": _make_metrics(0.1),
            "Linear": _make_metrics(0.15),
        }
        plot_per_step_metrics(metrics_dict, output_dir=str(tmp_path))
        assert any(tmp_path.glob("*.png"))


class TestPlotModelComparison:
    def test_generates_plot(self, tmp_path):
        metrics_dict = {
            "LSTM": _make_metrics(0.1),
            "Zero": _make_metrics(0.2),
        }
        plot_model_comparison(metrics_dict, output_dir=str(tmp_path))
        assert any(tmp_path.glob("*.png"))
