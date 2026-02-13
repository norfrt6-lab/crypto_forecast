"""
Visualization module.

Generates publication-quality plots for model evaluation:
- Predictions vs actuals
- Per-horizon-step error
- Training loss curves
- Model comparison bar charts
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STYLE = {
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}
plt.rcParams.update(STYLE)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex | None,
    horizon: int,
    output_dir: str,
    model_name: str = "LSTM",
    step: int = 0,
    target_label: str = "Daily Return",
) -> None:
    """
    Plot predicted vs actual values for a specific horizon step.

    Args:
        step: which horizon step to plot (0 = t+1, 1 = t+2, etc.)
        target_label: y-axis label (e.g., "Daily Return" or "Price (USD)")
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots()

    x_axis = dates[:len(y_true)] if dates is not None else np.arange(len(y_true))

    ax.plot(x_axis, y_true[:, step], label="Actual", linewidth=1.5, color="#2196F3")
    ax.plot(x_axis, y_pred[:, step], label=f"{model_name} Predicted", linewidth=1.2,
            color="#FF5722", alpha=0.85)

    ax.set_title(f"{target_label}: Actual vs {model_name} (t+{step + 1})")
    ax.set_ylabel(target_label)
    ax.legend()

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    fig.tight_layout()
    fname = out / f"pred_vs_actual_{model_name.lower()}_step{step + 1}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", fname)


def plot_all_horizon_steps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex | None,
    output_dir: str,
    model_name: str = "LSTM",
    target_label: str = "Daily Return",
) -> None:
    """Plot predictions for each horizon step in a grid."""
    out = _ensure_dir(output_dir)
    horizon = y_true.shape[1]
    cols = min(horizon, 3)
    rows = (horizon + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if horizon == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    x_axis = dates[:len(y_true)] if dates is not None else np.arange(len(y_true))

    for k in range(horizon):
        ax = axes[k]
        ax.plot(x_axis, y_true[:, k], label="Actual", linewidth=1.2, color="#2196F3")
        ax.plot(x_axis, y_pred[:, k], label="Predicted", linewidth=1.0,
                color="#FF5722", alpha=0.8)
        ax.set_title(f"t+{k + 1}")
        ax.set_ylabel(target_label)
        ax.legend(fontsize=8)
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    for k in range(horizon, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(f"{model_name}: Multi-Step Predictions", fontsize=14)
    fig.tight_layout()
    fname = out / f"all_steps_{model_name.lower()}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", fname)


def plot_training_loss(
    history: dict,
    output_dir: str,
) -> None:
    """Plot training and validation loss curves."""
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots()

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()

    fig.tight_layout()
    fname = out / "training_loss.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", fname)


def plot_per_step_metrics(
    metrics_dict: dict,
    output_dir: str,
    metric_unit: str = "Return",
) -> None:
    """
    Bar chart comparing per-step MAE across models.

    metrics_dict: {model_name: Metrics}
    """
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots()

    n_models = len(metrics_dict)
    first_metrics = next(iter(metrics_dict.values()))
    horizon = len(first_metrics.per_step_mae)
    x = np.arange(horizon)
    width = 0.8 / n_models

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]

    for i, (name, m) in enumerate(metrics_dict.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, m.per_step_mae, width, label=name, color=colors[i % len(colors)])

    ax.set_xlabel("Horizon Step")
    ax.set_ylabel(f"MAE ({metric_unit})")
    ax.set_title("Per-Step MAE by Model")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{k + 1}" for k in range(horizon)])
    ax.legend()

    fig.tight_layout()
    fname = out / "per_step_mae_comparison.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", fname)


def plot_model_comparison(
    metrics_dict: dict,
    output_dir: str,
    metric_unit: str = "Return",
) -> None:
    """Summary bar chart comparing overall metrics across models."""
    out = _ensure_dir(output_dir)

    model_names = list(metrics_dict.keys())
    rmses = [m.rmse for m in metrics_dict.values()]
    maes = [m.mae for m in metrics_dict.values()]
    dir_accs = [m.direction_accuracy * 100 for m in metrics_dict.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"][:len(model_names)]

    for ax, values, title, ylabel in [
        (axes[0], rmses, "RMSE", metric_unit),
        (axes[1], maes, "MAE", metric_unit),
        (axes[2], dir_accs, "Direction Accuracy", "%"),
    ]:
        bars = ax.bar(model_names, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}" if ylabel != "%" else f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Model Comparison on Test Set", fontsize=14)
    fig.tight_layout()
    fname = out / "model_comparison.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", fname)
