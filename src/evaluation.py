"""
Evaluation module.

Provides:
- Regression metrics (MSE, MAE, RMSE) in both scaled and original space
- Direction accuracy: for return targets, checks sign of each predicted return;
  for price targets, checks sign of price changes between horizon steps
- Per-horizon-step metrics (error at t+1, t+2, ..., t+H)
- Walk-forward validation
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

from .dataset import TimeSeriesDataset, create_sequences

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    mse: float
    mae: float
    rmse: float
    direction_accuracy: float
    per_step_mse: np.ndarray
    per_step_mae: np.ndarray
    per_step_direction_acc: np.ndarray


def predict_with_model(
    model: torch.nn.Module,
    loader: DataLoader,
) -> np.ndarray:
    """Run inference and collect all predictions."""
    model.eval()
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_direction_accuracy_returns(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Direction accuracy when target is returns.

    Each value is already a return — its sign directly indicates
    whether price went up (>0) or down (<0).
    """
    correct = (np.sign(y_true) == np.sign(y_pred))
    return float(correct.mean())


def compute_direction_accuracy_price(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Direction accuracy when target is raw price.

    Compares sign of step-to-step price changes.
    """
    true_dirs = np.diff(y_true, axis=1)
    pred_dirs = np.diff(y_pred, axis=1)
    if true_dirs.size == 0:
        return 0.0
    return float((np.sign(true_dirs) == np.sign(pred_dirs)).mean())


def compute_per_step_direction_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, target_is_return: bool
) -> np.ndarray:
    """
    Direction accuracy broken down by horizon step.

    Always returns an array of length H (horizon) for consistency with
    per_step_mse and per_step_mae. For price targets, step 0 compares
    against the implicit "no change" direction (the last known price is
    carried forward), while steps 1..H-1 compare consecutive changes.
    """
    horizon = y_true.shape[1]
    accs = np.zeros(horizon)

    if target_is_return:
        for k in range(horizon):
            accs[k] = (np.sign(y_true[:, k]) == np.sign(y_pred[:, k])).mean()
    else:
        # Step 0: direction from implicit last-known value (assume ~0 change baseline)
        # We don't have the actual previous price, so step 0 gets overall accuracy
        accs[0] = compute_direction_accuracy_price(y_true, y_pred)
        if horizon > 1:
            true_dirs = np.diff(y_true, axis=1)
            pred_dirs = np.diff(y_pred, axis=1)
            for k in range(horizon - 1):
                accs[k + 1] = (np.sign(true_dirs[:, k]) == np.sign(pred_dirs[:, k])).mean()

    return accs


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_is_return: bool = False,
) -> Metrics:
    """Compute full metric suite on (n_samples, horizon) arrays."""
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)

    if target_is_return:
        direction_acc = compute_direction_accuracy_returns(y_true, y_pred)
    else:
        direction_acc = compute_direction_accuracy_price(y_true, y_pred)

    horizon = y_true.shape[1]
    per_step_mse = np.array([
        mean_squared_error(y_true[:, k], y_pred[:, k]) for k in range(horizon)
    ])
    per_step_mae = np.array([
        mean_absolute_error(y_true[:, k], y_pred[:, k]) for k in range(horizon)
    ])
    per_step_dir_acc = compute_per_step_direction_accuracy(
        y_true, y_pred, target_is_return
    )

    return Metrics(
        mse=mse,
        mae=mae,
        rmse=rmse,
        direction_accuracy=direction_acc,
        per_step_mse=per_step_mse,
        per_step_mae=per_step_mae,
        per_step_direction_acc=per_step_dir_acc,
    )


def inverse_transform_predictions(
    y_scaled: np.ndarray,
    target_scaler,
) -> np.ndarray:
    """Convert scaled predictions back to original target space."""
    n_samples, horizon = y_scaled.shape
    result = np.zeros_like(y_scaled)
    for k in range(horizon):
        result[:, k] = target_scaler.inverse_transform(
            y_scaled[:, k].reshape(-1, 1)
        ).flatten()
    return result


def evaluate_split(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    target_scaler,
    batch_size: int = 32,
    split_name: str = "test",
    target_is_return: bool = False,
) -> tuple[Metrics, Metrics, np.ndarray, np.ndarray]:
    """
    Evaluate model on a data split.

    Returns metrics in both scaled and original space,
    plus the predictions in original space.
    """
    ds = TimeSeriesDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    y_pred_scaled = predict_with_model(model, loader)

    # Metrics in scaled space
    scaled_metrics = compute_metrics(y, y_pred_scaled, target_is_return=target_is_return)

    # Inverse transform to original space
    y_true_orig = inverse_transform_predictions(y, target_scaler)
    y_pred_orig = inverse_transform_predictions(y_pred_scaled, target_scaler)

    # Metrics in original space
    orig_metrics = compute_metrics(y_true_orig, y_pred_orig, target_is_return=target_is_return)

    unit = "Return" if target_is_return else "Price"
    logger.info(
        "[%s] Scaled — MSE=%.6f, MAE=%.6f, RMSE=%.6f, DirAcc=%.2f%%",
        split_name,
        scaled_metrics.mse, scaled_metrics.mae, scaled_metrics.rmse,
        scaled_metrics.direction_accuracy * 100,
    )
    logger.info(
        "[%s] %s  — MSE=%.6f, MAE=%.6f, RMSE=%.6f, DirAcc=%.2f%%",
        split_name, unit,
        orig_metrics.mse, orig_metrics.mae, orig_metrics.rmse,
        orig_metrics.direction_accuracy * 100,
    )

    return scaled_metrics, orig_metrics, y_true_orig, y_pred_orig


def walk_forward_evaluate(
    trained_model: torch.nn.Module,
    df_features: np.ndarray,
    target_col_idx: int,
    lookback: int,
    horizon: int,
    n_steps: int,
    scaler,
    target_scaler,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    target_is_return: bool = False,
) -> list[Metrics]:
    """
    Walk-forward validation using a pre-trained model.

    Slides an evaluation window through the test portion of the data.
    Uses the same scaler that the model was trained with (refitting the
    scaler while keeping model weights frozen would feed distribution-shifted
    inputs to the model, producing unreliable metrics).

    The window starts at train_ratio + val_ratio (i.e., the true test set
    boundary), ensuring no evaluation on data the model saw during training
    or early stopping.
    """
    n = len(df_features)
    test_start = int(n * (train_ratio + val_ratio))
    test_size = n - test_start
    step_size = test_size // n_steps

    all_metrics = []

    for i in range(n_steps):
        window_start = test_start + i * step_size
        window_end = min(window_start + step_size + lookback + horizon, n)

        if window_end - window_start < lookback + horizon:
            logger.warning("Walk-forward step %d: insufficient data, skipping", i + 1)
            continue

        test_window = df_features[window_start:window_end]
        test_scaled = scaler.transform(test_window)

        X_test, y_test = create_sequences(test_scaled, target_col_idx, lookback, horizon)

        if len(X_test) == 0:
            continue

        ds = TimeSeriesDataset(X_test, y_test)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y_pred_scaled = predict_with_model(trained_model, loader)

        y_true_orig = inverse_transform_predictions(y_test, target_scaler)
        y_pred_orig = inverse_transform_predictions(y_pred_scaled, target_scaler)

        metrics = compute_metrics(y_true_orig, y_pred_orig, target_is_return=target_is_return)
        all_metrics.append(metrics)

        logger.info(
            "Walk-forward step %d/%d — RMSE=%.6f, MAE=%.6f, DirAcc=%.2f%%",
            i + 1, n_steps, metrics.rmse, metrics.mae, metrics.direction_accuracy * 100,
        )

    return all_metrics
