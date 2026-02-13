"""
Crypto Time-Series Forecasting — Main Entry Point

Orchestrates the full pipeline:
1. Load configuration
2. Fetch and cache OHLCV data
3. Preprocess
4. Engineer features
5. Build train/val/test splits with scaling
6. Train LSTM model
7. Evaluate baselines and LSTM
8. Walk-forward validation
9. Generate comparison plots and metrics report
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure the project root is on sys.path regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_ohlcv
from src.preprocessing import preprocess
from src.features import build_features
from src.dataset import build_splits, build_dataloaders
from src.models.baseline import ZeroBaselineModel, NaiveRepeatModel, LinearBaselineModel, GBTBaselineModel
from src.models.lstm import LSTMForecaster
from src.trainer import train_model
from src.evaluation import (
    evaluate_split,
    compute_metrics,
    inverse_transform_predictions,
    walk_forward_evaluate,
)
from src.visualization import (
    plot_predictions_vs_actual,
    plot_all_horizon_steps,
    plot_training_loss,
    plot_per_step_metrics,
    plot_model_comparison,
)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    log_level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("file", "outputs/run.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w"),
    ]

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _is_return_target(target: str) -> bool:
    """Check if the target column represents returns rather than prices."""
    return "return" in target.lower()


def evaluate_baselines(splits, cfg, target_is_return: bool):
    """Evaluate all baselines: zero, naive repeat, and linear."""
    ds_cfg = cfg["dataset"]
    target = ds_cfg["target"]
    horizon = ds_cfg["horizon"]
    feature_names = splits.feature_names
    target_col_idx = feature_names.index(target)

    results = {}
    logger = logging.getLogger(__name__)

    y_true_orig = inverse_transform_predictions(splits.y_test, splits.target_scaler)

    # --- Zero Baseline (random walk null hypothesis for returns) ---
    # Predict literal zero in original space (not scaled space, which would
    # inverse-transform to the training mean and distort the comparison).
    logger.info("=" * 60)
    logger.info("Evaluating: Zero Baseline")
    logger.info("=" * 60)

    zero = ZeroBaselineModel(horizon=horizon)
    zero_pred_orig = zero.predict(splits.X_test)  # zeros in original space

    zero_metrics = compute_metrics(
        y_true_orig, zero_pred_orig, target_is_return=target_is_return
    )
    results["Zero (RW)"] = (zero_metrics, y_true_orig, zero_pred_orig)

    logger.info(
        "[Zero] RMSE=%.6f, MAE=%.6f, DirAcc=%.2f%%",
        zero_metrics.rmse, zero_metrics.mae, zero_metrics.direction_accuracy * 100,
    )

    # --- Naive Repeat Baseline (momentum hypothesis) ---
    logger.info("=" * 60)
    logger.info("Evaluating: Naive Repeat Baseline")
    logger.info("=" * 60)

    naive = NaiveRepeatModel(target_col_idx=target_col_idx, horizon=horizon)
    naive_pred_scaled = naive.predict(splits.X_test)
    naive_pred_orig = inverse_transform_predictions(naive_pred_scaled, splits.target_scaler)

    naive_metrics = compute_metrics(
        y_true_orig, naive_pred_orig, target_is_return=target_is_return
    )
    results["Naive Repeat"] = (naive_metrics, y_true_orig, naive_pred_orig)

    logger.info(
        "[Naive] RMSE=%.6f, MAE=%.6f, DirAcc=%.2f%%",
        naive_metrics.rmse, naive_metrics.mae, naive_metrics.direction_accuracy * 100,
    )

    # --- Linear Regression Baseline ---
    logger.info("=" * 60)
    logger.info("Evaluating: Linear Regression Baseline")
    logger.info("=" * 60)

    bl_cfg = cfg.get("baselines", {})
    linear = LinearBaselineModel(alpha=bl_cfg.get("ridge_alpha", 1.0))
    linear.fit(splits.X_train, splits.y_train)
    linear_pred_scaled = linear.predict(splits.X_test)
    linear_pred_orig = inverse_transform_predictions(
        linear_pred_scaled, splits.target_scaler
    )

    linear_metrics = compute_metrics(
        y_true_orig, linear_pred_orig, target_is_return=target_is_return
    )
    results["Linear (Ridge)"] = (linear_metrics, y_true_orig, linear_pred_orig)

    logger.info(
        "[Linear] RMSE=%.6f, MAE=%.6f, DirAcc=%.2f%%",
        linear_metrics.rmse, linear_metrics.mae, linear_metrics.direction_accuracy * 100,
    )

    # --- Gradient Boosted Trees Baseline ---
    logger.info("=" * 60)
    logger.info("Evaluating: Gradient Boosted Trees Baseline")
    logger.info("=" * 60)

    gbt = GBTBaselineModel(
        n_estimators=bl_cfg.get("gbt_n_estimators", 200),
        max_depth=bl_cfg.get("gbt_max_depth", 4),
        learning_rate=bl_cfg.get("gbt_learning_rate", 0.05),
        subsample=bl_cfg.get("gbt_subsample", 0.8),
    )
    gbt.fit(splits.X_train, splits.y_train)
    gbt_pred_scaled = gbt.predict(splits.X_test)
    gbt_pred_orig = inverse_transform_predictions(
        gbt_pred_scaled, splits.target_scaler
    )

    gbt_metrics = compute_metrics(
        y_true_orig, gbt_pred_orig, target_is_return=target_is_return
    )
    results["GBT"] = (gbt_metrics, y_true_orig, gbt_pred_orig)

    logger.info(
        "[GBT] RMSE=%.6f, MAE=%.6f, DirAcc=%.2f%%",
        gbt_metrics.rmse, gbt_metrics.mae, gbt_metrics.direction_accuracy * 100,
    )

    return results


def main(config_path: str = "config.yaml") -> None:
    if not Path(config_path).exists():
        config_path = PROJECT_ROOT / config_path

    cfg = load_config(config_path)
    setup_logging(cfg)
    set_seed(cfg.get("seed", 42))

    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded from %s", config_path)
    logger.info("Random seed: %d", cfg.get("seed", 42))

    # ---- 1. Data Loading ----
    data_cfg = cfg["data"]
    df = load_ohlcv(
        symbol=data_cfg["symbol"],
        interval=data_cfg["interval"],
        since=data_cfg.get("since", "2020-01-01T00:00:00Z"),
        cache_path=data_cfg.get("cache_path"),
        static_csv=data_cfg.get("static_csv"),
    )

    # ---- 2. Preprocessing ----
    df = preprocess(df, cfg)

    # ---- 3. Feature Engineering ----
    df = build_features(df, cfg)

    # ---- 4. Build Splits ----
    splits = build_splits(df, cfg)

    ds_cfg = cfg["dataset"]
    batch_size = cfg["training"]["batch_size"]
    horizon = ds_cfg["horizon"]
    target = ds_cfg["target"]
    target_is_return = _is_return_target(target)
    target_label = "Daily Return" if target_is_return else "Price (USD)"
    metric_unit = "Return" if target_is_return else "USD"

    logger.info("Target: %s (is_return=%s)", target, target_is_return)

    train_loader, val_loader, test_loader = build_dataloaders(
        splits, batch_size=batch_size
    )

    # ---- 5. Evaluate Baselines ----
    baseline_results = evaluate_baselines(splits, cfg, target_is_return)

    # ---- 6. Train LSTM ----
    model_cfg = cfg["model"]
    input_size = splits.X_train.shape[2]

    logger.info("=" * 60)
    logger.info("Training: LSTM Forecaster")
    logger.info(
        "  input_size=%d, hidden=%d, layers=%d, horizon=%d",
        input_size, model_cfg["hidden_size"], model_cfg["num_layers"], horizon,
    )
    logger.info("=" * 60)

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        horizon=horizon,
        bidirectional=model_cfg.get("bidirectional", False),
    )

    model, history = train_model(model, train_loader, val_loader, cfg)

    # ---- 7. Evaluate LSTM ----
    logger.info("=" * 60)
    logger.info("Evaluating: LSTM on Test Set")
    logger.info("=" * 60)

    _, orig_metrics, y_true_orig, y_pred_orig = evaluate_split(
        model,
        splits.X_test,
        splits.y_test,
        splits.target_scaler,
        batch_size=batch_size,
        split_name="test",
        target_is_return=target_is_return,
    )

    # ---- 8. Walk-Forward Validation ----
    eval_cfg = cfg.get("evaluation", {})
    wf_steps = eval_cfg.get("walk_forward_steps", 5)

    logger.info("=" * 60)
    logger.info("Walk-Forward Validation (%d steps)", wf_steps)
    logger.info("=" * 60)

    feature_names = splits.feature_names
    target_col_idx = feature_names.index(target)

    wf_metrics = walk_forward_evaluate(
        trained_model=model,
        df_features=df.values,
        target_col_idx=target_col_idx,
        lookback=ds_cfg["lookback"],
        horizon=horizon,
        n_steps=wf_steps,
        scaler=splits.scaler,
        target_scaler=splits.target_scaler,
        train_ratio=ds_cfg.get("train_ratio", 0.7),
        val_ratio=ds_cfg.get("val_ratio", 0.15),
        batch_size=batch_size,
        target_is_return=target_is_return,
    )

    if wf_metrics:
        avg_wf_rmse = np.mean([m.rmse for m in wf_metrics])
        avg_wf_mae = np.mean([m.mae for m in wf_metrics])
        avg_wf_dir = np.mean([m.direction_accuracy for m in wf_metrics])
        logger.info(
            "Walk-forward avg — RMSE=%.6f, MAE=%.6f, DirAcc=%.2f%%",
            avg_wf_rmse, avg_wf_mae, avg_wf_dir * 100,
        )

    # ---- 9. Plots ----
    plot_dir = eval_cfg.get("plot_output_dir", "outputs/plots")

    plot_training_loss(history, plot_dir)

    plot_predictions_vs_actual(
        y_true_orig, y_pred_orig, splits.dates_test,
        horizon=horizon, output_dir=plot_dir, model_name="LSTM", step=0,
        target_label=target_label,
    )

    plot_all_horizon_steps(
        y_true_orig, y_pred_orig, splits.dates_test,
        output_dir=plot_dir, model_name="LSTM", target_label=target_label,
    )

    for name, (_, yt, yp) in baseline_results.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_predictions_vs_actual(
            yt, yp, splits.dates_test,
            horizon=horizon, output_dir=plot_dir, model_name=safe_name, step=0,
            target_label=target_label,
        )

    # Comparison plots
    all_metrics = {name: m for name, (m, _, _) in baseline_results.items()}
    all_metrics["LSTM"] = orig_metrics

    plot_per_step_metrics(all_metrics, plot_dir, metric_unit=metric_unit)
    plot_model_comparison(all_metrics, plot_dir, metric_unit=metric_unit)

    # ---- 10. Save Metrics Report ----
    metrics_dir = Path(eval_cfg.get("metrics_output_dir", "outputs/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    for name, m in all_metrics.items():
        report[name] = {
            "rmse": round(float(m.rmse), 6),
            "mae": round(float(m.mae), 6),
            "mse": round(float(m.mse), 6),
            "direction_accuracy_pct": round(m.direction_accuracy * 100, 2),
            "per_step_mae": [round(float(v), 6) for v in m.per_step_mae.tolist()],
        }

    if wf_metrics:
        report["walk_forward"] = {
            "n_steps": len(wf_metrics),
            "avg_rmse": round(float(avg_wf_rmse), 6),
            "avg_mae": round(float(avg_wf_mae), 6),
            "avg_direction_accuracy_pct": round(float(avg_wf_dir * 100), 2),
        }

    report_path = metrics_dir / "metrics_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Metrics report saved to %s", report_path)

    # ---- 11. Summary ----
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY (target=%s)", target)
    logger.info("=" * 60)
    fmt = "  %-20s RMSE=%10.6f  MAE=%10.6f  DirAcc=%5.1f%%"
    for name, m in all_metrics.items():
        logger.info(fmt, name, m.rmse, m.mae, m.direction_accuracy * 100)
    if wf_metrics:
        logger.info(fmt, "Walk-Fwd (avg)", avg_wf_rmse, avg_wf_mae, avg_wf_dir * 100)
    logger.info("=" * 60)
    logger.info("Plots saved to: %s", plot_dir)
    logger.info("Metrics saved to: %s", report_path)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Time-Series Forecasting")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
