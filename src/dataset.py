"""
Dataset and sequence generation module.

Converts a feature DataFrame into sliding-window sequences suitable for
time-series models. Handles scaling, train/val/test splitting with strict
temporal ordering, and PyTorch Dataset creation.

Data leakage prevention:
1. Splits are purely chronological — no shuffling.
2. Scaler is fit ONLY on training data, then applied to val/test.
3. Target values are extracted BEFORE scaling to allow easy inverse transform.
4. No future information leaks into feature computation (handled in features.py).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class SplitArrays:
    """Container for train/val/test numpy arrays (pre-tensor)."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler | MinMaxScaler
    target_scaler: StandardScaler | MinMaxScaler
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset wrapping numpy sequence arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_sequences(
    data: np.ndarray,
    target_col_idx: int,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding-window sequences.

    For each valid index i, the input is data[i : i+lookback] (all features),
    and the target is the target column at data[i+lookback : i+lookback+horizon].

    This is a DIRECT multi-step forecasting approach: the model predicts all
    H future steps at once, rather than recursively feeding predictions back.
    Direct forecasting avoids error accumulation that plagues recursive methods.
    """
    X, y = [], []
    n = len(data)
    for i in range(n - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon, target_col_idx])
    return np.array(X), np.array(y)


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically.

    No shuffling — the first train_ratio fraction is training,
    the next val_ratio fraction is validation, the rest is test.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split) == 0:
            raise ValueError(
                f"Empty {name} split. Dataset has {n} rows, which is too small "
                f"for train_ratio={train_ratio}, val_ratio={val_ratio}."
            )

    logger.info(
        "Temporal split: train=%d (%s to %s), val=%d (%s to %s), test=%d (%s to %s)",
        len(train_df), train_df.index[0].date(), train_df.index[-1].date(),
        len(val_df), val_df.index[0].date(), val_df.index[-1].date(),
        len(test_df), test_df.index[0].date(), test_df.index[-1].date(),
    )
    return train_df, val_df, test_df


def build_splits(df: pd.DataFrame, cfg: dict) -> SplitArrays:
    """
    Full pipeline: split -> scale -> create sequences.

    Scaling is fit on training set only, then applied to validation and test.
    A separate scaler is maintained for the target column to allow
    inverse-transforming predictions back to price space.
    """
    ds_cfg = cfg.get("dataset", {})
    lookback = ds_cfg.get("lookback", 60)
    horizon = ds_cfg.get("horizon", 5)
    target = ds_cfg.get("target", "close")
    train_ratio = ds_cfg.get("train_ratio", 0.7)
    val_ratio = ds_cfg.get("val_ratio", 0.15)
    scaling = ds_cfg.get("scaling", "standard")

    feature_names = list(df.columns)
    if target not in feature_names:
        raise ValueError(
            f"Target column '{target}' not found in features. "
            f"Available columns: {feature_names}"
        )
    target_col_idx = feature_names.index(target)

    # 1. Temporal split on the full DataFrame (before sequencing)
    train_df, val_df, test_df = temporal_split(df, train_ratio, val_ratio)

    # 2. Fit scalers on training data only
    if scaling == "standard":
        scaler = StandardScaler()
        target_scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")

    scaler.fit(train_df.values)
    target_scaler.fit(train_df[[target]].values)

    train_scaled = scaler.transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)

    # 3. Create sequences from each split independently
    #    This means no sequence spans a split boundary — preventing leakage.
    X_train, y_train = create_sequences(train_scaled, target_col_idx, lookback, horizon)
    X_val, y_val = create_sequences(val_scaled, target_col_idx, lookback, horizon)
    X_test, y_test = create_sequences(test_scaled, target_col_idx, lookback, horizon)

    # Dates corresponding to prediction targets (the last date in each sequence's horizon)
    def _seq_dates(split_df: pd.DataFrame) -> pd.DatetimeIndex:
        n_seq = len(split_df) - lookback - horizon + 1
        # Each sequence's "target date" is the last predicted candle
        return split_df.index[lookback + horizon - 1 : lookback + horizon - 1 + n_seq]

    logger.info(
        "Sequences — train: %d, val: %d, test: %d  |  shape: X=%s, y=%s",
        len(X_train), len(X_val), len(X_test),
        X_train.shape, y_train.shape,
    )

    return SplitArrays(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
        target_scaler=target_scaler,
        dates_train=_seq_dates(train_df),
        dates_val=_seq_dates(val_df),
        dates_test=_seq_dates(test_df),
    )


def build_dataloaders(
    splits: SplitArrays,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from split arrays."""
    train_ds = TimeSeriesDataset(splits.X_train, splits.y_train)
    val_ds = TimeSeriesDataset(splits.X_val, splits.y_val)
    test_ds = TimeSeriesDataset(splits.X_test, splits.y_test)

    # Training loader shuffles — this is safe because sequences are already
    # independent samples (no overlap across split boundaries).
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
