"""
Preprocessing pipeline.

Handles missing values, outlier detection, and data quality checks.
All operations are stateless functions operating on DataFrames.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Run basic sanity checks on raw OHLCV data."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_before = len(df)
    # Drop rows where all OHLC values are identical AND zero (corrupt rows)
    mask = (df[["open", "high", "low", "close"]] == 0).all(axis=1)
    df = df[~mask]
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d zero-price rows", n_dropped)

    # Enforce chronological order
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        logger.info("Sorted index to chronological order")

    return df


def handle_missing_values(df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
    """
    Handle missing values in OHLCV data.

    Strategy: forward-fill then back-fill.
    Forward-fill is appropriate for financial time series because the last
    known price is the best estimate for a missing observation (the market
    was closed or data is missing, but the price didn't jump).
    Back-fill covers only the very first rows if they are NaN.
    """
    n_null_before = df.isnull().sum().sum()
    if n_null_before == 0:
        return df

    logger.info("Found %d null values across all columns", n_null_before)

    if fill_method == "ffill":
        df = df.ffill().bfill()
    elif fill_method == "interpolate":
        df = df.interpolate(method="time").bfill()
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    n_null_after = df.isnull().sum().sum()
    if n_null_after > 0:
        logger.warning("%d nulls remain after filling — dropping those rows", n_null_after)
        df = df.dropna()

    logger.info("Filled %d null values", n_null_before - n_null_after)
    return df


def remove_outliers(df: pd.DataFrame, z_thresh: float = 10.0) -> pd.DataFrame:
    """
    Flag extreme outliers in returns (not prices).

    Uses a rolling z-score with a 252-day window (approx 1 year) instead of
    full-sample statistics. This avoids look-ahead bias — each day's z-score
    is computed using only past data.

    We use a high z-threshold (10) because crypto legitimately has
    large moves. This only catches data errors, not real volatility.
    """
    returns = df["close"].pct_change()
    # Rolling stats — only uses past data (no future leakage)
    rolling_mean = returns.rolling(window=252, min_periods=30).mean()
    rolling_std = returns.rolling(window=252, min_periods=30).std()
    z_scores = (returns - rolling_mean) / rolling_std.replace(0, np.nan)

    outlier_mask = z_scores.abs() > z_thresh
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        logger.warning(
            "Flagged %d return outliers (|z| > %.1f) — replacing with NaN and forward-filling",
            n_outliers,
            z_thresh,
        )
        df.loc[outlier_mask, ["open", "high", "low", "close"]] = np.nan
        df = df.ffill()
    return df


def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    prep_cfg = cfg.get("preprocessing", {})
    fill_method = prep_cfg.get("fill_method", "ffill")

    df = validate_ohlcv(df)
    df = handle_missing_values(df, fill_method=fill_method)
    df = remove_outliers(df)

    logger.info("Preprocessing complete: %d rows, %d columns", len(df), len(df.columns))
    return df
