"""
Feature engineering module.

Computes technical indicators from OHLCV data. Every feature is derived
solely from past data — no look-ahead. Each function receives a full
DataFrame and returns it with new columns appended; NaN rows introduced
by indicator warm-up periods are dropped at the end.

Feature rationale:
- Returns & log-returns: stationary representation of price changes
- SMA/EMA: trend filters at multiple horizons
- RSI: momentum oscillator (mean-reverting signal)
- Bollinger Band width: volatility regime proxy
- ATR: volatility measure normalized to price range
- MACD: trend-following momentum
- Volume ratios: liquidity/interest regime changes
- Lagged returns: autoregressive components
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["return_1"] = df["close"].pct_change()
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_sma(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        # Ratio of close to SMA — more stationary than raw SMA
        sma = df["close"].rolling(window=w).mean()
        df[f"close_sma{w}_ratio"] = df["close"] / sma
    return df


def add_ema(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        ema = df["close"].ewm(span=w, adjust=False).mean()
        df[f"close_ema{w}_ratio"] = df["close"] / ema
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return df


def add_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    sma = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    # Percentage bandwidth — scale-invariant
    df[f"bb_width_{window}"] = (upper - lower) / sma
    # Position within bands (0 = lower, 1 = upper)
    df[f"bb_pos_{window}"] = (df["close"] - lower) / (upper - lower)
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    # Normalize by close price for scale-invariance
    df[f"atr_{window}_norm"] = atr / df["close"]
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    # Normalize by close for scale-invariance
    df["macd_norm"] = macd_line / df["close"]
    df["macd_signal_norm"] = signal_line / df["close"]
    df["macd_hist_norm"] = (macd_line - signal_line) / df["close"]
    return df


def add_volume_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        vol_sma = df["volume"].rolling(window=w).mean()
        df[f"vol_sma{w}_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)
    return df


def add_lagged_returns(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        # Skip lag=1 — already computed as return_1 in add_returns()
        if lag == 1:
            continue
        df[f"return_{lag}d"] = df["close"].pct_change(periods=lag)
    return df


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encoding of day-of-week (preserves continuity: Sun ≈ Mon)."""
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Returns a DataFrame with engineered features only. Raw OHLCV columns
    are dropped because:
    1. They are non-stationary (prices trend over time).
    2. The close price is trivially related to the return_1 target
       (return_1[t] = close[t]/close[t-1] - 1), which would let the
       model learn an identity shortcut rather than temporal patterns.
    3. All price information is already captured in stationary form by
       the engineered features (ratios, returns, normalized indicators).
    """
    feat_cfg = cfg.get("features", {})

    df = add_returns(df)
    df = add_sma(df, feat_cfg.get("sma_windows", [7, 21, 50]))
    df = add_ema(df, feat_cfg.get("ema_windows", [12, 26]))
    df = add_rsi(df, feat_cfg.get("rsi_window", 14))
    df = add_bollinger_bands(
        df,
        window=feat_cfg.get("bb_window", 20),
        num_std=feat_cfg.get("bb_std", 2.0),
    )
    df = add_atr(df, feat_cfg.get("atr_window", 14))
    df = add_macd(
        df,
        fast=feat_cfg.get("macd_fast", 12),
        slow=feat_cfg.get("macd_slow", 26),
        signal=feat_cfg.get("macd_signal", 9),
    )
    df = add_volume_features(df, feat_cfg.get("volume_sma_windows", [7, 21]))
    df = add_lagged_returns(df, feat_cfg.get("return_lags", [1, 3, 7, 14]))
    df = add_day_of_week(df)

    # Drop raw OHLCV — all information is captured by engineered features
    raw_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df.drop(columns=raw_cols)
    logger.info("Dropped raw OHLCV columns: %s", raw_cols)

    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    logger.info(
        "Feature engineering: %d features, dropped %d warm-up rows, %d rows remaining",
        len(df.columns),
        n_dropped,
        len(df),
    )
    return df
