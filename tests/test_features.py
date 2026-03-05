"""Tests for src.features."""

import numpy as np
import pandas as pd

from src.features import (
    add_atr,
    add_bollinger_bands,
    add_day_of_week,
    add_ema,
    add_lagged_returns,
    add_macd,
    add_returns,
    add_rsi,
    add_sma,
    add_volume_features,
    build_features,
)


class TestAddReturns:
    def test_adds_columns(self, ohlcv_df):
        result = add_returns(ohlcv_df.copy())
        assert "return_1" in result.columns
        assert "log_return_1" in result.columns

    def test_first_row_is_nan(self, ohlcv_df):
        result = add_returns(ohlcv_df.copy())
        assert np.isnan(result["return_1"].iloc[0])


class TestAddSMA:
    def test_adds_ratio_columns(self, ohlcv_df):
        result = add_sma(ohlcv_df.copy(), windows=[7, 21])
        assert "close_sma7_ratio" in result.columns
        assert "close_sma21_ratio" in result.columns


class TestAddEMA:
    def test_adds_ratio_columns(self, ohlcv_df):
        result = add_ema(ohlcv_df.copy(), windows=[12])
        assert "close_ema12_ratio" in result.columns


class TestAddRSI:
    def test_adds_rsi_column(self, ohlcv_df):
        result = add_rsi(ohlcv_df.copy(), window=14)
        assert "rsi_14" in result.columns

    def test_rsi_bounded(self, ohlcv_df):
        result = add_rsi(ohlcv_df.copy(), window=14)
        valid = result["rsi_14"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100


class TestAddBollingerBands:
    def test_adds_columns(self, ohlcv_df):
        result = add_bollinger_bands(ohlcv_df.copy())
        assert "bb_width_20" in result.columns
        assert "bb_pos_20" in result.columns


class TestAddATR:
    def test_adds_column(self, ohlcv_df):
        result = add_atr(ohlcv_df.copy(), window=14)
        assert "atr_14_norm" in result.columns


class TestAddMACD:
    def test_adds_columns(self, ohlcv_df):
        result = add_macd(ohlcv_df.copy())
        assert "macd_norm" in result.columns
        assert "macd_signal_norm" in result.columns
        assert "macd_hist_norm" in result.columns


class TestAddVolumeFeatures:
    def test_adds_columns(self, ohlcv_df):
        result = add_volume_features(ohlcv_df.copy(), windows=[7, 21])
        assert "vol_sma7_ratio" in result.columns
        assert "vol_sma21_ratio" in result.columns


class TestAddLaggedReturns:
    def test_skips_lag_1(self, ohlcv_df):
        df = add_returns(ohlcv_df.copy())
        result = add_lagged_returns(df, lags=[1, 3, 7])
        assert "return_3d" in result.columns
        assert "return_7d" in result.columns
        # lag=1 should be skipped (already return_1)
        assert "return_1d" not in result.columns


class TestAddDayOfWeek:
    def test_adds_cyclical_columns(self, ohlcv_df):
        result = add_day_of_week(ohlcv_df.copy())
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns

    def test_values_bounded(self, ohlcv_df):
        result = add_day_of_week(ohlcv_df.copy())
        assert result["dow_sin"].min() >= -1
        assert result["dow_sin"].max() <= 1


class TestBuildFeatures:
    def test_drops_raw_ohlcv(self, ohlcv_df):
        cfg = {"features": {"sma_windows": [7], "ema_windows": [12]}}
        result = build_features(ohlcv_df.copy(), cfg)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col not in result.columns

    def test_no_nans(self, ohlcv_df):
        result = build_features(ohlcv_df.copy(), {})
        assert result.isnull().sum().sum() == 0

    def test_returns_dataframe(self, ohlcv_df):
        result = build_features(ohlcv_df.copy(), {})
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
