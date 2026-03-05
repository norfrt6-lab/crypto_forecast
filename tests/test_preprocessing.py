"""Tests for src.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    handle_missing_values,
    preprocess,
    remove_outliers,
    validate_ohlcv,
)


class TestValidateOHLCV:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"open": [1], "high": [2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv(df)

    def test_drops_zero_price_rows(self, ohlcv_df):
        df = ohlcv_df.copy()
        df.iloc[0, :4] = 0  # set OHLC to zero
        result = validate_ohlcv(df)
        assert len(result) == len(df) - 1

    def test_sorts_unsorted_index(self, ohlcv_df):
        df = ohlcv_df.iloc[::-1]  # reverse order
        result = validate_ohlcv(df)
        assert result.index.is_monotonic_increasing

    def test_valid_data_passes_through(self, ohlcv_df):
        result = validate_ohlcv(ohlcv_df)
        assert len(result) == len(ohlcv_df)


class TestHandleMissingValues:
    def test_no_nulls_returns_unchanged(self, ohlcv_df):
        result = handle_missing_values(ohlcv_df)
        pd.testing.assert_frame_equal(result, ohlcv_df)

    def test_ffill_fills_nulls(self, ohlcv_df):
        df = ohlcv_df.copy()
        df.iloc[5, 0] = np.nan
        df.iloc[10, 1] = np.nan
        result = handle_missing_values(df, fill_method="ffill")
        assert result.isnull().sum().sum() == 0

    def test_interpolate_fills_nulls(self, ohlcv_df):
        df = ohlcv_df.copy()
        df.iloc[5, 0] = np.nan
        result = handle_missing_values(df, fill_method="interpolate")
        assert result.isnull().sum().sum() == 0

    def test_unknown_method_raises(self, ohlcv_df):
        df = ohlcv_df.copy()
        df.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="Unknown fill_method"):
            handle_missing_values(df, fill_method="bad")


class TestRemoveOutliers:
    def test_no_outliers_unchanged(self, ohlcv_df):
        result = remove_outliers(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_returns_dataframe(self, ohlcv_df):
        result = remove_outliers(ohlcv_df)
        assert isinstance(result, pd.DataFrame)


class TestPreprocess:
    def test_full_pipeline(self, ohlcv_df):
        cfg = {"preprocessing": {"fill_method": "ffill"}}
        result = preprocess(ohlcv_df, cfg)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert result.isnull().sum().sum() == 0
