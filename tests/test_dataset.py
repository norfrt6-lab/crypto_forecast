"""Tests for src.dataset."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import (
    TimeSeriesDataset,
    build_splits,
    create_sequences,
    temporal_split,
)


class TestCreateSequences:
    def test_output_shapes(self):
        data = np.random.randn(100, 5)
        X, y = create_sequences(data, target_col_idx=0, lookback=10, horizon=3)
        assert X.shape == (88, 10, 5)  # 100 - 10 - 3 + 1 = 88
        assert y.shape == (88, 3)

    def test_correct_values(self):
        data = np.arange(20).reshape(20, 1).astype(float)
        X, y = create_sequences(data, target_col_idx=0, lookback=3, horizon=2)
        # First sequence: X=[0,1,2], y=[3,4]
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
        np.testing.assert_array_equal(y[0], [3, 4])

    def test_empty_with_small_data(self):
        data = np.random.randn(5, 3)
        X, y = create_sequences(data, target_col_idx=0, lookback=10, horizon=3)
        assert len(X) == 0


class TestTemporalSplit:
    def test_split_sizes(self, features_df):
        train, val, test = temporal_split(features_df, 0.7, 0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(features_df)

    def test_chronological_order(self, features_df):
        train, val, test = temporal_split(features_df, 0.7, 0.15)
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_empty_split_raises(self):
        df = pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
        with pytest.raises(ValueError, match="Empty"):
            temporal_split(df, 0.99, 0.009)


class TestTimeSeriesDataset:
    def test_len(self):
        X = np.random.randn(50, 10, 5)
        y = np.random.randn(50, 3)
        ds = TimeSeriesDataset(X, y)
        assert len(ds) == 50

    def test_getitem_types(self):
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10, 2)
        ds = TimeSeriesDataset(X, y)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert x_item.dtype == torch.float32

    def test_getitem_shapes(self):
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10, 2)
        ds = TimeSeriesDataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.shape == (5, 3)
        assert y_item.shape == (2,)


class TestBuildSplits:
    def test_full_pipeline(self, features_df):
        cfg = {
            "dataset": {
                "lookback": 10,
                "horizon": 3,
                "target": "return_1",
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "scaling": "standard",
            }
        }
        splits = build_splits(features_df, cfg)
        assert splits.X_train.shape[1] == 10  # lookback
        assert splits.y_train.shape[1] == 3  # horizon
        assert splits.X_train.shape[2] == len(features_df.columns)

    def test_missing_target_raises(self, features_df):
        cfg = {"dataset": {"target": "nonexistent"}}
        with pytest.raises(ValueError, match="Target column"):
            build_splits(features_df, cfg)

    def test_unknown_scaling_raises(self, features_df):
        cfg = {"dataset": {"target": "return_1", "scaling": "bad"}}
        with pytest.raises(ValueError, match="Unknown scaling"):
            build_splits(features_df, cfg)
