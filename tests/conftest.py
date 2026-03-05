"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_df():
    """Minimal OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = np.cumsum(np.random.randn(n) * 0.5) + 100
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def features_df():
    """DataFrame with enough rows/features for dataset tests."""
    np.random.seed(42)
    n = 200
    n_features = 10
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    data = np.random.randn(n, n_features)
    cols = [f"feat_{i}" for i in range(n_features)]
    cols[0] = "return_1"
    return pd.DataFrame(data, columns=cols, index=dates)
