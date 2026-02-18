import pytest
import numpy as np


@pytest.fixture
def sample_prices():
    np.random.seed(42)
    return np.random.uniform(30000, 50000, size=100)


@pytest.fixture
def sample_features():
    np.random.seed(42)
    return np.random.randn(100, 10)
