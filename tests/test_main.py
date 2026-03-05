"""Tests for main module utility functions."""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _is_return_target, set_seed  # noqa: E402


class TestIsReturnTarget:
    def test_return_target(self):
        assert _is_return_target("return_1") is True
        assert _is_return_target("log_return_1") is True

    def test_price_target(self):
        assert _is_return_target("close") is False
        assert _is_return_target("price") is False

    def test_case_insensitive(self):
        assert _is_return_target("Return_5") is True


class TestSetSeed:
    def test_deterministic_numpy(self):
        import numpy as np

        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_torch(self):
        import torch

        set_seed(42)
        a = torch.rand(5)
        set_seed(42)
        b = torch.rand(5)
        assert torch.equal(a, b)
