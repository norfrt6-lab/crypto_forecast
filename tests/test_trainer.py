"""Tests for src.trainer."""

import numpy as np

from src.trainer import EarlyStopping


class TestEarlyStopping:
    def test_initial_state(self):
        es = EarlyStopping(patience=5)
        assert es.best_loss == np.inf
        assert es.counter == 0
        assert es.should_stop is False

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(0.9)
        assert es.counter == 0
        assert es.best_loss == 0.9

    def test_no_improvement_increments(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(1.1)
        assert es.counter == 1
        es.step(1.2)
        assert es.counter == 2

    def test_triggers_after_patience(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)
        assert not es.step(1.1)
        assert not es.step(1.2)
        assert es.step(1.3)  # patience=3 reached
        assert es.should_stop is True

    def test_min_delta(self):
        es = EarlyStopping(patience=3, min_delta=0.1)
        es.step(1.0)
        # 0.95 is not an improvement of 0.1 below 1.0
        result = es.step(0.95)
        assert es.counter == 1
        assert not result
