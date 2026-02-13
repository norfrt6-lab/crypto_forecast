"""
Baseline models for comparison.

Four baselines are provided:
1. ZeroBaseline: predicts zero for all horizon steps. For return targets,
   this is the true random-walk null hypothesis (expected return = 0).
2. NaiveRepeat: predicts the last known value for all future steps.
   For price targets, this is the random-walk baseline. For return targets,
   this is a momentum/autocorrelation hypothesis.
3. LinearBaseline: Ridge regression from flattened lookback window to
   the horizon. Tests whether the relationship is capturable without
   non-linear modeling.
4. GBTBaseline: Gradient-boosted trees (one per horizon step). Tests
   whether non-linear feature interactions improve over linear models,
   without the sequential modeling of an LSTM.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


class ZeroBaselineModel:
    """
    Predicts zero for all horizon steps.

    For return targets, this is the proper random-walk null hypothesis:
    the best prediction of future returns is zero (prices follow a
    martingale). Any model that cannot beat this is not learning signal.
    """

    def __init__(self, horizon: int):
        self.horizon = horizon

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X shape: (batch, lookback, features) -> (batch, horizon)"""
        return np.zeros((X.shape[0], self.horizon))


class NaiveRepeatModel:
    """
    Predicts the last observed target value for all horizon steps.

    For price targets, this is the random-walk baseline (best predictor
    of tomorrow's price is today's price). For return targets, this
    assumes autocorrelation (momentum).
    """

    def __init__(self, target_col_idx: int, horizon: int):
        self.target_col_idx = target_col_idx
        self.horizon = horizon

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X shape: (batch, lookback, features) -> (batch, horizon)"""
        last_val = X[:, -1, self.target_col_idx]  # (batch,)
        return np.repeat(last_val[:, np.newaxis], self.horizon, axis=1)


class LinearBaselineModel:
    """
    Ridge regression from flattened lookback window to horizon.

    Uses Ridge (L2) rather than OLS to reduce overfitting given the high
    dimensionality of flattened input (lookback * n_features).
    """

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (n_samples, lookback, features), y: (n_samples, horizon)"""
        n = X.shape[0]
        X_flat = X.reshape(n, -1)
        self.model.fit(X_flat, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fit before prediction")
        n = X.shape[0]
        X_flat = X.reshape(n, -1)
        return self.model.predict(X_flat)


class GBTBaselineModel:
    """
    Gradient-boosted trees for multi-step forecasting.

    Uses sklearn's GradientBoostingRegressor wrapped in MultiOutputRegressor
    (one tree ensemble per horizon step — direct multi-step strategy).

    This tests whether non-linear feature interactions capture signal that
    linear models miss, without relying on sequential modeling. Uses only
    the last timestep's features (not the full lookback window) to keep
    dimensionality manageable and training fast on CPU.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
    ):
        base = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
        )
        self.model = MultiOutputRegressor(base)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X: (n_samples, lookback, features), y: (n_samples, horizon)

        Uses only the last timestep of the lookback window as features.
        The full flattened window (lookback * features) would be too
        high-dimensional for tree models and slow to train.
        """
        X_last = X[:, -1, :]  # (n_samples, features)
        self.model.fit(X_last, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fit before prediction")
        X_last = X[:, -1, :]
        return self.model.predict(X_last)
