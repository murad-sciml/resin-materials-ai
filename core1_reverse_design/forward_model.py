"""
Forward DNN property-prediction model for Core Function #1.

Uses scikit-learn's MLPRegressor with solver='lbfgs' -- matching the
"Gradient-based backpropagation optimization (L-BFGS-B)" method specified in
the core-function table -- trained as a single multi-output regressor:
composition + process -> five major properties + cost.

With only ~150-200 training rows and a dozen input features, an
under-regularized network overfits badly (near-zero training loss, poor test
R^2). fit() runs a small cross-validated grid search over the L2 penalty
(`alpha`) rather than hardcoding a value, so this stays well-behaved as the
dataset grows.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# "Basic / Large / Wide DNN" per the core-function table
ARCHITECTURES = {
    "basic": (32,),
    "large": (64, 64),
    "wide": (128,),
}

DEFAULT_ALPHA_GRID = (0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 8.0, 12.0)


class ForwardDNN:
    """DNN wrapper handling input/output scaling, regularization selection,
    and multi-output regression."""

    def __init__(self, architecture: str = "large", max_iter: int = 5000, random_state: int = 42):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {list(ARCHITECTURES)}")
        self.architecture = architecture
        self.max_iter = max_iter
        self.random_state = random_state
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model: MLPRegressor | None = None
        self.best_alpha: float | None = None
        self._fitted = False

    def _make_model(self, alpha: float) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=ARCHITECTURES[self.architecture],
            activation="relu",
            solver="lbfgs",  # matches the table's specified L-BFGS-B optimization method
            alpha=alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def _select_alpha(self, X: np.ndarray, Y: np.ndarray, alpha_grid, n_splits: int = 5) -> float:
        n_splits = min(n_splits, max(2, len(X) // 10))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        best_alpha, best_score = alpha_grid[0], -np.inf
        for alpha in alpha_grid:
            scores = []
            for train_idx, val_idx in kf.split(X):
                xs, ys = StandardScaler(), StandardScaler()
                X_tr = xs.fit_transform(X[train_idx])
                Y_tr = ys.fit_transform(Y[train_idx])
                X_val = xs.transform(X[val_idx])
                model = self._make_model(alpha)
                model.fit(X_tr, Y_tr)
                pred = ys.inverse_transform(np.atleast_2d(model.predict(X_val)))
                scores.append(r2_score(Y[val_idx], pred))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score, best_alpha = mean_score, alpha
        return best_alpha

    def fit(self, X: np.ndarray, Y: np.ndarray, test_size: float = 0.2, random_state: int = 42,
            alpha_grid=DEFAULT_ALPHA_GRID) -> dict:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        self.best_alpha = self._select_alpha(X_train, Y_train, alpha_grid)

        Xs_train = self.x_scaler.fit_transform(X_train)
        Ys_train = self.y_scaler.fit_transform(Y_train)
        self.model = self._make_model(self.best_alpha)
        self.model.fit(Xs_train, Ys_train)
        self._fitted = True

        Y_pred = self.predict(X_test)
        metrics = {
            "r2": float(r2_score(Y_test, Y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(Y_test, Y_pred))),
            "mae": float(mean_absolute_error(Y_test, Y_pred)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "selected_alpha": self.best_alpha,
        }
        # per-target breakdown, since "R^2 >= 0.88" is typically judged per predicted property
        metrics["per_target_r2"] = {i: float(r2_score(Y_test[:, i], Y_pred[:, i])) for i in range(Y_test.shape[1])}
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        Xs = self.x_scaler.transform(np.atleast_2d(X))
        Ys_pred = np.atleast_2d(self.model.predict(Xs))
        # All targets (strength, viscosity, carbon emission, cost) are physically
        # non-negative, so clip predictions at zero.
        return np.maximum(self.y_scaler.inverse_transform(Ys_pred), 0.0)

