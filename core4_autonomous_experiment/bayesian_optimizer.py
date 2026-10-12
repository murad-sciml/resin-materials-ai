"""
Bayesian Optimization engine for Core Function #4 — Software-Based Autonomous
Experiment Module.

Uses a Gaussian Process surrogate (scikit-learn) + Expected Improvement (EI)
acquisition to recommend the next formulation/process condition to test.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class BayesianOptimizer:
    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        minimize: bool = False,
        xi: float = 0.01,
        random_state: int = 42,
    ):
        """
        bounds: one (low, high) tuple per design variable, e.g.
                [(0.05, 0.4), (1.0, 3.5), (0, 48)] for
                (resin_voc_ratio, sio2_na2o_ratio, curing_time_hr).
        minimize: True to minimize the target property, False to maximize it
                  (e.g. maximize R^2 / strength, minimize cost or carbon emission).
        xi: exploration-exploitation trade-off for Expected Improvement
            (higher = more exploration).
        """
        self.bounds = np.array(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.minimize = minimize
        self.xi = xi
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=random_state
        )
        self.X_obs = np.empty((0, self.dim))
        self.y_obs = np.empty((0,))
        self.rng = np.random.default_rng(random_state)

    def update(self, X_new, y_new) -> None:
        """Fold new experimental feedback into the surrogate model."""
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        y_new = np.atleast_1d(np.asarray(y_new, dtype=float))
        self.X_obs = np.vstack([self.X_obs, X_new])
        self.y_obs = np.concatenate([self.y_obs, y_new])
        self.gp.fit(self.X_obs, self.y_obs)

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        return self.gp.predict(np.atleast_2d(X), return_std=True)

    def _sample_uniform(self, n: int) -> np.ndarray:
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return self.rng.uniform(low, high, size=(n, self.dim))

    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        mu, sigma = self.predict(X)
        sigma = np.maximum(sigma, 1e-9)
        if self.minimize:
            best = self.y_obs.min()
            imp = best - mu - self.xi
        else:
            best = self.y_obs.max()
            imp = mu - best - self.xi
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
        return ei

    def suggest(self, n_candidates: int = 2000, n_suggest: int = 1) -> np.ndarray:
        """Return the top `n_suggest` candidate design points by Expected Improvement.
        Falls back to random sampling before any feedback has been observed
        (cold-start)."""
        candidates = self._sample_uniform(n_candidates)
        if len(self.y_obs) == 0:
            idx = self.rng.choice(len(candidates), size=n_suggest, replace=False)
            return candidates[idx]
        ei = self._expected_improvement(candidates)
        top_idx = np.argsort(-ei)[:n_suggest]
        return candidates[top_idx]
