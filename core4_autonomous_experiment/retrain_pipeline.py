"""
Closed-loop retraining pipeline for Core Function #4.

submit_feedback() : ingest a new experimental result -> retrain the surrogate
next_doe()        : rank and recommend the next-priority Design of Experiments
                     (DOE), blending Bayesian-Optimization exploitation with
                     uncertainty-sampling exploration
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from active_learning import uncertainty_reduction_rate, uncertainty_sampling
from bayesian_optimizer import BayesianOptimizer


class RetrainingPipeline:
    def __init__(self, bounds, minimize: bool = False, design_var_names: list[str] | None = None):
        self.optimizer = BayesianOptimizer(bounds=bounds, minimize=minimize)
        self.design_var_names = design_var_names or [f"x{i}" for i in range(len(bounds))]
        self.feedback_log: list[dict] = []
        self.history: list[dict] = []
        self._last_uncertainty_sample_sigma = None

    def submit_feedback(self, design_variables, measured_value) -> dict:
        """Ingest one (or more) experimental results and retrain the surrogate."""
        X = np.atleast_2d(np.asarray(design_variables, dtype=float))
        y = np.atleast_1d(np.asarray(measured_value, dtype=float))
        self.feedback_log.append({"X": X.tolist(), "y": y.tolist()})
        self.optimizer.update(X, y)
        metrics = self._evaluate_fit()
        self.history.append(metrics)
        return metrics

    def _evaluate_fit(self) -> dict:
        X_obs, y_obs = self.optimizer.X_obs, self.optimizer.y_obs
        if len(y_obs) < 3:
            return {"r2": None, "rmse": None, "n_observations": int(len(y_obs))}
        mu, _ = self.optimizer.predict(X_obs)
        r2 = float(r2_score(y_obs, mu))
        rmse = float(np.sqrt(mean_squared_error(y_obs, mu)))
        return {"r2": r2, "rmse": rmse, "n_observations": int(len(y_obs))}

    def next_doe(self, n_bo: int = 3, n_uncertain: int = 2, n_candidate_pool: int = 2000) -> list[dict]:
        """
        Next-priority DOE list: top Bayesian-Optimization candidates
        (exploitation, ranked first) plus top uncertainty-sampling candidates
        (exploration), each tagged with its recommendation source.
        """
        bo_candidates = self.optimizer.suggest(n_candidates=n_candidate_pool, n_suggest=n_bo)
        unc_candidates, sigma = uncertainty_sampling(self.optimizer, n_candidates=n_candidate_pool, n_select=n_uncertain)

        reduction = uncertainty_reduction_rate(self._last_uncertainty_sample_sigma, sigma)
        self._last_uncertainty_sample_sigma = sigma

        doe = []
        for i, x in enumerate(bo_candidates):
            doe.append(
                {
                    "rank": i + 1,
                    "design_variables": dict(zip(self.design_var_names, x.tolist())),
                    "source": "bayesian_optimization",
                }
            )
        offset = len(doe)
        for i, x in enumerate(unc_candidates):
            doe.append(
                {
                    "rank": offset + i + 1,
                    "design_variables": dict(zip(self.design_var_names, x.tolist())),
                    "source": "uncertainty_sampling",
                }
            )
        return {"doe": doe, "uncertainty_reduction_rate": reduction}

    def convergence_metrics(self) -> list[dict]:
        return self.history
