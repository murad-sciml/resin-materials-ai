"""
Uncertainty-sampling active learning for Core Function #4.

Selects the design points where the GP surrogate is *least* confident, so the
next physical experiment maximally informs the model (as opposed to Bayesian
Optimization's exploitation of the current best-predicted region).
"""
from __future__ import annotations

import numpy as np


def uncertainty_sampling(optimizer, n_candidates: int = 2000, n_select: int = 5):
    """
    Returns (selected_points, their_predicted_std). Before any feedback has
    been observed, selects uniformly at random since the GP has no signal yet.
    """
    candidate_pool = optimizer._sample_uniform(n_candidates)
    if len(optimizer.y_obs) == 0:
        idx = optimizer.rng.choice(len(candidate_pool), size=min(n_select, len(candidate_pool)), replace=False)
        return candidate_pool[idx], None

    _, sigma = optimizer.predict(candidate_pool)
    top_idx = np.argsort(-sigma)[:n_select]
    return candidate_pool[top_idx], sigma[top_idx]


def uncertainty_reduction_rate(prev_sigma, curr_sigma) -> float | None:
    """
    Average uncertainty reduction after a retraining round — one of the
    Performance Metrics for this core function.
    """
    if prev_sigma is None or curr_sigma is None or len(prev_sigma) == 0:
        return None
    prev_mean = float(np.mean(prev_sigma))
    curr_mean = float(np.mean(curr_sigma))
    if prev_mean == 0:
        return None
    return (prev_mean - curr_mean) / prev_mean
