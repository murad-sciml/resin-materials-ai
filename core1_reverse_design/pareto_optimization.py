"""
Multi-objective Pareto optimization for Core Function #1: simultaneously
considers performance (maximize compressive_strength), cost (minimize), and
carbon emissions (minimize).

Approach: randomly sample the design space, predict all three objectives with
the trained forward model, then filter down to the non-dominated (Pareto-
optimal) set -- designs where no other sampled design is at least as good on
every objective and strictly better on at least one.
"""
from __future__ import annotations

import itertools

import numpy as np

from data_prep import CURING_METHODS


def _sample_designs(n_samples, continuous_bounds, additive_vocab, max_additives, rng):
    n_methods = len(CURING_METHODS)
    n_additives = len(additive_vocab)
    low = np.array([b[0] for b in continuous_bounds])
    high = np.array([b[1] for b in continuous_bounds])

    additive_combos = [()]
    for r in range(1, max_additives + 1):
        additive_combos.extend(itertools.combinations(range(n_additives), r))

    designs = []
    for _ in range(n_samples):
        continuous_vals = rng.uniform(low, high)
        method_idx = int(rng.integers(0, n_methods))
        combo = additive_combos[int(rng.integers(0, len(additive_combos)))]
        method_onehot = np.zeros(n_methods)
        method_onehot[method_idx] = 1.0
        additive_flags = np.zeros(n_additives)
        for idx in combo:
            additive_flags[idx] = 1.0
        x = np.concatenate([continuous_vals, method_onehot, additive_flags])
        designs.append((x, CURING_METHODS[method_idx], [additive_vocab[i] for i in combo], continuous_vals))
    return designs


def _pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Vectorized non-domination filter. objectives: (n, k), all to be MINIMIZED
    (negate any objective that should be maximized before calling)."""
    n = objectives.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        le_all = np.all(objectives <= objectives[i], axis=1)
        lt_any = np.any(objectives < objectives[i], axis=1)
        dominates_i = le_all & lt_any
        dominates_i[i] = False
        if dominates_i.any():
            is_dominated[i] = True
    return ~is_dominated


def pareto_front(
    model,
    target_property_names: list[str],
    continuous_bounds: list[tuple[float, float]],
    additive_vocab: list[str],
    n_samples: int = 1500,
    max_additives: int = 2,
    seed: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    designs = _sample_designs(n_samples, continuous_bounds, additive_vocab, max_additives, rng)

    idx_strength = target_property_names.index("compressive_strength")
    idx_cost = target_property_names.index("cost")
    idx_carbon = target_property_names.index("carbon_emission")

    preds = np.vstack([model.predict(x)[0] for x, _, _, _ in designs])
    objectives = np.column_stack([-preds[:, idx_strength], preds[:, idx_cost], preds[:, idx_carbon]])

    mask = _pareto_mask(objectives)

    records = []
    for keep, (x, method, additives, continuous_vals), pred in zip(mask, designs, preds):
        if not keep:
            continue
        records.append(
            {
                "compressive_strength": float(pred[idx_strength]),
                "cost": float(pred[idx_cost]),
                "carbon_emission": float(pred[idx_carbon]),
                "curing_method": method,
                "additives": additives,
                "resin_voc_ratio": float(continuous_vals[0]),
                "sio2_na2o_ratio": float(continuous_vals[1]),
                "curing_time": float(continuous_vals[2]),
            }
        )
    records.sort(key=lambda r: -r["compressive_strength"])
    return records
