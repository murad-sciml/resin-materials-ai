"""
Reverse design for Core Function #1: given a target property value, search
for the input recipe (composition + process) that gets closest to it, using
gradient-based (L-BFGS-B) optimization on top of the trained forward DNN.

curing_method and additives are discrete choices, so they're enumerated; for
each discrete combination, scipy.optimize.minimize (L-BFGS-B) optimizes the
continuous inputs (resin_voc_ratio, sio2_na2o_ratio, curing_time). The best
result across all combinations is returned. This whole search is designed to
finish well under the table's "response time <= 1 s" target for a small
number of continuous variables and a handful of discrete combinations.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.optimize import minimize

from data_prep import CURING_METHODS


def _build_input_vector(continuous_vals, method_idx, additive_flags, n_methods):
    method_onehot = np.zeros(n_methods)
    method_onehot[method_idx] = 1.0
    return np.concatenate([continuous_vals, method_onehot, additive_flags])


def reverse_design(
    model,
    target_properties: dict[str, float],
    target_property_names: list[str],
    continuous_bounds: list[tuple[float, float]],
    additive_vocab: list[str],
    max_additives: int = 2,
    x0=None,
) -> dict:
    """
    target_properties: e.g. {"compressive_strength": 45.0} -- any subset of
    target_property_names. Returns the best-found recipe, its predicted
    properties, the optimization loss, and the response time in seconds.
    """
    start = time.perf_counter()
    n_methods = len(CURING_METHODS)
    n_additives = len(additive_vocab)
    target_idx = [target_property_names.index(k) for k in target_properties]
    target_vals = np.array([target_properties[k] for k in target_properties])

    def objective(continuous_vals, method_idx, additive_flags):
        x = _build_input_vector(continuous_vals, method_idx, additive_flags, n_methods)
        pred = model.predict(x)[0]
        pred_targets = pred[target_idx]
        return float(np.mean((pred_targets - target_vals) ** 2))

    additive_combos = [()]
    for r in range(1, max_additives + 1):
        additive_combos.extend(itertools.combinations(range(n_additives), r))

    best = None
    x0_cont = np.array(x0) if x0 is not None else np.mean(np.array(continuous_bounds), axis=1)

    for method_idx in range(n_methods):
        for combo in additive_combos:
            additive_flags = np.zeros(n_additives)
            for idx in combo:
                additive_flags[idx] = 1.0

            res = minimize(
                objective,
                x0=x0_cont,
                args=(method_idx, additive_flags),
                method="L-BFGS-B",
                bounds=continuous_bounds,
            )
            if best is None or res.fun < best["loss"]:
                x_full = _build_input_vector(res.x, method_idx, additive_flags, n_methods)
                pred = model.predict(x_full)[0]
                best = {
                    "loss": float(res.fun),
                    "continuous_values": {
                        "resin_voc_ratio": float(res.x[0]),
                        "sio2_na2o_ratio": float(res.x[1]),
                        "curing_time": float(res.x[2]),
                    },
                    "curing_method": CURING_METHODS[method_idx],
                    "additives": [additive_vocab[i] for i in combo],
                    "predicted_properties": {name: float(val) for name, val in zip(target_property_names, pred)},
                }

    best["response_time_s"] = time.perf_counter() - start
    return best
