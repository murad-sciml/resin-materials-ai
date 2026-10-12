"""
End-to-end demo of the Autonomous Experiment Module, independent of the API
layer. Simulates a closed loop: recommend -> "run experiment" (synthetic
ground-truth function) -> submit feedback -> retrain -> repeat.

Run: python example_usage.py
"""
import numpy as np

from retrain_pipeline import RetrainingPipeline

rng = np.random.default_rng(0)

DESIGN_VAR_NAMES = ["resin_voc_ratio", "sio2_na2o_ratio", "curing_time_hr"]
BOUNDS = [(0.05, 0.4), (1.0, 3.5), (1.0, 48.0)]


def ground_truth(x: np.ndarray) -> float:
    """Stand-in for a real physical experiment: a noisy function with a
    known optimum, used only to demonstrate closed-loop convergence."""
    voc, ratio, curing = x
    score = (
        -((voc - 0.2) ** 2) * 50
        - ((ratio - 2.2) ** 2) * 4
        - ((curing - 24) ** 2) * 0.01
        + 40
    )
    return float(score + rng.normal(0, 0.5))


pipeline = RetrainingPipeline(bounds=BOUNDS, minimize=False, design_var_names=DESIGN_VAR_NAMES)

print("=== Closed-loop autonomous experimentation ===")
for round_i in range(8):
    doe = pipeline.next_doe(n_bo=2, n_uncertain=1)
    for candidate in doe["doe"]:
        x = [candidate["design_variables"][name] for name in DESIGN_VAR_NAMES]
        y = ground_truth(np.array(x))
        fit_metrics = pipeline.submit_feedback(x, y)
    print(
        f"round {round_i+1:2d}: n_obs={fit_metrics['n_observations']:3d}  "
        f"R2={fit_metrics['r2']}  RMSE={fit_metrics['rmse']}  "
        f"uncertainty_reduction={doe['uncertainty_reduction_rate']}"
    )

best_idx = int(np.argmax(pipeline.optimizer.y_obs))
print("\nBest formulation found:")
for name, val in zip(DESIGN_VAR_NAMES, pipeline.optimizer.X_obs[best_idx]):
    print(f"  {name}: {val:.3f}")
print(f"  measured value: {pipeline.optimizer.y_obs[best_idx]:.3f}")
