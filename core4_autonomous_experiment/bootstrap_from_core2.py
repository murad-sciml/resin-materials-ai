"""
Bootstraps the Core Function #4 Bayesian Optimizer / Retraining Pipeline using
REAL formulations + measured properties from Core Function #2's unified
materials_wide.csv -- instead of the synthetic ground_truth() function used in
example_usage.py.

Usage:
    python bootstrap_from_core2.py --csv ../core2_data_integration/sample_data/materials_wide.csv
    python bootstrap_from_core2.py --csv ../core2_data_integration/sample_data/materials_wide.csv \
        --target carbon_emission --minimize
"""
from __future__ import annotations

import argparse

import pandas as pd

from retrain_pipeline import RetrainingPipeline

# These must match column names in materials_wide.csv (see core2_data_integration/schema.py)
DESIGN_VARS = ["resin_voc_ratio", "sio2_na2o_ratio", "curing_time"]

# Physically sensible default bounds; widened automatically if real data falls outside them.
DEFAULT_BOUNDS = {
    "resin_voc_ratio": (0.05, 0.4),
    "sio2_na2o_ratio": (1.0, 3.5),
    "curing_time": (1.0, 48.0),
}

TARGET_CHOICES = ["compressive_strength", "flexural_strength", "viscosity", "curing_time", "carbon_emission"]


def infer_bounds(df: pd.DataFrame, pad_frac: float = 0.05) -> list[tuple[float, float]]:
    """Use the wider of (default bounds, observed data range) so real data
    outside the defaults doesn't get silently clipped. Lower bounds are
    floored at 0 since all three design variables are physical quantities
    that can't go negative."""
    bounds = []
    for col in DESIGN_VARS:
        default_lo, default_hi = DEFAULT_BOUNDS[col]
        lo = min(default_lo, df[col].min())
        hi = max(default_hi, df[col].max())
        pad = (hi - lo) * pad_frac
        bounds.append((max(0.0, lo - pad), hi + pad))
    return bounds


def load_real_observations(csv_path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = DESIGN_VARS + [target]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{csv_path} is missing expected columns: {missing_cols}. "
            f"Run core2_data_integration/export_to_csv.py to regenerate materials_wide.csv."
        )
    before = len(df)
    df = df.dropna(subset=needed)
    dropped = before - len(df)
    if dropped:
        print(f"dropped {dropped}/{before} rows with missing values in {needed}")
    if len(df) < 3:
        raise ValueError(f"Only {len(df)} complete rows available; need at least 3 to fit a surrogate model.")
    return df[needed].reset_index(drop=True)


def bootstrap_pipeline(df: pd.DataFrame, target: str, minimize: bool) -> RetrainingPipeline:
    bounds = infer_bounds(df)
    pipeline = RetrainingPipeline(bounds=bounds, minimize=minimize, design_var_names=DESIGN_VARS)

    X = df[DESIGN_VARS].to_numpy()
    y = df[target].to_numpy()
    # submit_feedback accepts batched X/y (np.atleast_2d/atleast_1d internally), so this
    # is a SINGLE Gaussian Process fit on the whole dataset -- not a loop that would
    # refit from scratch after every individual row (O(n) refits instead of O(1), which
    # gets extremely slow as the real dataset grows past a couple hundred rows).
    metrics = pipeline.submit_feedback(X, y)

    print(f"Bootstrapped optimizer with {len(df)} real observations from Core Function #2")
    print(f"Surrogate fit on real data: R2={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="../core2_data_integration/sample_data/materials_wide.csv",
                         help="Path to Core Function #2's materials_wide.csv")
    parser.add_argument("--target", default="compressive_strength", choices=TARGET_CHOICES,
                         help="Which measured property to optimize")
    parser.add_argument("--minimize", action="store_true",
                         help="Minimize the target instead of maximizing it (e.g. --target carbon_emission --minimize)")
    parser.add_argument("--n-bo", type=int, default=3, help="Number of exploitation (Bayesian Opt.) candidates")
    parser.add_argument("--n-uncertain", type=int, default=2, help="Number of exploration (uncertainty sampling) candidates")
    args = parser.parse_args()

    df = load_real_observations(args.csv, args.target)
    pipeline = bootstrap_pipeline(df, args.target, minimize=args.minimize)

    direction = "minimize" if args.minimize else "maximize"
    print(f"\nNext-priority DOE ({direction} {args.target}):")
    doe = pipeline.next_doe(n_bo=args.n_bo, n_uncertain=args.n_uncertain)
    for candidate in doe["doe"]:
        vars_str = ", ".join(f"{k}={v:.3f}" for k, v in candidate["design_variables"].items())
        print(f"  rank {candidate['rank']}: {vars_str}   [{candidate['source']}]")

    print("\nSurrogate fit quality over the bootstrapped history (last 5 points added):")
    for m in pipeline.convergence_metrics()[-5:]:
        print(f"  n_obs={m['n_observations']:>3}  R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}")

    return pipeline


if __name__ == "__main__":
    main()
