"""
Synthesizes a plausible unit-cost value for every material in Core Function
#2's materials_wide.csv, since cost data isn't tracked anywhere in the Data
Integration module today. Cost is built from a simple, transparent formula
over composition, process, and additives -- swap this out for real supplier
cost data or a plant cost dashboard once available; nothing downstream cares
where the 'cost' column came from.

Run: python add_cost_data.py --in ../core2_data_integration/sample_data/materials_wide.csv \
                              --out sample_data/materials_wide_with_cost.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CURING_METHOD_COST = {"ambient": 0.0, "steam": 15.0, "autoclave": 35.0}
ADDITIVE_UNIT_COST = {
    "fly_ash": 2.0,
    "silica_fume": 6.0,
    "slag": 3.0,
    "fiber_reinforcement": 20.0,
    "plasticizer": 8.0,
}


def parse_additives(cell) -> list[str]:
    """additives is stored as a stringified JSON dict, e.g. '{"fly_ash": true}'."""
    if pd.isna(cell):
        return []
    if isinstance(cell, dict):
        return list(cell.keys())
    try:
        parsed = json.loads(cell)
        return list(parsed.keys()) if isinstance(parsed, dict) else []
    except (ValueError, TypeError):
        return []


def synthesize_cost(df: pd.DataFrame, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    additive_lists = df["additives"].apply(parse_additives)
    additive_cost = additive_lists.apply(lambda names: sum(ADDITIVE_UNIT_COST.get(n, 5.0) for n in names))
    method_cost = df["curing_method"].map(CURING_METHOD_COST).fillna(10.0)

    base = 50.0
    cost = (
        base
        + 120.0 * df["resin_voc_ratio"].fillna(df["resin_voc_ratio"].median())
        + 8.0 * df["sio2_na2o_ratio"].fillna(df["sio2_na2o_ratio"].median())
        + method_cost
        + additive_cost
        + 0.4 * df["curing_time"].fillna(df["curing_time"].median())
        + rng.normal(0, 4, len(df))
    )
    return cost.round(2).clip(lower=10.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default="../core2_data_integration/sample_data/materials_wide.csv")
    parser.add_argument("--out", dest="out_path", default="sample_data/materials_wide_with_cost.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    df["cost"] = synthesize_cost(df)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows) with synthesized 'cost' column")
    print(df[["name", "source_type", "resin_voc_ratio", "sio2_na2o_ratio", "curing_method", "cost"]].head())


if __name__ == "__main__":
    main()
