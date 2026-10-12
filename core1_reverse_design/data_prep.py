"""
Data preparation for Core Function #1 -- turns Core Function #2's unified
materials data (+ synthesized cost from add_cost_data.py) into model-ready
feature/target arrays for the forward DNN.

Inputs (composition + process):  resin_voc_ratio, sio2_na2o_ratio, curing_time
                                  (continuous) + curing_method (one-hot) +
                                  additives (multi-hot)
Targets (properties + objectives): compressive_strength, flexural_strength,
                                  viscosity, carbon_emission, cost
"""
from __future__ import annotations

import json

import pandas as pd

INPUT_NUMERIC = ["resin_voc_ratio", "sio2_na2o_ratio", "curing_time"]
CURING_METHODS = ["ambient", "steam", "autoclave"]
TARGET_PROPERTIES = ["compressive_strength", "flexural_strength", "viscosity", "carbon_emission", "cost"]


_INVALID_ADDITIVE_TOKENS = {"nan", "none", "null", ""}


def parse_additives(cell) -> list[str]:
    if pd.isna(cell):
        return []
    if isinstance(cell, dict):
        names = list(cell.keys())
    else:
        try:
            parsed = json.loads(cell)
            names = list(parsed.keys()) if isinstance(parsed, dict) else []
        except (ValueError, TypeError):
            return []
    return [n for n in names if n.lower() not in _INVALID_ADDITIVE_TOKENS]


def load_dataset(csv_path: str, additive_vocab: list[str] | None = None):
    """
    Returns (X, Y, feature_names, target_names, additive_vocab, curing_methods).
    additive_vocab is inferred from the data unless explicitly passed (pass the
    training vocabulary when preparing new inference inputs, so column order matches).
    """
    df = pd.read_csv(csv_path)
    needed = INPUT_NUMERIC + ["curing_method", "additives"] + TARGET_PROPERTIES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns: {missing}. "
            f"Run add_cost_data.py first if 'cost' is the missing one."
        )

    before = len(df)
    # Guard against duplicated ingestion: identical samples in both train and test
    # splits would inflate the reported test scores.
    if "name" in df.columns and "source_type" in df.columns:
        df = df.drop_duplicates(subset=["name", "source_type"], keep="first")
    df = df.dropna(subset=INPUT_NUMERIC + TARGET_PROPERTIES).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"dropped {dropped}/{before} rows with missing values in required columns")

    additive_lists = df["additives"].apply(parse_additives)
    if additive_vocab is None:
        additive_vocab = sorted({a for lst in additive_lists for a in lst})

    method_onehot = pd.get_dummies(
        df["curing_method"].astype(pd.CategoricalDtype(categories=CURING_METHODS)), prefix="method"
    ).astype(float)
    additive_onehot = pd.DataFrame(
        {f"additive_{a}": additive_lists.apply(lambda lst, a=a: float(a in lst)) for a in additive_vocab}
    )

    X = pd.concat(
        [df[INPUT_NUMERIC].reset_index(drop=True), method_onehot.reset_index(drop=True), additive_onehot.reset_index(drop=True)],
        axis=1,
    )
    Y = df[TARGET_PROPERTIES].reset_index(drop=True)

    feature_names = list(X.columns)
    return X.to_numpy(dtype=float), Y.to_numpy(dtype=float), feature_names, TARGET_PROPERTIES, additive_vocab, CURING_METHODS
