"""
ETL pipeline for Core Function #2 — Data Integration & Standardization.

Pipeline stages:
    load_raw            -> read csv / json / xlsx into a DataFrame
    harmonize_units      -> convert heterogeneous units to canonical units
    handle_missing_and_outliers -> impute missing values, cap/remove outliers
    normalize            -> z-score or min-max scale numeric columns
    unify_to_schema       -> reshape into (materials_df, properties_df) matching schema.py
    compute_quality_metrics -> data-quality KPIs used by the Performance Metrics column
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from schema import FIVE_MAJOR_PROPERTIES

# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_raw(file_path: str | Path) -> pd.DataFrame:
    """Load experimental / literature / process-log data regardless of format."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".json":
        with open(file_path) as f:
            data = json.load(f)
        return pd.json_normalize(data)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file format: {suffix}")


# ---------------------------------------------------------------------------
# 2. Unit harmonization
# ---------------------------------------------------------------------------

# Example conversion factors to canonical units. Extend as new sources arrive.
UNIT_CONVERSIONS = {
    ("curing_time", "min"): ("curing_time", "hr", 1 / 60),
    ("curing_time", "s"): ("curing_time", "hr", 1 / 3600),
    ("viscosity", "cP"): ("viscosity", "Pa.s", 1 / 1000),
    ("carbon_emission", "g_co2"): ("carbon_emission", "kg_co2", 1 / 1000),
}


def harmonize_units(df: pd.DataFrame, unit_column: str = "unit") -> pd.DataFrame:
    """
    Convert rows to canonical units using (property_name, unit) -> (canonical_name,
    canonical_unit, factor) lookups. Expects a long/tidy DataFrame with columns
    ['property_name', 'property_value', unit_column]. Rows whose unit already
    matches the canonical unit (or that aren't in the map) pass through unchanged.
    """
    out = df.copy()
    for i, row in out.iterrows():
        key = (row.get("property_name"), row.get(unit_column))
        if key in UNIT_CONVERSIONS:
            canonical_name, canonical_unit, factor = UNIT_CONVERSIONS[key]
            out.at[i, "property_name"] = canonical_name
            out.at[i, "property_value"] = row["property_value"] * factor
            out.at[i, unit_column] = canonical_unit
    return out


# ---------------------------------------------------------------------------
# 3. Missing-value & outlier handling
# ---------------------------------------------------------------------------

def handle_missing_and_outliers(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    outlier_method: str = "iqr",
    impute_method: str = "median",
    iqr_k: float = 1.5,
    nonnegative: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (cleaned_df, report). report includes counts needed for the
    'Missing-value rate' and 'Outlier reduction rate' performance metrics.
    """
    df = df.copy()
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    n_total_cells = len(df) * max(len(numeric_cols), 1)
    n_missing_before = df[numeric_cols].isna().sum().sum() if numeric_cols else 0

    outliers_removed = 0
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")

        if outlier_method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            is_outlier = (series < lower) | (series > upper)
        else:
            z = (series - series.mean()) / series.std(ddof=0)
            is_outlier = z.abs() > 3

        if nonnegative:
            # Tracked properties (strength, viscosity, carbon emission, curing time) are
            # physically non-negative, so a negative reading is invalid even when it sits
            # inside the statistical bounds.
            is_outlier = is_outlier | (series < 0)

        outliers_removed += int(is_outlier.sum())
        series[is_outlier] = np.nan  # treat outliers as missing, then impute

        if impute_method == "median":
            fill_value = series.median()
        elif impute_method == "mean":
            fill_value = series.mean()
        else:
            fill_value = 0.0
        df[col] = series.fillna(fill_value)

    n_missing_after = df[numeric_cols].isna().sum().sum() if numeric_cols else 0

    report = {
        "n_rows": len(df),
        "n_numeric_cells": n_total_cells,
        "missing_value_rate": (n_missing_before / n_total_cells) if n_total_cells else 0.0,
        "outliers_detected": outliers_removed,
        "outlier_reduction_rate": 1.0 if n_missing_after == 0 else 1 - (n_missing_after / n_total_cells),
    }
    return df, report


# ---------------------------------------------------------------------------
# 4. Normalization
# ---------------------------------------------------------------------------

def normalize(df: pd.DataFrame, cols: Iterable[str], method: str = "zscore") -> tuple[pd.DataFrame, dict]:
    """Normalize numeric columns; returns the transformed df plus scaler params
    so the transform can be inverted later (e.g. after model prediction)."""
    df = df.copy()
    params: dict[str, dict] = {}
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if method == "zscore":
            mu, sigma = series.mean(), series.std(ddof=0) or 1.0
            df[col] = (series - mu) / sigma
            params[col] = {"method": "zscore", "mean": mu, "std": sigma}
        elif method == "minmax":
            lo, hi = series.min(), series.max()
            span = (hi - lo) or 1.0
            df[col] = (series - lo) / span
            params[col] = {"method": "minmax", "min": lo, "max": hi}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    return df, params


def denormalize(values: np.ndarray, params: dict) -> np.ndarray:
    if params["method"] == "zscore":
        return values * params["std"] + params["mean"]
    if params["method"] == "minmax":
        return values * (params["max"] - params["min"]) + params["min"]
    raise ValueError(f"Unknown normalization method: {params['method']}")


# ---------------------------------------------------------------------------
# 5. Canonical schema mapping
# ---------------------------------------------------------------------------

def unify_to_schema(df: pd.DataFrame, source_type: str = "experimental") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reshape an arbitrary wide raw DataFrame into (materials_df, properties_df)
    matching schema.Material / schema.MaterialProperty.

    Expected wide input columns (any subset, missing ones become NaN/None):
        name, resin_voc_ratio, sio2_na2o_ratio, additives (dict or JSON str),
        curing_method, <one column per property in FIVE_MAJOR_PROPERTIES>
    """
    df = df.copy()
    if "name" not in df.columns:
        df["name"] = [f"{source_type}_{i}" for i in range(len(df))]

    material_cols = ["name", "resin_voc_ratio", "sio2_na2o_ratio", "additives", "curing_method"]
    for c in material_cols:
        if c not in df.columns:
            df[c] = None

    materials_df = df[material_cols].copy()
    materials_df["source_type"] = source_type
    materials_df["material_row_id"] = range(len(materials_df))

    long_records = []
    for _, row in df.iterrows():
        for prop in FIVE_MAJOR_PROPERTIES:
            if prop in df.columns and pd.notna(row[prop]):
                long_records.append(
                    {
                        "material_row_id": row.name if "material_row_id" not in df.columns else row["material_row_id"],
                        "property_name": prop,
                        "property_value": float(row[prop]),
                        "unit": None,
                        "is_synthetic": source_type == "synthetic",
                    }
                )
    properties_df = pd.DataFrame(
        long_records,
        columns=["material_row_id", "property_name", "property_value", "unit", "is_synthetic"],
    )
    return materials_df, properties_df


def persist(materials_df: pd.DataFrame, properties_df: pd.DataFrame, session) -> int:
    """Write unified DataFrames into the SQL schema. Returns count of materials written.

    Idempotent: a material whose (name, source_type) is already in the database is
    skipped, so re-running ingestion never duplicates samples (duplicates would leak
    identical rows into both the train and test splits downstream)."""
    from schema import Material, MaterialProperty  # local import avoids circularity in scripts

    existing = {(m.name, m.source_type) for m in session.query(Material.name, Material.source_type)}

    row_id_to_pk = {}
    for _, row in materials_df.iterrows():
        if (row["name"], row["source_type"]) in existing:
            continue
        existing.add((row["name"], row["source_type"]))
        mat = Material(
            name=row["name"],
            source_type=row["source_type"],
            resin_voc_ratio=row.get("resin_voc_ratio"),
            sio2_na2o_ratio=row.get("sio2_na2o_ratio"),
            additives=row.get("additives") if isinstance(row.get("additives"), dict) else None,
            curing_method=row.get("curing_method"),
        )
        session.add(mat)
        session.flush()  # assign mat.id
        row_id_to_pk[row["material_row_id"]] = mat.id

    for _, row in properties_df.iterrows():
        if row["material_row_id"] not in row_id_to_pk:  # material skipped as a duplicate
            continue
        session.add(
            MaterialProperty(
                material_id=row_id_to_pk[row["material_row_id"]],
                property_name=row["property_name"],
                property_value=row["property_value"],
                unit=row["unit"],
                is_synthetic=bool(row["is_synthetic"]),
            )
        )
    session.commit()
    return len(row_id_to_pk)


# ---------------------------------------------------------------------------
# 6. Quality metrics (maps directly to the Performance Metrics column)
# ---------------------------------------------------------------------------

def compute_quality_metrics(raw_df: pd.DataFrame, clean_df: pd.DataFrame, numeric_cols: Iterable[str]) -> dict:
    numeric_cols = [c for c in numeric_cols if c in raw_df.columns]
    n_total_cells = len(raw_df) * max(len(numeric_cols), 1)
    n_missing = raw_df[numeric_cols].isna().sum().sum() if numeric_cols else 0

    expected_cols = {"name", "resin_voc_ratio", "sio2_na2o_ratio", "curing_method", *numeric_cols}
    schema_consistency = len(expected_cols & set(raw_df.columns)) / len(expected_cols) if expected_cols else 1.0

    return {
        "data_completeness": 1 - (n_missing / n_total_cells) if n_total_cells else 1.0,
        "missing_value_rate": (n_missing / n_total_cells) if n_total_cells else 0.0,
        "schema_consistency": schema_consistency,
        "n_raw_records": len(raw_df),
        "n_clean_records": len(clean_df),
    }
