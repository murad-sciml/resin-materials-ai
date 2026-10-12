"""
Generates simulated RAW data for the eco-friendly-binder project, mimicking
three real-world sources that a materials program would actually have:

  1. experimental_raw.csv   - in-house lab measurements (fairly clean, canonical units)
  2. literature_raw.json    - digitized published papers (different field names,
                               percentages instead of fractions, cP instead of Pa.s,
                               minutes instead of hours, grams instead of kilograms)
  3. process_log_raw.xlsx   - plant/process-log sensor data (different field names,
                               sensor dropout -> missing values, occasional sensor
                               glitches -> outliers)

Underlying values follow a plausible physical relationship (higher SiO2/Na2O
ratio and lower VOC content -> higher strength, roughly) plus noise, so the
data is realistic enough to sanity-check both the ETL pipeline and later
modeling work — not just random numbers.

Run: python generate_simulated_data.py
Outputs written to ./sample_data/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "sample_data"
OUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(7)

CURING_METHODS = ["ambient", "steam", "autoclave"]
ADDITIVE_POOL = ["fly_ash", "silica_fume", "slag", "fiber_reinforcement", "plasticizer"]


def true_relationships(voc_ratio: np.ndarray, sio2_na2o: np.ndarray, curing_hr: np.ndarray):
    """Rough, plausible physical relationships used to generate correlated
    'ground truth' properties before adding measurement noise."""
    linear = 20 + 14 * (sio2_na2o - 1.0) / 2.5 - 35 * (voc_ratio - 0.05) / 0.35 + 0.12 * np.minimum(curing_hr, 36)
    # Smooth floor (softplus) keeps strength physically non-negative: ~linear above
    # ~10 MPa, tapering toward 0 MPa for weak, high-VOC / low-ratio recipes.
    compressive = 4.0 * np.logaddexp(0.0, linear / 4.0)
    flexural = 3 + 0.18 * compressive
    viscosity = 500 + 400 * (sio2_na2o - 1.0) / 2.5 + 200 * voc_ratio / 0.4
    carbon_emission = 180 - 90 * (sio2_na2o - 1.0) / 2.5 + 60 * voc_ratio / 0.4
    return compressive, flexural, viscosity, carbon_emission


def sample_additives(n: int) -> list[list[str]]:
    return [list(rng.choice(ADDITIVE_POOL, size=rng.integers(0, 3), replace=False)) for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. Experimental data (CSV) -- in-house lab, mostly canonical units
# ---------------------------------------------------------------------------
def generate_experimental(n: int = 80) -> pd.DataFrame:
    voc = rng.uniform(0.05, 0.40, n)
    ratio = rng.uniform(1.0, 3.5, n)
    curing_hr = rng.uniform(4, 36, n)
    compressive, flexural, viscosity, carbon = true_relationships(voc, ratio, curing_hr)

    df = pd.DataFrame(
        {
            "sample_id": [f"EXP-{i:03d}" for i in range(n)],
            "resin_voc_ratio": voc.round(3),
            "sio2_na2o_ratio": ratio.round(2),
            "additives": ["|".join(a) for a in sample_additives(n)],
            "curing_method": rng.choice(CURING_METHODS, n),
            "compressive_strength_MPa": (compressive + rng.normal(0, 1.5, n)).round(2),
            "flexural_strength_MPa": (flexural + rng.normal(0, 0.4, n)).round(2),
            "viscosity_Pa_s": ((viscosity + rng.normal(0, 30, n)) / 1000).round(3),
            "curing_time_hr": curing_hr.round(2),
            "carbon_emission_kgCO2": (carbon + rng.normal(0, 8, n)).round(2),
            "notes": "",
        }
    )

    # realistic messiness: dropout + a couple of transcription outliers
    for col in ["compressive_strength_MPa", "viscosity_Pa_s"]:
        missing_idx = rng.choice(n, size=max(1, n // 20), replace=False)
        df.loc[missing_idx, col] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "compressive_strength_MPa"] = [180.0, -5.0]  # bad readings
    df.loc[5, "notes"] = "recast after mixer fault"
    return df


# ---------------------------------------------------------------------------
# 2. Literature data (JSON) -- digitized from published papers, different
#    field names AND different units than the canonical schema
# ---------------------------------------------------------------------------
def generate_literature(n: int = 40) -> list[dict]:
    voc = rng.uniform(0.05, 0.40, n)
    ratio = rng.uniform(1.0, 3.5, n)
    curing_hr = rng.uniform(4, 48, n)
    compressive, flexural, viscosity, carbon = true_relationships(voc, ratio, curing_hr)

    records = []
    for i in range(n):
        record = {
            "paper_id": f"LIT-{2015 + rng.integers(0, 10)}-{i:03d}",
            "doi": f"10.1000/binder.{rng.integers(1000,9999)}",
            "VOC_ratio_pct": round(voc[i] * 100, 1),                    # percentage, not fraction
            "SiO2_Na2O_molar_ratio": round(ratio[i], 2),
            "additive_list": list(rng.choice(ADDITIVE_POOL, size=int(rng.integers(0, 3)), replace=False)),
            "cure_method": rng.choice(CURING_METHODS),
            "compressive_strength_MPa": round(float(compressive[i] + rng.normal(0, 2.0)), 2),
            "flexural_strength_MPa": round(float(flexural[i] + rng.normal(0, 0.6)), 2),
            "viscosity_cP": round(float(viscosity[i] + rng.normal(0, 40)), 1),   # centipoise, not Pa.s
            "cure_time_min": round(float(curing_hr[i] * 60 + rng.normal(0, 15)), 0),  # minutes, not hours
            "CO2_g": round(float((carbon[i] + rng.normal(0, 10)) * 1000), 0),          # grams, not kg
        }
        records.append(record)

    # literature reporting gaps: some papers don't report every property
    for rec in rng.choice(records, size=max(1, n // 8), replace=False):
        rec.pop(rng.choice(["flexural_strength_MPa", "viscosity_cP"]))
    return records


# ---------------------------------------------------------------------------
# 3. Process-log data (Excel) -- plant sensor/batch logs, different field
#    names, sensor dropout, and occasional sensor-glitch outliers
# ---------------------------------------------------------------------------
def generate_process_log(n: int = 60) -> pd.DataFrame:
    voc = rng.uniform(0.05, 0.40, n)
    ratio = rng.uniform(1.0, 3.5, n)
    curing_hr = rng.uniform(2, 30, n)
    compressive, flexural, viscosity, carbon = true_relationships(voc, ratio, curing_hr)

    timestamps = pd.date_range("2025-01-01", periods=n, freq="8h")
    df = pd.DataFrame(
        {
            "batch_id": [f"BATCH-{i:04d}" for i in range(n)],
            "timestamp": timestamps,
            "resin_ratio": voc.round(3),
            "molar_ratio": ratio.round(2),
            "curing_process": rng.choice(CURING_METHODS, n),
            "curing_duration_hr": curing_hr.round(2),
            "batch_compressive_MPa": (compressive + rng.normal(0, 2.5, n)).round(2),
            "batch_flexural_MPa": (flexural + rng.normal(0, 0.5, n)).round(2),
            "batch_viscosity_cP": (viscosity + rng.normal(0, 50, n)).round(1),
            "emission_kg": (carbon + rng.normal(0, 12, n)).round(2),
            "operator": rng.choice(["A. Kim", "J. Park", "S. Lee", "M. Cho"], n),
            "machine_id": rng.choice(["MX-1", "MX-2", "MX-3"], n),
        }
    )

    # sensor dropout
    for col in ["batch_compressive_MPa", "batch_viscosity_cP", "emission_kg"]:
        dropout_idx = rng.choice(n, size=max(1, n // 15), replace=False)
        df.loc[dropout_idx, col] = np.nan
    # sensor glitches (outliers)
    glitch_idx = rng.choice(n, size=2, replace=False)
    df.loc[glitch_idx, "batch_viscosity_cP"] = [9999.0, -100.0]
    return df


def main():
    exp_df = generate_experimental()
    exp_path = OUT_DIR / "experimental_raw.csv"
    exp_df.to_csv(exp_path, index=False)
    print(f"wrote {exp_path}  ({len(exp_df)} rows)")

    lit_records = generate_literature()
    lit_path = OUT_DIR / "literature_raw.json"
    with open(lit_path, "w") as f:
        json.dump(lit_records, f, indent=2)
    print(f"wrote {lit_path}  ({len(lit_records)} records)")

    proc_df = generate_process_log()
    proc_path = OUT_DIR / "process_log_raw.xlsx"
    proc_df.to_excel(proc_path, index=False)
    print(f"wrote {proc_path}  ({len(proc_df)} rows)")


if __name__ == "__main__":
    main()
