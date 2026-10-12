"""
End-to-end demo of the Data Integration & Standardization pipeline,
independent of the API layer. Run: python example_usage.py
"""
import numpy as np
import pandas as pd

import augmentation as aug
import etl
from schema import FIVE_MAJOR_PROPERTIES, get_engine, get_session, init_db

rng = np.random.default_rng(0)

# 1. Simulate a raw experimental data dump (with messy units, missing values, an outlier)
n = 60
raw = pd.DataFrame(
    {
        "name": [f"binder_{i}" for i in range(n)],
        "resin_voc_ratio": rng.uniform(0.05, 0.4, n),
        "sio2_na2o_ratio": rng.uniform(1.0, 3.5, n),
        "curing_method": rng.choice(["ambient", "steam", "autoclave"], n),
        "compressive_strength": rng.normal(35, 5, n),
        "flexural_strength": rng.normal(6, 1, n),
        "viscosity": rng.normal(800, 100, n),
        "curing_time": rng.normal(24, 3, n),
        "carbon_emission": rng.normal(120, 15, n),
    }
)
raw.loc[3, "compressive_strength"] = np.nan          # missing value
raw.loc[5, "compressive_strength"] = 500.0           # outlier
raw.loc[10, "viscosity"] = np.nan

print("=== 1. ETL: clean + quality metrics ===")
clean, missing_report = etl.handle_missing_and_outliers(raw, numeric_cols=FIVE_MAJOR_PROPERTIES)
quality = etl.compute_quality_metrics(raw, clean, numeric_cols=FIVE_MAJOR_PROPERTIES)
quality.update(missing_report)
for k, v in quality.items():
    print(f"  {k}: {v}")

print("\n=== 2. Unify to canonical schema + persist ===")
materials_df, properties_df = etl.unify_to_schema(clean, source_type="experimental")
engine = init_db(get_engine("sqlite:///:memory:"))
session = get_session(engine)
n_written = etl.persist(materials_df, properties_df, session)
print(f"  materials written: {n_written}, property rows: {len(properties_df)}")

print("\n=== 3. GAN-based augmentation ===")
real = clean[FIVE_MAJOR_PROPERTIES].to_numpy()
synthetic = aug.generate_synthetic(real, n_samples=20, epochs=400)
similarity = aug.distribution_similarity(real, synthetic, feature_names=FIVE_MAJOR_PROPERTIES)
print(f"  mean KS statistic across features: {similarity['mean_ks_statistic']:.3f} (closer to 0 is better)")
for name in FIVE_MAJOR_PROPERTIES:
    print(f"    {name}: KS={similarity[name]['ks_statistic']:.3f}  Wasserstein={similarity[name]['wasserstein_distance']:.3f}")

session.close()
print("\nDone.")
