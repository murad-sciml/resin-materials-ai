"""
Loads the three simulated raw sources (sample_data/) and pushes each through
the real ETL pipeline (etl.py + schema.py) -- renaming fields and converting
units to canonical form first, exactly like a production ingestion job would.

Run: python ingest_simulated_sources.py
"""
import json

import numpy as np
import pandas as pd

import etl
from schema import FIVE_MAJOR_PROPERTIES, get_engine, get_session, init_db

DATA_DIR = "sample_data"


def load_experimental() -> pd.DataFrame:
    df = pd.read_csv(f"{DATA_DIR}/experimental_raw.csv")
    df = df.rename(
        columns={
            "sample_id": "name",
            "compressive_strength_MPa": "compressive_strength",
            "flexural_strength_MPa": "flexural_strength",
            "viscosity_Pa_s": "viscosity",
            "curing_time_hr": "curing_time",
            "carbon_emission_kgCO2": "carbon_emission",
        }
    )
    df["additives"] = df["additives"].apply(
        lambda s: {} if pd.isna(s) else {a: True for a in str(s).split("|") if a}
    )
    return df


def load_literature() -> pd.DataFrame:
    with open(f"{DATA_DIR}/literature_raw.json") as f:
        records = json.load(f)
    df = pd.json_normalize(records)
    df = df.rename(
        columns={
            "paper_id": "name",
            "SiO2_Na2O_molar_ratio": "sio2_na2o_ratio",
            "cure_method": "curing_method",
        }
    )
    # unit harmonization: percentage -> fraction, cP -> Pa.s, minutes -> hours, grams -> kilograms
    df["resin_voc_ratio"] = df["VOC_ratio_pct"] / 100.0
    if "viscosity_cP" in df.columns:
        df["viscosity"] = df["viscosity_cP"] / 1000.0
    if "cure_time_min" in df.columns:
        df["curing_time"] = df["cure_time_min"] / 60.0
    if "CO2_g" in df.columns:
        df["carbon_emission"] = df["CO2_g"] / 1000.0
    df = df.rename(columns={"compressive_strength_MPa": "compressive_strength",
                              "flexural_strength_MPa": "flexural_strength"})
    df["additives"] = df["additive_list"].apply(
        lambda lst: {a: True for a in lst} if isinstance(lst, list) else {}
    )
    return df


def load_process_log() -> pd.DataFrame:
    df = pd.read_excel(f"{DATA_DIR}/process_log_raw.xlsx")
    df = df.rename(
        columns={
            "batch_id": "name",
            "resin_ratio": "resin_voc_ratio",
            "molar_ratio": "sio2_na2o_ratio",
            "curing_process": "curing_method",
            "curing_duration_hr": "curing_time",
            "batch_compressive_MPa": "compressive_strength",
            "batch_flexural_MPa": "flexural_strength",
            "emission_kg": "carbon_emission",
        }
    )
    df["viscosity"] = df["batch_viscosity_cP"] / 1000.0  # cP -> Pa.s
    df["additives"] = [{} for _ in range(len(df))]        # not tracked at the plant level
    return df


def run_pipeline_for_source(raw_df: pd.DataFrame, source_type: str, session) -> dict:
    clean_df, missing_report = etl.handle_missing_and_outliers(raw_df, numeric_cols=FIVE_MAJOR_PROPERTIES)
    materials_df, properties_df = etl.unify_to_schema(clean_df, source_type=source_type)
    n_written = etl.persist(materials_df, properties_df, session)
    quality = etl.compute_quality_metrics(raw_df, clean_df, numeric_cols=FIVE_MAJOR_PROPERTIES)
    quality.update(missing_report)
    quality["materials_written"] = n_written
    return quality


def main():
    engine = init_db(get_engine("sqlite:///sample_data_ingested.db"))
    session = get_session(engine)

    sources = {
        "experimental": load_experimental(),
        "literature": load_literature(),
        "process_log": load_process_log(),
    }

    print(f"{'source':<14}{'rows':>6}{'written':>10}{'completeness':>14}{'missing_rate':>14}{'outliers':>10}")
    for source_type, raw_df in sources.items():
        quality = run_pipeline_for_source(raw_df, source_type, session)
        print(
            f"{source_type:<14}{len(raw_df):>6}{quality['materials_written']:>10}"
            f"{quality['data_completeness']:>14.3f}{quality['missing_value_rate']:>14.3f}"
            f"{quality['outliers_detected']:>10}"
        )

    session.close()
    print("\nAll three heterogeneous sources unified into sample_data_ingested.db")


if __name__ == "__main__":
    main()
