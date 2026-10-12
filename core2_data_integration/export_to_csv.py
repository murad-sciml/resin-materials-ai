"""
Exports the ingested SQLite database to CSV.

Produces three files in --out-dir (default: ./sample_data):
  materials.csv            - one row per material (raw table)
  material_properties.csv  - one row per (material, property) pair (raw table, long format)
  materials_wide.csv       - one row per material, properties pivoted into columns
                              (materials.csv joined with material_properties.csv) --
                              this is usually the one you actually want for analysis/Excel.

Usage:
    python export_to_csv.py --db sample_data_ingested.db --out-dir sample_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def export(db_path: str, out_dir: str) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    materials = pd.read_sql("SELECT * FROM materials", engine)
    properties = pd.read_sql("SELECT * FROM material_properties", engine)

    materials.to_csv(out / "materials.csv", index=False)
    properties.to_csv(out / "material_properties.csv", index=False)
    print(f"wrote {out / 'materials.csv'}            ({len(materials)} rows)")
    print(f"wrote {out / 'material_properties.csv'}  ({len(properties)} rows)")

    if not properties.empty:
        wide_props = properties.pivot_table(
            index="material_id", columns="property_name", values="property_value"
        )
        wide = materials.merge(wide_props, left_on="id", right_index=True, how="left")
        wide.to_csv(out / "materials_wide.csv", index=False)
        print(f"wrote {out / 'materials_wide.csv'}    ({len(wide)} rows)")
    else:
        print("no properties found; skipped materials_wide.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sample_data_ingested.db", help="Path to the SQLite .db file")
    parser.add_argument("--out-dir", default="sample_data", help="Directory to write CSVs into")
    args = parser.parse_args()
    export(args.db, args.out_dir)
