"""
REST API for Core Function #2 — Data Integration & Standardization.

Run locally with:
    uvicorn api:app --reload --port 8002
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

import augmentation as aug
import etl
from schema import FIVE_MAJOR_PROPERTIES, IngestLog, MaterialProperty, get_session, init_db

app = FastAPI(title="Data Integration & Standardization API")
engine = init_db()


@app.post("/ingest")
async def ingest(file: UploadFile, source_type: str = "experimental"):
    """Upload a raw csv/json/xlsx file; runs the full ETL pipeline and stores
    canonical records. Returns the data-quality metrics for this batch."""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    raw_df = etl.load_raw(tmp_path)
    clean_df, missing_report = etl.handle_missing_and_outliers(
        raw_df, numeric_cols=FIVE_MAJOR_PROPERTIES + ["resin_voc_ratio", "sio2_na2o_ratio"]
    )
    materials_df, properties_df = etl.unify_to_schema(clean_df, source_type=source_type)

    session = get_session(engine)
    n_written = etl.persist(materials_df, properties_df, session)

    quality = etl.compute_quality_metrics(raw_df, clean_df, numeric_cols=FIVE_MAJOR_PROPERTIES)
    quality.update(missing_report)

    session.add(
        IngestLog(
            source_format=suffix.lstrip("."),
            source_type=source_type,
            record_count=n_written,
            completeness=quality["data_completeness"],
            missing_value_rate=quality["missing_value_rate"],
            outlier_reduction_rate=missing_report["outlier_reduction_rate"],
            schema_consistency=quality["schema_consistency"],
        )
    )
    session.commit()
    session.close()

    return {"status": "ok", "materials_written": n_written, "quality_metrics": quality}


class AugmentRequest(BaseModel):
    property_names: list[str] = FIVE_MAJOR_PROPERTIES
    n_samples: int = 50
    epochs: int = 800


@app.post("/augment")
def augment(req: AugmentRequest):
    """Train the lightweight GAN on existing property records and generate
    synthetic samples, flagged is_synthetic=True. Returns distribution-similarity
    metrics so a human can judge augmentation quality before trusting the data."""
    session = get_session(engine)
    rows = (
        session.query(MaterialProperty)
        .filter(MaterialProperty.property_name.in_(req.property_names))
        .filter(MaterialProperty.is_synthetic.is_(False))
        .all()
    )
    session.close()

    df = pd.DataFrame(
        [{"material_id": r.material_id, "property_name": r.property_name, "value": r.property_value} for r in rows]
    )
    if df.empty:
        return {"status": "error", "message": "No real data available to train the GAN yet."}

    wide = df.pivot_table(index="material_id", columns="property_name", values="value").dropna()
    if len(wide) < 10:
        return {"status": "error", "message": f"Need >=10 complete records, found {len(wide)}."}

    real = wide[req.property_names].to_numpy()
    synthetic = aug.generate_synthetic(real, n_samples=req.n_samples, epochs=req.epochs)
    similarity = aug.distribution_similarity(real, synthetic, feature_names=req.property_names)

    return {
        "status": "ok",
        "n_real_records": len(real),
        "n_synthetic_generated": req.n_samples,
        "distribution_similarity": similarity,
        "synthetic_preview": synthetic[:5].tolist(),
    }


@app.get("/metrics")
def latest_metrics(limit: int = 5):
    session = get_session(engine)
    logs = session.query(IngestLog).order_by(IngestLog.ingested_at.desc()).limit(limit).all()
    session.close()
    return {
        "recent_ingests": [
            {
                "ingested_at": str(l.ingested_at),
                "source_type": l.source_type,
                "record_count": l.record_count,
                "completeness": l.completeness,
                "missing_value_rate": l.missing_value_rate,
                "outlier_reduction_rate": l.outlier_reduction_rate,
                "schema_consistency": l.schema_consistency,
            }
            for l in logs
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Lets you run this file directly (e.g. via an IDE's "Run" button) instead
    # of typing `uvicorn api:app --reload --port 8002` in a terminal.
    uvicorn.run("api:app", host="127.0.0.1", port=8002, reload=True)
