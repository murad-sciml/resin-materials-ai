"""
REST API for Core Function #3 -- Validation Protocol Engine.

Run: python api.py (or: uvicorn api:app --reload --port 8003)
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from feedback import FeedbackRecord, reproducibility_score, submit_feedback
from protocol_generator import DEFAULT_TARGET_PROPERTIES, generate_protocol
from standards_mapper import map_all, mapping_accuracy

app = FastAPI(title="Validation Protocol Engine API")

_protocol_store: dict[str, dict] = {}  # protocol_id -> protocol dict (in-memory; swap for a DB in production)


class RecipeIn(BaseModel):
    resin_voc_ratio: float
    sio2_na2o_ratio: float
    curing_time: float
    curing_method: str
    additives: list[str] = []


class GenerateProtocolRequest(BaseModel):
    recipe: RecipeIn
    target_properties: list[str] = DEFAULT_TARGET_PROPERTIES
    predicted_properties: dict[str, float] = {}  # optional: Core Function #1's predictions, for later reproducibility scoring


@app.post("/generate_protocol")
def generate_protocol_endpoint(req: GenerateProtocolRequest):
    protocol = generate_protocol(req.recipe.model_dump(), req.target_properties)
    record = protocol.__dict__.copy()
    record["predicted_properties"] = req.predicted_properties
    _protocol_store[protocol.protocol_id] = record
    return record


@app.get("/map_standards")
def map_standards_endpoint(properties: str):
    """properties: comma-separated list, e.g. ?properties=compressive_strength,viscosity"""
    prop_list = [p.strip() for p in properties.split(",") if p.strip()]
    mappings = map_all(prop_list)
    return {"mappings": mappings, "mean_confidence": mapping_accuracy(mappings)}


class FeedbackIn(BaseModel):
    protocol_id: str
    recipe: RecipeIn
    measured_properties: dict[str, float]
    expert_rating: float | None = None
    reproducibility_note: str | None = None


@app.post("/submit_feedback")
def submit_feedback_endpoint(req: FeedbackIn):
    record = FeedbackRecord(
        protocol_id=req.protocol_id,
        recipe=req.recipe.model_dump(),
        measured_properties=req.measured_properties,
        expert_rating=req.expert_rating,
        reproducibility_note=req.reproducibility_note,
    )
    material_id = submit_feedback(record)

    stored = _protocol_store.get(req.protocol_id, {})
    predicted = stored.get("predicted_properties", {})
    repro = reproducibility_score(predicted, req.measured_properties) if predicted else None

    return {"status": "ok", "material_id": material_id, "reproducibility_score": repro}


@app.get("/health")
def health():
    return {"status": "ok", "protocols_generated": len(_protocol_store)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8003, reload=True)
