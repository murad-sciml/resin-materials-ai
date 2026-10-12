"""
REST API for Core Function #1 -- AI Reverse Design Core.

Run: python api.py (or: uvicorn api:app --reload --port 8001)
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from data_prep import CURING_METHODS, TARGET_PROPERTIES, load_dataset
from forward_model import ForwardDNN
from pareto_optimization import pareto_front as run_pareto_front
from reverse_design import reverse_design as run_reverse_design

app = FastAPI(title="AI Reverse Design Core API")

DATA_CSV = "sample_data/materials_wide_with_cost.csv"
CONTINUOUS_BOUNDS = [(0.05, 0.4), (1.0, 3.5), (1.0, 48.0)]  # resin_voc_ratio, sio2_na2o_ratio, curing_time

state: dict = {"model": None, "additive_vocab": None, "metrics": None}


def _ensure_trained():
    if state["model"] is None:
        train_model()


@app.post("/train")
def train_model(architecture: str = "large"):
    """(Re)trains the forward DNN on Core Function #2's unified data + synthesized cost."""
    X, Y, feature_names, target_names, additive_vocab, methods = load_dataset(DATA_CSV)
    model = ForwardDNN(architecture=architecture)
    metrics = model.fit(X, Y)
    state.update({"model": model, "additive_vocab": additive_vocab, "metrics": metrics, "feature_names": feature_names})
    return {"status": "ok", "metrics": metrics, "meets_r2_target": metrics["r2"] >= 0.88}


class PredictRequest(BaseModel):
    resin_voc_ratio: float
    sio2_na2o_ratio: float
    curing_time: float
    curing_method: str
    additives: list[str] = []


@app.post("/predict")
def predict(req: PredictRequest):
    """Forward prediction: composition + process -> five properties + cost."""
    _ensure_trained()
    model = state["model"]
    additive_vocab = state["additive_vocab"]
    method_onehot = [1.0 if m == req.curing_method else 0.0 for m in CURING_METHODS]
    additive_flags = [1.0 if a in req.additives else 0.0 for a in additive_vocab]
    x = np.array([req.resin_voc_ratio, req.sio2_na2o_ratio, req.curing_time] + method_onehot + additive_flags)
    pred = model.predict(x)[0]
    return {name: float(val) for name, val in zip(TARGET_PROPERTIES, pred)}


class ReverseDesignRequest(BaseModel):
    target_properties: dict[str, float]


@app.post("/reverse_design")
def reverse_design_endpoint(req: ReverseDesignRequest):
    """Inverse design: given target property value(s), find the recipe that
    best achieves them via L-BFGS-B optimization on the trained forward model."""
    _ensure_trained()
    model = state["model"]
    additive_vocab = state["additive_vocab"]
    result = run_reverse_design(model, req.target_properties, TARGET_PROPERTIES, CONTINUOUS_BOUNDS, additive_vocab)
    return result


@app.get("/pareto")
def pareto_endpoint(n_samples: int = 1500):
    """Multi-objective Pareto front: performance (strength) vs. cost vs. carbon emission."""
    _ensure_trained()
    model = state["model"]
    additive_vocab = state["additive_vocab"]
    front = run_pareto_front(model, TARGET_PROPERTIES, CONTINUOUS_BOUNDS, additive_vocab, n_samples=n_samples)
    return {"n_pareto_points": len(front), "pareto_front": front}


@app.get("/health")
def health():
    return {"status": "ok", "trained": state["model"] is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=True)
