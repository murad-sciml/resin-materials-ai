"""
REST API for Core Function #4 — Software-Based Autonomous Experiment Module.

Run locally with:
    uvicorn api:app --reload --port 8004
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retrain_pipeline import RetrainingPipeline

app = FastAPI(title="Autonomous Experiment Module API")

# Example design space: (resin_voc_ratio, sio2_na2o_ratio, curing_time_hr).
# Replace with the real bounds for the material system being optimized.
DESIGN_VAR_NAMES = ["resin_voc_ratio", "sio2_na2o_ratio", "curing_time_hr"]
BOUNDS = [(0.05, 0.4), (1.0, 3.5), (1.0, 48.0)]

pipeline = RetrainingPipeline(bounds=BOUNDS, minimize=False, design_var_names=DESIGN_VAR_NAMES)


class FeedbackIn(BaseModel):
    design_variables: list[float] = Field(..., description="Values in the order of DESIGN_VAR_NAMES")
    measured_value: float = Field(..., description="Measured target property (e.g. compressive strength)")


@app.post("/experiments/feedback")
def submit_feedback(payload: FeedbackIn):
    """Submit one completed experiment's result; triggers automatic retraining."""
    metrics = pipeline.submit_feedback(payload.design_variables, payload.measured_value)
    return {"status": "ok", "fit_metrics": metrics}


@app.get("/experiments/next")
def next_doe(n_bo: int = 3, n_uncertain: int = 2):
    """Get the next-priority Design of Experiments recommendation."""
    return pipeline.next_doe(n_bo=n_bo, n_uncertain=n_uncertain)


@app.get("/experiments/metrics")
def convergence_metrics():
    """R^2 / RMSE trend over successive retraining rounds (convergence speed)."""
    return {"history": pipeline.convergence_metrics()}


@app.get("/health")
def health():
    return {"status": "ok", "design_variables": DESIGN_VAR_NAMES, "bounds": BOUNDS}


if __name__ == "__main__":
    import uvicorn

    # Lets you run this file directly (e.g. via an IDE's "Run" button) instead
    # of typing `uvicorn api:app --reload --port 8004` in a terminal.
    uvicorn.run("api:app", host="127.0.0.1", port=8004, reload=True)
