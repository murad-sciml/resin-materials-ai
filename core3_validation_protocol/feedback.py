"""
Structured experimental-feedback interface for Core Function #3.

Captures the actual measured results for a generated protocol once the
recipe has been physically synthesized and tested, plus an optional expert
rating of protocol quality. Feedback records are written back into Core
Function #2's unified database as new 'experimental' Material +
MaterialProperty rows -- so validated recipes become real training data for
Core Functions #1 and #4, closing the loop.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from dataclasses import dataclass, field

import numpy as np

# Reuse Core Function #2's schema/persistence layer rather than duplicating it.
CORE2_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core2_data_integration"))
if CORE2_PATH not in sys.path:
    sys.path.insert(0, CORE2_PATH)

from schema import Material, MaterialProperty, get_engine, get_session, init_db  # noqa: E402

DEFAULT_DB_PATH = f"sqlite:///{CORE2_PATH}/sample_data_ingested.db"


@dataclass
class FeedbackRecord:
    protocol_id: str
    recipe: dict
    measured_properties: dict  # e.g. {"compressive_strength": 31.2, "viscosity": 0.9, ...}
    expert_rating: float | None = None  # 0-1 "expert agreement" on protocol quality
    reproducibility_note: str | None = None
    submitted_at: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat(timespec="seconds"))


def validate_measurements(measured: dict, session, iqr_k: float = 3.0) -> None:
    """Rejects implausible measured values before they reach the training data.

    Feedback bypasses Core Function #2's ETL outlier handling, so it is checked here:
    values must be non-negative and lie within [Q1 - k*IQR, Q3 + k*IQR] of the values
    already stored for that property (k=3 flags only extreme values, e.g. a viscosity
    of 33 Pa.s when existing measurements are ~0.5-1.1 Pa.s). Raises ValueError."""
    problems = []
    for prop, value in measured.items():
        value = float(value)
        if value < 0:
            problems.append(f"{prop}={value} is negative")
            continue
        stored = [
            v for (v,) in session.query(MaterialProperty.property_value)
            .filter(MaterialProperty.property_name == prop, MaterialProperty.property_value.isnot(None))
        ]
        if len(stored) < 10:  # too little history to judge
            continue
        q1, q3 = np.percentile(stored, [25, 75])
        lower, upper = q1 - iqr_k * (q3 - q1), q3 + iqr_k * (q3 - q1)
        if not lower <= value <= upper:
            problems.append(f"{prop}={value} is outside the plausible range [{lower:.3g}, {upper:.3g}]")
    if problems:
        raise ValueError("Feedback rejected: " + "; ".join(problems) + ". Check units and values.")


def submit_feedback(record: FeedbackRecord, db_path: str = DEFAULT_DB_PATH) -> int:
    """Writes the recipe + measured properties back into Core Function #2's
    database as a new 'experimental' Material. Returns the new material's id.
    Raises ValueError if any measured value is implausible (see validate_measurements)."""
    engine = init_db(get_engine(db_path))
    session = get_session(engine)
    try:
        validate_measurements(record.measured_properties, session)
    except ValueError:
        session.close()
        raise

    material = Material(
        name=f"VALIDATED-{record.protocol_id}",
        source_type="experimental",
        resin_voc_ratio=record.recipe.get("resin_voc_ratio"),
        sio2_na2o_ratio=record.recipe.get("sio2_na2o_ratio"),
        additives={a: True for a in record.recipe.get("additives", [])},
        curing_method=record.recipe.get("curing_method"),
    )
    session.add(material)
    session.flush()  # assign material.id

    for prop_name, value in record.measured_properties.items():
        session.add(
            MaterialProperty(
                material_id=material.id,
                property_name=prop_name,
                property_value=float(value),
                unit=None,
                is_synthetic=False,
            )
        )
    # curing_time is tracked as a property too, matching Core Function #2's schema convention
    if "curing_time" not in record.measured_properties and record.recipe.get("curing_time") is not None:
        session.add(
            MaterialProperty(
                material_id=material.id,
                property_name="curing_time",
                property_value=float(record.recipe["curing_time"]),
                unit="hr",
                is_synthetic=False,
            )
        )

    session.commit()
    material_id = material.id
    session.close()
    return material_id


def reproducibility_score(predicted: dict, measured: dict) -> float:
    """
    Reproducibility metric: 1 - mean absolute relative error between the
    model's predicted properties (from Core Function #1) and what was
    actually measured. Clipped to [0, 1]; NaN if there's no overlap.
    """
    common = [k for k in predicted if k in measured and measured[k] != 0]
    if not common:
        return float("nan")
    rel_errors = [abs(predicted[k] - measured[k]) / abs(measured[k]) for k in common]
    return float(np.clip(1 - np.mean(rel_errors), 0.0, 1.0))
