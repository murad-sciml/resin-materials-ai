"""
Rule-based synthesis + testing protocol generator for Core Function #3.

Given a recommended recipe (composition + process parameters, typically from
Core Function #1's reverse-design or Pareto output) and the set of target
properties to validate, generates a step-by-step protocol referencing the
correct standard test method for each property.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from standards_kb import get_standard

DEFAULT_TARGET_PROPERTIES = ["compressive_strength", "flexural_strength", "viscosity", "carbon_emission", "cost"]


@dataclass
class Protocol:
    protocol_id: str
    generated_at: str
    recipe: dict
    target_properties: list[str]
    synthesis_steps: list[str]
    evaluation_steps: list[dict]
    completeness_score: float
    missing_standards: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"PROTOCOL {self.protocol_id}  (generated {self.generated_at})",
            f"Recipe: {self.recipe}",
            "",
            "SYNTHESIS",
        ]
        lines += [f"  {s}" for s in self.synthesis_steps]
        lines += ["", "EVALUATION (standard-mapped)"]
        for i, ev in enumerate(self.evaluation_steps, 1):
            lines.append(f"  {i}. {ev['property']} -- {ev['standard']} ({ev['standard_body']})")
            lines.append(f"     {ev['title']}")
            lines.append(f"     Equipment: {', '.join(ev['equipment'])}")
            lines.append(f"     Procedure: {ev['procedure_summary']}")
        if self.missing_standards:
            lines += ["", f"WARNING: no standard mapped for: {', '.join(self.missing_standards)}"]
        lines += ["", f"Protocol completeness: {self.completeness_score:.0%}"]
        return "\n".join(lines)


def generate_protocol(recipe: dict, target_properties: list[str] | None = None, protocol_id: str | None = None) -> Protocol:
    """
    recipe: {"resin_voc_ratio":..., "sio2_na2o_ratio":..., "curing_time":...,
             "curing_method":..., "additives": [...]}
    """
    target_properties = target_properties or DEFAULT_TARGET_PROPERTIES
    protocol_id = protocol_id or f"PROT-{int(dt.datetime.utcnow().timestamp())}"

    additives = recipe.get("additives", [])
    synthesis_steps = [
        f"Weigh out resin at a VOC ratio of {recipe['resin_voc_ratio']:.3f} and combine with the "
        f"alkaline activator at a SiO2/Na2O molar ratio of {recipe['sio2_na2o_ratio']:.2f}.",
        f"Incorporate additive(s): {', '.join(additives) if additives else 'none'}.",
        "Mix per standard mixing procedure until homogeneous; cast into the molds required by each "
        "evaluation method below.",
        f"Cure specimens using the '{recipe['curing_method']}' method for {recipe['curing_time']:.1f} "
        f"hours, per ASTM C511 / ISO 9229 conditions.",
    ]

    evaluation_steps = []
    missing_standards = []
    for prop in target_properties:
        try:
            std = get_standard(prop)
            evaluation_steps.append(
                {
                    "property": prop,
                    "standard": std["standard"],
                    "standard_body": std["body"],
                    "title": std["title"],
                    "equipment": std["equipment"],
                    "procedure_summary": std["procedure_summary"],
                }
            )
        except KeyError:
            missing_standards.append(prop)

    completeness_score = (len(evaluation_steps) / len(target_properties)) if target_properties else 1.0

    return Protocol(
        protocol_id=protocol_id,
        generated_at=dt.datetime.utcnow().isoformat(timespec="seconds"),
        recipe=recipe,
        target_properties=target_properties,
        synthesis_steps=synthesis_steps,
        evaluation_steps=evaluation_steps,
        completeness_score=completeness_score,
        missing_standards=missing_standards,
    )
