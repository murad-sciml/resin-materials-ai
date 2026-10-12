"""
Knowledge base mapping each tracked material property to a real standard test
method (ASTM/ISO), for Core Function #3 -- Validation Protocol Engine.

IMPORTANT: these are common, real standard designations used for testing
cementitious/resin-based binder systems, included as a plausible STARTER SET
so the protocol generator has something concrete to work with. The exact
applicable standard depends on your specific material system and regulatory
context -- this table should be reviewed and corrected by a domain expert
before being used to drive real lab protocols.
"""
from __future__ import annotations

STANDARDS_KB: dict[str, dict] = {
    "compressive_strength": {
        "standard": "ASTM C109 / ASTM C39",
        "title": "Standard Test Method for Compressive Strength of Hydraulic Cement Mortars / Concrete Cylinders",
        "body": "ASTM",
        "equipment": ["Compression testing machine", "50 mm cube molds or cylinder molds"],
        "procedure_summary": (
            "Cast specimens, cure per the specified regime, then load to failure in a calibrated "
            "compression machine; report peak load divided by cross-sectional area."
        ),
    },
    "flexural_strength": {
        "standard": "ASTM C348 / ASTM C78",
        "title": "Standard Test Method for Flexural Strength of Hydraulic Cement Mortars / Concrete",
        "body": "ASTM",
        "equipment": ["Three/four-point flexural test fixture", "Prism molds"],
        "procedure_summary": (
            "Cast prism specimens, cure per the specified regime, then load in three- or four-point "
            "bending to failure; compute flexural strength from beam theory."
        ),
    },
    "viscosity": {
        "standard": "ASTM D2196 / ISO 2555",
        "title": "Standard Test Methods for Rheological Properties of Non-Newtonian Materials by Rotational Viscometer",
        "body": "ASTM / ISO",
        "equipment": ["Rotational (Brookfield-type) viscometer", "Temperature-controlled bath"],
        "procedure_summary": (
            "Equilibrate the sample to test temperature, measure viscosity at the specified spindle "
            "speed(s), and report in Pa\u00b7s."
        ),
    },
    "curing_time": {
        "standard": "ASTM C511 / ISO 9229",
        "title": "Standard Specification for Moist Cabinets, Rooms, and Water Storage Tanks Used in Testing Hydraulic Cements and Concretes",
        "body": "ASTM / ISO",
        "equipment": ["Curing chamber/cabinet with controlled temperature & humidity"],
        "procedure_summary": (
            "Cure specimens using the specified method (ambient / steam / autoclave) for the target "
            "duration under controlled conditions before evaluating other properties."
        ),
    },
    "carbon_emission": {
        "standard": "ISO 14040 / ISO 14044",
        "title": "Environmental Management -- Life Cycle Assessment",
        "body": "ISO",
        "equipment": ["LCA software or emission-factor database"],
        "procedure_summary": (
            "Estimate embodied CO2 from raw-material quantities and process energy using standard "
            "emission factors and LCA methodology."
        ),
    },
    "cost": {
        "standard": "Internal costing procedure (no external test standard)",
        "title": "Formulation Unit-Cost Estimation",
        "body": "Internal",
        "equipment": ["Bill-of-materials cost sheet"],
        "procedure_summary": (
            "Estimate unit cost from raw-material quantities, additive costs, and process energy cost "
            "per the internal costing procedure."
        ),
    },
}


def get_standard(property_name: str) -> dict:
    if property_name not in STANDARDS_KB:
        raise KeyError(f"No standard mapping registered for '{property_name}'. Add it to STANDARDS_KB.")
    return STANDARDS_KB[property_name]
