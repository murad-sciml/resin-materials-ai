"""
Standard-to-property mapping engine for Core Function #3.

Maps a (possibly loosely named) evaluation item to the correct standard test
method in standards_kb.py -- exact match first, falling back to fuzzy string
matching against the KB's known property names. Reports a confidence score
per mapping, used for the 'Standard mapping accuracy' performance metric.
"""
from __future__ import annotations

import difflib

from standards_kb import STANDARDS_KB, get_standard


def map_property_to_standard(property_name: str, cutoff: float = 0.6) -> dict:
    key = property_name.strip().lower().replace(" ", "_")
    if key in STANDARDS_KB:
        return {"input": property_name, "matched_property": key, "confidence": 1.0, **get_standard(key)}

    candidates = difflib.get_close_matches(key, STANDARDS_KB.keys(), n=1, cutoff=cutoff)
    if candidates:
        matched = candidates[0]
        confidence = round(difflib.SequenceMatcher(None, key, matched).ratio(), 3)
        return {"input": property_name, "matched_property": matched, "confidence": confidence, **get_standard(matched)}

    return {"input": property_name, "matched_property": None, "confidence": 0.0, "error": "no standard found"}


def map_all(property_names: list[str]) -> list[dict]:
    return [map_property_to_standard(p) for p in property_names]


def mapping_accuracy(mappings: list[dict], expected: dict[str, str] | None = None) -> float:
    """
    If `expected` (input property name -> correct KB key) is supplied, computes
    exact-match accuracy against it (use this with a human-labeled test set).
    Otherwise falls back to mean mapping confidence as a proxy metric.
    """
    if not mappings:
        return 1.0
    if expected:
        correct = sum(1 for m in mappings if expected.get(m["input"]) == m["matched_property"])
        return correct / len(mappings)
    return sum(m["confidence"] for m in mappings) / len(mappings)
