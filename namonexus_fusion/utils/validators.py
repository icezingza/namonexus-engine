"""
Input validators for NamoNexus Fusion Engine.
Patent-Pending Technology | NamoNexus Research Team
"""
from __future__ import annotations


def validate_score(value: float, strict: bool = True) -> float:
    """Validate that score is in [0, 1]."""
    if not (0.0 <= value <= 1.0):
        if strict:
            raise ValueError(f"score must be in [0, 1], got {value}")
        return float(max(0.0, min(1.0, value)))
    return float(value)


def validate_confidence(value: float, strict: bool = True) -> float:
    """Validate that confidence is in [0, 1]."""
    if not (0.0 <= value <= 1.0):
        if strict:
            raise ValueError(f"confidence must be in [0, 1], got {value}")
        return float(max(0.0, min(1.0, value)))
    return float(value)
