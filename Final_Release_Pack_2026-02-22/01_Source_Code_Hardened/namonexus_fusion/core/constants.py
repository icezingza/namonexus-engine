"""
Core constants for NamoNexus Fusion Engine.
Patent-Pending Technology | NamoNexus Research Team
"""

import math

# ─── Golden Ratio ──────────────────────────────────────────────────────────────
GOLDEN_RATIO: float = (1.0 + math.sqrt(5.0)) / 2.0          # φ ≈ 1.6180339887
GOLDEN_RATIO_RECIPROCAL: float = 1.0 / GOLDEN_RATIO          # 1/φ ≈ 0.6180339887

# ─── Engine versioning ─────────────────────────────────────────────────────────
ENGINE_VERSION: str = "4.0.0-phase2"

# ─── Bayesian defaults ─────────────────────────────────────────────────────────
DEFAULT_PRIOR_STRENGTH: float = 1.0        # s in α₀ = φ·s
DEFAULT_MAX_TRIALS: int = 100
DEFAULT_CONFIDENCE_SCALE: float = 10.0

# ─── Risk thresholds (fused_score) ─────────────────────────────────────────────
RISK_LOW_THRESHOLD: float = 0.35
RISK_MEDIUM_THRESHOLD: float = 0.60
RISK_HIGH_THRESHOLD: float = 0.80

# ─── Validation Thresholds ─────────────────────────────────────────────────────
LOW_CONF_THRESHOLD: float = 0.15  # Adjusted from 0.60 to 0.15 (Issue 4)
