"""
NamoNexus Fusion Engine
=======================
Patent-Pending Multimodal Bayesian Fusion Technology.

Quick Start
-----------
from namonexus_fusion import NamoNexusEngine

engine = NamoNexusEngine()
engine.update(0.85, 0.70, "text")
engine.update(0.25, 0.90, "voice")
engine.update(0.60, 0.85, "face")

print(engine.fused_score)        # → float in (0, 1)
print(engine.risk_level)       # → "low" | "medium" | "high" | "critical"

report = engine.explain()        # GDPR / PDPA audit trail
print(report.summary_text)

© 2025 NamoNexus Research Team. All rights reserved.
Patent Pending — Proprietary & Confidential.
"""

from .engine import NamoNexusEngine
from .core.golden_bayesian import GoldenBayesianFusion, FusionState
from .core.temporal_golden_fusion import TemporalGoldenFusion
from .core.phase2_fusion import Phase2GoldenFusion
from .core.phase3_fusion import Phase3GoldenFusion
from .core.phase4_fusion import Phase4GoldenFusion
from .core.constants import GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL, ENGINE_VERSION
from .config.settings import FusionConfig
from .core.explainability import ShapleyExplainer, ExplanationConfig, ExplanationReport
from .core.hierarchical_bayes import PopulationModel, LocalModel, FederatedAggregator, HierarchicalConfig

# Additional imports for convenience
from .core.golden_bayesian import GoldenBayesianFusion as BayesianFusion
from .core.temporal_golden_fusion import TemporalGoldenFusion as TemporalBayesianFusion
from .core.phase2_fusion import Phase2GoldenFusion as Phase2BayesianFusion
from .core.phase3_fusion import Phase3GoldenFusion as Phase3BayesianFusion
__version__ = "4.0.0"
__author__  = "NamoNexus Research Team"
__license__ = "Proprietary"

__all__ = [
    # Recommended entry point
    "NamoNexusEngine",
    # Individual phase engines
    "GoldenBayesianFusion",
    "TemporalGoldenFusion",
    "Phase2GoldenFusion",
    "Phase3GoldenFusion",
    "Phase4GoldenFusion",
    # Config + constants
    "ShapleyExplainer",
    "ExplanationConfig",
    "ExplanationReport",
    "PopulationModel",
    "LocalModel",
    "FederatedAggregator",
    "HierarchicalConfig",
    "FusionConfig",
    "FusionState",
    "GOLDEN_RATIO",
    "GOLDEN_RATIO_RECIPROCAL",
    "ENGINE_VERSION",
]
